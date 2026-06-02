import functools as ft
import os
import pickle
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from flax.training.train_state import TrainState

from .ddpg import DDPG, _tree_first_device
from ..env.base import MultiAgentEnv
from ..trainer.data import Rollout
from ..trainer.utils import has_any_nan_or_inf, compute_norm_and_clip
from ..utils.typing import Array, Params


class DDPGLagr(DDPG):
    """Lagrangian 版 DDPG。

    critic 分别学习 reward return 和 cost return：
        Qr(s, a), Qc(s, a)

    actor 最大化拉格朗日目标：
        Qr(s, mu(s)) - lambda * Qc(s, mu(s))

    代码中等价最小化：
        L_actor = mean(-Qr + lambda * Qc)

    lambda 使用 projected gradient 更新：
        lambda <- relu(lambda + lr_lagr * violation)
    """

    def __init__(
            self,
            env: MultiAgentEnv,
            node_dim: int,
            edge_dim: int,
            state_dim: int,
            action_dim: int,
            n_agents: int,
            lagr_init: float = 0.78,
            lr_lagr: float = 1e-7,
            **kwargs
    ):
        super().__init__(env, node_dim, edge_dim, state_dim, action_dim, n_agents, **kwargs)
        self.lagr_init = lagr_init
        self.lr_lagr = lr_lagr
        self.lagr = jnp.ones((n_agents, env.n_cost), dtype=jnp.float32) * lagr_init

    @property
    def config(self) -> dict:
        config = super().config
        config.update({
            "safety_mode": "lagr",
            "lagr_init": self.lagr_init,
            "lr_lagr": self.lr_lagr,
        })
        return config

    def update(self, rollouts: Rollout, iter_index: int) -> dict:
        assert iter_index == self.iter_index
        self.replay.append(rollouts)
        num_devices = int(rollouts.rewards.shape[0])
        can_update = self.replay.length >= max(self.batch_size, self.replay_warmup_transitions)
        if not can_update:
            self.iter_index += 1
            return {
                "replay/size": float(self.replay.length),
                "ddpg/skipped_update": 1.0,
            }

        info_acc = None
        for _ in range(self.updates_per_iter):
            batch = self.replay.sample(self.batch_size, num_devices)
            critic, safety, actor, target_actor, target_critic, target_safety, lagr, info = self.pmap_update(
                self.critic_train_state,
                self.safety_train_state,
                self.actor_train_state,
                self.target_actor_params,
                self.target_critic_params,
                self.target_safety_params,
                self.lagr,
                batch,
            )
            self.critic_train_state = _tree_first_device(critic)
            self.safety_train_state = _tree_first_device(safety)
            self.actor_train_state = _tree_first_device(actor)
            self.target_actor_params = _tree_first_device(target_actor)
            self.target_critic_params = _tree_first_device(target_critic)
            self.target_safety_params = _tree_first_device(target_safety)
            self.lagr = _tree_first_device(lagr)
            info_single = _tree_first_device(info)
            info_acc = info_single if info_acc is None else jtu.tree_map(lambda a, b: a + b, info_acc, info_single)

        info_acc = jtu.tree_map(lambda x: x / self.updates_per_iter, info_acc)
        info_acc.update({
            "replay/size": float(self.replay.length),
            "ddpg/skipped_update": 0.0,
        })
        self.iter_index += 1
        return info_acc

    @ft.partial(jax.pmap, in_axes=(None, None, None, None, None, None, None, None, 0),
                axis_name="n_gpu", static_broadcasted_argnums=(0,))
    def pmap_update(
            self,
            critic_state: TrainState,
            safety_state: TrainState,
            actor_state: TrainState,
            target_actor_params: Params,
            target_critic_params: Params,
            target_safety_params: Params,
            lagr: Array,
            batch: Rollout,
    ):
        critic_state, safety_state, critic_info = self.update_critics(
            critic_state, safety_state, target_actor_params, target_critic_params, target_safety_params, batch
        )
        actor_state, lagr, actor_info = self.update_actor(actor_state, critic_state, safety_state, lagr, batch)
        target_actor_params = optax.incremental_update(actor_state.params, target_actor_params, self.tau)
        target_critic_params = optax.incremental_update(critic_state.params, target_critic_params, self.tau)
        target_safety_params = optax.incremental_update(safety_state.params, target_safety_params, self.tau)
        critic_info.update(actor_info)
        return critic_state, safety_state, actor_state, target_actor_params, target_critic_params, target_safety_params, lagr, critic_info

    @ft.partial(jax.jit, static_argnums=(0,))
    def update_actor(
            self,
            actor_state: TrainState,
            critic_state: TrainState,
            safety_state: TrainState,
            lagr: Array,
            batch: Rollout,
    ):
        zs = batch.zs

        def actor_loss_fn(actor_params):
            actions = self._batched_actor(actor_params, batch.graph, zs)
            q = self._batched_q(critic_state.params, batch.graph, actions, zs).squeeze(-1)
            safety_q = self._batched_safety_q(safety_state.params, batch.graph, actions, zs)
            penalty = (jnp.maximum(lagr, 0.0)[None, :, :] * safety_q).mean(axis=-1)
            loss = (-q + penalty).mean()
            info = {
                "policy/loss": jax.lax.pmean(loss, axis_name="n_gpu"),
                "policy/q": jax.lax.pmean(q.mean(), axis_name="n_gpu"),
                "policy/safety_q": jax.lax.pmean(safety_q.mean(), axis_name="n_gpu"),
                "policy/action_abs": jax.lax.pmean(jnp.abs(actions).mean(), axis_name="n_gpu"),
            }
            return loss, (safety_q, info)

        grad, (safety_q, info) = jax.grad(actor_loss_fn, has_aux=True)(actor_state.params)
        grad_has_nan = jax.lax.pmax(has_any_nan_or_inf(grad).astype(jnp.float32), axis_name="n_gpu")
        grad, grad_norm = compute_norm_and_clip(grad, self.max_grad_norm)
        actor_state = actor_state.apply_gradients(grads=grad)

        # violation 由当前策略下 cost critic 的均值近似。
        violation = jax.lax.pmean(safety_q.mean(axis=0), axis_name="n_gpu")
        lagr = nn.relu(lagr + self.lr_lagr * violation)
        info.update({
            "policy/mean_lagr": jax.lax.pmean(lagr.mean(), axis_name="n_gpu"),
            "policy/has_nan": grad_has_nan,
            "policy/grad_norm": jax.lax.pmean(grad_norm, axis_name="n_gpu"),
        })
        return actor_state, lagr, info

    def save(self, save_dir: str, step: int, params_to_save: dict = None):
        super().save(save_dir, step, params_to_save=params_to_save)
        model_dir = os.path.join(save_dir, str(step))
        pickle.dump(self.lagr, open(os.path.join(model_dir, "lagr.pkl"), "wb"))

    def load(self, load_dir: str, step: int):
        super().load(load_dir, step)
        lagr_path = os.path.join(load_dir, str(step), "lagr.pkl")
        if os.path.exists(lagr_path):
            self.lagr = pickle.load(open(lagr_path, "rb"))
