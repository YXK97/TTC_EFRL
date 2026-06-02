import functools as ft
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from .ddpg import DDPG
from .module.root_finder import RootFinder
from ..env.base import MultiAgentEnv
from ..trainer.data import Rollout
from ..trainer.utils import has_any_nan_or_inf, compute_norm_and_clip
from ..utils.graph import GraphsTuple
from ..utils.typing import Array, Params


class DDPGEFRL(DDPG):
    """EFRL 版 DDPG。

    该版本沿用 DefMARL 的 EFRL 预算变量 z：
        z' = clip((z + reward) / gamma, z_min, z_max)

    critic 不再解释为 reward return，而是拆成：
        Ql(s, a, z): 累计损失，单步损失 l = -reward
        Qh(s, a, z): safety value，对应环境返回的 cost h

    actor 最小化 Q 版 EFRL value：
        V(s, z) = max(max_h Qh_h(s, mu(s,z), z), Ql(s, mu(s,z), z) - z)
    """

    use_ef: bool = True

    def __init__(
            self,
            env: MultiAgentEnv,
            node_dim: int,
            edge_dim: int,
            state_dim: int,
            action_dim: int,
            n_agents: int,
            **kwargs
    ):
        super().__init__(env, node_dim, edge_dim, state_dim, action_dim, n_agents, **kwargs)
        self.root_finder = RootFinder(
            z_min=self.z_min,
            z_max=self.z_max,
            n_agent=self.n_agents,
        )

    @property
    def config(self) -> dict:
        config = super().config
        config.update({"safety_mode": "efrl"})
        return config

    def get_opt_z(self, graph: GraphsTuple, Vh_rnn_state: Array, params: Optional[Params] = None) -> Tuple[Array, Array]:
        params = self.params if params is None else params

        def Vh_fn(z: Array):
            # DDPG 没有单独的 Vh 网络，用当前确定性 actor 诱导：
            #   Vh(s,z) = Qh(s, mu(s,z), z)
            action = self.actor.apply(params["policy"], graph, z)
            Vh = self.safety_critic.apply(params["safety_critic"], graph, action, z)
            return Vh, Vh_rnn_state

        return self.root_finder.get_dec_opt_z(Vh_fn, graph)

    @ft.partial(jax.jit, static_argnums=(0,))
    def update_critics(
            self,
            critic_state: TrainState,
            safety_state: TrainState,
            target_actor_params: Params,
            target_critic_params: Params,
            target_safety_params: Params,
            batch: Rollout,
    ):
        rewards = jnp.nan_to_num(batch.rewards, nan=-1e3, posinf=1e3, neginf=-1e3)
        if rewards.ndim == 1:
            rewards = rewards[:, None].repeat(self.n_agents, axis=1)
        costs = jnp.nan_to_num(batch.costs, nan=1e3, posinf=1e3, neginf=-1e3)
        dones = batch.dones.astype(jnp.float32)
        zs = batch.zs

        # EFRL 的 z dynamics 来自预算递推。
        next_zs = jnp.clip((zs + rewards[:, :, None]) / self.gamma, self.z_min, self.z_max)
        next_actions = self._batched_actor(target_actor_params, batch.next_graph, next_zs)

        # Ql 学累计损失，单步损失 l = -reward。
        next_q = self._batched_q(target_critic_params, batch.next_graph, next_actions, next_zs).squeeze(-1)
        q_target = -rewards + self.gamma * (1.0 - dones[:, None]) * next_q
        q_target = jnp.clip(q_target, -1e4, 1e4)

        # Qh 对齐 compute_dec_efocp_gae 中的 stabilize-avoid DP。
        next_safety_q = self._batched_safety_q(target_safety_params, batch.next_graph, next_actions, next_zs)
        h_disc = jnp.max(costs, axis=-1, keepdims=True)
        safety_target = jnp.maximum(costs, (1.0 - self.gamma) * h_disc + self.gamma * next_safety_q)
        safety_target = jnp.clip(safety_target, -1e4, 1e4)

        def critic_loss_fn(q_params, safety_params):
            q_pred = self._batched_q(q_params, batch.graph, batch.actions, zs).squeeze(-1)
            safety_pred = self._batched_safety_q(safety_params, batch.graph, batch.actions, zs)
            loss_q = optax.l2_loss(q_pred, q_target).mean()
            loss_safety = optax.l2_loss(safety_pred, safety_target).mean()
            info = {
                "critic/loss": jax.lax.pmean(loss_q, axis_name="n_gpu"),
                "critic/loss_safety": jax.lax.pmean(loss_safety, axis_name="n_gpu"),
                "critic/target_q": jax.lax.pmean(q_target.mean(), axis_name="n_gpu"),
                "critic/target_safety": jax.lax.pmean(safety_target.mean(), axis_name="n_gpu"),
            }
            return loss_q + loss_safety, info

        (grad_q, grad_safety), info = jax.grad(critic_loss_fn, argnums=(0, 1), has_aux=True)(
            critic_state.params, safety_state.params
        )
        grad_q_has_nan = jax.lax.pmax(has_any_nan_or_inf(grad_q).astype(jnp.float32), axis_name="n_gpu")
        grad_safety_has_nan = jax.lax.pmax(has_any_nan_or_inf(grad_safety).astype(jnp.float32), axis_name="n_gpu")
        grad_q, grad_q_norm = compute_norm_and_clip(grad_q, self.max_grad_norm)
        grad_safety, grad_safety_norm = compute_norm_and_clip(grad_safety, self.max_grad_norm)
        critic_state = critic_state.apply_gradients(grads=grad_q)
        safety_state = safety_state.apply_gradients(grads=grad_safety)
        info.update({
            "critic/has_nan": grad_q_has_nan,
            "critic/safety_has_nan": grad_safety_has_nan,
            "critic/grad_norm": jax.lax.pmean(grad_q_norm, axis_name="n_gpu"),
            "critic/safety_grad_norm": jax.lax.pmean(grad_safety_norm, axis_name="n_gpu"),
        })
        return critic_state, safety_state, info

    @ft.partial(jax.jit, static_argnums=(0,))
    def update_actor(
            self,
            actor_state: TrainState,
            critic_state: TrainState,
            safety_state: TrainState,
            batch: Rollout,
    ):
        zs = batch.zs

        def actor_loss_fn(actor_params):
            actions = self._batched_actor(actor_params, batch.graph, zs)
            q = self._batched_q(critic_state.params, batch.graph, actions, zs).squeeze(-1)
            safety_q = self._batched_safety_q(safety_state.params, batch.graph, actions, zs)
            z = zs.squeeze(-1)
            max_safety_q = jnp.max(safety_q, axis=-1)
            q_minus_z = q - z
            objective = jnp.maximum(max_safety_q, q_minus_z)
            loss = objective.mean()
            info = {
                "policy/loss": jax.lax.pmean(loss, axis_name="n_gpu"),
                "policy/q": jax.lax.pmean(q.mean(), axis_name="n_gpu"),
                "policy/safety_q": jax.lax.pmean(safety_q.mean(), axis_name="n_gpu"),
                "policy/max_safety_q": jax.lax.pmean(max_safety_q.mean(), axis_name="n_gpu"),
                "policy/q_minus_z": jax.lax.pmean(q_minus_z.mean(), axis_name="n_gpu"),
                "policy/safety_branch_frac": jax.lax.pmean((max_safety_q > q_minus_z).mean(), axis_name="n_gpu"),
                "policy/z_mean": jax.lax.pmean(z.mean(), axis_name="n_gpu"),
                "policy/z_min": jax.lax.pmin(z.min(), axis_name="n_gpu"),
                "policy/z_max": jax.lax.pmax(z.max(), axis_name="n_gpu"),
                "policy/action_abs": jax.lax.pmean(jnp.abs(actions).mean(), axis_name="n_gpu"),
            }
            return loss, (safety_q, info)

        grad, (_, info) = jax.grad(actor_loss_fn, has_aux=True)(actor_state.params)
        grad_has_nan = jax.lax.pmax(has_any_nan_or_inf(grad).astype(jnp.float32), axis_name="n_gpu")
        grad, grad_norm = compute_norm_and_clip(grad, self.max_grad_norm)
        actor_state = actor_state.apply_gradients(grads=grad)
        info.update({
            "policy/has_nan": grad_has_nan,
            "policy/grad_norm": jax.lax.pmean(grad_norm, axis_name="n_gpu"),
        })
        return actor_state, info
