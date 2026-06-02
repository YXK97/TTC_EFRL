import functools as ft
import os
import pickle
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import optax
from flax.training.train_state import TrainState

from .base import Algorithm
from .module.ef_wrapper import ZEncoder
from .utils import val_to_optax_schedule
from ..env.base import MultiAgentEnv
from ..nn.gnn import GraphTransformerGNN
from ..nn.mlp import MLP
from ..nn.utils import default_nn_init, get_default_tx
from ..trainer.data import Rollout
from ..trainer.utils import rollout as rollout_fn
from ..trainer.utils import has_any_nan_or_inf, compute_norm_and_clip
from ..utils.graph import GraphsTuple
from ..utils.typing import Action, Array, Params, PRNGKey


class DeterministicActor(nn.Module):
    """确定性 actor。

    DDPG 不学习随机策略 pi(a|s)，而是直接学习确定性策略：
        a = mu_theta(s, z)
    这里输出的是归一化 action，范围为 [-1, 1]；环境 step 中会通过 transform_action
    映射到真实控制量范围。
    """

    action_dim: int
    n_agents: int
    gnn_layers: int
    use_ef: bool = False

    @nn.compact
    def __call__(self, graph: GraphsTuple, z: Array = None) -> Action:
        x = GraphTransformerGNN(msg_dim=32, out_dim=64, n_heads=6, n_layers=self.gnn_layers)(
            graph, node_type=0, n_type=self.n_agents
        )
        if self.use_ef:
            z_enc = ZEncoder(nz=8, z_mean=1.0, z_scale=1.0)(z)
            x = jnp.concatenate([x, z_enc], axis=-1)
        x = MLP(hid_sizes=(128, 128), act=nn.relu, act_final=True, name="DDPGActorHead")(x)
        action = nn.Dense(self.action_dim, kernel_init=default_nn_init())(x)
        return jnp.tanh(action)


class GraphQNet(nn.Module):
    """集中式 graph critic。

    每个 agent 的 Q_i 都可以看到局部 graph embedding、全局 graph embedding 和所有 agent action 的均值。
    这样比只看自身 action 更稳定，也不需要修改环境 graph 结构。

    EFRL 模式下：
        n_out=1      时表示 Ql(s, a, z)，即累计损失 critic；
        n_out=n_cost 时表示 Qh(s, a, z)，即安全约束 critic。

    Lagrangian 模式下：
        n_out=1      时表示 Qr(s, a)，即累计 reward critic；
        n_out=n_cost 时表示 Qc(s, a)，即累计 cost critic。
    """

    n_agents: int
    n_out: int
    gnn_layers: int
    use_ef: bool = False

    @nn.compact
    def __call__(self, graph: GraphsTuple, action: Action, z: Array = None) -> Array:
        x = GraphTransformerGNN(msg_dim=32, out_dim=64, n_heads=6, n_layers=self.gnn_layers)(
            graph, node_type=0, n_type=self.n_agents
        )
        x_global = jnp.mean(x, axis=0, keepdims=True).repeat(self.n_agents, axis=0)
        a_global = jnp.mean(action, axis=0, keepdims=True).repeat(self.n_agents, axis=0)
        feats = [x, x_global, action, a_global]
        if self.use_ef:
            feats.append(ZEncoder(nz=8, z_mean=1.0, z_scale=1.0)(z))
        x = jnp.concatenate(feats, axis=-1)
        x = MLP(hid_sizes=(128, 128), act=nn.relu, act_final=True, name="DDPGQHead")(x)
        q = nn.Dense(self.n_out, kernel_init=default_nn_init())(x)
        return q


def _tree_first_device(tree):
    return jtu.tree_map(lambda x: x[0], tree)


def _tree_flatten_rollout(rollout: Rollout) -> Rollout:
    """把 (device, env, T, ...) 或 (env, T, ...) 的 rollout 展平成 transition batch。"""

    def flat(x):
        if x is None:
            return None
        if x.ndim <= 1:
            return x.reshape((-1,))
        if x.ndim >= 3:
            return x.reshape((-1,) + x.shape[3:])
        return x.reshape((-1,) + x.shape[2:])

    return jtu.tree_map(flat, rollout)


class _TransitionReplay:
    """简单 host replay buffer，内部存 Rollout 结构，第一维是 transition 数。"""

    def __init__(self, size: int):
        self.size = size
        self.buffer = None

    @property
    def length(self) -> int:
        if self.buffer is None:
            return 0
        return int(self.buffer.rewards.shape[0])

    def append(self, rollout: Rollout):
        data = jax.device_get(_tree_flatten_rollout(rollout))
        data = jtu.tree_map(lambda x: None if x is None else np.asarray(x), data)
        if self.buffer is None:
            self.buffer = data
        else:
            self.buffer = jtu.tree_map(
                lambda a, b: None if a is None else np.concatenate([a, b], axis=0),
                self.buffer,
                data,
            )
        if self.length > self.size:
            self.buffer = jtu.tree_map(
                lambda x: None if x is None else x[-self.size:],
                self.buffer,
            )

    def sample(self, batch_size: int, num_devices: int) -> Rollout:
        per_device = max(1, batch_size // num_devices)
        total = per_device * num_devices
        idx = np.random.randint(0, self.length, size=total)
        batch = jtu.tree_map(lambda x: None if x is None else x[idx], self.buffer)
        batch = jtu.tree_map(
            lambda x: None if x is None else jnp.asarray(x).reshape((num_devices, per_device) + x.shape[1:]),
            batch,
        )
        return batch


class DDPG(Algorithm):
    """普通 DDPG，使用固定权重的 reward-cost 目标。

    actor 是确定性策略：
        a = mu_theta(s)

    critic 分别学习 reward return 和 cost return：
        Qr(s, a), Qc(s, a)

    actor 最大化固定权重目标：
        Qr(s, mu(s)) - cost_weight * Qc(s, mu(s))

    代码中等价最小化：
        L_actor = mean(-Qr + cost_weight * Qc)
    """

    use_ef: bool = False

    def __init__(
            self,
            env: MultiAgentEnv,
            node_dim: int,
            edge_dim: int,
            state_dim: int,
            action_dim: int,
            n_agents: int,
            cost_weight: float = 0.,
            actor_gnn_layers: int = 2,
            critic_gnn_layers: int = 2,
            gamma: float = 0.99,
            lr_actor: float = 3e-4,
            lr_actor_decay: bool = False,
            lr_actor_init: Optional[float] = None,
            lr_actor_decay_ratio: Optional[float] = None,
            lr_actor_warmup_iters: Optional[int] = None,
            lr_actor_trans_iters: Optional[int] = None,
            lr_critic: float = 1e-3,
            lr_critic_decay: bool = False,
            lr_critic_init: Optional[float] = None,
            lr_critic_decay_ratio: Optional[float] = None,
            lr_critic_warmup_iters: Optional[int] = None,
            lr_critic_trans_iters: Optional[int] = None,
            max_grad_norm: float = 2.0,
            seed: int = 0,
            batch_size: int = 4096,
            tau: float = 0.005,
            replay_size: int = 250000,
            expl_noise: float = 0.15,
            replay_warmup_transitions: int = 8192,
            updates_per_iter: int = 1,
            iter_index: int = 0,
            **kwargs
    ):
        del state_dim, kwargs
        super().__init__(env, node_dim, edge_dim, action_dim, n_agents)

        self.cost_weight = cost_weight
        self.actor_gnn_layers = actor_gnn_layers
        self.critic_gnn_layers = critic_gnn_layers
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        self.batch_size = batch_size
        self.tau = tau
        self.replay_size = replay_size
        self.expl_noise = expl_noise
        self.replay_warmup_transitions = replay_warmup_transitions
        self.updates_per_iter = updates_per_iter
        self.z_min = -env.reward_max
        self.z_max = -env.reward_min
        self.iter_index = iter_index

        self.lr_actor_val = lr_actor
        self.lr_critic_val = lr_critic
        self.lr_actor_sched = val_to_optax_schedule(
            lr_actor, lr_actor_decay, lr_actor_init, lr_actor_decay_ratio,
            lr_actor_warmup_iters, lr_actor_trans_iters
        )
        self.lr_critic_sched = val_to_optax_schedule(
            lr_critic, lr_critic_decay, lr_critic_init, lr_critic_decay_ratio,
            lr_critic_warmup_iters, lr_critic_trans_iters
        )

        self.actor = DeterministicActor(action_dim, n_agents, actor_gnn_layers, use_ef=self.use_ef)
        self.critic = GraphQNet(n_agents, 1, critic_gnn_layers, use_ef=self.use_ef)
        self.safety_critic = GraphQNet(n_agents, env.n_cost, critic_gnn_layers, use_ef=self.use_ef)

        n_nodes = n_agents
        self.nominal_graph = GraphsTuple(
            n_node=jnp.array(n_nodes),
            n_edge=jnp.array(n_nodes),
            nodes=jnp.zeros((n_nodes, node_dim), dtype=jnp.float32),
            edges=jnp.zeros((n_nodes, edge_dim), dtype=jnp.float32),
            states=jnp.zeros((n_nodes, env.state_dim), dtype=jnp.float32),
            receivers=jnp.arange(n_nodes),
            senders=jnp.arange(n_nodes),
            node_type=jnp.zeros((n_nodes,), dtype=jnp.int32),
            env_states=jnp.zeros((n_nodes,), dtype=jnp.float32),
        )
        self.nominal_z = jnp.zeros((n_agents, 1), dtype=jnp.float32)
        self.nominal_action = jnp.zeros((n_agents, action_dim), dtype=jnp.float32)

        key = jr.PRNGKey(seed)
        actor_key, critic_key, safety_key, key = jr.split(key, 4)
        actor_params = self.actor.init(actor_key, self.nominal_graph, self.nominal_z)
        critic_params = self.critic.init(critic_key, self.nominal_graph, self.nominal_action, self.nominal_z)
        safety_params = self.safety_critic.init(safety_key, self.nominal_graph, self.nominal_action, self.nominal_z)

        self.actor_train_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=actor_params,
            tx=get_default_tx(self.lr_actor_sched),
        )
        self.critic_train_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params,
            tx=get_default_tx(self.lr_critic_sched),
        )
        self.safety_train_state = TrainState.create(
            apply_fn=self.safety_critic.apply,
            params=safety_params,
            tx=get_default_tx(self.lr_critic_sched),
        )
        self.target_actor_params = actor_params
        self.target_critic_params = critic_params
        self.target_safety_params = safety_params

        self.init_rnn_state = jnp.zeros((n_agents, 1, 64), dtype=jnp.float32)
        self.replay = _TransitionReplay(replay_size)
        self.key = key

        def rollout_fn_single_(cur_params, cur_key):
            return rollout_fn(
                self._env,
                ft.partial(self.step, params=cur_params),
                self.init_rnn_state,
                cur_key,
                self.gamma,
            )

        self.rollout_fn = jax.jit(lambda cur_params, cur_keys: jax.vmap(
            ft.partial(rollout_fn_single_, cur_params)
        )(cur_keys))

    @property
    def config(self) -> dict:
        return {
            "cost_weight": self.cost_weight,
            "actor_gnn_layers": self.actor_gnn_layers,
            "critic_gnn_layers": self.critic_gnn_layers,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "replay_size": self.replay_size,
            "expl_noise": self.expl_noise,
            "replay_warmup_transitions": self.replay_warmup_transitions,
            "updates_per_iter": self.updates_per_iter,
            "safety_mode": "weighted",
            "seed": self.seed,
            "use_rnn": False,
            "iter_index": self.iter_index,
        }

    @property
    def params(self) -> Params:
        return {
            "policy": self.actor_train_state.params,
            "critic": self.critic_train_state.params,
            "safety_critic": self.safety_train_state.params,
            "target_policy": self.target_actor_params,
            "target_critic": self.target_critic_params,
            "target_safety_critic": self.target_safety_params,
        }

    def act(self, graph: GraphsTuple, z: Array, rnn_state: Array, params: Optional[Params] = None) -> Tuple[Action, Array]:
        params = self.params if params is None else params
        action = self.actor.apply(params["policy"], graph, z)
        return action, rnn_state

    def step(
            self, graph: GraphsTuple, z: Array, rnn_state: Array, key: PRNGKey, params: Optional[Params] = None
    ) -> Tuple[Action, Array, Array]:
        params = self.params if params is None else params
        action = self.actor.apply(params["policy"], graph, z)
        # 训练时加入探索噪声：
        #   a_explore = clip(mu(s,z) + eps, -1, 1), eps ~ N(0, sigma^2)
        # DDPG 的 actor 本身是确定性的，所以探索必须显式加在 action 上。
        noise = jr.normal(key, action.shape) * self.expl_noise
        action = jnp.clip(action + noise, -1.0, 1.0)
        log_pi = jnp.zeros((self.n_agents,), dtype=jnp.float32)
        return action, log_pi, rnn_state

    def get_opt_z(self, graph: GraphsTuple, Vh_rnn_state: Array, params: Optional[Params] = None) -> Tuple[Array, Array]:
        # params = self.params if params is None else params
        # 非efrl环境不需要使用变化的z
        del graph, params
        z = jnp.ones((self.n_agents, 1), dtype=jnp.float32) * self.z_max
        return z, Vh_rnn_state

    @ft.partial(jax.pmap, in_axes=(None, None, 0), axis_name="n_gpu", static_broadcasted_argnums=(0,))
    def collect(self, params: Params, b_key: PRNGKey) -> Rollout:
        return self.rollout_fn(params, b_key)

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
            critic, safety, actor, target_actor, target_critic, target_safety, info = self.pmap_update(
                self.critic_train_state,
                self.safety_train_state,
                self.actor_train_state,
                self.target_actor_params,
                self.target_critic_params,
                self.target_safety_params,
                batch,
            )
            self.critic_train_state = _tree_first_device(critic)
            self.safety_train_state = _tree_first_device(safety)
            self.actor_train_state = _tree_first_device(actor)
            self.target_actor_params = _tree_first_device(target_actor)
            self.target_critic_params = _tree_first_device(target_critic)
            self.target_safety_params = _tree_first_device(target_safety)
            info_single = _tree_first_device(info)
            info_acc = info_single if info_acc is None else jtu.tree_map(lambda a, b: a + b, info_acc, info_single)

        info_acc = jtu.tree_map(lambda x: x / self.updates_per_iter, info_acc)
        info_acc.update({
            "replay/size": float(self.replay.length),
            "ddpg/skipped_update": 0.0,
        })
        self.iter_index += 1
        return info_acc

    @ft.partial(jax.pmap, in_axes=(None, None, None, None, None, None, None, 0),
                axis_name="n_gpu", static_broadcasted_argnums=(0,))
    def pmap_update(
            self,
            critic_state: TrainState,
            safety_state: TrainState,
            actor_state: TrainState,
            target_actor_params: Params,
            target_critic_params: Params,
            target_safety_params: Params,
            batch: Rollout,
    ):
        critic_state, safety_state, critic_info = self.update_critics(
            critic_state, safety_state, target_actor_params, target_critic_params, target_safety_params, batch
        )
        actor_state, actor_info = self.update_actor(actor_state, critic_state, safety_state, batch)
        # DDPG 使用 target network 降低 bootstrapping 震荡：
        #   theta_bar <- tau * theta + (1 - tau) * theta_bar
        target_actor_params = optax.incremental_update(actor_state.params, target_actor_params, self.tau)
        target_critic_params = optax.incremental_update(critic_state.params, target_critic_params, self.tau)
        target_safety_params = optax.incremental_update(safety_state.params, target_safety_params, self.tau)
        critic_info.update(actor_info)
        return critic_state, safety_state, actor_state, target_actor_params, target_critic_params, target_safety_params, critic_info

    def _batched_actor(self, params: Params, graphs: GraphsTuple, zs: Array) -> Action:
        return jax.vmap(lambda g, z: self.actor.apply(params, g, z))(graphs, zs)

    def _batched_q(self, params: Params, graphs: GraphsTuple, actions: Action, zs: Array) -> Array:
        return jax.vmap(lambda g, a, z: self.critic.apply(params, g, a, z))(graphs, actions, zs)

    def _batched_safety_q(self, params: Params, graphs: GraphsTuple, actions: Action, zs: Array) -> Array:
        return jax.vmap(lambda g, a, z: self.safety_critic.apply(params, g, a, z))(graphs, actions, zs)

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

        # Bellman target 中的下一步 action 使用 target actor：
        #   a' = mu_target(s')
        next_zs = zs
        next_actions = self._batched_actor(target_actor_params, batch.next_graph, next_zs)
        next_q = self._batched_q(target_critic_params, batch.next_graph, next_actions, next_zs).squeeze(-1)
        q_target = rewards + self.gamma * (1.0 - dones[:, None]) * next_q
        q_target = jnp.clip(q_target, -1e4, 1e4)

        next_safety_q = self._batched_safety_q(target_safety_params, batch.next_graph, next_actions, next_zs)
        # cost critic 学折扣累计正 cost：
        #   y_c = max(cost, 0) + gamma * Qc_target(s', mu_target(s'))
        safety_target = jnp.maximum(costs, 0.0) + self.gamma * (1.0 - dones[:, None, None]) * next_safety_q
        safety_target = jnp.clip(safety_target, -1e4, 1e4)

        def critic_loss_fn(q_params, safety_params):
            q_pred = self._batched_q(q_params, batch.graph, batch.actions, zs).squeeze(-1)
            safety_pred = self._batched_safety_q(safety_params, batch.graph, batch.actions, zs)
            # critic 使用均方 Bellman error：
            #   L_Qr = mean((Qr - y_r)^2)
            #   L_Qc = mean((Qc - y_c)^2)
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
            # 固定权重 actor objective：
            #   max_mu Qr(s,mu(s)) - cost_weight * Qc(s,mu(s))
            # 代码中转为最小化 -Qr + cost_weight * Qc。
            penalty = self.cost_weight * safety_q.mean(axis=-1)
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

        info.update({
            "policy/has_nan": grad_has_nan,
            "policy/grad_norm": jax.lax.pmean(grad_norm, axis_name="n_gpu"),
        })
        return actor_state, info

    def save(self, save_dir: str, step: int, params_to_save: dict = None):
        model_dir = os.path.join(save_dir, str(step))
        os.makedirs(model_dir, exist_ok=True)
        params = self.params if params_to_save is None else params_to_save
        pickle.dump(params["policy"], open(os.path.join(model_dir, "actor.pkl"), "wb"))
        pickle.dump(params["critic"], open(os.path.join(model_dir, "critic.pkl"), "wb"))
        pickle.dump(params["safety_critic"], open(os.path.join(model_dir, "safety_critic.pkl"), "wb"))

    def load(self, load_dir: str, step: int):
        path = os.path.join(load_dir, str(step))
        actor_params = pickle.load(open(os.path.join(path, "actor.pkl"), "rb"))
        critic_params = pickle.load(open(os.path.join(path, "critic.pkl"), "rb"))
        safety_params = pickle.load(open(os.path.join(path, "safety_critic.pkl"), "rb"))
        self.actor_train_state = self.actor_train_state.replace(params=actor_params)
        self.critic_train_state = self.critic_train_state.replace(params=critic_params)
        self.safety_train_state = self.safety_train_state.replace(params=safety_params)
        self.target_actor_params = actor_params
        self.target_critic_params = critic_params
        self.target_safety_params = safety_params
