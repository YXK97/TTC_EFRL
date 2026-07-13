import functools as ft
import pathlib
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrow

from .designed_scene_gen_two_lane import gen_scene_randomly
from .mve import MVE, MVEEnvBoundState, MVEEnvState
from .utils import process_lane_marks
from defmarl.trainer.data import Record, Rollout
from defmarl.utils.graph import EdgeBlock, GetGraph, GraphsTuple
from defmarl.utils.scaling_lowspeed import scaling_calc, scaling_calc_bound
from defmarl.utils.typing import Action, AgentState, Array, Cost, ObstState, Reward, State
from defmarl.utils.utils import find_closest_goal_indices, gen_i_j_pairs, save_anim, tree_index


INF = jnp.inf
EPS = 1e-6


class LowSpeedAccelMixin(MVE):
    """Shared low-speed vehicle env logic.

    State: [x, y, heading_x, heading_y, v, delta].
    Geometry parameters such as bw, bh, and lr are read from env params.
    """

    includes_bound_nodes = False
    use_cbf_cost = False

    @property
    def state_dim(self) -> int:
        return 6

    @property
    def node_dim(self) -> int:
        return self.state_dim + 3

    @property
    def edge_dim(self) -> int:
        return self.state_dim

    @property
    def action_dim(self) -> int:
        return 2  # longitudinal acceleration a (m/s^2), steering command delta (rad)

    @property
    def n_cost(self) -> int:
        return 4

    @property
    def cost_components(self) -> Tuple[str, ...]:
        return "agent collisions", "obs collisions", "bound exceeds y low", "bound exceeds y high"

    @property
    def num_goals(self) -> int:
        return 4800

    def _observable(self, states: State) -> State:
        return states

    def _normalize_heading(self, heading: Array) -> Array:
        norm = jnp.linalg.norm(heading, axis=1, keepdims=True)
        return heading / jnp.maximum(norm, EPS)

    def _filter_delta(self, delta_prev: Array, delta_cmd: Array) -> Array:
        alpha = self.params.get("delta_filter_alpha", 0.5)
        max_delta_step = self.params.get("max_delta", 0.2)
        delta_m = alpha * delta_prev + (1.0 - alpha) * delta_cmd
        ddelta = jnp.clip(delta_m - delta_prev, -max_delta_step, max_delta_step)
        return delta_prev + ddelta

    def _speed_lim(self) -> Tuple[Array, Array]:
        return self.params.get("v_min", 0.0), self.params.get("v_max", 30.0 / 3.6)

    def reset(self, key: Array) -> Tuple[GraphsTuple, jnp.ndarray]:
        c_ycs = self.params["lane_centers"]
        xrange = self.params["default_state_range"][:2]
        yrange = self.params["default_state_range"][2:4]
        lanewidth = self.params["lane_width"]
        agents, obsts, all_goals, all_dsYddts = gen_scene_randomly(
            key, self.num_agents, self.num_goals, xrange, yrange, lanewidth, c_ycs
        )
        obsts = obsts if obsts.shape[0] > 0 else jnp.empty((0, self.state_dim))
        self.all_goals = all_goals
        self.all_dsYddts = all_dsYddts
        goals_init_indices = find_closest_goal_indices(self._observable(agents), self._observable(all_goals))
        agents_indices = jnp.arange(agents.shape[0])
        goals = all_goals[agents_indices, goals_init_indices, :]
        dsYddts = all_dsYddts[agents_indices, goals_init_indices, :]
        self.num_obsts = obsts.shape[0]

        if self.includes_bound_nodes:
            bounds = self.generate_bound(agents, self.params["bound_bb_size"])
            env_state = MVEEnvBoundState(agents, goals, bounds, obsts)
        else:
            env_state = MVEEnvState(agents, goals, obsts)
        return self.get_graph(env_state), dsYddts

    def generate_bound(self, agent_states: AgentState, bound_bb_size: Array) -> State:
        num_agents = agent_states.shape[0]
        y_low = self.params["default_state_range"][2]
        y_high = self.params["default_state_range"][3]
        bound_bh = jnp.asarray(bound_bb_size)[1]
        x = agent_states[:, 0]
        hx = jnp.ones((num_agents,))
        hy = jnp.zeros((num_agents,))
        v = jnp.zeros((num_agents,))
        delta = jnp.zeros((num_agents,))
        lower = jnp.stack([x, jnp.ones(num_agents) * (y_low - 0.5 * bound_bh), hx, hy, v, delta], axis=1)
        upper = jnp.stack([x, jnp.ones(num_agents) * (y_high + 0.5 * bound_bh), hx, hy, v, delta], axis=1)
        return jnp.stack([lower, upper], axis=1).reshape(num_agents * 2, self.state_dim)

    def agent_step_euler(self, agent_states: AgentState, action: Action) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        x, y = agent_states[:, 0], agent_states[:, 1]
        h = self._normalize_heading(agent_states[:, 2:4])
        v = agent_states[:, 4]
        delta_prev = agent_states[:, 5]
        accel = action[:, 0]
        delta = self._filter_delta(delta_prev, action[:, 1])

        v_min, v_max = self._speed_lim()
        v_mid = jnp.clip(v + 0.5 * accel * self.dt, v_min, v_max)
        v_next = jnp.clip(v + accel * self.dt, v_min, v_max)
        omega = v_mid / self.params["ego_L"] * jnp.tan(delta)

        x_new = x + v_mid * h[:, 0] * self.dt
        y_new = y + v_mid * h[:, 1] * self.dt
        h_dot = jnp.stack([-h[:, 1] * omega, h[:, 0] * omega], axis=1)
        h_new = self._normalize_heading(h + h_dot * self.dt)
        out = jnp.concatenate([x_new[:, None], y_new[:, None], h_new, v_next[:, None], delta[:, None]], axis=1)
        return self.clip_internal_state(out)

    def obst_step_euler(self, obst_states: ObstState) -> ObstState:
        num_obsts = obst_states.shape[0]
        assert obst_states.shape == (num_obsts, self.state_dim)
        h = self._normalize_heading(obst_states[:, 2:4])
        x_new = obst_states[:, 0] + obst_states[:, 4] * h[:, 0] * self.dt
        y_new = obst_states[:, 1] + obst_states[:, 4] * h[:, 1] * self.dt
        return obst_states.at[:, 0].set(x_new).at[:, 1].set(y_new)

    def goal_dsYddt_step(self, agent_states_new: AgentState) -> Tuple[State, jnp.ndarray]:
        goal_indices = find_closest_goal_indices(self._observable(agent_states_new), self._observable(self.all_goals))
        agent_indices = jnp.arange(agent_states_new.shape[0])
        return self.all_goals[agent_indices, goal_indices, :], self.all_dsYddts[agent_indices, goal_indices, :]

    def step(self, graph: GraphsTuple, action: Action, get_eval_info: bool = False):
        env_state = graph.env_states
        action = self.transform_action(action)
        next_agent_states = self.agent_step_euler(env_state.agent, action)
        next_obst_states = self.obst_step_euler(env_state.obstacle)
        next_goal_states, next_dsYddts = self.goal_dsYddt_step(next_agent_states)
        if self.includes_bound_nodes:
            next_bounds = self.generate_bound(next_agent_states, self.params["bound_bb_size"])
            next_env_state = MVEEnvBoundState(next_agent_states, next_goal_states, next_bounds, next_obst_states)
        else:
            next_env_state = MVEEnvState(next_agent_states, next_goal_states, next_obst_states)
        reward = self.get_reward(graph, action)
        cost, cost_real = self.get_cost(graph, action) if self.use_cbf_cost else self.get_cost(graph)
        return self.get_graph(next_env_state), next_dsYddts, reward, cost, cost_real, jnp.array(False), {}

    def get_reward(self, graph: GraphsTuple, action: Action) -> Reward:
        agent = self._observable(graph.env_states.agent)
        goal = self._observable(graph.env_states.goal)
        e = agent - goal
        W = jnp.diag(jnp.array([5e-3, 5e-3, 0, 0, 5e-3, 0]))
        reward = -jnp.einsum("ai,ij,ja->a", e, W, e.transpose()).mean()
        reward -= (action[:, 0] ** 2).mean() * 0.005
        reward -= (action[:, 1] ** 2).mean() * 0.005
        return reward

    def _scaling_cost(self, graph: GraphsTuple) -> Tuple[Cost, Cost]:
        thresh = self.params["alpha_thresh"]
        agent_states = graph.env_states.agent
        num_agents = agent_states.shape[0]
        num_obsts = graph.env_states.obstacle.shape[0]
        a_agent_cost = -jnp.ones((num_agents,), dtype=jnp.float32)
        a_agent_cost_real = -jnp.ones((num_agents,), dtype=jnp.float32)

        if num_obsts == 0:
            a_obst_cost = -jnp.ones((num_agents,), dtype=jnp.float32)
            a_obst_cost_real = -jnp.ones((num_agents,), dtype=jnp.float32)
        else:
            i_pairs, j_pairs = gen_i_j_pairs(num_agents, num_obsts)
            alpha_pairs = jax.vmap(scaling_calc, in_axes=(0, 0, None, None, None, None))(
                agent_states[i_pairs],
                graph.env_states.obstacle[j_pairs],
                self.params["ego_bb_size"],
                self.params["ego_lr"],
                self.params["obst_bb_size"],
                self.params["obst_lr"],
            )
            alpha_pairs = jnp.nan_to_num(alpha_pairs, nan=0.0, posinf=1e6, neginf=0.0)
            alpha_matrix = alpha_pairs.reshape((num_agents, num_obsts))
            a_obst_cost = jnp.max(thresh - alpha_matrix, axis=1)
            a_obst_cost_real = jnp.max(1 - alpha_matrix, axis=1)

        if self.includes_bound_nodes:
            agent_idx = jnp.arange(num_agents)
            alpha_low = jax.vmap(scaling_calc, in_axes=(0, 0, None, None, None, None))(
                agent_states,
                graph.env_states.bound[agent_idx * 2],
                self.params["ego_bb_size"],
                self.params["ego_lr"],
                self.params["bound_bb_size"],
                0.0,
            )
            alpha_high = jax.vmap(scaling_calc, in_axes=(0, 0, None, None, None, None))(
                agent_states,
                graph.env_states.bound[agent_idx * 2 + 1],
                self.params["ego_bb_size"],
                self.params["ego_lr"],
                self.params["bound_bb_size"],
                0.0,
            )
            alpha_low = jnp.nan_to_num(alpha_low, nan=0.0, posinf=1e6, neginf=0.0)
            alpha_high = jnp.nan_to_num(alpha_high, nan=0.0, posinf=1e6, neginf=0.0)
            a_bound_low_cost = thresh - alpha_low
            a_bound_high_cost = thresh - alpha_high
            a_bound_low_cost_real = 1 - alpha_low
            a_bound_high_cost_real = 1 - alpha_high
        else:
            yl = self.params["default_state_range"][2]
            yh = self.params["default_state_range"][3]
            alpha_low = jax.vmap(scaling_calc_bound, in_axes=(0, None, None, None, None))(
                agent_states,
                self.params["ego_bb_size"],
                self.params["ego_lr"],
                jnp.array([[0.0, 1.0]]),
                jnp.array([yl]),
            )
            alpha_high = jax.vmap(scaling_calc_bound, in_axes=(0, None, None, None, None))(
                agent_states,
                self.params["ego_bb_size"],
                self.params["ego_lr"],
                jnp.array([[0.0, -1.0]]),
                jnp.array([-yh]),
            )
            alpha_low = jnp.nan_to_num(alpha_low, nan=0.0, posinf=1e6, neginf=0.0)
            alpha_high = jnp.nan_to_num(alpha_high, nan=0.0, posinf=1e6, neginf=0.0)
            a_bound_low_cost = thresh - alpha_low
            a_bound_high_cost = thresh - alpha_high
            a_bound_low_cost_real = 1 - alpha_low
            a_bound_high_cost_real = 1 - alpha_high

        cost = jnp.stack([a_agent_cost, a_obst_cost, a_bound_low_cost, a_bound_high_cost], axis=1)
        cost_real = jnp.stack([a_agent_cost_real, a_obst_cost_real, a_bound_low_cost_real, a_bound_high_cost_real], axis=1)
        cost = jnp.where(cost <= 0.0, cost, cost + 1.0)
        return jnp.clip(cost, a_min=-10.0), cost_real

    def _cbf_cost(self, graph: GraphsTuple, action: Action) -> Tuple[Cost, Cost]:
        thresh = self.params["alpha_thresh"]
        gamma = self.params["gamma"]
        num_agents = graph.env_states.agent.shape[0]
        num_obsts = graph.env_states.obstacle.shape[0]
        action_delta = self._filter_delta(graph.env_states.agent[:, 5], action[:, 1])

        def cbf_between(s1, s2, delta_rad, is_bound):
            bb = self.params["bound_bb_size"] if is_bound else self.params["obst_bb_size"]
            lr = 0.0 if is_bound else self.params["obst_lr"]

            def alpha_fn(z):
                full = jnp.array([z[0], z[1], z[2], z[3], s1[4], s1[5]])
                return scaling_calc(full, s2, self.params["ego_bb_size"], self.params["ego_lr"], bb, lr)

            z = s1[:4]
            alpha, grad_z = jax.value_and_grad(alpha_fn)(z)
            alpha = jnp.nan_to_num(alpha, nan=0.0, posinf=1e6, neginf=0.0)
            grad_z = jnp.nan_to_num(grad_z, nan=0.0, posinf=0.0, neginf=0.0)
            hvec = z[2:4] / jnp.maximum(jnp.linalg.norm(z[2:4]), EPS)
            omega = s1[4] / self.params["ego_L"] * jnp.tan(delta_rad)
            z_dot = jnp.array([s1[4] * hvec[0], s1[4] * hvec[1], -hvec[1] * omega, hvec[0] * omega])
            cost = -(jnp.dot(grad_z, z_dot) + gamma * (alpha - thresh)) / gamma
            cost = jnp.nan_to_num(cost, nan=10.0, posinf=10.0, neginf=-3.0)
            return cost, 1 - alpha

        a_agent_cost = -jnp.ones((num_agents,), dtype=jnp.float32) * 3.0
        a_agent_cost_real = -jnp.ones((num_agents,), dtype=jnp.float32) * 3.0

        if num_obsts == 0:
            a_obst_cost = -jnp.ones((num_agents,), dtype=jnp.float32) * 3.0
            a_obst_cost_real = -jnp.ones((num_agents,), dtype=jnp.float32) * 3.0
        else:
            i_pairs, j_pairs = gen_i_j_pairs(num_agents, num_obsts)
            costs, reals = jax.vmap(cbf_between, in_axes=(0, 0, 0, None))(
                graph.env_states.agent[i_pairs], graph.env_states.obstacle[j_pairs], action_delta[i_pairs], False
            )
            a_obst_cost = jnp.max(costs.reshape((num_agents, num_obsts)), axis=1)
            a_obst_cost_real = jnp.max(reals.reshape((num_agents, num_obsts)), axis=1)

        yl = self.params["default_state_range"][2]
        yh = self.params["default_state_range"][3]
        bound_size = self.params.get("bound_bb_size", jnp.array([5.0, 1.0]))
        lower = self.generate_bound(graph.env_states.agent, bound_size)[::2]
        upper = self.generate_bound(graph.env_states.agent, bound_size)[1::2]
        a_low_cost, a_low_real = jax.vmap(cbf_between, in_axes=(0, 0, 0, None))(
            graph.env_states.agent, lower, action_delta, True
        )
        a_high_cost, a_high_real = jax.vmap(cbf_between, in_axes=(0, 0, 0, None))(
            graph.env_states.agent, upper, action_delta, True
        )

        cost = jnp.stack([a_agent_cost, a_obst_cost, a_low_cost, a_high_cost], axis=1)
        cost_real = jnp.stack([a_agent_cost_real, a_obst_cost_real, a_low_real, a_high_real], axis=1)
        cost = jnp.where(cost <= 0.0, cost, cost + 1.0)
        return jnp.clip(cost, a_min=-10.0, a_max=10.0), cost_real

    def get_cost(self, graph: GraphsTuple, action: Optional[Action] = None) -> Tuple[Cost, Cost]:
        if self.use_cbf_cost:
            if action is None:
                return self._scaling_cost(graph)
            return self._cbf_cost(graph, action)
        return self._scaling_cost(graph)

    def edge_blocks(self, state) -> List[EdgeBlock]:
        num_agents = state.agent.shape[0]
        num_goals = state.goal.shape[0]
        num_obsts = state.obstacle.shape[0]
        agent_obs = self._observable(state.agent)
        id_agent = jnp.arange(num_agents)
        edges = []

        for i_agent in range(num_agents):
            rel = self._observable(state.agent[i_agent][None, :])[0] - self._observable(state.goal[i_agent][None, :])[0]
            edges.append(EdgeBlock(rel[None, None, :], jnp.ones((1, 1)), jnp.array([i_agent]), jnp.array([i_agent + num_agents])))

        if self.includes_bound_nodes:
            num_bounds = state.bound.shape[0]
            for i_agent in range(num_agents):
                for offset in range(2):
                    bound_id = 2 * i_agent + offset
                    rel = self._observable(state.agent[i_agent][None, :])[0] - self._observable(state.bound[bound_id][None, :])[0]
                    edges.append(EdgeBlock(rel[None, None, :], jnp.ones((1, 1)),
                                           jnp.array([i_agent]), jnp.array([num_agents + num_goals + bound_id])))
            obst_offset = num_agents + num_goals + num_bounds
        else:
            obst_offset = num_agents + num_goals

        if num_obsts > 0:
            obs_obs = self._observable(state.obstacle)
            dist = jnp.linalg.norm(agent_obs[:, None, :2] - obs_obs[None, :, :2], axis=-1)
            mask = jnp.less(dist, self.params["comm_radius"])
            i_pairs, j_pairs = gen_i_j_pairs(num_agents, num_obsts)
            rel_pairs = agent_obs[i_pairs] - obs_obs[j_pairs]
            rel = rel_pairs.reshape((num_agents, num_obsts, self.state_dim))
            edges.append(EdgeBlock(rel, mask, id_agent, jnp.arange(num_obsts) + obst_offset))
        return edges

    def get_graph(self, env_state, obst_as_agent: bool = False) -> GraphsTuple:
        num_agents = env_state.agent.shape[0]
        num_goals = env_state.goal.shape[0]
        num_obsts = env_state.obstacle.shape[0]
        num_bounds = env_state.bound.shape[0] if self.includes_bound_nodes else 0
        total = num_agents + num_goals + num_bounds + num_obsts
        node_feats = jnp.zeros((total, self.node_dim))
        node_type = -jnp.ones((total,), dtype=jnp.int32)

        node_feats = node_feats.at[:num_agents, :self.state_dim].set(self._observable(env_state.agent))
        node_feats = node_feats.at[num_agents:num_agents + num_goals, :self.state_dim].set(self._observable(env_state.goal))
        node_type = node_type.at[:num_agents].set(MVE.AGENT)
        node_type = node_type.at[num_agents:num_agents + num_goals].set(MVE.GOAL)
        node_feats = node_feats.at[:num_agents, -1].set(1.0)
        node_feats = node_feats.at[num_agents:num_agents + num_goals, -2].set(1.0)

        cursor = num_agents + num_goals
        if self.includes_bound_nodes:
            node_feats = node_feats.at[cursor:cursor + num_bounds, :self.state_dim].set(self._observable(env_state.bound))
            node_feats = node_feats.at[cursor:cursor + num_bounds, -3].set(1.0)
            node_type = node_type.at[cursor:cursor + num_bounds].set(MVE.BOUND)
            cursor += num_bounds
        if num_obsts > 0:
            node_feats = node_feats.at[cursor:, :self.state_dim].set(self._observable(env_state.obstacle))
            node_feats = node_feats.at[cursor:, -3].set(1.0)
            node_type = node_type.at[cursor:].set(MVE.OBST)

        states = node_feats[:, :self.state_dim]
        return GetGraph(node_feats, node_type, self.edge_blocks(env_state), env_state, states).to_padded()

    def clip_internal_state(self, state: State) -> State:
        x_l, x_h = self.params["rollout_state_range"][0], self.params["rollout_state_range"][1]
        y_l, y_h = self.params["rollout_state_range"][2], self.params["rollout_state_range"][3]
        v_l, v_h = self._speed_lim()
        state = state.at[:, 0].set(jnp.clip(state[:, 0], x_l, x_h))
        state = state.at[:, 1].set(jnp.clip(state[:, 1], y_l, y_h))
        state = state.at[:, 2:4].set(self._normalize_heading(state[:, 2:4]))
        state = state.at[:, 4].set(jnp.clip(state[:, 4], v_l, v_h))
        state = state.at[:, 5].set(jnp.clip(state[:, 5], -self.params["delta_abs_max"], self.params["delta_abs_max"]))
        return state

    def state_lim(self, state: Optional[State]) -> Tuple[State, State]:
        v_l, v_h = self._speed_lim()
        lower = jnp.array([
            self.params["rollout_state_range"][0],
            self.params["rollout_state_range"][2],
            -1.0,
            -1.0,
            v_l,
            -self.params["delta_abs_max"],
        ])
        upper = jnp.array([
            self.params["rollout_state_range"][1],
            self.params["rollout_state_range"][3],
            1.0,
            1.0,
            v_h,
            self.params["delta_abs_max"],
        ])
        return lower, upper

    def action_lim(self) -> Tuple[Action, Action]:
        lower = jnp.array([self.params["min_accel"], -self.params["delta_abs_max"]])[None, :].repeat(self.num_agents, axis=0)
        upper = jnp.array([self.params["max_accel"], self.params["delta_abs_max"]])[None, :].repeat(self.num_agents, axis=0)
        return lower, upper

    @ft.partial(jax.jit, static_argnums=(0,))
    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        _, cost_real = self.get_cost(graph)
        return jnp.any(cost_real >= 0.0, axis=-1)

    def _plot_pose(self, ax, states, bb_size, lr, color, zorder):
        h = self._normalize_heading(states[:, 2:4])
        pos = states[:, :2] + lr * h
        theta = np.asarray(jnp.arctan2(h[:, 1], h[:, 0]) * 180.0 / jnp.pi)
        bb = np.asarray(bb_size)
        radius = float(jnp.linalg.norm(jnp.asarray(bb_size)))
        arrows = [FancyArrow(float(pos[i, 0]), float(pos[i, 1]), float(h[i, 0]) * radius / 2,
                             float(h[i, 1]) * radius / 2, length_includes_head=True,
                             width=0.3, color=color, alpha=1.0) for i in range(states.shape[0])]
        rects = [plt.Rectangle((float(pos[i, 0] - bb[0] / 2), float(pos[i, 1] - bb[1] / 2)),
                               width=float(bb[0]), height=float(bb[1]), angle=float(theta[i]),
                               rotation_point="center", color=color, linewidth=0.0, alpha=0.6)
                 for i in range(states.shape[0])]
        for patch in arrows + rects:
            patch.set_zorder(zorder)
            ax.add_patch(patch)
        return arrows, rects

    def render_video(self, rollout: Rollout, video_path: pathlib.Path, Ta_is_unsafe=None,
                     n_goals: Optional[int] = None, **kwargs) -> None:
        n_goals = self.num_agents if n_goals is None else n_goals
        xlim = self.params["rollout_state_range"][:2]
        ylim = self.params["default_state_range"][2:4]
        fig, ax = plt.subplots(1, 1, figsize=(30, (ylim[1] + 3 - (ylim[0] - 3)) * 20 / (xlim[1] + 3 - (xlim[0] - 3)) + 4), dpi=100)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0] - 3, ylim[1] + 3)
        ax.set(aspect="equal")
        two_yms_bold, l_yms_scatter = process_lane_marks(self.params["default_state_range"][2:4], self.params["lane_width"])
        ax.axhline(y=two_yms_bold[0], linewidth=1.5, color="b")
        ax.axhline(y=two_yms_bold[1], linewidth=1.5, color="b")
        if l_yms_scatter is not None:
            for ym in l_yms_scatter:
                ax.axhline(y=ym, linewidth=1, color="b", linestyle="--")

        T_graph = rollout.graph
        graph0 = tree_index(T_graph, 0)
        agent_arrows, agent_rects = self._plot_pose(ax, graph0.env_states.agent, self.params["ego_bb_size"], self.params["ego_lr"], "#0068ff", 6)
        obst_arrows, obst_rects = self._plot_pose(ax, graph0.env_states.obstacle, self.params["obst_bb_size"], self.params["obst_lr"], "#8a0000", 5) if self.num_obsts > 0 else ([], [])
        ref_goals = np.asarray(rollout.graph.env_states.goal[:, :, :2])
        ax.scatter(
            ref_goals[:, :, 0].reshape(-1),
            ref_goals[:, :, 1].reshape(-1),
            color="#2fdd00",
            zorder=7,
            s=5,
            alpha=1.0,
            marker=".",
        )

        def edge_segments(graph):
            all_pos = np.asarray(graph.states[:, :2])
            edge_index = np.stack([np.asarray(graph.senders), np.asarray(graph.receivers)], axis=0)
            pad_id = int(np.asarray(graph.n_node)) - 1
            is_pad = np.any(edge_index == pad_id, axis=0)
            edge_index = edge_index[:, ~is_pad]
            if edge_index.shape[1] == 0:
                return np.zeros((0, 2, 2))
            starts = all_pos[edge_index[0]]
            ends = all_pos[edge_index[1]]
            return np.stack([starts, ends], axis=1)

        col_edges = LineCollection(edge_segments(graph0), colors="0.2", linewidths=2, alpha=0.5, zorder=3)
        ax.add_collection(col_edges)

        cost_text = ax.text(0.02, 1.00, "Cost: 1.0\nReward: 1.0", va="bottom", size=16, color="k", transform=ax.transAxes)
        kk_text = ax.text(0.99, 1.14, "kk=0", va="bottom", ha="right", size=16, color="k", transform=ax.transAxes)
        safe_text = ax.text(0.99, 1.05, "Unsafe: {}", va="bottom", ha="right", size=16, color="k", transform=ax.transAxes)

        def update_pose(arrows, rects, states, bb_size, lr):
            h = np.asarray(self._normalize_heading(states[:, 2:4]))
            pos = np.asarray(states[:, :2] + lr * self._normalize_heading(states[:, 2:4]))
            theta = np.arctan2(h[:, 1], h[:, 0]) * 180.0 / np.pi
            bb = np.asarray(bb_size)
            radius = float(jnp.linalg.norm(jnp.asarray(bb_size)))
            for i in range(states.shape[0]):
                arrows[i].set_data(x=pos[i, 0], y=pos[i, 1], dx=h[i, 0] * radius / 2, dy=h[i, 1] * radius / 2)
                rects[i].set_xy((pos[i, 0] - bb[0] / 2, pos[i, 1] - bb[1] / 2))
                rects[i].set_angle(theta[i])

        def update(kk: int):
            graph = tree_index(T_graph, kk)
            update_pose(agent_arrows, agent_rects, graph.env_states.agent, self.params["ego_bb_size"], self.params["ego_lr"])
            if self.num_obsts > 0:
                update_pose(obst_arrows, obst_rects, graph.env_states.obstacle, self.params["obst_bb_size"], self.params["obst_lr"])
            col_edges.set_segments(edge_segments(graph))
            if kk < len(rollout.costs):
                all_costs = "\n".join(
                    f"    {self.cost_components[i]}: {rollout.costs[kk][:, i].max():5.4f}"
                    for i in range(rollout.costs[kk].shape[1])
                )
                cost_text.set_text(f"Cost:\n{all_costs}\nReward: {rollout.rewards[kk]:5.4f}")
            if Ta_is_unsafe is not None and kk < len(Ta_is_unsafe):
                safe_text.set_text("Unsafe: {}".format(np.where(Ta_is_unsafe[kk])[0]))
            kk_text.set_text("kk={:04}".format(kk))
            return [col_edges, *agent_arrows, *agent_rects, *obst_arrows, *obst_rects, cost_text, safe_text, kk_text]

        ani = FuncAnimation(fig, update, frames=len(T_graph.n_node), interval=1000 / 30.0, blit=True)
        try:
            save_anim(ani, video_path)
        finally:
            plt.close(fig)

    def plot_agent_speed_from_rollout(self, rollout: Rollout, record: Record, save_path=None, use_body_frame=False):
        T = len(rollout.graph.n_node)
        speeds = np.zeros((T, self.num_agents), dtype=np.float32)
        xs = np.zeros(T, dtype=np.float32)
        for t in range(T):
            g = tree_index(rollout.graph, t)
            speeds[t] = np.asarray(g.env_states.agent[:, 4])
            xs[t] = np.asarray(g.env_states.agent[:, 0]).mean()
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        for a in range(self.num_agents):
            ax.plot(xs, speeds[:, a], label=f"agent{a}")
        ax.set_ylabel("v (m/s)")
        ax.set_xlabel("World X Position (m)")
        ax.legend(ncol=4, fontsize=8)
        fig.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=150)
            plt.close(fig)
        else:
            plt.show()
