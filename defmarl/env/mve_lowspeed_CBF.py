from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from typing_extensions import override

from .designed_scene_gen_two_lane_split import gen_scene_randomly_split
from .mve import MVEEnvState
from .mve_lowspeed_base import LowSpeedAccelMixin
from .utils import process_lane_centers
from defmarl.utils.graph import GraphsTuple
from defmarl.utils.scaling_lowspeed import scaling_calc
from defmarl.utils.typing import Action, Array, Cost, State
from defmarl.utils.utils import find_closest_goal_indices, gen_i_j_pairs


INF = jnp.inf


class MVELaneChangeAndOverTake_LowSpeed_CBF(LowSpeedAccelMixin):
    """Low-speed env with action-dependent CBF costs."""

    PARAMS = {
        "ego_lf": 0.8475,
        "ego_lr": 0.9025,
        "ego_bb_size": jnp.array([2.625, 1.647]),
        "ego_m": 940.0,
        "ego_Iz": 752.25333,
        "ego_Cf": 47850.0,
        "ego_Cr": 46510.0,
        "comm_radius": 100,
        "obst_bb_size": jnp.array([2.625, 1.647]),
        "obst_lr": 0.9025,
        "bound_bb_size": jnp.array([5.0, 1.0]),
        "rollout_state_range": jnp.array([-500.0, 1500.0, -10.0, 10.0]),
        # "agent_init_state_range": jnp.array([-100.0, -50.0, -3.7, 3.7]),
        # "terminal_state_range": jnp.array([50.0, 100.0, -3.7, 3.7]),
        "default_state_range": jnp.array([0.0, 130.0, -3.7, 3.7]),
        "lane_width": 3.7,
        "alpha_thresh": 1.05,
        "delta_filter_alpha": 0.5,
        "max_delta": 1 * jnp.pi / 180.0,
        "delta_abs_max": 10.0 * jnp.pi / 180.0,
        "min_accel": -5.0,
        "max_accel": 5.0,
        "v_min": 0.0,
        "v_max": 40.0 / 3.6,
        "gamma": 3.0,
    }
    PARAMS.update({
        "ego_radius": jnp.linalg.norm(PARAMS["ego_bb_size"] / 2),
        "ego_L": PARAMS["ego_lf"] + PARAMS["ego_lr"],
        "lane_centers": process_lane_centers(PARAMS["default_state_range"][2:4], PARAMS["lane_width"]),
        "obst_radius": jnp.linalg.norm(PARAMS["obst_bb_size"] / 2),
    })

    def __init__(
        self,
        num_agents: int,
        area_size: Optional[float] = None,
        max_step: int = 512,
        max_travel: Optional[float] = None,
        dt: float = 0.05,
        reward_min: float = -17.0,
        reward_max: float = 0.5,
        params: dict = None,
    ):
        area_size = self.PARAMS["rollout_state_range"][:4] if area_size is None else area_size
        params = self.PARAMS if params is None else params
        super().__init__(num_agents, area_size, max_step, max_travel, dt, reward_min, reward_max, params)
        self.all_goals = jnp.zeros((num_agents, self.num_goals, self.state_dim))
        self.all_dsYddts = jnp.zeros((num_agents, self.num_goals, 4))
        self.num_obsts = 0

    @override
    def reset(self, key):
        c_ycs = self.params["lane_centers"]
        xrange = self.params["default_state_range"][:2]
        yrange = self.params["default_state_range"][2:4]
        lanewidth = self.params["lane_width"]
        agents, obsts, all_goals, all_dsYddts = gen_scene_randomly_split(
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
        env_state = self._create_env_state(agents, goals, obsts)
        return self.get_graph(env_state), dsYddts

    @override
    def _create_env_state(self, agents: State, goals: State, obsts: State) -> MVEEnvState:
        return MVEEnvState(agents, goals, obsts)

    @override
    def _boundary_scaling_cost(self, graph: GraphsTuple, thresh: Array):
        return self._road_boundary_scaling_cost(graph, thresh)

    @override
    def get_cost(self, graph: GraphsTuple, action: Action) -> Tuple[Cost, Cost]:
        return self._cbf_cost(graph, action)

    def _cbf_cost(self, graph: GraphsTuple, action: Action) -> Tuple[Cost, Cost]:
        thresh = self.params["alpha_thresh"]
        gamma = self.params["gamma"]
        num_agents = graph.env_states.agent.shape[0]
        num_obsts = graph.env_states.obstacle.shape[0]
        action_delta = self._filter_delta(graph.env_states.agent[:, 5], action[:, 1])

        def cbf_between(s1, s2, delta_rad, bb_size, lr):
            def alpha_fn(z):
                full = jnp.array([z[0], z[1], z[2], z[3], s1[4], s1[5]])
                return scaling_calc(
                    full, s2, self.params["ego_bb_size"], self.params["ego_lr"], bb_size, lr
                )

            z = s1[:4]
            alpha, grad_z = jax.value_and_grad(alpha_fn)(z)
            alpha = jnp.nan_to_num(alpha, nan=0.0, posinf=1e6, neginf=0.0)
            grad_z = jnp.nan_to_num(grad_z, nan=0.0, posinf=0.0, neginf=0.0)
            hvec = z[2:4] / jnp.maximum(jnp.linalg.norm(z[2:4]), 1e-6)
            omega = s1[4] / self.params["ego_L"] * jnp.tan(delta_rad)
            z_dot = jnp.array(
                [s1[4] * hvec[0], s1[4] * hvec[1], -hvec[1] * omega, hvec[0] * omega]
            )
            cost = -(jnp.dot(grad_z, z_dot) / gamma + alpha - thresh)
            cost = jnp.nan_to_num(cost, nan=10.0, posinf=10.0, neginf=-3.0)
            return cost, 1 - alpha

        a_agent_cost = -jnp.ones((num_agents,), dtype=jnp.float32) * 3.0
        a_agent_cost_real = -jnp.ones((num_agents,), dtype=jnp.float32) * 3.0

        if num_obsts == 0:
            a_obst_cost = -jnp.ones((num_agents,), dtype=jnp.float32) * 3.0
            a_obst_cost_real = -jnp.ones((num_agents,), dtype=jnp.float32) * 3.0
        else:
            i_pairs, j_pairs = gen_i_j_pairs(num_agents, num_obsts)
            costs, reals = jax.vmap(cbf_between, in_axes=(0, 0, 0, None, None))(
                graph.env_states.agent[i_pairs],
                graph.env_states.obstacle[j_pairs],
                action_delta[i_pairs],
                self.params["obst_bb_size"],
                self.params["obst_lr"],
            )
            a_obst_cost = jnp.max(costs.reshape((num_agents, num_obsts)), axis=1)
            a_obst_cost_real = jnp.max(reals.reshape((num_agents, num_obsts)), axis=1)

        bounds = self.generate_bound(graph.env_states.agent, self.params["bound_bb_size"])
        a_low_cost, a_low_real = jax.vmap(cbf_between, in_axes=(0, 0, 0, None, None))(
            graph.env_states.agent,
            bounds[::2],
            action_delta,
            self.params["bound_bb_size"],
            0.0,
        )
        a_high_cost, a_high_real = jax.vmap(cbf_between, in_axes=(0, 0, 0, None, None))(
            graph.env_states.agent,
            bounds[1::2],
            action_delta,
            self.params["bound_bb_size"],
            0.0,
        )

        cost = jnp.stack([a_agent_cost, a_obst_cost, a_low_cost, a_high_cost], axis=1)
        cost_real = jnp.stack([a_agent_cost_real, a_obst_cost_real, a_low_real, a_high_real], axis=1)
        cost = jnp.where(cost <= 0.0, cost, cost + 1.0)
        return jnp.clip(cost, a_min=-10.0, a_max=10.0), cost_real

    @override
    def _boundary_edge_blocks(self, state, node_offset: int):
        del state
        return [], node_offset

    @override
    def _num_boundary_nodes(self, env_state: MVEEnvState) -> int:
        del env_state
        return 0

    @override
    def _add_boundary_nodes(self, node_feats: Array, node_type: Array, env_state: MVEEnvState, cursor: int):
        del env_state
        return node_feats, node_type, cursor
