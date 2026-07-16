from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from typing_extensions import override

from .mve import MVE, MVEEnvBoundState
from .mve_lowspeed_base import LowSpeedAccelMixin
from .utils import process_lane_centers
from defmarl.utils.graph import EdgeBlock, GraphsTuple
from defmarl.utils.scaling_lowspeed import scaling_calc
from defmarl.utils.typing import Action, Array, Cost, State


INF = jnp.inf


class MVELaneChangeAndOverTake_LowSpeed_Bound(LowSpeedAccelMixin):
    """Low-speed env with explicit road-boundary nodes in the graph."""

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
        "default_state_range": jnp.array([0.0, 100.0, -3.0, 3.0, -180.0, 180.0, -INF, INF, 0.0, INF, 0.0, INF, 0.0, INF]),
        "lane_width": 3,
        "alpha_thresh": 1.05,
        "delta_filter_alpha": 0.5,
        "max_delta": 0.2 * jnp.pi / 180.0,
        "delta_abs_max": 10.0 * jnp.pi / 180.0,
        "min_accel": -4.0,
        "max_accel": 2.0,
        "v_min": 0.0,
        "v_max": 30.0 / 3.6,
    }
    PARAMS.update({
        "rollout_state_range": jnp.array([
            -5.0,
            150.0,
            PARAMS["default_state_range"][2] - PARAMS["bound_bb_size"][1],
            PARAMS["default_state_range"][3] + PARAMS["bound_bb_size"][1],
            -1.0,
            1.0,
            -1.0,
            1.0,
        ]),
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
    def _create_env_state(self, agents: State, goals: State, obsts: State) -> MVEEnvBoundState:
        bounds = self.generate_bound(agents, self.params["bound_bb_size"])
        return MVEEnvBoundState(agents, goals, bounds, obsts)

    @override
    def _boundary_scaling_cost(self, graph: GraphsTuple, thresh: Array):
        agent_states = graph.env_states.agent
        agent_idx = jnp.arange(agent_states.shape[0])
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
        return thresh - alpha_low, thresh - alpha_high, 1 - alpha_low, 1 - alpha_high

    @override
    def get_cost(self, graph: GraphsTuple, action: Optional[Action] = None) -> Tuple[Cost, Cost]:
        del action
        return self._scaling_cost(graph)

    @override
    def _boundary_edge_blocks(self, state: MVEEnvBoundState, node_offset: int):
        edges = []
        for i_agent in range(state.agent.shape[0]):
            for offset in range(2):
                bound_id = 2 * i_agent + offset
                rel = self._observable(state.agent[i_agent][None, :])[0] - self._observable(
                    state.bound[bound_id][None, :]
                )[0]
                edges.append(
                    EdgeBlock(
                        rel[None, None, :],
                        jnp.ones((1, 1)),
                        jnp.array([i_agent]),
                        jnp.array([node_offset + bound_id]),
                    )
                )
        return edges, node_offset + state.bound.shape[0]

    @override
    def _num_boundary_nodes(self, env_state: MVEEnvBoundState) -> int:
        return env_state.bound.shape[0]

    @override
    def _add_boundary_nodes(
        self, node_feats: Array, node_type: Array, env_state: MVEEnvBoundState, cursor: int
    ):
        num_bounds = env_state.bound.shape[0]
        node_feats = node_feats.at[cursor:cursor + num_bounds, :self.state_dim].set(
            self._observable(env_state.bound)
        )
        node_feats = node_feats.at[cursor:cursor + num_bounds, -3].set(1.0)
        node_type = node_type.at[cursor:cursor + num_bounds].set(MVE.BOUND)
        return node_feats, node_type, cursor + num_bounds
