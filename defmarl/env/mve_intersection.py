import jax
import jax.numpy as jnp
import functools as ft

from typing import Optional, Tuple
from typing_extensions import override

from .mve import MVEEnvState, MVEEnvGraphsTuple
from .mve_lanechangeANDovertake import MVELaneChangeAndOverTake
from .designed_scene_gen_intersection import gen_scene_randomly, gen_handmade_scene
from defmarl.utils.graph import GraphsTuple
from defmarl.utils.typing import Array, AgentState, ObstState, Cost
from defmarl.utils.utils import find_closest_goal_indices
from ..utils.scaling import scaling_calc, scaling_calc_bound


INF = jnp.inf


class MVEIntersection(MVELaneChangeAndOverTake):
    """Intersection turning environment.

    This class keeps the graph/state/action contract of MVELaneChangeAndOverTake,
    but replaces the straight-road assumptions that are unsafe for intersection
    turns: scene generation, rollout bounds, obstacle motion, and x/y boundary
    costs.
    """

    PARAMS = dict(MVELaneChangeAndOverTake.PARAMS)
    PARAMS.update({
        "default_state_range": jnp.array([
            -100., 100., -100., 100.,
            -INF, INF, -INF, INF,
            -180., 180., -INF, INF,
            -INF, INF, -INF, INF,
        ]),
        "rollout_state_range": jnp.array([
            -110., 110., -110., 110.,
            -INF, INF, -INF, INF,
            -180., 180., -INF, INF,
            -INF, INF, -INF, INF,
        ]),
        "rollout_state_b_range": jnp.array([
            -INF, INF, -INF, INF,
            5., 100., -INF, INF,
            -INF, INF, -INF, INF,
            -INF, INF, -INF, INF,
        ]),
        "lane_width": 3.,
        "intersection_radius": 14.5,
        "alpha_thresh": 1.4,
    })
    PARAMS.update({
        "lane_centers": jnp.array([-3., 0., 3.], dtype=jnp.float32),
    })

    def __init__(self,
                 num_agents: int,
                 area_size: Optional[float] = None,
                 max_step: int = 256,
                 max_travel: Optional[float] = None,
                 dt: float = 0.05,
                 reward_min: float = -17.,
                 reward_max: float = 0.5,
                 params: dict = None):
        area_size = MVEIntersection.PARAMS["rollout_state_range"][:4] if area_size is None else area_size
        params = MVEIntersection.PARAMS if params is None else params
        super().__init__(num_agents, area_size, max_step, max_travel, dt, reward_min, reward_max, params)

    @override
    @property
    def cost_components(self) -> Tuple[str, ...]:
        return (
            "agent collisions",
            "obs collisions",
            "bound exceeds x low",
            "bound exceeds x high",
            "bound exceeds y low",
            "bound exceeds y high",
        )

    @override
    @property
    def n_cost(self) -> int:
        return 6

    @override
    def reset(self, key: Array) -> Tuple[GraphsTuple, jnp.ndarray]:
        xrange = self.params["default_state_range"][:2]
        yrange = self.params["default_state_range"][2:4]
        lane_width = self.params["lane_width"]
        lane_centers = self.params["lane_centers"]

        agents, obsts, all_goals, all_dsYddts = gen_scene_randomly(
            key, self.num_agents, self.num_goals, xrange, yrange, lane_width, lane_centers
        )
        # agents, obsts, all_goals, all_dsYddts = gen_handmade_scene(
        #     key, self.num_agents, self.num_goals, xrange, yrange, lane_width, lane_centers
        # )
        self.all_goals = all_goals
        self.all_dsYddts = all_dsYddts
        goals_init_indices = find_closest_goal_indices(agents, all_goals)
        agents_indices = jnp.arange(agents.shape[0])
        goals = all_goals[agents_indices, goals_init_indices, :]
        dsYddts = all_dsYddts[agents_indices, goals_init_indices, :]
        env_state = MVEEnvState(agents, goals, obsts)
        self.num_obsts = obsts.shape[0]
        return self.get_graph(env_state), dsYddts

    @override
    def obst_step_euler(self, o_obst_states: ObstState) -> ObstState:
        num_obsts = o_obst_states.shape[0]
        assert o_obst_states.shape == (num_obsts, self.state_dim)
        o_x = o_obst_states[:, 0]
        o_y = o_obst_states[:, 1]
        o_vx = o_obst_states[:, 2]
        o_vy = o_obst_states[:, 3]
        o_obst_states_new = o_obst_states.at[:, 0].set(o_x + o_vx / 3.6 * self.dt)
        o_obst_states_new = o_obst_states_new.at[:, 1].set(o_y + o_vy / 3.6 * self.dt)
        return o_obst_states_new

    def _bound_cost(self, agent_states: AgentState, A: Array, b: Array) -> Tuple[Array, Array]:
        thresh = self.params["alpha_thresh"]
        alpha = jax.vmap(scaling_calc_bound, in_axes=(0, None, None))(agent_states, A, b)
        return thresh - alpha, 1.0 - alpha

    @override
    def get_cost(self, graph: MVEEnvGraphsTuple) -> Tuple[Cost, Cost]:
        thresh = self.params["alpha_thresh"]
        num_agents = graph.env_states.agent.shape[0]
        num_obsts = graph.env_states.obstacle.shape[0]
        agent_states = graph.type_states(type_idx=self.AGENT, n_type=num_agents)

        if num_agents == 1:
            a_agent_cost = -jnp.ones((num_agents,), dtype=jnp.float32)
            a_agent_cost_real = -jnp.ones((num_agents,), dtype=jnp.float32)
        else:
            a_agent_cost = -jnp.ones((num_agents,), dtype=jnp.float32)
            a_agent_cost_real = -jnp.ones((num_agents,), dtype=jnp.float32)

        if num_obsts == 0:
            a_obst_cost = -jnp.ones((num_agents,), dtype=jnp.float32)
            a_obst_cost_real = -jnp.ones((num_agents,), dtype=jnp.float32)
        else:
            obstacle_states = graph.type_states(type_idx=self.OBST, n_type=num_obsts)
            i_grid, j_grid = jnp.meshgrid(jnp.arange(num_agents), jnp.arange(num_obsts), indexing="ij")
            state_i_pairs = agent_states[i_grid.flatten(), :]
            state_j_pairs = obstacle_states[j_grid.flatten(), :]
            alpha_pairs = jax.vmap(scaling_calc, in_axes=(0, 0))(state_i_pairs, state_j_pairs)
            alpha_matrix = alpha_pairs.reshape((num_agents, num_obsts))
            a_obst_cost = jnp.max(thresh - alpha_matrix, axis=1)
            a_obst_cost_real = jnp.max(1.0 - alpha_matrix, axis=1)

        state_range = self.params["default_state_range"]
        xl, xh, yl, yh = state_range[0], state_range[1], state_range[2], state_range[3]
        a_bound_xl_cost, a_bound_xl_cost_real = self._bound_cost(agent_states, jnp.array([[1., 0.]]), jnp.array([xl]))
        a_bound_xh_cost, a_bound_xh_cost_real = self._bound_cost(agent_states, jnp.array([[-1., 0.]]), jnp.array([-xh]))
        a_bound_yl_cost, a_bound_yl_cost_real = self._bound_cost(agent_states, jnp.array([[0., 1.]]), jnp.array([yl]))
        a_bound_yh_cost, a_bound_yh_cost_real = self._bound_cost(agent_states, jnp.array([[0., -1.]]), jnp.array([-yh]))

        cost = jnp.stack([
            a_agent_cost,
            a_obst_cost,
            a_bound_xl_cost,
            a_bound_xh_cost,
            a_bound_yl_cost,
            a_bound_yh_cost,
        ], axis=1)
        cost_real = jnp.stack([
            a_agent_cost_real,
            a_obst_cost_real,
            a_bound_xl_cost_real,
            a_bound_xh_cost_real,
            a_bound_yl_cost_real,
            a_bound_yh_cost_real,
        ], axis=1)
        assert cost.shape == (num_agents, self.n_cost)
        assert cost_real.shape == (num_agents, self.n_cost)

        eps = 1.0
        cost = jnp.where(cost <= 0.0, cost, cost + eps)
        cost = jnp.clip(cost, a_min=-3.0)
        return cost, cost_real

    @override
    @ft.partial(jax.jit, static_argnums=(0,))
    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        _, cost_real = self.get_cost(graph)
        return jnp.any(cost_real >= 0.0, axis=-1)
