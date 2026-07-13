from typing import Optional

import jax.numpy as jnp

from .designed_scene_gen_two_lane_split import gen_scene_randomly_split
from .mve import MVEEnvState
from .mve_lowspeed_base import LowSpeedAccelMixin
from .utils import process_lane_centers
from defmarl.utils.utils import find_closest_goal_indices


INF = jnp.inf


class MVELaneChangeAndOverTake_LowSpeed_CBF(LowSpeedAccelMixin):
    """Low-speed env with action-dependent CBF costs."""

    use_cbf_cost = True

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
        "rollout_state_range": jnp.array([-5.0, 150.0, -10.0, 10.0]),
        # "agent_init_state_range": jnp.array([-100.0, -50.0, -3.7, 3.7]),
        # "terminal_state_range": jnp.array([50.0, 100.0, -3.7, 3.7]),
        "default_state_range": jnp.array([0.0, 100.0, -3.7, 3.7]),
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
        env_state = MVEEnvState(agents, goals, obsts)
        return self.get_graph(env_state), dsYddts
