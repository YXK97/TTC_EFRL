from typing import Optional, Tuple

import jax.numpy as jnp
from typing_extensions import override

from .designed_scene_gen_two_lane_split_dynamic import (
    DYNAMIC_OBST_ACCEL,
    DYNAMIC_OBST_TARGET_SPEED,
    gen_scene_randomly_split_dynamic,
)
from .mve_lowspeed_CBF import MVELaneChangeAndOverTake_LowSpeed_CBF
from defmarl.utils.graph import GraphsTuple
from defmarl.utils.typing import Array, ObstState
from defmarl.utils.utils import find_closest_goal_indices


class MVELaneChangeAndOverTake_LowSpeed_CBF_Dynamic(MVELaneChangeAndOverTake_LowSpeed_CBF):
    """Ordinary low-speed CBF environment with one accelerating obstacle."""

    def __init__(
        self,
        num_agents: int,
        area_size: Optional[float] = None,
        max_step: int = 256,
        max_travel: Optional[float] = None,
        dt: float = 0.05,
        reward_min: float = -17.0,
        reward_max: float = 0.5,
        params: dict = None,
    ):
        super().__init__(
            num_agents,
            area_size,
            max_step,
            max_travel,
            dt,
            reward_min,
            reward_max,
            params,
        )

    @override
    def reset(self, key: Array) -> Tuple[GraphsTuple, jnp.ndarray]:
        agents, obsts, all_goals, all_dsYddts = gen_scene_randomly_split_dynamic(
            key,
            self.num_agents,
            self.num_goals,
            self.params["default_state_range"][:2],
            self.params["default_state_range"][2:4],
            self.params["lane_width"],
            self.params["lane_centers"],
        )
        self.all_goals = all_goals
        self.all_dsYddts = all_dsYddts
        goal_indices = find_closest_goal_indices(
            self._observable(agents), self._observable(all_goals)
        )
        agent_indices = jnp.arange(agents.shape[0])
        goals = all_goals[agent_indices, goal_indices, :]
        dsYddts = all_dsYddts[agent_indices, goal_indices, :]
        self.num_obsts = obsts.shape[0]
        env_state = self._create_env_state(agents, goals, obsts)
        return self.get_graph(env_state), dsYddts

    @override
    def obst_step_euler(self, obst_states: ObstState) -> ObstState:
        assert obst_states.shape == (2, self.state_dim)
        static_obstacle = obst_states[0]
        dynamic_obstacle = obst_states[1]

        heading = dynamic_obstacle[2:4] / jnp.maximum(
            jnp.linalg.norm(dynamic_obstacle[2:4]), 1e-6
        )
        speed = dynamic_obstacle[4]
        speed_next = jnp.minimum(
            speed + DYNAMIC_OBST_ACCEL * self.dt,
            DYNAMIC_OBST_TARGET_SPEED,
        )
        speed_mid = 0.5 * (speed + speed_next)
        position_next = dynamic_obstacle[:2] + speed_mid * heading * self.dt
        dynamic_obstacle_next = (
            dynamic_obstacle.at[:2]
            .set(position_next)
            .at[2:4]
            .set(heading)
            .at[4]
            .set(speed_next)
        )
        return jnp.stack([static_obstacle, dynamic_obstacle_next], axis=0)
