from typing import NamedTuple, Optional, Tuple

import jax.numpy as jnp
from typing_extensions import override

from .designed_scene_gen_two_lane_split_dynamic import (
    gen_scene_randomly_split_dynamic_with_id,
)
from .mve_lowspeed_CBF import MVELaneChangeAndOverTake_LowSpeed_CBF
from defmarl.utils.graph import GraphsTuple
from defmarl.utils.typing import Action, Array, ObstState, Reward, State
from defmarl.utils.utils import find_closest_goal_indices


class MVEDynamicEnvState(NamedTuple):
    agent: State
    goal: State
    obstacle: State
    dynamic_obstacle_accel: Array
    dynamic_obstacle_max_speed: Array
    scene_id: Array

    @property
    def n_agent(self) -> int:
        return self.agent.shape[0]


class MVELaneChangeAndOverTake_LowSpeed_CBF_Dynamic2(MVELaneChangeAndOverTake_LowSpeed_CBF):
    """Ordinary low-speed CBF environment with one accelerating obstacle."""

    PARAMS = MVELaneChangeAndOverTake_LowSpeed_CBF.PARAMS.copy()
    PARAMS.update({
        "gamma": 10.0,
    })

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
        agents, obsts, all_goals, all_dsYddts, dynamic_accel, dynamic_max_speed, scene_id = (
            gen_scene_randomly_split_dynamic_with_id(
                key,
                self.num_agents,
                self.num_goals,
                self.params["default_state_range"][:2],
                self.params["default_state_range"][2:4],
                self.params["lane_width"],
                self.params["lane_centers"],
            )
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
        env_state = MVEDynamicEnvState(
            agents,
            goals,
            obsts,
            dynamic_accel,
            dynamic_max_speed,
            scene_id,
        )
        return self.get_graph(env_state), dsYddts

    @override
    def obst_step_euler(
        self,
        obst_states: ObstState,
        dynamic_accel: Array,
        dynamic_max_speed: Array,
    ) -> ObstState:
        assert obst_states.shape == (2, self.state_dim)
        static_obstacle = obst_states[0]
        dynamic_obstacle = obst_states[1]

        heading = dynamic_obstacle[2:4] / jnp.maximum(
            jnp.linalg.norm(dynamic_obstacle[2:4]), 1e-6
        )
        speed = dynamic_obstacle[4]
        speed_next = jnp.minimum(
            speed + dynamic_accel * self.dt,
            dynamic_max_speed,
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

    @override
    def step(self, graph: GraphsTuple, action: Action, get_eval_info: bool = False):
        del get_eval_info
        env_state = graph.env_states
        action = self.transform_action(action)
        next_agent_states = self.agent_step_euler(env_state.agent, action)
        next_obst_states = self.obst_step_euler(
            env_state.obstacle,
            env_state.dynamic_obstacle_accel,
            env_state.dynamic_obstacle_max_speed,
        )
        next_goal_states, next_dsYddts = self.goal_dsYddt_step(next_agent_states)
        next_env_state = MVEDynamicEnvState(
            next_agent_states,
            next_goal_states,
            next_obst_states,
            env_state.dynamic_obstacle_accel,
            env_state.dynamic_obstacle_max_speed,
            env_state.scene_id,
        )
        reward = self.get_reward(graph, action)
        cost, cost_real = self.get_cost(graph, action)
        return (
            self.get_graph(next_env_state),
            next_dsYddts,
            reward,
            cost,
            cost_real,
            jnp.array(False),
            {},
        )

    @override
    def get_reward(self, graph: GraphsTuple, action: Action) -> Reward:
        agent = self._observable(graph.env_states.agent)
        goal = self._observable(graph.env_states.goal)
        e = agent - goal
        W = jnp.diag(jnp.array([1e-3, 1e-3, 0, 0, 1e-3, 0]))
        reward = -jnp.sqrt(jnp.einsum("ai,ij,ja->a", e, W, e.transpose())).mean()
        reward -= (action[:, 0] ** 2).mean() * 0.0001
        reward -= (action[:, 1] ** 2).mean() * 0.0001
        return reward
