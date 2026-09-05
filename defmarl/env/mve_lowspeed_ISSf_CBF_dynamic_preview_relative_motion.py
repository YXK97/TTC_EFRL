"""Straight-road preview environment with relative-motion ISSf obstacle costs."""

from typing import Tuple

import jax
import jax.numpy as jnp
from typing_extensions import override

from .mve_lowspeed_ISSf_CBF import LowSpeedSafetyDiagnostics
from .mve_lowspeed_ISSf_CBF_dynamic import (
    _safe_compressed_diagnostic_terms,
)
from .mve_lowspeed_ISSf_CBF_dynamic_preview import (
    MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF_Dynamic_Preview,
)
from .mve_lowspeed_relative_motion_issf import (
    relative_obstacle_diagnostic_terms,
    relative_obstacle_issf_constraint,
)
from defmarl.utils.graph import GraphsTuple
from defmarl.utils.issf_barrier import compress_safe_barrier
from defmarl.utils.scaling_lowspeed import (
    scaling_calc_parameterized,
    scaling_calc_unbounded_bound,
)
from defmarl.utils.typing import Action, Array, Cost
from defmarl.utils.utils import gen_i_j_pairs


class MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF_Dynamic_Preview_RelativeMotion(
    MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF_Dynamic_Preview
):
    """Use both ego and obstacle pose velocities in obstacle CBF derivatives."""

    PARAMS = (
        MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF_Dynamic_Preview.PARAMS.copy()
    )
    PARAMS.update(
        {
            "preview_base_distance": 5.0,
            "preview_time": 0.5,
            "preview_max_distance": 10.0,
            # Both straight and lane-change generators sample references at
            # approximately 0.1 m path-progress intervals.
            "preview_reference_point_interval": 0.1,

            "gamma": 10.0,
            "issf_epsilon_0": 2.0,
            "issf_epsilon_rate": 2.0,
            "issf_epsilon_min": 10.0,
            "pre_static_penalty": 0.00,
            "v_min": 5.0 / 3.6,
            "v_max": 30.0 / 3.6,
            "issf_safe_barrier_kappa": 0.4,
            "deterministic_scene_train_probability": 0.05,
        }
    )

    def _obstacle_scaling(self, ego_state, obstacle_state):
        return scaling_calc_parameterized(
            ego_state,
            obstacle_state,
            self.params["ego_bb_size"],
            self.params["ego_lr"],
            self.params["obst_bb_size"],
            self.params["obst_lr"],
        )

    @override
    def get_cost(self, graph: GraphsTuple, action: Action) -> Tuple[Cost, Cost]:
        """Keep existing geometry and bounds; change obstacle h_dot only."""
        num_agents = graph.env_states.agent.shape[0]
        num_obstacles = graph.env_states.obstacle.shape[0]
        steering = self._filter_delta(
            graph.env_states.agent[:, 5], action[:, 1]
        )

        agent_cost = -jnp.ones((num_agents,), dtype=jnp.float32) * 3.0
        agent_real = -jnp.ones((num_agents,), dtype=jnp.float32) * 3.0
        if num_obstacles == 0:
            obstacle_cost = -jnp.ones((num_agents,), dtype=jnp.float32) * 3.0
            obstacle_real = obstacle_cost
        else:
            i_pairs, j_pairs = gen_i_j_pairs(num_agents, num_obstacles)
            pair_cost, pair_real = jax.vmap(
                lambda state, obstacle, steering_value:
                relative_obstacle_issf_constraint(
                    self,
                    self._obstacle_scaling,
                    state,
                    obstacle,
                    steering_value,
                )
            )(
                graph.env_states.agent[i_pairs],
                graph.env_states.obstacle[j_pairs],
                steering[i_pairs],
            )
            obstacle_cost = jnp.max(
                pair_cost.reshape((num_agents, num_obstacles)), axis=1
            )
            obstacle_real = jnp.max(
                pair_real.reshape((num_agents, num_obstacles)), axis=1
            )

        def boundary_constraint(state, steering_value, A, b):
            def alpha_fn(ego_state):
                return scaling_calc_unbounded_bound(
                    ego_state,
                    self.params["ego_bb_size"],
                    self.params["ego_lr"],
                    A,
                    b,
                )

            alpha, _, barrier_dot, steering_dot = (
                _safe_compressed_diagnostic_terms(
                    self, alpha_fn, state, steering_value
                )
            )
            barrier = self._compressed_barrier(alpha)
            epsilon = self.params["issf_epsilon_min"] + self.params[
                "issf_epsilon_0"
            ] * jax.nn.softplus(
                self.params["issf_epsilon_rate"] * barrier
            )
            residual = (
                barrier_dot / self.params["gamma"]
                + barrier
                - jnp.square(steering_dot)
                / (self.params["gamma"] * epsilon)
            )
            return (
                jnp.nan_to_num(
                    -residual, nan=3.0, posinf=3.0, neginf=-3.0
                ),
                1.0 - alpha,
            )

        y_low = self.params["default_state_range"][2]
        y_high = self.params["default_state_range"][3]
        lower_A, lower_b = jnp.array([[0.0, 1.0]]), jnp.array([y_low])
        upper_A, upper_b = jnp.array([[0.0, -1.0]]), jnp.array([-y_high])
        lower_cost, lower_real = jax.vmap(
            boundary_constraint, in_axes=(0, 0, None, None)
        )(graph.env_states.agent, steering, lower_A, lower_b)
        upper_cost, upper_real = jax.vmap(
            boundary_constraint, in_axes=(0, 0, None, None)
        )(graph.env_states.agent, steering, upper_A, upper_b)

        cost = jnp.stack(
            [agent_cost, obstacle_cost, lower_cost, upper_cost], axis=1
        )
        cost_real = jnp.stack(
            [agent_real, obstacle_real, lower_real, upper_real], axis=1
        )
        cost = jnp.where(cost <= 0.0, cost, cost + 1.0)
        return jnp.clip(cost, a_min=-3.0, a_max=3.0), cost_real

    def _compressed_barrier(self, alpha: Array) -> Array:
        return compress_safe_barrier(
            alpha,
            self.params["alpha_thresh"],
            self.params["issf_safe_barrier_kappa"],
        )

    @override
    def get_safety_diagnostics(
        self, graph: GraphsTuple, transformed_action: Action
    ) -> LowSpeedSafetyDiagnostics:
        """Export total relative h_dot through the existing CSV interface."""
        agents = graph.env_states.agent
        obstacles = graph.env_states.obstacle
        num_agents = agents.shape[0]
        num_obstacles = obstacles.shape[0]
        steering = self._filter_delta(agents[:, 5], transformed_action[:, 1])
        i_pairs, j_pairs = gen_i_j_pairs(num_agents, num_obstacles)
        obstacle_terms = jax.vmap(
            lambda state, obstacle, steering_value:
            relative_obstacle_diagnostic_terms(
                self,
                self._obstacle_scaling,
                state,
                obstacle,
                steering_value,
            )
        )(agents[i_pairs], obstacles[j_pairs], steering[i_pairs])
        obstacle_alpha = obstacle_terms[0].reshape(
            (num_agents, num_obstacles)
        )
        obstacle_grad = obstacle_terms[1].reshape(
            (num_agents, num_obstacles, 4)
        )
        obstacle_h_dot = obstacle_terms[2].reshape(
            (num_agents, num_obstacles)
        )
        obstacle_g_dot = obstacle_terms[3].reshape(
            (num_agents, num_obstacles)
        )

        def boundary_terms(state, steering_value, A, b):
            def alpha_fn(ego_state):
                return scaling_calc_unbounded_bound(
                    ego_state,
                    self.params["ego_bb_size"],
                    self.params["ego_lr"],
                    A,
                    b,
                )

            return _safe_compressed_diagnostic_terms(
                self, alpha_fn, state, steering_value
            )

        y_low = self.params["default_state_range"][2]
        y_high = self.params["default_state_range"][3]
        lower_A, lower_b = jnp.array([[0.0, 1.0]]), jnp.array([y_low])
        upper_A, upper_b = jnp.array([[0.0, -1.0]]), jnp.array([-y_high])
        lower = jax.vmap(boundary_terms, in_axes=(0, 0, None, None))(
            agents, steering, lower_A, lower_b
        )
        upper = jax.vmap(boundary_terms, in_axes=(0, 0, None, None))(
            agents, steering, upper_A, upper_b
        )
        return LowSpeedSafetyDiagnostics(
            steering,
            obstacle_alpha,
            obstacle_grad,
            obstacle_h_dot,
            obstacle_g_dot,
            lower[0],
            lower[1],
            lower[2],
            lower[3],
            upper[0],
            upper[1],
            upper[2],
            upper[3],
        )


MVELaneChangeAndOverTakeLowSpeedISSfCBFDynamicPreviewRelativeMotion = (
    MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF_Dynamic_Preview_RelativeMotion
)
