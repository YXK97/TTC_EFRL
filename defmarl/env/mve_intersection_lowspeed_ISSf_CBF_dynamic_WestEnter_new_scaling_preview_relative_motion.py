"""WestEnter preview environment with relative-motion obstacle ISSf costs."""

from typing import Tuple

import jax
import jax.numpy as jnp
from typing_extensions import override

from .mve_intersection_lowspeed_ISSf_CBF_dynamic import (
    IntersectionSafetyDiagnostics,
)
from .mve_intersection_lowspeed_ISSf_CBF_dynamic_WestEnter_new_scaling_preview import (
    MVEIntersection_LowSpeed_ISSf_CBF_Dynamic_WestEnter_NewScaling_Preview,
)
from .mve_lowspeed_relative_motion_issf import (
    relative_obstacle_diagnostic_terms,
    relative_obstacle_issf_constraint,
)
from defmarl.utils.graph import GraphsTuple
from defmarl.utils.scaling_lowspeed import scaling_calc_parameterized
from defmarl.utils.typing import Action, Cost
from defmarl.utils.utils import gen_i_j_pairs


class MVEIntersection_LowSpeed_ISSf_CBF_Dynamic_WestEnter_NewScaling_Preview_RelativeMotion(
    MVEIntersection_LowSpeed_ISSf_CBF_Dynamic_WestEnter_NewScaling_Preview
):
    """Add known obstacle translation to each obstacle barrier derivative."""

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
        """Use relative motion for obstacles and the existing static boundary."""
        num_agents = graph.env_states.agent.shape[0]
        if not self.SAFETY_SIGNALS_ENABLED:
            fixed_cost = -jnp.ones(
                (num_agents, self.n_cost), dtype=jnp.float32
            )
            return fixed_cost, fixed_cost

        obstacles = graph.env_states.obstacle
        num_obstacles = obstacles.shape[0]
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
                obstacles[j_pairs],
                steering[i_pairs],
            )
            obstacle_cost = jnp.max(
                pair_cost.reshape((num_agents, num_obstacles)), axis=1
            )
            obstacle_real = jnp.max(
                pair_real.reshape((num_agents, num_obstacles)), axis=1
            )

        boundary_cost, boundary_real = jax.vmap(
            lambda state, steering_value: self._issf_constraint(
                self._intersection_alpha, state, steering_value
            )
        )(graph.env_states.agent, steering)
        cost = jnp.stack(
            [agent_cost, obstacle_cost, boundary_cost], axis=1
        )
        cost_real = jnp.stack(
            [agent_real, obstacle_real, boundary_real], axis=1
        )
        cost = jnp.where(cost <= 0.0, cost, cost + 1.0)
        return jnp.clip(cost, a_min=-3.0, a_max=3.0), cost_real

    @override
    def get_safety_diagnostics(
        self, graph: GraphsTuple, transformed_action: Action
    ) -> IntersectionSafetyDiagnostics:
        """Keep CSV shapes while reporting total relative obstacle h_dot."""
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

        boundary_terms = jax.vmap(
            lambda state, steering_value: self._safety_diagnostic_terms(
                self._intersection_alpha, state, steering_value
            )
        )(agents, steering)
        return IntersectionSafetyDiagnostics(
            steering,
            obstacle_alpha,
            obstacle_grad,
            obstacle_h_dot,
            obstacle_g_dot,
            boundary_terms[0],
            boundary_terms[1],
            boundary_terms[2],
            boundary_terms[3],
        )


MVEIntersectionLowSpeedISSfCBFDynamicWestEnterNewScalingPreviewRelativeMotion = (
    MVEIntersection_LowSpeed_ISSf_CBF_Dynamic_WestEnter_NewScaling_Preview_RelativeMotion
)
