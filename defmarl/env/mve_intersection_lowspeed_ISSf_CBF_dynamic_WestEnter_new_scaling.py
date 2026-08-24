"""Fixed-west-entry intersection environment with parameterized ray scaling."""

import jax.numpy as jnp
from typing_extensions import override

from defmarl.utils.graph import GraphsTuple
from defmarl.utils.typing import Action, Array, Reward

from .designed_scene_gen_intersection_split_dynamic_WestEnter import (
    gen_scene_randomly_split_dynamic_WestEnter_with_id,
)
from .mve_intersection_lowspeed_ISSf_CBF_dynamic_new_scaling import (
    MVEIntersection_LowSpeed_ISSf_CBF_Dynamic_NewScaling,
)


class MVEIntersection_LowSpeed_ISSf_CBF_Dynamic_WestEnter_NewScaling(
    MVEIntersection_LowSpeed_ISSf_CBF_Dynamic_NewScaling
):
    """West-entry task whose road-boundary cost uses parameterized rays.

    Dynamics, obstacle behavior, ISSf-CBF equations, graph construction,
    diagnostics and rendering come from the new-scaling parent.  Only the scene
    source and the lower speed-tracking reward weight remain WestEnter-specific.
    """

    SAFETY_SIGNALS_ENABLED = True

    @override
    def _generate_scene(self, key: Array):
        return gen_scene_randomly_split_dynamic_WestEnter_with_id(
            key,
            self.num_agents,
            self.num_goals,
            self.params["default_state_range"][:2],
            self.params["default_state_range"][2:4],
            self.params["lane_width"],
            self.params["lane_centers"],
        )

    @override
    def get_reward(self, graph: GraphsTuple, action: Action) -> Reward:
        agent = self._observable(graph.env_states.agent)
        goal = self._observable(graph.env_states.goal)
        error = agent - goal
        weight = jnp.diag(jnp.array([1e-3, 1e-3, 0.0, 0.0, 2e-4, 0.0]))
        return -jnp.sqrt(
            jnp.einsum("ai,ij,ja->a", error, weight, error.transpose())
        ).mean()


MVEIntersectionLowSpeedISSfCBFDynamicWestEnterNewScaling = (
    MVEIntersection_LowSpeed_ISSf_CBF_Dynamic_WestEnter_NewScaling
)
