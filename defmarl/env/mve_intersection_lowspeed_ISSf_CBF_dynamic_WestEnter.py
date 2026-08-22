"""Low-speed ISSf-CBF intersection environment with a fixed west entrance."""
import jax.numpy as jnp

from typing_extensions import override

from defmarl.utils.typing import Array, Action, Reward
from defmarl.utils.graph import GraphsTuple

from .designed_scene_gen_intersection_split_dynamic_WestEnter import (
    gen_scene_randomly_split_dynamic_WestEnter_with_id,
)
from .mve_intersection_lowspeed_ISSf_CBF_dynamic import (
    MVEIntersection_LowSpeed_ISSf_CBF_Dynamic,
)


class MVEIntersection_LowSpeed_ISSf_CBF_Dynamic_WestEnter(
    MVEIntersection_LowSpeed_ISSf_CBF_Dynamic
):
    """Intersection task restricted to routes entering from the west road.

    Dynamics, rewards, graph construction and rendering are inherited unchanged.
    The parent's trajectory-only fixed-cost mode is mirrored explicitly below;
    only the reset scene source differs.
    """

    # Explicitly mirror the parent's complete safety-signal ablation.
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
        e = agent - goal
        W = jnp.diag(jnp.array([1e-3, 1e-3, 0, 0, 2e-4, 0]))
        reward = -jnp.sqrt(jnp.einsum("ai,ij,ja->a", e, W, e.transpose())).mean()
        # reward -= (action[:, 0] ** 2).mean() * 0.0001
        # reward -= (action[:, 1] ** 2).mean() * 0.0001
        return reward


MVEIntersectionLowSpeedISSfCBFDynamicWestEnter = (
    MVEIntersection_LowSpeed_ISSf_CBF_Dynamic_WestEnter
)
