"""Low-speed ISSf-CBF intersection environment with a fixed west entrance."""

from typing_extensions import override

from defmarl.utils.typing import Array

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
    SAFETY_SIGNALS_ENABLED = False

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


MVEIntersectionLowSpeedISSfCBFDynamicWestEnter = (
    MVEIntersection_LowSpeed_ISSf_CBF_Dynamic_WestEnter
)
