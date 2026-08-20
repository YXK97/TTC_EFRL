"""West-entry variant of the low-speed split intersection scene generator.

Only ego's entrance road and maneuver probabilities are constrained here.
Phase selection, lane selection, reference generation, static-obstacle
placement, dynamic-obstacle relation and motion parameters all remain
implemented by ``designed_scene_gen_intersection_split_dynamic``.
"""

from typing import Optional, Tuple

from defmarl.utils.typing import AgentState, Array, ObstState, PathRefs, PRNGKey

from .designed_scene_gen_intersection_split_dynamic import (
    IntersectionSplitDynamicScene,
)


# Road indices follow the shared generator convention:
# 0 south, 1 east, 2 north, 3 west.
WEST_ROAD_IDX = 3
WEST_MANEUVER_PROBS = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)


class IntersectionSplitDynamicWestEnterScene(IntersectionSplitDynamicScene):
    """Split intersection scene whose ego route always enters from the west."""

    def __init__(
        self,
        key: PRNGKey,
        num_agents: int,
        num_ref_points: int,
        xrange: Array,
        yrange: Array,
        lane_width: float,
        lane_centers: Array,
        maneuver: Optional[int] = None,
        dynamic_relation: Optional[int] = None,
        phase: Optional[int] = None,
    ):
        super().__init__(
            key,
            num_agents,
            num_ref_points,
            xrange,
            yrange,
            lane_width,
            lane_centers,
            maneuver=maneuver,
            dynamic_relation=dynamic_relation,
            phase=phase,
            fixed_start_road_idx=WEST_ROAD_IDX,
            maneuver_probs=WEST_MANEUVER_PROBS,
        )


def gen_scene_randomly_split_dynamic_WestEnter(
    key: PRNGKey,
    num_agents: int,
    num_ref_points: int,
    xrange: Array,
    yrange: Array,
    lane_width: float,
    lane_centers: Array,
) -> Tuple[AgentState, ObstState, PathRefs, Array, Array, Array]:
    """Generate a random split scene with ego entering from the west road."""
    return IntersectionSplitDynamicWestEnterScene(
        key,
        num_agents,
        num_ref_points,
        xrange,
        yrange,
        lane_width,
        lane_centers,
    ).make()


# Keep the conventional generator names available inside this dedicated module.
gen_scene_randomly_split_dynamic = gen_scene_randomly_split_dynamic_WestEnter
gen_scene_randomly_split = gen_scene_randomly_split_dynamic_WestEnter
