from typing import Optional, Tuple

from typing_extensions import override

from .mve_lowspeed_CBF_dynamic import MVELaneChangeAndOverTake_LowSpeed_CBF_Dynamic
from defmarl.utils.graph import GraphsTuple
from defmarl.utils.typing import Action, Cost


class MVELaneChangeAndOverTake_LowSpeed_Dynamic(MVELaneChangeAndOverTake_LowSpeed_CBF_Dynamic):
    """Low-speed dynamic-obstacle environment without action-dependent CBF costs."""

    @override
    def get_cost(self, graph: GraphsTuple, action: Optional[Action] = None) -> Tuple[Cost, Cost]:
        del action
        return self._scaling_cost(graph)
