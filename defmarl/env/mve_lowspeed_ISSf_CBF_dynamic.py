from typing import Tuple

from typing_extensions import override

from .mve_lowspeed_CBF_dynamic import MVELaneChangeAndOverTake_LowSpeed_CBF_Dynamic
from .mve_lowspeed_ISSf_CBF import MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF
from defmarl.utils.graph import GraphsTuple
from defmarl.utils.typing import Action, Cost


class MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF_Dynamic(MVELaneChangeAndOverTake_LowSpeed_CBF_Dynamic):
    """Dynamic-obstacle low-speed environment with ego-only ISSf-CBF costs."""

    PARAMS = MVELaneChangeAndOverTake_LowSpeed_CBF_Dynamic.PARAMS.copy()
    PARAMS.update({
        "gamma": 100.0,
        "issf_epsilon_0": 1.0,
        "issf_epsilon_rate": 1.0,
        "issf_epsilon_min": 100.0,
    })

    @override
    def get_cost(self, graph: GraphsTuple, action: Action) -> Tuple[Cost, Cost]:
        return MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF.get_cost(self, graph, action)
