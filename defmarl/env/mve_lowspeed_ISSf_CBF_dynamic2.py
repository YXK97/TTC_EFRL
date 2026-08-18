from typing import Tuple
import jax.numpy as jnp

from typing_extensions import override

from .mve_lowspeed_CBF_dynamic import MVELaneChangeAndOverTake_LowSpeed_CBF_Dynamic
from .mve_lowspeed_ISSf_CBF import MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF
from .mve_lowspeed_ISSf_CBF_dynamic import (
    MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF_Dynamic,
)
from defmarl.utils.graph import GraphsTuple
from defmarl.utils.typing import Action, Cost, Reward


class MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF_Dynamic2(MVELaneChangeAndOverTake_LowSpeed_CBF_Dynamic):
    """Dynamic-obstacle low-speed environment with ego-only ISSf-CBF costs."""

    PARAMS = MVELaneChangeAndOverTake_LowSpeed_CBF_Dynamic.PARAMS.copy()
    PARAMS.update({
        "delta_abs_max": 20.0 * jnp.pi / 180.0,
        "obst_bb_size": jnp.array([4, 2]),
        "gamma": 20.0,
        "issf_epsilon_0": 1.0,
        "issf_epsilon_rate": 1.0,
        "issf_epsilon_min": 100.0,
        "pre_static_penalty": 0.05,
    })

    @override
    def get_cost(self, graph: GraphsTuple, action: Action) -> Tuple[Cost, Cost]:
        return MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF.get_cost(self, graph, action)

    @override
    def get_reward(self, graph: GraphsTuple, action: Action) -> Reward:
        return MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF_Dynamic.get_reward(
            self, graph, action
        )
