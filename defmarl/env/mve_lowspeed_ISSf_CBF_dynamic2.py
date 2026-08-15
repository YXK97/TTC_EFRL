from typing import Tuple
import jax.numpy as jnp

from typing_extensions import override

from .mve_lowspeed_CBF_dynamic import MVELaneChangeAndOverTake_LowSpeed_CBF_Dynamic
from .mve_lowspeed_ISSf_CBF import MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF
from defmarl.utils.graph import GraphsTuple
from defmarl.utils.typing import Action, Cost, Reward


class MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF_Dynamic2(MVELaneChangeAndOverTake_LowSpeed_CBF_Dynamic):
    """Dynamic-obstacle low-speed environment with ego-only ISSf-CBF costs."""

    PARAMS = MVELaneChangeAndOverTake_LowSpeed_CBF_Dynamic.PARAMS.copy()
    PARAMS.update({
        "obst_bb_size": jnp.array([4, 2]),
        "gamma": 20.0,
        "issf_epsilon_0": 1.0,
        "issf_epsilon_rate": 1.0,
        "issf_epsilon_min": 100.0,
    })

    @override
    def get_cost(self, graph: GraphsTuple, action: Action) -> Tuple[Cost, Cost]:
        return MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF.get_cost(self, graph, action)

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