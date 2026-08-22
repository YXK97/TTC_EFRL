from typing import Tuple
import numpy as np
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
        "obst_bb_size": jnp.array([4, 2]),
        "gamma": 5.0,
        "issf_epsilon_0": 1.0,
        "issf_epsilon_rate": 1.0,
        "issf_epsilon_min": 10.0,
        "pre_static_penalty": 0.05,
    })

    _SCENE_PHASE_NAMES = (
        "START",
        "APPROACH",
        "SIDE",
        "PASSED",
        "DONE",
        "YIELD_RESUME",
        "EGO_FIRST",
    )

    def get_render_scene_label(self, graph: GraphsTuple) -> str:
        scene_id = int(np.asarray(graph.env_states.scene_id))
        reference_type = "LANE_CHANGE" if scene_id < len(self._SCENE_PHASE_NAMES) else "OVERTAKE"
        phase = self._SCENE_PHASE_NAMES[scene_id % len(self._SCENE_PHASE_NAMES)]
        return f"Scene: {reference_type} / {phase}"

    @override
    def get_cost(self, graph: GraphsTuple, action: Action) -> Tuple[Cost, Cost]:
        return MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF.get_cost(self, graph, action)

    @override
    def get_reward(self, graph: GraphsTuple, action: Action) -> Reward:
        agent = self._observable(graph.env_states.agent)
        goal = self._observable(graph.env_states.goal)
        e = agent - goal
        W = jnp.diag(jnp.array([2.5e-4, 2.5e-4, 0, 0, 1e-4, 0]))
        reward = -jnp.sqrt(jnp.einsum("ai,ij,ja->a", e, W, e.transpose())).mean()
        # reward -= (action[:, 0] ** 2).mean() * 0.0001
        # reward -= (action[:, 1] ** 2).mean() * 0.0001
        static_x = graph.env_states.obstacle[0, 0]
        ego_not_past_static = (agent[:, 0] <= static_x).astype(jnp.float32)
        reward -= self.params["pre_static_penalty"] * ego_not_past_static.mean()
        return reward
