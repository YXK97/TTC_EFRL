from typing import Tuple
import numpy as np
import jax.numpy as jnp

from typing_extensions import override

from .mve_lowspeed_CBF_dynamic import MVELaneChangeAndOverTake_LowSpeed_CBF_Dynamic
from .mve_lowspeed_ISSf_CBF import MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF
from .mve_lowspeed_ISSf_CBF_dynamic import (
    _get_safe_compressed_cost,
    _reset_deterministic_two_lane,
    _reset_training_scene_mixture,
    _safe_compressed_diagnostic_terms,
)
from .mve_lowspeed_ISSf_CBF import LowSpeedSafetyDiagnostics
from defmarl.utils.graph import GraphsTuple
from defmarl.utils.typing import Action, Cost, Reward


class MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF_Dynamic2(MVELaneChangeAndOverTake_LowSpeed_CBF_Dynamic):
    """Dynamic-obstacle low-speed environment with ego-only ISSf-CBF costs."""

    # Keep the gamma=5 variant geometrically identical to Dynamic1: both use
    # parameterized obstacle rays and direct unbounded road half-planes.
    USE_UNBOUNDED_ISSF_ROAD_BOUNDS = True
    USE_PARAMETERIZED_ISSF_OBSTACLE_SCALING = True

    PARAMS = MVELaneChangeAndOverTake_LowSpeed_CBF_Dynamic.PARAMS.copy()
    PARAMS.update({
        "obst_bb_size": jnp.array([4, 2]),
        "v_min": 1.0 / 3.6,
        "v_max": 30.0 / 3.6,
        "gamma": 2.0,
        "issf_epsilon_0": 1.0,
        "issf_epsilon_rate": 1.0,
        "issf_epsilon_min": 50.0,
        "issf_safe_barrier_kappa": 1.0,
        "pre_static_penalty": 0.02,
        "deterministic_scene_train_probability": 0.02,
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

    def reset_deterministic(self, scene_index):
        """Reset to one of the four fixed two-lane demonstration scenes."""
        return _reset_deterministic_two_lane(self, scene_index)

    @override
    def reset(self, key):
        """Mix the four fixed scenes into training with a small probability."""
        return _reset_training_scene_mixture(self, key)

    @override
    def get_cost(self, graph: GraphsTuple, action: Action) -> Tuple[Cost, Cost]:
        return _get_safe_compressed_cost(self, graph, action)

    def _safety_diagnostic_terms(self, alpha_fn, state, steering):
        return _safe_compressed_diagnostic_terms(
            self, alpha_fn, state, steering
        )

    def get_safety_diagnostics(
        self, graph: GraphsTuple, transformed_action: Action
    ) -> LowSpeedSafetyDiagnostics:
        return MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF.get_safety_diagnostics(
            self, graph, transformed_action
        )

    @override
    def get_reward(self, graph: GraphsTuple, action: Action) -> Reward:
        agent = self._observable(graph.env_states.agent)
        goal = self._observable(graph.env_states.goal)
        e = agent - goal
        W = jnp.diag(jnp.array([2.5e-6, 2.5e-6, 0, 0, 5e-6, 0]))
        reward = -jnp.sqrt(jnp.einsum("ai,ij,ja->a", e, W, e.transpose())).mean()
        # reward -= (action[:, 0] ** 2).mean() * 0.0001
        # reward -= (action[:, 1] ** 2).mean() * 0.0001
        static_x = graph.env_states.obstacle[0, 0]
        ego_not_past_static = (agent[:, 0] <= static_x).astype(jnp.float32)
        reward -= self.params["pre_static_penalty"] * ego_not_past_static.mean()
        return reward
