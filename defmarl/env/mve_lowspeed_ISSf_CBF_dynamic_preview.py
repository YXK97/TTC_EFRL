"""Straight-road dynamic ISSf-CBF environment with two reference goals."""

from typing import Tuple

import jax
import jax.numpy as jnp
from typing_extensions import override

from .designed_scene_gen_two_lane_deterministic import (
    gen_deterministic_scene_two_lane_with_id,
)
from .designed_scene_gen_two_lane_split_dynamic import (
    gen_scene_randomly_split_dynamic_with_id,
)
from .mve_lowspeed_CBF_dynamic import MVEDynamicEnvState
from .mve_lowspeed_ISSf_CBF_dynamic import (
    MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF_Dynamic,
)
from .mve_lowspeed_preview_goal import LowSpeedPreviewGoalMixin
from defmarl.utils.graph import GraphsTuple
from defmarl.utils.typing import Action, Array, Reward


class MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF_Dynamic_Preview(
    LowSpeedPreviewGoalMixin,
    MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF_Dynamic,
):
    """Observe the nearest reference and one speed-dependent preview point."""

    PARAMS = (
        MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF_Dynamic.PARAMS.copy()
    )
    PARAMS.update(
        {
            "preview_base_distance": 5.0,
            "preview_time": 0.5,
            "preview_max_distance": 10.0,
            # Both straight and lane-change generators sample references at
            # approximately 0.1 m path-progress intervals.
            "preview_reference_point_interval": 0.1,

            "gamma": 10.0,
            "issf_epsilon_0": 2.0,
            "issf_epsilon_rate": 2.0,
            "issf_epsilon_min": 10.0,
            "pre_static_penalty": 0.00,
            "v_min": 5.0 / 3.6,
            "v_max": 30.0 / 3.6,
            "issf_safe_barrier_kappa": 0.4,
            "deterministic_scene_train_probability": 0.05,
        }
    )

    def _build_preview_reset(self, scene) -> Tuple[GraphsTuple, Array]:
        """Convert either split or deterministic scene to a two-goal graph."""
        (
            agents,
            obstacles,
            all_goals,
            all_derivatives,
            dynamic_accel,
            dynamic_max_speed,
            scene_id,
        ) = scene
        self.all_goals = all_goals
        self.all_dsYddts = all_derivatives
        goals, tracking_derivatives = self._select_preview_goals(
            agents, all_goals, all_derivatives
        )
        self.num_obsts = obstacles.shape[0]
        env_state = MVEDynamicEnvState(
            agents,
            goals,
            obstacles,
            dynamic_accel,
            dynamic_max_speed,
            scene_id,
        )
        return self.get_graph(env_state), tracking_derivatives

    @override
    def reset_deterministic(
        self, scene_index: Array
    ) -> Tuple[GraphsTuple, Array]:
        scene = gen_deterministic_scene_two_lane_with_id(
            scene_index,
            self.num_agents,
            self.num_goals,
            self.params["default_state_range"][:2],
            self.params["default_state_range"][2:4],
            self.params["lane_width"],
            self.params["lane_centers"],
        )
        return self._build_preview_reset(scene)

    @override
    def reset(self, key: Array) -> Tuple[GraphsTuple, Array]:
        """Retain the original split/fixed training-scene mixture."""
        select_key, fixed_index_key, split_key = jax.random.split(key, 3)
        fixed_probability = jnp.clip(
            jnp.asarray(
                self.params.get(
                    "deterministic_scene_train_probability", 0.05
                ),
                dtype=jnp.float32,
            ),
            0.0,
            1.0,
        )
        use_fixed_scene = jax.random.bernoulli(
            select_key, p=fixed_probability
        )
        fixed_scene_index = jax.random.randint(
            fixed_index_key, (), minval=0, maxval=4, dtype=jnp.int32
        )

        def make_fixed_scene(_):
            return gen_deterministic_scene_two_lane_with_id(
                fixed_scene_index,
                self.num_agents,
                self.num_goals,
                self.params["default_state_range"][:2],
                self.params["default_state_range"][2:4],
                self.params["lane_width"],
                self.params["lane_centers"],
            )

        def make_split_scene(_):
            return gen_scene_randomly_split_dynamic_with_id(
                split_key,
                self.num_agents,
                self.num_goals,
                self.params["default_state_range"][:2],
                self.params["default_state_range"][2:4],
                self.params["lane_width"],
                self.params["lane_centers"],
            )

        scene = jax.lax.cond(
            use_fixed_scene,
            make_fixed_scene,
            make_split_scene,
            operand=None,
        )
        return self._build_preview_reset(scene)

    @override
    def get_reward(self, graph: GraphsTuple, action: Action) -> Reward:
        """Keep reward tied to the nearest point; preview is observation only."""
        return super().get_reward(
            self._graph_with_tracking_goals(graph), action
        )


MVELaneChangeAndOverTakeLowSpeedISSfCBFDynamicPreview = (
    MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF_Dynamic_Preview
)
