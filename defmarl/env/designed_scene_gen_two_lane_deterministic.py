"""Four deterministic two-lane scenes for evaluation and demonstration."""

from typing import Tuple

import jax
import jax.numpy as jnp

from .designed_scene_gen_two_lane_split import (
    generate_horizontal_path_points,
    generate_lanechange_path_points,
    make_state,
)
from defmarl.utils.typing import AgentState, Array, ObstState, PathRefs


EGO_MIN_SPEED = 1.0 / 3.6
EGO_TARGET_SPEED = 20.0 / 3.6
STATIC_X = 50.0
EGO_X = 10.0
LANE_CHANGE_START_X = 55.0
LANE_CHANGE_END_X = 95.0

# Ego reaches static_x in about 8.45 s.  The fast vehicle starts behind ego,
# reaches static_x in about 7.72 s and lane-change start in about 8.32 s.  The
# slow vehicle reaches static_x only after about 15.43 s.
FAST_ACCEL = 2.0
FAST_SPEED = 30.0 / 3.6
FAST_INITIAL_X = 3.0
SLOW_ACCEL = 0.3
SLOW_SPEED = 10.0 / 3.6
SLOW_INITIAL_X = 20.0

YIELD_RESUME = 5
EGO_FIRST = 6
NUM_PHASES = 7


def gen_deterministic_scene_two_lane_with_id(
    scene_index: Array,
    num_agents: int,
    num_ref_points: int,
    xrange: Array,
    yrange: Array,
    lane_width: float,
    lane_centers: Array,
) -> Tuple[AgentState, ObstState, PathRefs, Array, Array, Array, Array]:
    """Build straight/fast, straight/slow, lane-change/fast or change/slow."""
    del yrange, lane_width
    scene_index = jnp.clip(jnp.asarray(scene_index, dtype=jnp.int32), 0, 3)
    lane_change = scene_index >= 2
    fast_dynamic = (scene_index % 2) == 0
    lower_y = lane_centers[jnp.argmin(lane_centers)]
    upper_y = lane_centers[jnp.argmax(lane_centers)]
    target_speed = jnp.asarray(EGO_TARGET_SPEED, dtype=jnp.float32)

    def make_lane_change_reference(_):
        start = make_state(
            jnp.asarray(LANE_CHANGE_START_X, dtype=jnp.float32),
            lower_y,
            jnp.asarray(0.0, dtype=jnp.float32),
            target_speed,
        )
        terminal = make_state(
            jnp.asarray(LANE_CHANGE_END_X, dtype=jnp.float32),
            upper_y,
            jnp.asarray(0.0, dtype=jnp.float32),
            target_speed,
        )
        return generate_lanechange_path_points(
            xrange, num_agents, num_ref_points, start, terminal
        )

    def make_straight_reference(_):
        return generate_horizontal_path_points(
            xrange, num_agents, num_ref_points, lower_y, target_speed
        )

    goals, derivatives = jax.lax.cond(
        lane_change,
        make_lane_change_reference,
        make_straight_reference,
        operand=None,
    )
    ego_state = make_state(
        jnp.asarray(EGO_X, dtype=jnp.float32),
        lower_y,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(EGO_MIN_SPEED, dtype=jnp.float32),
    )
    static_state = make_state(
        jnp.asarray(STATIC_X, dtype=jnp.float32),
        lower_y,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    dynamic_x = jnp.where(fast_dynamic, FAST_INITIAL_X, SLOW_INITIAL_X)
    dynamic_state = make_state(
        jnp.asarray(dynamic_x, dtype=jnp.float32),
        upper_y,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    agents = jnp.repeat(ego_state[None, :], num_agents, axis=0)
    obstacles = jnp.stack([static_state, dynamic_state], axis=0)
    dynamic_accel = jnp.where(fast_dynamic, FAST_ACCEL, SLOW_ACCEL).astype(
        jnp.float32
    )
    dynamic_max_speed = jnp.where(
        fast_dynamic, FAST_SPEED, SLOW_SPEED
    ).astype(jnp.float32)
    phase = jnp.where(fast_dynamic, YIELD_RESUME, EGO_FIRST)
    scene_id = jnp.where(lane_change, 0, NUM_PHASES) + phase
    return (
        agents,
        obstacles,
        goals,
        derivatives,
        dynamic_accel,
        dynamic_max_speed,
        scene_id,
    )
