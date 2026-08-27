"""Four deterministic WestEnter intersection scenes for demonstration."""

from typing import Tuple

import jax.numpy as jnp

from . import designed_scene_gen_intersection_split_dynamic as _geometry
from defmarl.utils.typing import AgentState, Array, ObstState, PathRefs


WEST_ROAD_IDX = 3
EGO_MIN_SPEED = 1.0 / 3.6
EGO_TARGET_SPEED = 20.0 / 3.6
DYNAMIC_ACCEL = 0.6
DYNAMIC_TARGET_SPEED = 10.0 / 3.6
EGO_PROGRESS = -5.0
STATIC_PROGRESS = 25.0
# Ego and moving obstacle reach the center in about 11.15 s and 10.95 s.
DYNAMIC_ENTRY_DISTANCE = 24.0


def gen_deterministic_scene_WestEnter_with_id(
    scene_index: Array,
    num_agents: int,
    num_ref_points: int,
    xrange: Array,
    yrange: Array,
    lane_width: float,
    lane_centers: Array,
) -> Tuple[AgentState, ObstState, PathRefs, Array, Array, Array, Array]:
    """Build straight/opposite, straight/crossing, left/opposite or left/crossing."""
    del xrange, yrange, lane_width, lane_centers
    scene_index = jnp.clip(jnp.asarray(scene_index, dtype=jnp.int32), 0, 3)
    maneuver = jnp.array(
        [
            _geometry.MANEUVER_STRAIGHT,
            _geometry.MANEUVER_STRAIGHT,
            _geometry.MANEUVER_LEFT,
            _geometry.MANEUVER_LEFT,
        ],
        dtype=jnp.int32,
    )[scene_index]
    dynamic_road = jnp.array([1, 2, 1, 2], dtype=jnp.int32)[scene_index]
    relation = jnp.array(
        [
            _geometry.DYNAMIC_OPPOSITE_DIRECTION,
            _geometry.DYNAMIC_PERPENDICULAR,
            _geometry.DYNAMIC_OPPOSITE_DIRECTION,
            _geometry.DYNAMIC_PERPENDICULAR,
        ],
        dtype=jnp.int32,
    )[scene_index]

    start_road_idx = jnp.asarray(WEST_ROAD_IDX, dtype=jnp.int32)
    ego_lane_idx = jnp.asarray(1, dtype=jnp.int32)
    ego_xy, ego_heading, _, _ = _geometry._path_geometry(
        jnp.asarray(EGO_PROGRESS, dtype=jnp.float32),
        start_road_idx,
        maneuver,
        ego_lane_idx,
    )
    ego_state = _geometry._make_state(
        ego_xy, ego_heading, jnp.asarray(EGO_MIN_SPEED, dtype=jnp.float32)
    )
    agents = jnp.repeat(ego_state[None, :], num_agents, axis=0)
    goals, derivatives, _ = _geometry._generate_reference(
        num_agents,
        num_ref_points,
        start_road_idx,
        maneuver,
        ego_lane_idx,
        jnp.asarray(EGO_TARGET_SPEED, dtype=jnp.float32),
        jnp.asarray(EGO_PROGRESS, dtype=jnp.float32),
    )
    static_obstacle = _geometry._make_static_obstacle(
        jnp.asarray(STATIC_PROGRESS, dtype=jnp.float32),
        start_road_idx,
        maneuver,
        ego_lane_idx,
    )

    dynamic_lane_idx = jnp.where(dynamic_road == 1, 1, 0)
    dynamic_lane_offset = _geometry._lane_centers(dynamic_road)[
        dynamic_lane_idx
    ]
    dynamic_xy = _geometry._road_point(
        dynamic_road,
        jnp.asarray(-DYNAMIC_ENTRY_DISTANCE, dtype=jnp.float32),
        dynamic_lane_offset,
    )
    dynamic_obstacle = _geometry._make_state(
        dynamic_xy,
        _geometry.ROAD_DIRS[dynamic_road],
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    obstacles = jnp.stack([static_obstacle, dynamic_obstacle], axis=0)
    scene_id = (
        (start_road_idx * _geometry.NUM_MANEUVERS + maneuver)
        * _geometry.NUM_DYNAMIC_RELATIONS
        + relation
    ) * _geometry.NUM_PHASES + _geometry.PHASE_START
    return (
        agents,
        obstacles,
        goals,
        derivatives,
        jnp.asarray(DYNAMIC_ACCEL, dtype=jnp.float32),
        jnp.asarray(DYNAMIC_TARGET_SPEED, dtype=jnp.float32),
        scene_id,
    )
