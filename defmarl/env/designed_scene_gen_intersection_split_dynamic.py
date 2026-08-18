"""Low-speed split-scene generator for an asymmetric two-lane intersection.

The generator deliberately does not inherit from the high-speed intersection
scene classes.  Its public call signature follows the other designed-scene
generators, while every returned vehicle state uses the low-speed convention:

    [x, y, heading_x, heading_y, speed_mps, steering_rad]

The five training phases are defined relative to the static obstacle along the
ego reference path.  They are therefore independent of the world origin and of
whether the ego vehicle has already crossed the intersection.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr

from defmarl.utils.typing import AgentState, Array, ObstState, PathRefs, PRNGKey


# +/-50 m is the nominal scene-generation window.  Roads and references do not
# terminate there; both continue beyond this window during rollout.
ROAD_HALF = 50.0
TURN_HALF = 12.5
POINT_INTERVAL = 0.1

MAIN_LANE_WIDTH = 3.7
AUX_LANE_WIDTH = 3.3
MAIN_LANE_CENTERS = jnp.array([-MAIN_LANE_WIDTH / 2.0, MAIN_LANE_WIDTH / 2.0], dtype=jnp.float32)
AUX_LANE_CENTERS = jnp.array([-AUX_LANE_WIDTH / 2.0, AUX_LANE_WIDTH / 2.0], dtype=jnp.float32)

REFERENCE_SPEED_RANGE_KMH = (10.0, 40.0)
EGO_MIN_INITIAL_SPEED = 1.0 / 3.6
EGO_WHEELBASE = 1.75
DYNAMIC_INITIAL_SPEED_RANGE_KMH = (10.0, 40.0)
DYNAMIC_TARGET_SPEED_RANGE_KMH = (0.0, 40.0)
DYNAMIC_ACCEL_MAGNITUDE_RANGE = (0.1, 3.0)

# road: 0 south, 1 east, 2 north, 3 west.  ROAD_DIRS always points from
# the named entrance towards the intersection center.
ROAD_DIRS = jnp.array(
    [[0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 0.0]],
    dtype=jnp.float32,
)

MANEUVER_LEFT = 0
MANEUVER_RIGHT = 1
MANEUVER_STRAIGHT = 2

DYNAMIC_SAME_DIRECTION = 0
DYNAMIC_OPPOSITE_DIRECTION = 1
DYNAMIC_PERPENDICULAR = 2

PHASE_START = 0
PHASE_APPROACH = 1
PHASE_SIDE = 2
PHASE_PASSED = 3
PHASE_DONE = 4

# The first five IDs are turning scenes and the second five are straight scenes;
# ID % 5 is the
# split phase.  A turning scene samples left/right with equal probability.
SCENE_PROBS = jnp.array([0.075, 0.175, 0.1, 0.1, 0.05] * 2, dtype=jnp.float32)
TURN_DIRECTION_PROBS = jnp.array([0.5, 0.5], dtype=jnp.float32)
DYNAMIC_RELATION_PROBS = jnp.array([1.0 / 3.0] * 3, dtype=jnp.float32)


def _right_normal(direction: Array) -> Array:
    """Return the unit normal on the vehicle's right-hand side."""
    return jnp.stack([direction[1], -direction[0]])


def _lane_centers(road_idx: Array) -> Array:
    """Select main-road or auxiliary-road lane centers for one entrance."""
    is_main_road = jnp.logical_or(road_idx == 1, road_idx == 3)
    return jnp.where(is_main_road, MAIN_LANE_CENTERS, AUX_LANE_CENTERS)


def _road_point(road_idx: Array, longitudinal: Array, lane_offset: Array) -> Array:
    """Map road-local longitudinal/lateral coordinates into world coordinates."""
    direction = ROAD_DIRS[road_idx]
    return direction * longitudinal + _right_normal(direction) * lane_offset


def _make_state(xy: Array, heading: Array, speed: Array, steering: Array = 0.0) -> Array:
    """Create one normalized six-dimensional low-speed vehicle state."""
    heading = heading / jnp.maximum(jnp.linalg.norm(heading), 1e-6)
    return jnp.stack(
        [xy[0], xy[1], heading[0], heading[1], speed, jnp.asarray(steering, dtype=jnp.float32)]
    )


def _bezier_geometry(t: Array, p0: Array, p1: Array, p2: Array, p3: Array) -> Tuple[Array, Array, Array]:
    """Evaluate a cubic Bezier point, first derivative, and signed curvature."""
    omt = 1.0 - t
    point = omt ** 3 * p0 + 3.0 * omt ** 2 * t * p1 + 3.0 * omt * t ** 2 * p2 + t ** 3 * p3
    d1 = 3.0 * omt ** 2 * (p1 - p0) + 6.0 * omt * t * (p2 - p1) + 3.0 * t ** 2 * (p3 - p2)
    d2 = 6.0 * omt * (p2 - 2.0 * p1 + p0) + 6.0 * t * (p3 - 2.0 * p2 + p1)
    speed_param = jnp.maximum(jnp.linalg.norm(d1), 1e-6)
    curvature = (d1[0] * d2[1] - d1[1] * d2[0]) / speed_param ** 3
    return point, d1 / speed_param, curvature


def _path_geometry(
    path_s: Array,
    start_road_idx: Array,
    maneuver: Array,
    lane_idx: Array,
) -> Tuple[Array, Array, Array, Array]:
    """Evaluate ego path position, heading, curvature, and total path length.

    The turn is represented by a tangent-continuous cubic Bezier curve.  This is
    important here because a single circular arc cannot connect main-road lane
    offsets (1.85 m) and auxiliary-road offsets (1.65 m) exactly.
    """
    start_dir = ROAD_DIRS[start_road_idx]
    start_offset = _lane_centers(start_road_idx)[lane_idx]
    turn_sign = jnp.where(maneuver == MANEUVER_LEFT, 1.0, -1.0)
    is_straight = maneuver == MANEUVER_STRAIGHT

    terminal_road_idx = jnp.where(
        is_straight,
        (start_road_idx + 2) % 4,
        (start_road_idx - turn_sign.astype(jnp.int32)) % 4,
    )
    terminal_offset = _lane_centers(terminal_road_idx)[lane_idx]
    terminal_dir_turn = jnp.array(
        [
            start_dir[0] * jnp.cos(turn_sign * jnp.pi / 2.0) - start_dir[1] * jnp.sin(turn_sign * jnp.pi / 2.0),
            start_dir[0] * jnp.sin(turn_sign * jnp.pi / 2.0) + start_dir[1] * jnp.cos(turn_sign * jnp.pi / 2.0),
        ]
    )
    terminal_dir = jnp.where(is_straight, start_dir, terminal_dir_turn)

    # Straight paths are parameterized exactly by travelled distance.  Do not
    # clamp path_s at straight_total: +/-50 m only marks the nominal generation
    # window, while the road itself is unbounded in its travel direction.
    straight_total = 2.0 * ROAD_HALF
    straight_s = path_s
    straight_xy = _road_point(start_road_idx, -ROAD_HALF + straight_s, start_offset)

    approach_len = ROAD_HALF - TURN_HALF
    p0 = _road_point(start_road_idx, -TURN_HALF, start_offset)
    p3 = terminal_dir * TURN_HALF + _right_normal(terminal_dir) * terminal_offset

    # The standard quarter-circle Bezier handle factor gives a smooth and
    # conservative turn.  The chord-derived radius also handles unequal lanes.
    radius_estimate = jnp.linalg.norm(p3 - p0) / jnp.sqrt(2.0)
    handle_len = 4.0 / 3.0 * jnp.tan(jnp.pi / 8.0) * radius_estimate
    p1 = p0 + handle_len * start_dir
    p2 = p3 - handle_len * terminal_dir
    curve_len = jnp.pi * radius_estimate / 2.0
    turn_total = approach_len + curve_len + (ROAD_HALF - TURN_HALF)

    # The same rule applies to turns.  Before the Bezier interval the approach
    # continues backwards; after it the exit ray continues forwards forever.
    turn_s = path_s
    curve_t = jnp.clip((turn_s - approach_len) / jnp.maximum(curve_len, 1e-6), 0.0, 1.0)
    curve_xy, curve_heading, curve_curvature = _bezier_geometry(curve_t, p0, p1, p2, p3)
    approach_xy = _road_point(start_road_idx, -ROAD_HALF + turn_s, start_offset)
    exit_distance = turn_s - approach_len - curve_len
    exit_xy = p3 + terminal_dir * exit_distance

    in_approach = turn_s < approach_len
    in_curve = jnp.logical_and(turn_s >= approach_len, turn_s < approach_len + curve_len)
    turn_xy = jnp.where(in_approach, approach_xy, jnp.where(in_curve, curve_xy, exit_xy))
    turn_heading = jnp.where(in_approach, start_dir, jnp.where(in_curve, curve_heading, terminal_dir))
    turn_curvature = jnp.where(in_curve, curve_curvature, 0.0)

    xy = jnp.where(is_straight, straight_xy, turn_xy)
    heading = jnp.where(is_straight, start_dir, turn_heading)
    curvature = jnp.where(is_straight, 0.0, turn_curvature)
    total = jnp.where(is_straight, straight_total, turn_total)
    return xy, heading, curvature, total


def _map_progress_to_adjacent_lane(
    path_s: Array,
    start_road_idx: Array,
    maneuver: Array,
    lane_idx: Array,
) -> Array:
    """Map canonical path progress to the same cross-section of the other lane.

    ``path_s`` remains the logical progress on the original reference path.  On
    straight entrance and exit rays, both lane paths use the same longitudinal
    distance.  Their Bezier portions have different lengths, however, because
    the two lane offsets produce different turning radii.  Inside the turn we
    therefore preserve normalized Bezier progress; after the turn we preserve
    distance travelled beyond the original curve endpoint.

    Keeping this conversion separate is important for SIDE scenes: ego is
    physically initialized on the adjacent path, while its goals must stay on
    the original path so that it learns to merge back after passing the static
    obstacle.
    """
    adjacent_lane_idx = 1 - lane_idx
    zero_s = jnp.array(0.0, dtype=jnp.float32)
    _, _, _, original_total = _path_geometry(
        zero_s, start_road_idx, maneuver, lane_idx
    )
    _, _, _, adjacent_total = _path_geometry(
        zero_s, start_road_idx, maneuver, adjacent_lane_idx
    )

    approach_len = ROAD_HALF - TURN_HALF
    original_curve_len = original_total - 2.0 * approach_len
    adjacent_curve_len = adjacent_total - 2.0 * approach_len

    # Use original-path progress to decide which geometric section contains the
    # requested cross-section.  This also keeps the mapping continuous at both
    # Bezier endpoints.
    normalized_turn_progress = (
        (path_s - approach_len) / jnp.maximum(original_curve_len, 1e-6)
    )
    adjacent_turn_s = (
        approach_len + normalized_turn_progress * adjacent_curve_len
    )
    original_exit_distance = path_s - approach_len - original_curve_len
    adjacent_exit_s = approach_len + adjacent_curve_len + original_exit_distance
    mapped_turn_s = jnp.where(
        path_s < approach_len,
        path_s,
        jnp.where(
            path_s < approach_len + original_curve_len,
            adjacent_turn_s,
            adjacent_exit_s,
        ),
    )

    # Straight paths have no Bezier section, so their two lane progress
    # coordinates are identical.
    return jnp.where(maneuver == MANEUVER_STRAIGHT, path_s, mapped_turn_s)


def _generate_reference(
    num_agents: int,
    num_points: int,
    start_road_idx: Array,
    maneuver: Array,
    lane_idx: Array,
    reference_speed: Array,
    reference_start_s: Array,
    ego_wheelbase: float = EGO_WHEELBASE,
) -> Tuple[PathRefs, Array, Array]:
    """Generate a forward reference beginning at ego's initial path progress.

    A fixed 0.1 m interval and ``num_points`` determine the horizon.  The
    nominal +/-50 m scene window never truncates or repeats the final point.
    """
    path_s = reference_start_s + jnp.arange(num_points, dtype=jnp.float32) * POINT_INTERVAL
    xy, heading, curvature, total = jax.vmap(
        _path_geometry, in_axes=(0, None, None, None)
    )(path_s, start_road_idx, maneuver, lane_idx)

    steering = jnp.arctan(jnp.asarray(ego_wheelbase, dtype=jnp.float32) * curvature)
    speeds = jnp.full((num_points,), reference_speed, dtype=jnp.float32)
    one_goals = jnp.concatenate([xy, heading, speeds[:, None], steering[:, None]], axis=1)
    goals = jnp.repeat(one_goals[None, :, :], num_agents, axis=0)

    # The trainer still expects [y, dy/dt, d2y/dt2, d3y/dt3].  Constant speed
    # makes the lateral acceleration v^2*kappa*heading_x; jerk is left at zero.
    lateral_velocity = reference_speed * heading[:, 1]
    lateral_accel = reference_speed ** 2 * curvature * heading[:, 0]
    derivatives_one = jnp.stack([xy[:, 1], lateral_velocity, lateral_accel, jnp.zeros_like(lateral_accel)], axis=1)
    derivatives = jnp.repeat(derivatives_one[None, :, :], num_agents, axis=0)
    # Geometry parameters are scene-wide, so vmap repeats the same total for
    # every reference point.  Return one scalar for later split sampling.
    return goals, derivatives, total[0]


def _sample_path_positions(key: PRNGKey, phase: Array, path_total: Array) -> Tuple[Array, Array]:
    """Sample static-obstacle and ego progress for one split phase.

    Each branch first chooses a feasible static-obstacle position and then puts
    ego at the requested signed path-distance interval.  No branch refers to the
    intersection center.
    """
    static_key, gap_key = jr.split(key)

    def start_phase(_):
        static_s = jr.uniform(static_key, (), minval=40.0, maxval=path_total - 5.0)
        ego_s = jr.uniform(gap_key, (), minval=0.0, maxval=jnp.maximum(static_s - 35.0, 0.1))
        return static_s, ego_s

    def approach_phase(_):
        static_s = jr.uniform(static_key, (), minval=32.0, maxval=path_total - 2.0)
        gap = jr.uniform(gap_key, (), minval=18.0, maxval=32.0)
        return static_s, static_s - gap

    def side_phase(_):
        static_s = jr.uniform(static_key, (), minval=4.0, maxval=path_total - 4.0)
        relative_s = jr.uniform(gap_key, (), minval=-4.0, maxval=4.0)
        return static_s, static_s + relative_s

    def passed_phase(_):
        static_s = jr.uniform(static_key, (), minval=2.0, maxval=path_total - 18.0)
        gap = jr.uniform(gap_key, (), minval=8.0, maxval=18.0)
        return static_s, static_s + gap

    def done_phase(_):
        static_s = jr.uniform(static_key, (), minval=2.0, maxval=path_total - 32.0)
        gap = jr.uniform(gap_key, (), minval=18.0, maxval=32.0)
        return static_s, static_s + gap

    return jax.lax.switch(
        phase,
        [start_phase, approach_phase, side_phase, passed_phase, done_phase],
        operand=None,
    )


def _make_agents(
    key: PRNGKey,
    phase: Array,
    num_agents: int,
    ego_s: Array,
    start_road_idx: Array,
    maneuver: Array,
    lane_idx: Array,
) -> AgentState:
    """Place ego vehicles at the phase-specific path progress.

    SIDE is the bypass state: its logical progress remains within +/-4 m of the
    static obstacle, but its physical pose lies on the adjacent lane path.  All
    other phases initialize ego directly on its original reference path.
    """
    speed_key, lateral_key = jr.split(key)
    is_side = phase == PHASE_SIDE
    adjacent_lane_idx = 1 - lane_idx
    adjacent_s = _map_progress_to_adjacent_lane(
        ego_s, start_road_idx, maneuver, lane_idx
    )
    physical_s = jnp.where(is_side, adjacent_s, ego_s)
    physical_lane_idx = jnp.where(is_side, adjacent_lane_idx, lane_idx)
    xy, heading, curvature, _ = _path_geometry(
        physical_s, start_road_idx, maneuver, physical_lane_idx
    )
    speeds_random = jr.uniform(
        speed_key,
        (num_agents,),
        minval=EGO_MIN_INITIAL_SPEED,
        maxval=REFERENCE_SPEED_RANGE_KMH[1] / 3.6,
    )
    speeds = jnp.where(phase == PHASE_START, EGO_MIN_INITIAL_SPEED, speeds_random)
    lateral_noise = jr.uniform(lateral_key, (num_agents,), minval=-0.1, maxval=0.1)
    xys = xy[None, :] + lateral_noise[:, None] * _right_normal(heading)[None, :]
    headings = jnp.repeat(heading[None, :], num_agents, axis=0)
    # A SIDE scene can start midway through a turn.  Initialize its steering to
    # the adjacent curve's kinematic-bicycle value instead of forcing a zero
    # steering angle that would immediately point the vehicle off that curve.
    side_steering = jnp.arctan(EGO_WHEELBASE * curvature)
    steering = jnp.where(is_side, side_steering, 0.0)
    steerings = jnp.full((num_agents, 1), steering, dtype=jnp.float32)
    return jnp.concatenate(
        [xys, headings, speeds[:, None], steerings],
        axis=1,
    )


def _make_static_obstacle(
    static_s: Array,
    start_road_idx: Array,
    maneuver: Array,
    lane_idx: Array,
) -> ObstState:
    """Place the static vehicle exactly on the ego reference path."""
    xy, heading, _, _ = _path_geometry(static_s, start_road_idx, maneuver, lane_idx)
    return _make_state(xy, heading, jnp.array(0.0, dtype=jnp.float32))


def _make_dynamic_obstacle(
    key: PRNGKey,
    phase: Array,
    relation: Array,
    start_road_idx: Array,
    agents: AgentState,
    static_obstacle: ObstState,
) -> Tuple[ObstState, Array, Array]:
    """Create a straight-moving obstacle with an independent random lane.

    Relation constrains only the travel direction.  In particular, a same-
    direction obstacle is free to use either lane and is not forced into ego's
    lane.  The returned target speed may be below the initial speed so the
    environment can produce gradual deceleration.
    """
    (
        perpendicular_key,
        lane_key,
        initial_speed_key,
        target_speed_key,
        accel_key,
        time_key,
        gap_sign_key,
        gap_size_key,
    ) = jr.split(key, 8)
    perpendicular_side = jr.choice(
        perpendicular_key, jnp.array([-1, 1], dtype=jnp.int32)
    )
    obstacle_road = jnp.where(
        relation == DYNAMIC_SAME_DIRECTION,
        start_road_idx,
        jnp.where(
            relation == DYNAMIC_OPPOSITE_DIRECTION,
            (start_road_idx + 2) % 4,
            (start_road_idx + perpendicular_side) % 4,
        ),
    )
    obstacle_direction = ROAD_DIRS[obstacle_road]
    obstacle_lane_idx = jr.randint(lane_key, (), minval=0, maxval=2)
    obstacle_lane_offset = _lane_centers(obstacle_road)[obstacle_lane_idx]

    random_initial_speed = jr.uniform(
        initial_speed_key,
        (),
        minval=DYNAMIC_INITIAL_SPEED_RANGE_KMH[0] / 3.6,
        maxval=DYNAMIC_INITIAL_SPEED_RANGE_KMH[1] / 3.6,
    )
    initial_speed = jnp.where(phase == PHASE_START, 0.0, random_initial_speed)
    target_speed = jr.uniform(
        target_speed_key,
        (),
        minval=DYNAMIC_TARGET_SPEED_RANGE_KMH[0] / 3.6,
        maxval=DYNAMIC_TARGET_SPEED_RANGE_KMH[1] / 3.6,
    )
    # START must be able to move after reset, so avoid sampling a second zero.
    target_speed = jnp.where(phase == PHASE_START, jnp.maximum(target_speed, 10.0 / 3.6), target_speed)
    accel_magnitude = jr.uniform(
        accel_key,
        (),
        minval=DYNAMIC_ACCEL_MAGNITUDE_RANGE[0],
        maxval=DYNAMIC_ACCEL_MAGNITUDE_RANGE[1],
    )

    mean_agent_xy = jnp.mean(agents[:, :2], axis=0)
    interaction_time = jr.uniform(time_key, (), minval=2.0, maxval=6.0)
    nominal_dynamic_speed = 0.5 * (initial_speed + target_speed)

    # Same-direction traffic is placed ahead of or behind ego in the selected
    # lane.  Opposite/perpendicular traffic is timed around the central conflict
    # area.  Positioning never changes the independently sampled lane.
    ego_longitudinal = jnp.dot(mean_agent_xy, obstacle_direction)
    signed_gap = jr.choice(gap_sign_key, jnp.array([-1.0, 1.0])) * jr.uniform(
        gap_size_key, (), minval=10.0, maxval=30.0
    )
    same_longitudinal = ego_longitudinal + signed_gap
    conflict_longitudinal = -nominal_dynamic_speed * interaction_time
    obstacle_longitudinal = jnp.where(
        relation == DYNAMIC_SAME_DIRECTION,
        same_longitudinal,
        conflict_longitudinal,
    )
    obstacle_longitudinal = jnp.clip(obstacle_longitudinal, -ROAD_HALF, ROAD_HALF)
    xy = _road_point(obstacle_road, obstacle_longitudinal, obstacle_lane_offset)

    # Compare the original, forward-shifted, and backward-shifted candidates.
    # This also works at a nominal road-window endpoint, where shifting in only
    # one direction could be clipped back to the original unsafe position.
    candidate_longitudinals = jnp.array(
        [
            obstacle_longitudinal,
            jnp.clip(obstacle_longitudinal - 15.0, -ROAD_HALF, ROAD_HALF),
            jnp.clip(obstacle_longitudinal + 15.0, -ROAD_HALF, ROAD_HALF),
        ]
    )
    candidate_xys = jax.vmap(
        lambda longitudinal: _road_point(
            obstacle_road, longitudinal, obstacle_lane_offset
        )
    )(candidate_longitudinals)
    agent_clearance = jnp.min(
        jnp.linalg.norm(candidate_xys[:, None, :] - agents[None, :, :2], axis=2),
        axis=1,
    )
    static_clearance = jnp.linalg.norm(
        candidate_xys - static_obstacle[None, :2], axis=1
    )
    candidate_clearance = jnp.minimum(agent_clearance, static_clearance)
    xy = candidate_xys[jnp.argmax(candidate_clearance)]
    state = _make_state(xy, obstacle_direction, initial_speed)
    return state, accel_magnitude, target_speed


def _make_scene(
    key: PRNGKey,
    num_agents: int,
    num_ref_points: int,
    maneuver: Array,
    relation: Array,
    phase: Array,
) -> Tuple[AgentState, ObstState, PathRefs, Array, Array, Array]:
    """Build one complete scene with two obstacles and fixed output shapes."""
    road_key, lane_key, speed_key, progress_key, agent_key, dynamic_key = jr.split(key, 6)
    start_road_idx = jr.randint(road_key, (), minval=0, maxval=4)
    lane_idx = jr.randint(lane_key, (), minval=0, maxval=2)
    reference_speed = jr.uniform(
        speed_key,
        (),
        minval=REFERENCE_SPEED_RANGE_KMH[0] / 3.6,
        maxval=REFERENCE_SPEED_RANGE_KMH[1] / 3.6,
    )
    # path_total describes only the nominal entrance-to-exit window and is used
    # to place the static obstacle.  Reference generation itself starts later,
    # after ego_s is known, and continues beyond this total without clamping.
    _, _, _, path_total = _path_geometry(
        jnp.array(0.0, dtype=jnp.float32), start_road_idx, maneuver, lane_idx
    )
    static_s, ego_s = _sample_path_positions(progress_key, phase, path_total)
    agents = _make_agents(agent_key, phase, num_agents, ego_s, start_road_idx, maneuver, lane_idx)
    goals, derivatives, _ = _generate_reference(
        num_agents,
        num_ref_points,
        start_road_idx,
        maneuver,
        lane_idx,
        reference_speed,
        ego_s,
    )
    static_obstacle = _make_static_obstacle(static_s, start_road_idx, maneuver, lane_idx)
    dynamic_obstacle, dynamic_accel, dynamic_target_speed = _make_dynamic_obstacle(
        dynamic_key,
        phase,
        relation,
        start_road_idx,
        agents,
        static_obstacle,
    )
    obstacles = jnp.stack([static_obstacle, dynamic_obstacle], axis=0)
    return agents, obstacles, goals, derivatives, dynamic_accel, dynamic_target_speed


class IntersectionSplitDynamicScene:
    """Standalone configurable scene with the conventional ``make`` method."""

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
        # xrange/yrange/lane arguments remain in the constructor for compatibility
        # with the designed-scene interface.  Intersection geometry is asymmetric
        # and therefore uses the explicit constants above instead.
        del xrange, yrange, lane_width, lane_centers
        self.key = key
        self.num_agents = num_agents
        self.num_ref_points = num_ref_points
        self.maneuver = maneuver
        self.dynamic_relation = dynamic_relation
        self.phase = phase

    def make(self) -> Tuple[AgentState, ObstState, PathRefs, Array, Array, Array]:
        choose_key, scene_key = jr.split(self.key)
        scene_id_key, turn_key, relation_key = jr.split(choose_key, 3)
        scene_id = jr.choice(scene_id_key, 10, p=SCENE_PROBS)
        sampled_turn = jr.choice(
            turn_key,
            jnp.array([MANEUVER_LEFT, MANEUVER_RIGHT], dtype=jnp.int32),
            p=TURN_DIRECTION_PROBS,
        )
        sampled_maneuver = jnp.where(
            scene_id < 5, sampled_turn, MANEUVER_STRAIGHT
        )
        maneuver = (
            sampled_maneuver
            if self.maneuver is None
            else jnp.asarray(self.maneuver, dtype=jnp.int32)
        )
        relation = (
            jr.choice(relation_key, 3, p=DYNAMIC_RELATION_PROBS)
            if self.dynamic_relation is None
            else jnp.asarray(self.dynamic_relation, dtype=jnp.int32)
        )
        phase = (
            scene_id % 5
            if self.phase is None
            else jnp.asarray(self.phase, dtype=jnp.int32)
        )
        return _make_scene(
            scene_key,
            self.num_agents,
            self.num_ref_points,
            maneuver,
            relation,
            phase,
        )


class IntersectionSameDirectionDynamicScene:
    """Explicit same-direction scene facade retaining a ``make`` interface."""

    def __init__(self, *args, maneuver: Optional[int] = None, phase: Optional[int] = None):
        self._scene = IntersectionSplitDynamicScene(
            *args,
            maneuver=maneuver,
            dynamic_relation=DYNAMIC_SAME_DIRECTION,
            phase=phase,
        )

    def make(self):
        return self._scene.make()


class IntersectionOppositeDirectionDynamicScene:
    """Explicit opposite-direction scene facade retaining a ``make`` interface."""

    def __init__(self, *args, maneuver: Optional[int] = None, phase: Optional[int] = None):
        self._scene = IntersectionSplitDynamicScene(
            *args,
            maneuver=maneuver,
            dynamic_relation=DYNAMIC_OPPOSITE_DIRECTION,
            phase=phase,
        )

    def make(self):
        return self._scene.make()


class IntersectionPerpendicularDynamicScene:
    """Explicit perpendicular scene facade retaining a ``make`` interface."""

    def __init__(self, *args, maneuver: Optional[int] = None, phase: Optional[int] = None):
        self._scene = IntersectionSplitDynamicScene(
            *args,
            maneuver=maneuver,
            dynamic_relation=DYNAMIC_PERPENDICULAR,
            phase=phase,
        )

    def make(self):
        return self._scene.make()


def gen_scene_randomly_split_dynamic(
    key: PRNGKey,
    num_agents: int,
    num_ref_points: int,
    xrange: Array,
    yrange: Array,
    lane_width: float,
    lane_centers: Array,
):
    """Generate a random maneuver/relation/phase intersection scene."""
    return IntersectionSplitDynamicScene(
        key,
        num_agents,
        num_ref_points,
        xrange,
        yrange,
        lane_width,
        lane_centers,
    ).make()


def gen_scene_randomly_split(
    key: PRNGKey,
    num_agents: int,
    num_ref_points: int,
    xrange: Array,
    yrange: Array,
    lane_width: float,
    lane_centers: Array,
):
    """Compatibility alias matching the other split dynamic generator."""
    return gen_scene_randomly_split_dynamic(
        key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers
    )
