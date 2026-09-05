"""West-entry curriculum for the low-speed split intersection.

The shared intersection generator still owns all road and reference geometry.
This module only specializes the WestEnter curriculum: route probabilities,
early-path initialization, and dynamic-obstacle arrival timing.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr

from defmarl.utils.typing import AgentState, Array, ObstState, PathRefs, PRNGKey

from . import designed_scene_gen_intersection_split_dynamic as _base


# Geometry contract: path samples and vehicle x/y states denote rear-axle
# centers.  The environment's scaling functions apply ego_lr/obst_lr when
# constructing geometric centers and bounding-box vertices.  Applying those
# offsets here as well would shift every collision shape twice.

# Road indices follow the shared generator convention:
# 0 south, 1 east, 2 north, 3 west.
WEST_ROAD_IDX = 3
WEST_MANEUVER_PROBS = jnp.array([1.0 / 3.0] * 3, dtype=jnp.float32)
# Ego always enters from the west.  Exclude SAME_DIRECTION because that would
# also place the dynamic vehicle on the west road.  OPPOSITE_DIRECTION selects
# the east road, while PERPENDICULAR is split evenly between south and north;
# the resulting road probabilities are east/south/north = 1/3 each.
WEST_DYNAMIC_RELATION_PROBS = jnp.array(
    [0.0, 1.0 / 3.0, 2.0 / 3.0], dtype=jnp.float32
)
# Only part of the random curriculum should create an exact arrival-time
# conflict.  The remaining scenes retain interacting traffic, but move its
# center-arrival time far enough ahead of or behind ego to provide easier
# negotiation examples as well.
DYNAMIC_EXACT_ARRIVAL_PROBABILITY = 0.5
DYNAMIC_NON_EXACT_TIME_GAP_RANGE = (3.0, 6.0)
# When ego is already bypassing a static vehicle inside the turn, do not stack
# an exact dynamic-traffic conflict on top of that maneuver.  A later arrival
# and an explicit distance floor keep these compound scenes useful but easier.
DYNAMIC_TURN_OVERLAP_TIME_GAP_RANGE = (6.0, 10.0)
DYNAMIC_TURN_OVERLAP_MIN_DISTANCE_RANGE = (30.0, 45.0)
EGO_MIN_SPEED = 5.0 / 3.6
EGO_MAX_SPEED = 30.0 / 3.6
EGO_REFERENCE_SPEED_RANGE_KMH = (10.0, 30.0)

# PASSED and DONE do not help the early avoidance curriculum.  Renormalizing
# the former START/APPROACH/SIDE weights (0.20/0.35/0.20) preserves their
# relative frequency while assigning exactly zero mass to the last two phases.
WEST_PHASE_PROBS = jnp.array(
    [0.2, 0.4, 0.4, 0.0, 0.0], dtype=jnp.float32
)


def _static_path_limit(maneuver: Array, path_total: Array) -> Array:
    """End of the central segment plus the first third of the exit road."""
    approach_len = _base.ROAD_HALF - _base.TURN_HALF
    turn_curve_len = path_total - 2.0 * approach_len
    # A straight route crosses a 2*TURN_HALF-long central region.  Treating
    # that interval like a turn makes all three maneuvers use the same rule.
    central_segment_len = jnp.where(
        maneuver == _base.MANEUVER_STRAIGHT,
        2.0 * _base.TURN_HALF,
        turn_curve_len,
    )
    exit_first_third = approach_len / 3.0
    return approach_len + central_segment_len + exit_first_third


def _sample_west_path_positions(
    key: PRNGKey,
    phase: Array,
    maneuver: Array,
    path_total: Array,
) -> Tuple[Array, Array]:
    """Sample ego/static progress inside the early WestEnter curriculum.

    Random training only selects the first three branches.  PASSED and DONE
    remain implemented so callers can still force those phases for debugging.
    """
    static_key, gap_key = jr.split(key)
    static_limit = _static_path_limit(maneuver, path_total)

    def start_phase(_):
        # Preserve the established entrance-road lower bound and ego gap while
        # allowing the static vehicle to extend through the complete central
        # segment and into the first third of the exit road.
        static_s = jr.uniform(static_key, (), minval=15.0, maxval=static_limit)
        max_gap = jnp.minimum(24.0, static_s)
        gap = jr.uniform(
            gap_key, (), minval=12.0, maxval=max_gap
        )
        return static_s, static_s - gap

    def approach_phase(_):
        # APPROACH must leave enough distance for steering-rate-limited ego to
        # enter the adjacent lane before reaching the static vehicle.  Do not
        # cap the gap by static_s: negative ego progress is valid and places
        # ego west of the nominal x=-50 m initialization window.
        static_s = jr.uniform(static_key, (), minval=10.0, maxval=static_limit)
        gap = jr.uniform(gap_key, (), minval=24.0, maxval=36.0)
        return static_s, static_s - gap

    def side_phase(_):
        # SIDE represents an in-progress bypass in the adjacent lane.  Ego is
        # never initialized beyond the static vehicle in path progress.
        static_s = jr.uniform(static_key, (), minval=8.0, maxval=static_limit)
        relative_s = jr.uniform(gap_key, (), minval=-8.0, maxval=0.0)
        return static_s, static_s + relative_s

    def passed_phase(_):
        static_s = jr.uniform(
            static_key, (), minval=2.0, maxval=path_total - 18.0
        )
        gap = jr.uniform(gap_key, (), minval=8.0, maxval=18.0)
        return static_s, static_s + gap

    def done_phase(_):
        static_s = jr.uniform(
            static_key, (), minval=2.0, maxval=path_total - 32.0
        )
        gap = jr.uniform(gap_key, (), minval=18.0, maxval=32.0)
        return static_s, static_s + gap

    return jax.lax.switch(
        phase,
        [start_phase, approach_phase, side_phase, passed_phase, done_phase],
        operand=None,
    )


def _distance_under_speed_controller(
    initial_speed: Array,
    target_speed: Array,
    accel_magnitude: Array,
    duration: Array,
) -> Array:
    """Integrate the obstacle's constant-acceleration speed controller."""
    speed_delta = target_speed - initial_speed
    accel_time = jnp.abs(speed_delta) / jnp.maximum(accel_magnitude, 1e-6)
    active_time = jnp.minimum(duration, accel_time)
    signed_accel = jnp.sign(speed_delta) * accel_magnitude
    active_distance = (
        initial_speed * active_time
        + 0.5 * signed_accel * active_time**2
    )
    return active_distance + target_speed * (duration - active_time)


def _make_timed_dynamic_obstacle(
    key: PRNGKey,
    phase: Array,
    relation: Array,
    start_road_idx: Array,
    agents: AgentState,
    static_obstacle: ObstState,
    ego_arrival_time: Array,
    ego_lane_idx: Array,
    ego_and_static_in_turn: Array,
) -> Tuple[ObstState, Array, Array]:
    """Place dynamic traffic using a mixture of exact and offset arrivals."""
    (
        perpendicular_key,
        lane_key,
        initial_speed_key,
        target_speed_key,
        accel_key,
        exact_arrival_key,
        arrival_side_key,
        arrival_gap_key,
        turn_distance_key,
    ) = jr.split(key, 9)
    perpendicular_side = jr.choice(
        perpendicular_key, jnp.array([-1, 1], dtype=jnp.int32)
    )
    obstacle_road = jnp.where(
        relation == _base.DYNAMIC_SAME_DIRECTION,
        start_road_idx,
        jnp.where(
            relation == _base.DYNAMIC_OPPOSITE_DIRECTION,
            (start_road_idx + 2) % 4,
            (start_road_idx + perpendicular_side) % 4,
        ),
    )
    obstacle_direction = _base.ROAD_DIRS[obstacle_road]
    random_obstacle_lane_idx = jr.randint(lane_key, (), minval=0, maxval=2)
    # East-entering traffic travels opposite to ego on the east-west main road.
    # Because the two road directions have opposite right normals, equal numeric
    # lane indices denote different physical lanes.  Selecting ego_lane_idx here
    # therefore prevents an east vehicle from meeting ego head-on on its route.
    obstacle_lane_idx = jnp.where(
        obstacle_road == 1, ego_lane_idx, random_obstacle_lane_idx
    )
    obstacle_lane_offset = _base._lane_centers(obstacle_road)[obstacle_lane_idx]

    random_initial_speed = jr.uniform(
        initial_speed_key,
        (),
        minval=_base.DYNAMIC_INITIAL_SPEED_RANGE_KMH[0] / 3.6,
        maxval=_base.DYNAMIC_INITIAL_SPEED_RANGE_KMH[1] / 3.6,
    )
    initial_speed = jnp.where(phase == _base.PHASE_START, 0.0, random_initial_speed)
    target_speed = jr.uniform(
        target_speed_key,
        (),
        minval=_base.DYNAMIC_TARGET_SPEED_RANGE_KMH[0] / 3.6,
        maxval=_base.DYNAMIC_TARGET_SPEED_RANGE_KMH[1] / 3.6,
    )
    target_speed = jnp.where(
        phase == _base.PHASE_START,
        jnp.maximum(target_speed, 10.0 / 3.6),
        target_speed,
    )
    accel_magnitude = jr.uniform(
        accel_key,
        (),
        minval=_base.DYNAMIC_ACCEL_MAGNITUDE_RANGE[0],
        maxval=_base.DYNAMIC_ACCEL_MAGNITUDE_RANGE[1],
    )

    # Exact-conflict samples preserve the original curriculum.  Other samples
    # arrive 3-6 seconds before or after ego.  An early arrival is used only
    # when it still leaves at least one second of obstacle travel; otherwise it
    # is converted to a late arrival instead of being clipped back toward an
    # accidental near-synchronous conflict.
    sampled_exact_arrival = jr.bernoulli(
        exact_arrival_key, p=DYNAMIC_EXACT_ARRIVAL_PROBABILITY
    )
    exact_arrival = jnp.logical_and(
        sampled_exact_arrival, jnp.logical_not(ego_and_static_in_turn)
    )
    prefer_early = jr.bernoulli(arrival_side_key, p=0.5)
    normal_arrival_gap = jr.uniform(
        arrival_gap_key,
        (),
        minval=DYNAMIC_NON_EXACT_TIME_GAP_RANGE[0],
        maxval=DYNAMIC_NON_EXACT_TIME_GAP_RANGE[1],
    )
    turn_arrival_gap = jr.uniform(
        arrival_gap_key,
        (),
        minval=DYNAMIC_TURN_OVERLAP_TIME_GAP_RANGE[0],
        maxval=DYNAMIC_TURN_OVERLAP_TIME_GAP_RANGE[1],
    )
    arrival_gap = jnp.where(
        ego_and_static_in_turn, turn_arrival_gap, normal_arrival_gap
    )
    can_arrive_early = ego_arrival_time - arrival_gap >= 1.0
    # In the compound turn case always make the dynamic vehicle arrive later.
    use_early = jnp.logical_and(
        jnp.logical_and(prefer_early, can_arrive_early),
        jnp.logical_not(ego_and_static_in_turn),
    )
    offset_arrival_time = jnp.where(
        use_early,
        ego_arrival_time - arrival_gap,
        ego_arrival_time + arrival_gap,
    )
    obstacle_arrival_time = jnp.where(
        exact_arrival, ego_arrival_time, offset_arrival_time
    )

    travel_distance = _distance_under_speed_controller(
        initial_speed, target_speed, accel_magnitude, obstacle_arrival_time
    )
    turn_minimum_distance = jr.uniform(
        turn_distance_key,
        (),
        minval=DYNAMIC_TURN_OVERLAP_MIN_DISTANCE_RANGE[0],
        maxval=DYNAMIC_TURN_OVERLAP_MIN_DISTANCE_RANGE[1],
    )
    travel_distance = jnp.where(
        ego_and_static_in_turn,
        jnp.maximum(travel_distance, turn_minimum_distance),
        travel_distance,
    )
    # Roads are unbounded beyond the nominal +/-50 m initialization window.
    # Do not clip here: a fast obstacle paired with a distant START ego may
    # need to begin outside that window to arrive at the center at the same
    # time.  Clipping would make it cross the conflict point much too early.
    nominal_longitudinal = -travel_distance

    # The nominal candidate gives the requested center-arrival time.  Offset
    # candidates are only fallbacks for an unsafe initial coincidence with
    # ego/static.  They are ordered by timing error, so the smallest safe shift
    # is always selected.
    candidate_longitudinals = nominal_longitudinal + jnp.array(
        [0.0, -6.0, 6.0, -12.0, 12.0], dtype=jnp.float32
    )
    candidate_xys = jax.vmap(
        lambda longitudinal: _base._road_point(
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
    clearance = jnp.minimum(agent_clearance, static_clearance)
    valid = clearance >= 6.0
    # Prefer nominal, then +/-6 m, then +/-12 m whenever one is safe.
    valid_priority = jnp.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=jnp.float32)
    selected_idx = jnp.where(
        jnp.any(valid),
        jnp.argmax(jnp.where(valid, valid_priority, -1.0)),
        jnp.argmax(clearance),
    )
    state = _base._make_state(
        candidate_xys[selected_idx], obstacle_direction, initial_speed
    )
    return state, accel_magnitude, target_speed


def _make_west_scene(
    key: PRNGKey,
    num_agents: int,
    num_ref_points: int,
    maneuver: Array,
    relation: Array,
    phase: Array,
) -> Tuple[AgentState, ObstState, PathRefs, Array, Array, Array, Array]:
    """Build one WestEnter scene without changing the shared generator."""
    (
        lane_key,
        speed_key,
        progress_key,
        agent_key,
        ego_speed_key,
        dynamic_key,
    ) = jr.split(key, 6)
    start_road_idx = jnp.asarray(WEST_ROAD_IDX, dtype=jnp.int32)
    lane_idx = jr.randint(lane_key, (), minval=0, maxval=2)
    reference_speed = jr.uniform(
        speed_key,
        (),
        minval=EGO_REFERENCE_SPEED_RANGE_KMH[0] / 3.6,
        maxval=EGO_REFERENCE_SPEED_RANGE_KMH[1] / 3.6,
    )
    _, _, _, path_total = _base._path_geometry(
        jnp.array(0.0, dtype=jnp.float32), start_road_idx, maneuver, lane_idx
    )
    static_s, ego_s = _sample_west_path_positions(
        progress_key, phase, maneuver, path_total
    )
    agents = _base._make_agents(
        agent_key,
        phase,
        num_agents,
        ego_s,
        start_road_idx,
        maneuver,
        lane_idx,
    )
    # The shared generator still permits ego speeds up to 40 km/h.  Resample
    # only the WestEnter ego speed here so the shared intersection curriculum
    # remains untouched.  START uses the nonzero kinematic lower limit.
    random_ego_speeds = jr.uniform(
        ego_speed_key,
        (num_agents,),
        minval=EGO_MIN_SPEED,
        maxval=EGO_MAX_SPEED,
    )
    ego_speeds = jnp.where(
        phase == _base.PHASE_START,
        jnp.full((num_agents,), EGO_MIN_SPEED, dtype=jnp.float32),
        random_ego_speeds,
    )
    agents = agents.at[:, 4].set(ego_speeds)
    goals, derivatives, _ = _base._generate_reference(
        num_agents,
        num_ref_points,
        start_road_idx,
        maneuver,
        lane_idx,
        reference_speed,
        ego_s,
    )
    static_obstacle = _base._make_static_obstacle(
        static_s, start_road_idx, maneuver, lane_idx
    )

    approach_len = _base.ROAD_HALF - _base.TURN_HALF
    curve_len = path_total - 2.0 * approach_len
    conflict_s = jnp.where(
        maneuver == _base.MANEUVER_STRAIGHT,
        _base.ROAD_HALF,
        approach_len + 0.5 * curve_len,
    )
    # Ego generally accelerates toward its reference speed.  Using 85% of that
    # speed predicts center arrival more faithfully than averaging with the
    # deliberately tiny START speed.
    estimated_ego_speed = jnp.maximum(
        jnp.maximum(jnp.mean(agents[:, 4]), 0.85 * reference_speed), 2.0
    )
    ego_arrival_time = jnp.clip(
        (conflict_s - ego_s) / estimated_ego_speed, 1.0, 12.0
    )
    turn_start_s = approach_len
    turn_end_s = approach_len + curve_len
    static_in_turn = jnp.logical_and(
        static_s >= turn_start_s, static_s <= turn_end_s
    )
    ego_in_turn = jnp.logical_and(
        ego_s >= turn_start_s, ego_s <= turn_end_s
    )
    ego_and_static_in_turn = jnp.logical_and(static_in_turn, ego_in_turn)
    dynamic_obstacle, dynamic_accel, dynamic_target_speed = (
        _make_timed_dynamic_obstacle(
            dynamic_key,
            phase,
            relation,
            start_road_idx,
            agents,
            static_obstacle,
            ego_arrival_time,
            lane_idx,
            ego_and_static_in_turn,
        )
    )
    obstacles = jnp.stack([static_obstacle, dynamic_obstacle], axis=0)
    scene_id = (
        (start_road_idx * _base.NUM_MANEUVERS + maneuver)
        * _base.NUM_DYNAMIC_RELATIONS
        + relation
    ) * _base.NUM_PHASES + phase
    return (
        agents,
        obstacles,
        goals,
        derivatives,
        dynamic_accel,
        dynamic_target_speed,
        scene_id,
    )


class IntersectionSplitDynamicWestEnterScene(_base.IntersectionSplitDynamicScene):
    """Split intersection scene whose ego route always enters from the west."""

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
        super().__init__(
            key,
            num_agents,
            num_ref_points,
            xrange,
            yrange,
            lane_width,
            lane_centers,
            maneuver=maneuver,
            dynamic_relation=dynamic_relation,
            phase=phase,
            fixed_start_road_idx=WEST_ROAD_IDX,
            maneuver_probs=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        )

    def make_with_id(
        self,
    ) -> Tuple[AgentState, ObstState, PathRefs, Array, Array, Array, Array]:
        """Build a WestEnter scene using the dedicated early curriculum."""
        choose_key, scene_key = jr.split(self.key)
        phase_key, maneuver_key, relation_key = jr.split(choose_key, 3)
        sampled_phase = jr.choice(
            phase_key,
            jnp.arange(_base.NUM_PHASES, dtype=jnp.int32),
            p=WEST_PHASE_PROBS,
        )
        sampled_maneuver = jr.choice(
            maneuver_key,
            jnp.array(
                [
                    _base.MANEUVER_LEFT,
                    _base.MANEUVER_STRAIGHT,
                    _base.MANEUVER_RIGHT,
                ],
                dtype=jnp.int32,
            ),
            p=WEST_MANEUVER_PROBS,
        )
        sampled_relation = jr.choice(
            relation_key, 3, p=WEST_DYNAMIC_RELATION_PROBS
        )
        phase = (
            sampled_phase
            if self.phase is None
            else jnp.asarray(self.phase, dtype=jnp.int32)
        )
        maneuver = (
            sampled_maneuver
            if self.maneuver is None
            else jnp.asarray(self.maneuver, dtype=jnp.int32)
        )
        relation = (
            sampled_relation
            if self.dynamic_relation is None
            else jnp.asarray(self.dynamic_relation, dtype=jnp.int32)
        )
        return _make_west_scene(
            scene_key,
            self.num_agents,
            self.num_ref_points,
            maneuver,
            relation,
            phase,
        )


def gen_scene_randomly_split_dynamic_WestEnter(
    key: PRNGKey,
    num_agents: int,
    num_ref_points: int,
    xrange: Array,
    yrange: Array,
    lane_width: float,
    lane_centers: Array,
) -> Tuple[AgentState, ObstState, PathRefs, Array, Array, Array]:
    """Generate a random split scene with ego entering from the west road."""
    return IntersectionSplitDynamicWestEnterScene(
        key,
        num_agents,
        num_ref_points,
        xrange,
        yrange,
        lane_width,
        lane_centers,
    ).make()


def gen_scene_randomly_split_dynamic_WestEnter_with_id(
    key: PRNGKey,
    num_agents: int,
    num_ref_points: int,
    xrange: Array,
    yrange: Array,
    lane_width: float,
    lane_centers: Array,
) -> Tuple[AgentState, ObstState, PathRefs, Array, Array, Array, Array]:
    """Generate a west-entry scene and retain its category for rendering."""
    return IntersectionSplitDynamicWestEnterScene(
        key,
        num_agents,
        num_ref_points,
        xrange,
        yrange,
        lane_width,
        lane_centers,
    ).make_with_id()


# Keep the conventional generator names available inside this dedicated module.
gen_scene_randomly_split_dynamic = gen_scene_randomly_split_dynamic_WestEnter
gen_scene_randomly_split = gen_scene_randomly_split_dynamic_WestEnter
