from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jr

from .designed_scene_gen_two_lane_split import (
    generate_horizontal_path_points,
    generate_lanechange_path_points,
    make_state,
)
from defmarl.utils.typing import AgentState, Array, ObstState, PathRefs, PRNGKey


# Geometry contract: every generated state's x/y coordinate is the rear-axle
# center.  ``ego_lr`` and ``obst_lr`` are applied exactly once by the scaling
# functions when they convert a state to its geometric center; the scene
# generator must not pre-shift positions by either offset.

DYNAMIC_OBST_ACCEL_RANGE = (0.1, 7.0)
DYNAMIC_OBST_MAX_SPEED_RANGE_KMH = (10.0, 60.0)

EGO_MIN_INITIAL_SPEED = 5.0 / 3.6
EGO_MAX_SPEED = 30.0 / 3.6
EGO_REFERENCE_SPEED_RANGE_KMH = (10.0, 30.0)
_EGO_NOMINAL_ACCEL = 2.0
_INITIAL_LONGITUDINAL_GAP = 10.0
_APPROACH_MAX_INITIAL_GAP = 70.0
_INITIAL_X_SOFT_MARGIN = 10.0
# Phase probabilities, summing over the two reference types:
# START APPROACH SIDE PASSED DONE YIELD_RESUME EGO_FIRST
_SCENE_PROBS = jnp.array(
    [0.075, 0.125, 0.125, 0, 0, 0.1, 0.075] * 2
)

_START = 0
_APPROACH = 1
_SIDE = 2
_PASSED = 3
_DONE = 4
_YIELD_RESUME = 5
_EGO_FIRST = 6
_NUM_PHASES = 7

def _accelerating_distance(v0: Array, target_v: Array, accel: float, t: Array) -> Array:
    """Distance travelled while accelerating uniformly and then cruising."""
    accel = jnp.asarray(accel, dtype=jnp.float32)
    accel_time = jnp.maximum((target_v - v0) / accel, 0.0)
    time_accelerating = jnp.minimum(t, accel_time)
    distance_accelerating = v0 * time_accelerating + 0.5 * accel * time_accelerating ** 2
    return distance_accelerating + target_v * jnp.maximum(t - accel_time, 0.0)


def _arrival_time(
    distance: Array, v0: Array, target_v: Array, accel: Array
) -> Array:
    """Time needed to cover distance while accelerating and then cruising."""
    distance = jnp.maximum(distance, 0.0)
    target_v = jnp.maximum(target_v, v0)
    accel = jnp.maximum(accel, 1e-6)
    accel_time = (target_v - v0) / accel
    accel_distance = v0 * accel_time + 0.5 * accel * accel_time ** 2
    time_if_accelerating = (
        -v0 + jnp.sqrt(jnp.maximum(v0 ** 2 + 2.0 * accel * distance, 0.0))
    ) / accel
    time_if_cruising = accel_time + (distance - accel_distance) / jnp.maximum(
        target_v, 1e-6
    )
    return jnp.where(distance <= accel_distance, time_if_accelerating, time_if_cruising)


def _make_reference(
    key: PRNGKey,
    lane_change: bool,
    num_agents: int,
    num_ref_points: int,
    xrange: Array,
    lane_centers: Array,
):
    start_x_key, terminal_x_key, lane_key, speed_key = jr.split(key, 4)
    range_width = xrange[1] - xrange[0]
    start_x = jr.uniform(
        start_x_key,
        shape=(),
        dtype=jnp.float32,
        minval=xrange[0],
        maxval=xrange[0] + range_width / 3,
    )
    terminal_x = jr.uniform(
        terminal_x_key,
        shape=(),
        dtype=jnp.float32,
        minval=xrange[0] + 2 * range_width / 3,
        maxval=xrange[1],
    )
    start_lane = jr.choice(lane_key, jnp.array([0, 1], dtype=jnp.int32), shape=())
    terminal_lane = jnp.where(lane_change, 1 - start_lane, start_lane)
    start_y = lane_centers[start_lane]
    terminal_y = lane_centers[terminal_lane]
    terminal_v = jr.uniform(
        speed_key,
        shape=(),
        dtype=jnp.float32,
        minval=EGO_REFERENCE_SPEED_RANGE_KMH[0] / 3.6,
        maxval=EGO_REFERENCE_SPEED_RANGE_KMH[1] / 3.6,
    )

    def make_lane_change_reference(_):
        start = make_state(start_x, start_y, jnp.array(0.0), terminal_v)
        terminal = make_state(terminal_x, terminal_y, jnp.array(0.0), terminal_v)
        return generate_lanechange_path_points(
            xrange, num_agents, num_ref_points, start, terminal
        )

    def make_overtake_reference(_):
        return generate_horizontal_path_points(
            xrange, num_agents, num_ref_points, start_y, terminal_v
        )

    goals, derivatives = jax.lax.cond(
        lane_change,
        make_lane_change_reference,
        make_overtake_reference,
        operand=None,
    )

    return start_x, start_y, terminal_y, terminal_v, goals, derivatives


def _make_static_obstacle(key: PRNGKey, xrange: Array, lane_centers: Array):
    x_key, lane_key = jr.split(key)
    static_x = jr.uniform(
        x_key,
        shape=(),
        dtype=jnp.float32,
        minval=xrange[0] + 35.0,
        maxval=xrange[1] - 15.0,
    )
    static_lane = jr.choice(lane_key, jnp.array([0, 1], dtype=jnp.int32), shape=())
    static_state = make_state(
        static_x,
        lane_centers[static_lane],
        jnp.array(0.0),
        jnp.array(0.0),
    )
    return static_x, static_lane, static_state


def _make_agents(
    key: PRNGKey,
    phase: int,
    num_agents: int,
    xrange: Array,
    lane_centers: Array,
    start_x: Array,
    start_y: Array,
    terminal_y: Array,
    static_x: Array,
    static_lane: Array,
) -> AgentState:
    x_key, y_key, speed_key = jr.split(key, 3)
    side_y = lane_centers[1 - static_lane]

    def make_start(_):
        agent_x = jr.uniform(x_key, shape=(), dtype=jnp.float32, minval=xrange[0], maxval=start_x)
        agent_v = jnp.full((num_agents,), EGO_MIN_INITIAL_SPEED, dtype=jnp.float32)
        return agent_x, start_y, agent_v, y_key

    def make_approach(_):
        agent_x = static_x - jr.uniform(
            x_key,
            shape=(),
            dtype=jnp.float32,
            minval=15.0,
            maxval=25.0,
        )
        agent_v = jr.uniform(
            speed_key,
            (num_agents,),
            dtype=jnp.float32,
            minval=EGO_MIN_INITIAL_SPEED,
            maxval=EGO_MAX_SPEED,
        )
        return agent_x, start_y, agent_v, y_key

    def make_side(_):
        agent_x = static_x + jr.uniform(x_key, shape=(), dtype=jnp.float32, minval=-8.0, maxval=0.0)
        agent_v = jr.uniform(
            speed_key,
            (num_agents,),
            dtype=jnp.float32,
            minval=EGO_MIN_INITIAL_SPEED,
            maxval=EGO_MAX_SPEED,
        )
        return agent_x, side_y, agent_v, y_key

    def make_passed(_):
        center_key, noise_key = jr.split(y_key)
        agent_x = static_x + jr.uniform(x_key, shape=(), dtype=jnp.float32, minval=8.0, maxval=18.0)
        agent_y = jr.uniform(
            center_key,
            shape=(),
            dtype=jnp.float32,
            minval=jnp.minimum(lane_centers[0], lane_centers[1]),
            maxval=jnp.maximum(lane_centers[0], lane_centers[1]),
        )
        agent_v = jr.uniform(
            speed_key,
            (num_agents,),
            dtype=jnp.float32,
            minval=EGO_MIN_INITIAL_SPEED,
            maxval=EGO_MAX_SPEED,
        )
        return agent_x, agent_y, agent_v, noise_key

    def make_done(_):
        agent_x = static_x + jr.uniform(x_key, shape=(), dtype=jnp.float32, minval=18.0, maxval=32.0)
        agent_v = jr.uniform(
            speed_key,
            (num_agents,),
            dtype=jnp.float32,
            minval=EGO_MIN_INITIAL_SPEED,
            maxval=EGO_MAX_SPEED,
        )
        return agent_x, terminal_y, agent_v, y_key

    def make_yield_resume(_):
        # Low-speed ego must first yield and then resume after the other lane clears.
        agent_x = static_x - jr.uniform(
            x_key, shape=(), dtype=jnp.float32, minval=15.0, maxval=25.0
        )
        agent_v = jr.uniform(
            speed_key,
            (num_agents,),
            dtype=jnp.float32,
            minval=EGO_MIN_INITIAL_SPEED,
            maxval=15.0 / 3.6,
        )
        return agent_x, lane_centers[static_lane], agent_v, y_key

    def make_ego_first(_):
        # Ego has a feasible acceleration window to pass a slower moving vehicle.
        agent_x = static_x - jr.uniform(
            x_key, shape=(), dtype=jnp.float32, minval=15.0, maxval=25.0
        )
        agent_v = jr.uniform(
            speed_key,
            (num_agents,),
            dtype=jnp.float32,
            minval=5.0 / 3.6,
            maxval=20.0 / 3.6,
        )
        return agent_x, lane_centers[static_lane], agent_v, y_key

    agent_x, agent_y, agent_v, agent_y_key = jax.lax.switch(
        phase,
        [
            make_start,
            make_approach,
            make_side,
            make_passed,
            make_done,
            make_yield_resume,
            make_ego_first,
        ],
        operand=None,
    )

    agent_xs = jnp.full((num_agents,), agent_x, dtype=jnp.float32)
    agent_ys = jnp.full((num_agents,), agent_y, dtype=jnp.float32) + jr.uniform(
        agent_y_key, (num_agents,), dtype=jnp.float32, minval=-0.1, maxval=0.1
    )
    return jnp.stack(
        [
            agent_xs,
            agent_ys,
            jnp.ones((num_agents,), dtype=jnp.float32),
            jnp.zeros((num_agents,), dtype=jnp.float32),
            agent_v,
            jnp.zeros((num_agents,), dtype=jnp.float32),
        ],
        axis=1,
    )


def _make_dynamic_obstacle(
    key: PRNGKey,
    phase: int,
    agents: AgentState,
    static_x: Array,
    static_lane: Array,
    terminal_v: Array,
    xrange: Array,
    lane_centers: Array,
):
    (
        mode_key,
        lane_key,
        time_key,
        relevant_gap_key,
        weak_gap_key,
        side_key,
        accel_key,
        max_speed_key,
        arrival_offset_key,
        slow_accel_key,
        slow_speed_key,
        rear_mode_key,
        fast_accel_key,
        fast_speed_key,
        initial_order_key,
        approach_gap_key,
        repair_agent_side_key,
        repair_static_side_key,
        repair_both_side_key,
    ) = jr.split(key, 19)
    is_yield_resume = phase == _YIELD_RESUME
    is_ego_first = phase == _EGO_FIRST
    is_priority_interaction = jnp.logical_or(is_yield_resume, is_ego_first)
    dynamic_accel = jr.uniform(
        accel_key,
        shape=(),
        dtype=jnp.float32,
        minval=DYNAMIC_OBST_ACCEL_RANGE[0],
        maxval=DYNAMIC_OBST_ACCEL_RANGE[1],
    )
    dynamic_max_speed = jr.uniform(
        max_speed_key,
        shape=(),
        dtype=jnp.float32,
        minval=DYNAMIC_OBST_MAX_SPEED_RANGE_KMH[0],
        maxval=DYNAMIC_OBST_MAX_SPEED_RANGE_KMH[1],
    ) / 3.6
    slow_accel = jr.uniform(
        slow_accel_key,
        shape=(),
        dtype=jnp.float32,
        minval=0.1,
        maxval=0.6,
    )
    slow_max_speed = jr.uniform(
        slow_speed_key,
        shape=(),
        dtype=jnp.float32,
        minval=10.0,
        maxval=25.0,
    ) / 3.6
    rear_overtake_mode = jnp.logical_and(
        jr.bernoulli(rear_mode_key, p=0.5),
        is_yield_resume,
    )
    fast_accel = jr.uniform(
        fast_accel_key,
        shape=(),
        dtype=jnp.float32,
        minval=6.0,
        maxval=7.0,
    )
    fast_max_speed = jr.uniform(
        fast_speed_key,
        shape=(),
        dtype=jnp.float32,
        minval=45.0,
        maxval=60.0,
    ) / 3.6
    dynamic_accel = jnp.where(is_ego_first, slow_accel, dynamic_accel)
    dynamic_max_speed = jnp.where(
        is_ego_first, slow_max_speed, dynamic_max_speed
    )
    dynamic_accel = jnp.where(rear_overtake_mode, fast_accel, dynamic_accel)
    dynamic_max_speed = jnp.where(
        rear_overtake_mode, fast_max_speed, dynamic_max_speed
    )
    mode = jr.choice(mode_key, 3, p=jnp.array([0.7, 0.2, 0.1]))
    mean_agent_x = jnp.mean(agents[:, 0])
    mean_agent_y = jnp.mean(agents[:, 1])
    mean_agent_v = jnp.mean(agents[:, 4])
    current_lane = jnp.argmin(jnp.abs(lane_centers - mean_agent_y))
    random_lane = jr.choice(lane_key, jnp.array([0, 1], dtype=jnp.int32), shape=())

    # Most samples place the moving vehicle in the lane available around the static obstacle.
    dynamic_lane = jnp.where(
        phase == _APPROACH,
        random_lane,
        jnp.where(
            mode == 0,
            1 - static_lane,
            jnp.where(mode == 1, current_lane, random_lane),
        ),
    )
    time_low = jnp.where(phase <= _APPROACH, 2.5, 1.5)
    time_high = jnp.where(phase <= _APPROACH, 6.0, 4.5)
    interaction_time = jr.uniform(
        time_key, shape=(), dtype=jnp.float32, minval=time_low, maxval=time_high
    )

    ego_distance = _accelerating_distance(
        mean_agent_v, terminal_v, _EGO_NOMINAL_ACCEL, interaction_time
    )
    dynamic_distance = _accelerating_distance(
        jnp.array(0.0),
        dynamic_max_speed,
        dynamic_accel,
        interaction_time,
    )
    relevant_gap = jr.uniform(relevant_gap_key, shape=(), dtype=jnp.float32, minval=10.0, maxval=20.0)
    weak_gap = jr.uniform(weak_gap_key, shape=(), dtype=jnp.float32, minval=30.0, maxval=45.0)
    gap_magnitude = jnp.where(mode == 2, weak_gap, relevant_gap)
    gap_sign = jnp.where(jr.bernoulli(side_key), 1.0, -1.0)
    desired_gap = gap_sign * gap_magnitude
    interaction_x = mean_agent_x + ego_distance + desired_gap
    dynamic_x = interaction_x - dynamic_distance

    # The future-interaction equation tends to initialize the moving vehicle in
    # front of ego. For broad non-priority phases, retain the sampled magnitude
    # but independently choose its initial front/rear side with equal probability.
    initial_order_sign = jnp.where(
        jr.bernoulli(initial_order_key), 1.0, -1.0
    )
    # APPROACH is the broad static-obstacle bypass curriculum rather than a
    # prescribed right-of-way interaction. Give its moving vehicle a much
    # wider longitudinal spread so it does not collapse onto the two timed
    # YIELD_RESUME/EGO_FIRST scenarios.
    generic_initial_gap = jnp.clip(
        jnp.abs(dynamic_x - mean_agent_x),
        _INITIAL_LONGITUDINAL_GAP,
        45.0,
    )
    approach_initial_gap = jr.uniform(
        approach_gap_key,
        shape=(),
        dtype=jnp.float32,
        minval=_INITIAL_LONGITUDINAL_GAP,
        maxval=_APPROACH_MAX_INITIAL_GAP,
    )
    broad_initial_gap = jnp.where(
        phase == _APPROACH, approach_initial_gap, generic_initial_gap
    )
    balanced_dynamic_x = mean_agent_x + initial_order_sign * broad_initial_gap
    dynamic_x = jnp.where(
        is_priority_interaction, dynamic_x, balanced_dynamic_x
    )

    same_agent_lane = dynamic_lane == current_lane
    same_static_lane = dynamic_lane == static_lane
    safe_from_agents = jnp.logical_or(
        jnp.logical_not(same_agent_lane),
        jnp.abs(dynamic_x - mean_agent_x) >= _INITIAL_LONGITUDINAL_GAP,
    )
    # Repair unsafe same-lane gaps on a randomly selected side. Unlike the old
    # ego+15 m fallback (always in front) or a rear-only repair, this keeps the
    # initial front/rear distribution approximately balanced.
    agent_repair_sign = jnp.where(
        jr.bernoulli(repair_agent_side_key), 1.0, -1.0
    )
    dynamic_x = jnp.where(
        safe_from_agents,
        dynamic_x,
        mean_agent_x + agent_repair_sign * _INITIAL_LONGITUDINAL_GAP,
    )
    safe_from_static_after_agent = jnp.logical_or(
        jnp.logical_not(same_static_lane),
        jnp.abs(dynamic_x - static_x) >= _INITIAL_LONGITUDINAL_GAP,
    )
    dynamic_x = jnp.where(
        safe_from_static_after_agent,
        dynamic_x,
        static_x
        + jnp.where(jr.bernoulli(repair_static_side_key), 1.0, -1.0)
        * _INITIAL_LONGITUDINAL_GAP,
    )
    # When ego and the static obstacle share a lane, repairing against one can
    # make the other gap unsafe. Put such rare cases outside both vehicles,
    # choosing the front or rear side with equal probability.
    final_agent_unsafe = jnp.logical_and(
        same_agent_lane,
        jnp.abs(dynamic_x - mean_agent_x) < _INITIAL_LONGITUDINAL_GAP,
    )
    final_static_unsafe = jnp.logical_and(
        same_static_lane,
        jnp.abs(dynamic_x - static_x) < _INITIAL_LONGITUDINAL_GAP,
    )
    needs_joint_repair = jnp.logical_and(
        jnp.logical_and(same_agent_lane, same_static_lane),
        jnp.logical_or(final_agent_unsafe, final_static_unsafe),
    )
    joint_repair_x = jnp.where(
        jr.bernoulli(repair_both_side_key),
        jnp.maximum(mean_agent_x, static_x) + _INITIAL_LONGITUDINAL_GAP,
        jnp.minimum(mean_agent_x, static_x) - _INITIAL_LONGITUDINAL_GAP,
    )
    dynamic_x = jnp.where(needs_joint_repair, joint_repair_x, dynamic_x)

    # Only the two explicit priority-decision phases use the static obstacle as
    # a shared conflict point. APPROACH deliberately keeps the broad generic
    # moving-vehicle distribution so ego primarily learns to bypass the static car.
    ego_target_v = jnp.maximum(terminal_v, mean_agent_v)
    ego_arrival_time = _arrival_time(
        static_x - mean_agent_x,
        mean_agent_v,
        ego_target_v,
        jnp.asarray(_EGO_NOMINAL_ACCEL, dtype=jnp.float32),
    )
    yield_lead = jr.uniform(
        arrival_offset_key,
        shape=(),
        dtype=jnp.float32,
        minval=0.3,
        maxval=1.0,
    )
    ego_first_delay = jr.uniform(
        arrival_offset_key,
        shape=(),
        dtype=jnp.float32,
        minval=8.0,
        maxval=10.0,
    )
    dynamic_arrival_time = jnp.where(
        is_yield_resume,
        ego_arrival_time - yield_lead,
        jnp.where(
            is_ego_first,
            ego_arrival_time + ego_first_delay,
            ego_arrival_time,
        ),
    )
    dynamic_arrival_time = jnp.maximum(dynamic_arrival_time, 0.5)
    bottleneck_dynamic_distance = _accelerating_distance(
        jnp.array(0.0),
        dynamic_max_speed,
        dynamic_accel,
        dynamic_arrival_time,
    )
    bottleneck_dynamic_x = static_x - bottleneck_dynamic_distance
    dynamic_lane = jnp.where(
        is_priority_interaction, 1 - static_lane, dynamic_lane
    )
    dynamic_x = jnp.where(
        is_priority_interaction, bottleneck_dynamic_x, dynamic_x
    )
    # START must keep ego near the beginning of the road for long-horizon
    # evaluation. Limit only an excessively distant rear moving vehicle rather
    # than translating the complete START scene.
    dynamic_x = jnp.where(
        phase == _START,
        jnp.maximum(dynamic_x, xrange[0] - _INITIAL_X_SOFT_MARGIN),
        dynamic_x,
    )

    dynamic_state = make_state(
        dynamic_x,
        lane_centers[dynamic_lane],
        jnp.array(0.0),
        jnp.array(0.0),
    )
    return dynamic_state, dynamic_accel, dynamic_max_speed


def _make_scene(
    key: PRNGKey,
    lane_change: bool,
    phase: int,
    num_agents: int,
    num_ref_points: int,
    xrange: Array,
    yrange: Array,
    lane_width: float,
    lane_centers: Array,
):
    del yrange, lane_width
    reference_key, static_key, agent_key, dynamic_key, translation_key = jr.split(key, 5)
    start_x, start_y, terminal_y, terminal_v, goals, derivatives = _make_reference(
        reference_key, lane_change, num_agents, num_ref_points, xrange, lane_centers
    )
    static_x, static_lane, static_state = _make_static_obstacle(static_key, xrange, lane_centers)
    # The targeted scenario must actually block ego's reference-start lane.
    # Other phases retain the independently sampled obstacle lane.
    reference_start_lane = jnp.argmin(jnp.abs(lane_centers - start_y))
    # APPROACH still guarantees that the static car blocks ego's current lane,
    # but its moving vehicle remains broadly randomized.
    is_static_blocking_phase = jnp.logical_or(
        phase == _APPROACH,
        jnp.logical_or(phase == _YIELD_RESUME, phase == _EGO_FIRST),
    )
    static_lane = jnp.where(
        is_static_blocking_phase, reference_start_lane, static_lane
    )
    static_state = make_state(
        static_x,
        lane_centers[static_lane],
        jnp.array(0.0),
        jnp.array(0.0),
    )
    agents = _make_agents(
        agent_key,
        phase,
        num_agents,
        xrange,
        lane_centers,
        start_x,
        start_y,
        terminal_y,
        static_x,
        static_lane,
    )
    dynamic_state, dynamic_accel, dynamic_max_speed = _make_dynamic_obstacle(
        dynamic_key,
        phase,
        agents,
        static_x,
        static_lane,
        terminal_v,
        xrange,
        lane_centers,
    )
    obstacles = jnp.stack([static_state, dynamic_state], axis=0)

    # Non-START phases otherwise cluster around the latter half of the road
    # because their x positions are defined relative to the static obstacle.
    # Translate the complete scene together so training also covers smaller
    # absolute x values without changing any relative gaps or encounter timing.
    range_width = xrange[1] - xrange[0]
    desired_agent_x = jr.uniform(
        translation_key,
        shape=(),
        dtype=jnp.float32,
        minval=xrange[0] + 0.25 * range_width,
        maxval=xrange[0] + 0.45 * range_width,
    )
    mean_agent_x = jnp.mean(agents[:, 0])
    desired_x_translation = jnp.where(
        phase == _START,
        0.0,
        jnp.maximum(mean_agent_x - desired_agent_x, 0.0),
    )
    # Preserve relative geometry while enforcing a soft lower x bound. A
    # negative maximum translation shifts the complete scene to the right when
    # it was already generated below the bound.
    vehicle_xs = jnp.concatenate([agents[:, 0], obstacles[:, 0]], axis=0)
    max_left_translation = (
        jnp.min(vehicle_xs) - (xrange[0] - _INITIAL_X_SOFT_MARGIN)
    )
    bounded_x_translation = jnp.minimum(
        desired_x_translation, max_left_translation
    )
    x_translation = jnp.where(
        phase == _START, 0.0, bounded_x_translation
    )
    agents = agents.at[:, 0].add(-x_translation)
    obstacles = obstacles.at[:, 0].add(-x_translation)
    goals = goals.at[:, :, 0].add(-x_translation)
    return agents, obstacles, goals, derivatives, dynamic_accel, dynamic_max_speed


def _gen_scene_randomly_split_dynamic_with_id(
    key: PRNGKey,
    num_agents: int,
    num_ref_points: int,
    xrange: Array,
    yrange: Array,
    lane_width: float,
    lane_centers: Array,
):
    """Generate a dynamic split scene and retain its sampled scene id."""
    choose_key, scene_key = jr.split(key)
    scene_id = jr.choice(choose_key, 2 * _NUM_PHASES, p=_SCENE_PROBS)
    lane_change = scene_id < _NUM_PHASES
    phase = scene_id % _NUM_PHASES
    scene = _make_scene(
        scene_key,
        lane_change,
        phase,
        num_agents,
        num_ref_points,
        xrange,
        yrange,
        lane_width,
        lane_centers,
    )
    return (*scene, scene_id)


def gen_scene_randomly_split_dynamic(
    key: PRNGKey,
    num_agents: int,
    num_ref_points: int,
    xrange: Array,
    yrange: Array,
    lane_width: float,
    lane_centers: Array,
):
    """Generate the seven-phase dynamic split scene."""
    return _gen_scene_randomly_split_dynamic_with_id(
        key,
        num_agents,
        num_ref_points,
        xrange,
        yrange,
        lane_width,
        lane_centers,
    )[:-1]


def gen_scene_randomly_split_dynamic_with_id(
    key: PRNGKey,
    num_agents: int,
    num_ref_points: int,
    xrange: Array,
    yrange: Array,
    lane_width: float,
    lane_centers: Array,
):
    """Generate a dynamic split scene and return its id for visualization."""
    return _gen_scene_randomly_split_dynamic_with_id(
        key,
        num_agents,
        num_ref_points,
        xrange,
        yrange,
        lane_width,
        lane_centers,
    )


def gen_scene_randomly_split(
    key: PRNGKey,
    num_agents: int,
    num_ref_points: int,
    xrange: Array,
    yrange: Array,
    lane_width: float,
    lane_centers: Array,
):
    """Drop-in dynamic replacement for the original split scene generator."""
    return gen_scene_randomly_split_dynamic(
        key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers
    )
