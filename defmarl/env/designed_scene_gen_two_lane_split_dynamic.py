import functools as ft
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


DYNAMIC_OBST_ACCEL = 3.0
DYNAMIC_OBST_TARGET_SPEED = 30.0 / 3.6

_EGO_NOMINAL_ACCEL = 2.0
_INITIAL_LONGITUDINAL_GAP = 10.0
_SCENE_PROBS = jnp.array([0.075, 0.175, 0.1, 0.1, 0.05] * 2)

_START = 0
_APPROACH = 1
_SIDE = 2
_PASSED = 3
_DONE = 4


def _accelerating_distance(v0: Array, target_v: Array, accel: float, t: Array) -> Array:
    """Distance travelled while accelerating uniformly and then cruising."""
    accel = jnp.asarray(accel, dtype=jnp.float32)
    accel_time = jnp.maximum((target_v - v0) / accel, 0.0)
    time_accelerating = jnp.minimum(t, accel_time)
    distance_accelerating = v0 * time_accelerating + 0.5 * accel * time_accelerating ** 2
    return distance_accelerating + target_v * jnp.maximum(t - accel_time, 0.0)


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
    terminal_lane = 1 - start_lane if lane_change else start_lane
    start_y = lane_centers[start_lane]
    terminal_y = lane_centers[terminal_lane]
    terminal_v = jr.uniform(speed_key, shape=(), dtype=jnp.float32, minval=20.0, maxval=40.0) / 3.6

    if lane_change:
        start = make_state(start_x, start_y, jnp.array(0.0), terminal_v)
        terminal = make_state(terminal_x, terminal_y, jnp.array(0.0), terminal_v)
        goals, derivatives = generate_lanechange_path_points(
            xrange, num_agents, num_ref_points, start, terminal
        )
    else:
        goals, derivatives = generate_horizontal_path_points(
            xrange, num_agents, num_ref_points, start_y, terminal_v
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

    if phase == _START:
        agent_x = jr.uniform(x_key, shape=(), dtype=jnp.float32, minval=xrange[0], maxval=start_x)
        agent_y = start_y
        agent_v = jnp.zeros((num_agents,), dtype=jnp.float32)
    elif phase == _APPROACH:
        agent_x = static_x - jr.uniform(x_key, shape=(), dtype=jnp.float32, minval=18.0, maxval=32.0)
        agent_y = start_y
        agent_v = jr.uniform(speed_key, (num_agents,), dtype=jnp.float32, minval=0.0, maxval=40.0) / 3.6
    elif phase == _SIDE:
        agent_x = static_x + jr.uniform(x_key, shape=(), dtype=jnp.float32, minval=-4.0, maxval=4.0)
        agent_y = side_y
        agent_v = jr.uniform(speed_key, (num_agents,), dtype=jnp.float32, minval=0.0, maxval=40.0) / 3.6
    elif phase == _PASSED:
        center_key, noise_key = jr.split(y_key)
        agent_x = static_x + jr.uniform(x_key, shape=(), dtype=jnp.float32, minval=8.0, maxval=18.0)
        agent_y = jr.uniform(
            center_key,
            shape=(),
            dtype=jnp.float32,
            minval=jnp.minimum(lane_centers[0], lane_centers[1]),
            maxval=jnp.maximum(lane_centers[0], lane_centers[1]),
        )
        y_key = noise_key
        agent_v = jr.uniform(speed_key, (num_agents,), dtype=jnp.float32, minval=0.0, maxval=40.0) / 3.6
    else:
        agent_x = static_x + jr.uniform(x_key, shape=(), dtype=jnp.float32, minval=18.0, maxval=32.0)
        agent_y = terminal_y
        agent_v = jr.uniform(speed_key, (num_agents,), dtype=jnp.float32, minval=0.0, maxval=40.0) / 3.6

    agent_xs = jnp.full((num_agents,), agent_x, dtype=jnp.float32)
    agent_ys = jnp.full((num_agents,), agent_y, dtype=jnp.float32) + jr.uniform(
        y_key, (num_agents,), dtype=jnp.float32, minval=-0.1, maxval=0.1
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
    mode_key, lane_key, time_key, relevant_gap_key, weak_gap_key, side_key = jr.split(key, 6)
    mode = jr.choice(mode_key, 3, p=jnp.array([0.7, 0.2, 0.1]))
    mean_agent_x = jnp.mean(agents[:, 0])
    mean_agent_y = jnp.mean(agents[:, 1])
    mean_agent_v = jnp.mean(agents[:, 4])
    current_lane = jnp.argmin(jnp.abs(lane_centers - mean_agent_y))
    random_lane = jr.choice(lane_key, jnp.array([0, 1], dtype=jnp.int32), shape=())

    # Most samples place the moving vehicle in the lane available around the static obstacle.
    dynamic_lane = jnp.where(mode == 0, 1 - static_lane, jnp.where(mode == 1, current_lane, random_lane))
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
        jnp.array(DYNAMIC_OBST_TARGET_SPEED),
        DYNAMIC_OBST_ACCEL,
        interaction_time,
    )
    relevant_gap = jr.uniform(relevant_gap_key, shape=(), dtype=jnp.float32, minval=10.0, maxval=20.0)
    weak_gap = jr.uniform(weak_gap_key, shape=(), dtype=jnp.float32, minval=30.0, maxval=45.0)
    gap_magnitude = jnp.where(mode == 2, weak_gap, relevant_gap)
    gap_sign = jnp.where(jr.bernoulli(side_key), 1.0, -1.0)
    desired_gap = gap_sign * gap_magnitude
    interaction_x = mean_agent_x + ego_distance + desired_gap
    dynamic_x = interaction_x - dynamic_distance

    same_agent_lane = dynamic_lane == current_lane
    same_static_lane = dynamic_lane == static_lane
    safe_from_agents = jnp.logical_or(
        jnp.logical_not(same_agent_lane),
        jnp.abs(dynamic_x - mean_agent_x) >= _INITIAL_LONGITUDINAL_GAP,
    )
    safe_from_static = jnp.logical_or(
        jnp.logical_not(same_static_lane),
        jnp.abs(dynamic_x - static_x) >= _INITIAL_LONGITUDINAL_GAP,
    )
    in_generation_range = jnp.logical_and(
        dynamic_x >= xrange[0] - 5.0,
        dynamic_x <= xrange[1] + 45.0,
    )
    valid = jnp.logical_and(jnp.logical_and(safe_from_agents, safe_from_static), in_generation_range)

    # Guaranteed-safe fallback: use the lane not occupied by the static obstacle and start ahead of ego.
    fallback_lane = 1 - static_lane
    fallback_x = mean_agent_x + 15.0
    dynamic_lane = jnp.where(valid, dynamic_lane, fallback_lane)
    dynamic_x = jnp.where(valid, dynamic_x, fallback_x)
    dynamic_state = make_state(
        dynamic_x,
        lane_centers[dynamic_lane],
        jnp.array(0.0),
        jnp.array(0.0),
    )
    return dynamic_state


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
) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
    del yrange, lane_width
    reference_key, static_key, agent_key, dynamic_key = jr.split(key, 4)
    start_x, start_y, terminal_y, terminal_v, goals, derivatives = _make_reference(
        reference_key, lane_change, num_agents, num_ref_points, xrange, lane_centers
    )
    static_x, static_lane, static_state = _make_static_obstacle(static_key, xrange, lane_centers)
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
    dynamic_state = _make_dynamic_obstacle(
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
    return agents, obstacles, goals, derivatives


def gen_scene_randomly_split_dynamic(
    key: PRNGKey,
    num_agents: int,
    num_ref_points: int,
    xrange: Array,
    yrange: Array,
    lane_width: float,
    lane_centers: Array,
) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
    """Generate one of ten split scenes with one static and one moving obstacle."""
    choose_key, scene_key = jr.split(key)
    scene_fns = [
        ft.partial(_make_scene, scene_key, task, phase, num_agents, num_ref_points, xrange, yrange,
                   lane_width, lane_centers)
        for task in (True, False)
        for phase in (_START, _APPROACH, _SIDE, _PASSED, _DONE)
    ]
    scene_id = jr.choice(choose_key, len(scene_fns), p=_SCENE_PROBS)
    return jax.lax.switch(scene_id, scene_fns)


def gen_scene_randomly_split(
    key: PRNGKey,
    num_agents: int,
    num_ref_points: int,
    xrange: Array,
    yrange: Array,
    lane_width: float,
    lane_centers: Array,
) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
    """Drop-in dynamic replacement for the original split scene generator."""
    return gen_scene_randomly_split_dynamic(
        key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers
    )
