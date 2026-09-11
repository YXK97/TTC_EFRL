"""Render an intersection video directly from three vehicle-state CSV files.

The CSV state convention matches the low-speed intersection environments:

    [rear_x, rear_y, heading_x, heading_y, speed, steering]

Only the asymmetric intersection, the recorded reference-point cloud, and the
three vehicle poses are rendered. Graph edges, on-frame statistics, scene
labels, and step counters are intentionally omitted. Reward and geometric
``cost_real`` values are instead exported as a separate CSV.
"""

import argparse
import csv
import pathlib
import sys
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import FancyArrow, Rectangle
import jax
import jax.numpy as jnp
import numpy as np


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from defmarl.env.designed_scene_gen_intersection_split_dynamic import ROAD_HALF
from defmarl.env.mve_intersection_lowspeed_ISSf_CBF_dynamic import (
    intersection_corner_extreme_points,
    intersection_corner_halfspaces,
)
from defmarl.env.mve_intersection_lowspeed_ISSf_CBF_dynamic_WestEnter_new_scaling import (
    _parameterized_corner_alpha_without_gate,
)
from defmarl.env.mve_intersection_lowspeed_ISSf_CBF_dynamic_WestEnter_new_scaling_preview import (
    MVEIntersection_LowSpeed_ISSf_CBF_Dynamic_WestEnter_NewScaling_Preview,
)
from defmarl.utils.scaling_lowspeed import (
    scaling_calc_parameterized,
)


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "video_csv" / "intersection"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "video" / "intersection"
DEFAULT_STEM = "0905-1136_epi00"
DEFAULT_AGENT_CSV = DEFAULT_INPUT_DIR / f"{DEFAULT_STEM}_agent00_states.csv"
DEFAULT_OBSTACLE0_CSV = DEFAULT_INPUT_DIR / f"{DEFAULT_STEM}_obst00_states.csv"
DEFAULT_OBSTACLE1_CSV = DEFAULT_INPUT_DIR / f"{DEFAULT_STEM}_obst01_states.csv"
DEFAULT_GOAL_CSV = DEFAULT_INPUT_DIR / f"{DEFAULT_STEM}_goal00_states.csv"
DEFAULT_OUTPUT_VIDEO = (
    DEFAULT_OUTPUT_DIR / f"{DEFAULT_STEM}_intersection_with_reference.mp4"
)

STATE_COLUMNS = (
    "x",
    "y",
    "heading_x",
    "heading_y",
    "speed",
    "steering",
)
LEGACY_STATE_COLUMNS = tuple(f"s{index}" for index in range(6))
COST_REAL_COLUMNS = (
    "cost_real_agent_collision",
    "cost_real_obstacle_collision",
    "cost_real_road_boundary",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the reference path plus ego, static-obstacle, and "
            "dynamic-obstacle poses from intersection CSV files."
        )
    )
    parser.add_argument(
        "--agent-csv",
        type=pathlib.Path,
        default=DEFAULT_AGENT_CSV,
        help="CSV containing the ego state sequence.",
    )
    parser.add_argument(
        "--obstacle0-csv",
        type=pathlib.Path,
        default=DEFAULT_OBSTACLE0_CSV,
        help="CSV containing obstacle 0, normally the static vehicle.",
    )
    parser.add_argument(
        "--obstacle1-csv",
        type=pathlib.Path,
        default=DEFAULT_OBSTACLE1_CSV,
        help="CSV containing obstacle 1, normally the dynamic vehicle.",
    )
    parser.add_argument(
        "--goal-csv",
        type=pathlib.Path,
        default=DEFAULT_GOAL_CSV,
        help="CSV containing the ego's recorded reference-goal sequence.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT_VIDEO,
        help="Output MP4 path.",
    )
    parser.add_argument(
        "--statistics-output",
        type=pathlib.Path,
        default=None,
        help=(
            "Output per-step statistics CSV. By default it is written beside "
            "the video as <video_stem>_statistics.csv."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output video frame rate.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="Output video resolution scaling.",
    )
    parser.add_argument(
        "--view-half",
        type=float,
        default=float(ROAD_HALF) + 5.0,
        help=(
            "Half-width of the square viewport in metres (default: 55, "
            "matching the environment renderer)."
        ),
    )
    return parser.parse_args()


def load_state_csv(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load either semantic or legacy six-dimensional state columns."""
    if not path.is_file():
        raise FileNotFoundError(f"State CSV does not exist: {path}")

    table = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if table.size == 0:
        raise ValueError(f"State CSV is empty: {path}")
    table = np.atleast_1d(table)
    available_columns = set(table.dtype.names or ())
    if set(STATE_COLUMNS).issubset(available_columns):
        state_columns = STATE_COLUMNS
    elif set(LEGACY_STATE_COLUMNS).issubset(available_columns):
        state_columns = LEGACY_STATE_COLUMNS
    else:
        expected = ", ".join(STATE_COLUMNS)
        legacy = ", ".join(LEGACY_STATE_COLUMNS)
        raise ValueError(
            f"{path} must contain time_step and either [{expected}] "
            f"or legacy columns [{legacy}]."
        )
    if "time_step" not in available_columns:
        raise ValueError(f"{path} is missing column: time_step")

    time_steps = np.asarray(table["time_step"], dtype=np.int64)
    states = np.column_stack(
        [np.asarray(table[column], dtype=np.float64) for column in state_columns]
    )
    if not np.all(np.isfinite(states)):
        raise ValueError(f"State CSV contains NaN or Inf: {path}")
    heading_norms = np.linalg.norm(states[:, 2:4], axis=1)
    if np.any(heading_norms <= 1e-8):
        bad_frame = int(np.flatnonzero(heading_norms <= 1e-8)[0])
        raise ValueError(
            f"State CSV has a zero heading vector at row {bad_frame}: {path}"
        )
    return time_steps, states


def validate_timelines(
    agent_times: np.ndarray,
    obstacle0_times: np.ndarray,
    obstacle1_times: np.ndarray,
    goal_times: np.ndarray,
) -> None:
    """Require exact frame alignment instead of silently truncating data."""
    if not (
        np.array_equal(agent_times, obstacle0_times)
        and np.array_equal(agent_times, obstacle1_times)
        and np.array_equal(agent_times, goal_times)
    ):
        raise ValueError(
            "The four CSV files must contain identical time_step sequences."
        )


def normalize_heading(state: np.ndarray) -> np.ndarray:
    heading = state[2:4]
    return heading / np.linalg.norm(heading)


def geometric_center(state: np.ndarray, rear_offset: float) -> np.ndarray:
    """Convert the recorded rear-axle position to the rectangle center."""
    return state[:2] + rear_offset * normalize_heading(state)


def add_vehicle(
    axis,
    state: np.ndarray,
    bounding_box: np.ndarray,
    rear_offset: float,
    color: str,
    zorder: int,
):
    """Create the oriented rectangle and heading arrow for one vehicle."""
    heading = normalize_heading(state)
    center = geometric_center(state, rear_offset)
    angle_degrees = np.degrees(np.arctan2(heading[1], heading[0]))
    arrow_length = np.linalg.norm(bounding_box) / 2.0

    rectangle = Rectangle(
        center - bounding_box / 2.0,
        width=bounding_box[0],
        height=bounding_box[1],
        angle=angle_degrees,
        rotation_point="center",
        color=color,
        linewidth=0.0,
        alpha=0.6,
        zorder=zorder,
    )
    arrow = FancyArrow(
        center[0],
        center[1],
        heading[0] * arrow_length,
        heading[1] * arrow_length,
        length_includes_head=True,
        width=0.3,
        color=color,
        alpha=1.0,
        zorder=zorder + 1,
    )
    axis.add_patch(rectangle)
    axis.add_patch(arrow)
    return arrow, rectangle


def update_vehicle(
    arrow: FancyArrow,
    rectangle: Rectangle,
    state: np.ndarray,
    bounding_box: np.ndarray,
    rear_offset: float,
) -> None:
    """Move existing artists to the pose stored in the current CSV row."""
    heading = normalize_heading(state)
    center = geometric_center(state, rear_offset)
    angle_degrees = np.degrees(np.arctan2(heading[1], heading[0]))
    arrow_length = np.linalg.norm(bounding_box) / 2.0
    arrow.set_data(
        x=center[0],
        y=center[1],
        dx=heading[0] * arrow_length,
        dy=heading[1] * arrow_length,
    )
    rectangle.set_xy(center - bounding_box / 2.0)
    rectangle.set_angle(angle_degrees)


def draw_intersection(
    axis,
    view_half: float,
    turn_half: float,
    main_half: float,
    auxiliary_half: float,
) -> None:
    """Draw the same asymmetric road geometry as the environment renderer."""
    corner_polygons = [
        [
            (-view_half, -view_half),
            (-view_half, -main_half),
            (-turn_half, -main_half),
            (-auxiliary_half, -turn_half),
            (-auxiliary_half, -view_half),
        ],
        [
            (auxiliary_half, -view_half),
            (auxiliary_half, -turn_half),
            (turn_half, -main_half),
            (view_half, -main_half),
            (view_half, -view_half),
        ],
        [
            (auxiliary_half, turn_half),
            (turn_half, main_half),
            (view_half, main_half),
            (view_half, view_half),
            (auxiliary_half, view_half),
        ],
        [
            (-view_half, main_half),
            (-turn_half, main_half),
            (-auxiliary_half, turn_half),
            (-auxiliary_half, view_half),
            (-view_half, view_half),
        ],
    ]
    for polygon in corner_polygons:
        axis.fill(
            *zip(*polygon),
            facecolor="#e6e6e6",
            edgecolor="#666666",
            linewidth=1.0,
            zorder=0,
        )

    road_color = "#1f4e79"
    dash_style = (0, (7, 7))
    for y_coordinate in (-main_half, main_half):
        axis.plot(
            [-view_half, -turn_half],
            [y_coordinate, y_coordinate],
            color=road_color,
            linewidth=1.4,
        )
        axis.plot(
            [turn_half, view_half],
            [y_coordinate, y_coordinate],
            color=road_color,
            linewidth=1.4,
        )
    axis.plot(
        [-view_half, -turn_half],
        [0.0, 0.0],
        color=road_color,
        linestyle=dash_style,
    )
    axis.plot(
        [turn_half, view_half],
        [0.0, 0.0],
        color=road_color,
        linestyle=dash_style,
    )

    for x_coordinate in (-auxiliary_half, auxiliary_half):
        axis.plot(
            [x_coordinate, x_coordinate],
            [-view_half, -turn_half],
            color=road_color,
            linewidth=1.4,
        )
        axis.plot(
            [x_coordinate, x_coordinate],
            [turn_half, view_half],
            color=road_color,
            linewidth=1.4,
        )
    axis.plot(
        [0.0, 0.0],
        [-view_half, -turn_half],
        color=road_color,
        linestyle=dash_style,
    )
    axis.plot(
        [0.0, 0.0],
        [turn_half, view_half],
        color=road_color,
        linestyle=dash_style,
    )


def compute_statistics(
    agent_states: np.ndarray,
    obstacle0_states: np.ndarray,
    obstacle1_states: np.ndarray,
    goal_states: np.ndarray,
    params: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reproduce preview get_reward and geometric cost_real only.

    The ISSf residual, its derivatives, and action-dependent terms are not
    evaluated. The two obstacle costs and four corner-region costs are reduced
    with the same hard maxima/minima used by the environment.
    """
    error = agent_states - goal_states
    reward_weights = np.asarray(
        [2e-5, 2e-5, 0.0, 0.0, 1e-4, 0.0], dtype=np.float64
    )
    rewards = -np.sqrt(np.sum(error * error * reward_weights, axis=1))

    # Preserve the environment's static-pass term. It is zero with the current
    # preview parameters, but retaining it keeps this offline calculation valid
    # if pre_static_penalty is enabled later.
    ego_headings = agent_states[:, 2:4] / np.linalg.norm(
        agent_states[:, 2:4], axis=1, keepdims=True
    )
    static_headings = obstacle0_states[:, 2:4] / np.linalg.norm(
        obstacle0_states[:, 2:4], axis=1, keepdims=True
    )
    ego_centers = agent_states[:, :2] + float(params["ego_lr"]) * ego_headings
    static_centers = (
        obstacle0_states[:, :2]
        + float(params["obst_lr"]) * static_headings
    )
    relative_longitudinal = np.sum(
        (ego_centers - static_centers) * static_headings, axis=1
    )
    ego_not_past_static = (
        relative_longitudinal <= float(params["static_pass_margin"])
    )
    rewards -= float(params.get("pre_static_penalty", 0.0)) * ego_not_past_static

    agents = jnp.asarray(agent_states, dtype=jnp.float32)
    obstacles = jnp.asarray(
        np.stack([obstacle0_states, obstacle1_states], axis=1),
        dtype=jnp.float32,
    )
    ego_bb = jnp.asarray(params["ego_bb_size"], dtype=jnp.float32)
    obst_bb = jnp.asarray(params["obst_bb_size"], dtype=jnp.float32)
    ego_lr = jnp.asarray(params["ego_lr"], dtype=jnp.float32)
    obst_lr = jnp.asarray(params["obst_lr"], dtype=jnp.float32)

    def obstacle_alphas(agent, frame_obstacles):
        return jax.vmap(
            scaling_calc_parameterized,
            in_axes=(None, 0, None, None, None, None),
        )(agent, frame_obstacles, ego_bb, ego_lr, obst_bb, obst_lr)

    obstacle_alpha = jax.vmap(obstacle_alphas)(agents, obstacles)
    obstacle_alpha = jnp.nan_to_num(
        obstacle_alpha, nan=0.0, posinf=1e6, neginf=0.0
    )
    obstacle_cost_real = jnp.max(1.0 - obstacle_alpha, axis=1)

    all_A, all_b = intersection_corner_halfspaces(
        params["main_road_half_width"],
        params["auxiliary_road_half_width"],
        params["intersection_radius"],
    )
    all_extreme_points = intersection_corner_extreme_points(
        params["main_road_half_width"],
        params["auxiliary_road_half_width"],
        params["intersection_radius"],
    )

    def boundary_alpha(agent):
        corner_alpha = jax.vmap(
            _parameterized_corner_alpha_without_gate,
            in_axes=(None, 0, 0, 0, None, None),
        )(agent, all_A, all_b, all_extreme_points, ego_bb, ego_lr)
        return jnp.min(corner_alpha)

    boundary_alpha_values = jax.vmap(boundary_alpha)(agents)
    boundary_alpha_values = jnp.nan_to_num(
        boundary_alpha_values, nan=0.0, posinf=1e6, neginf=0.0
    )

    # There is only one controlled ego, so the ego-to-ego channel is fixed.
    agent_cost_real = -jnp.ones(agents.shape[0], dtype=jnp.float32) * 3.0
    cost_real = jnp.stack(
        [
            agent_cost_real,
            obstacle_cost_real,
            1.0 - boundary_alpha_values,
        ],
        axis=1,
    )
    return rewards, np.asarray(cost_real, dtype=np.float64)


def write_statistics_csv(
    path: pathlib.Path,
    time_steps: np.ndarray,
    rewards: np.ndarray,
    cost_real: np.ndarray,
) -> None:
    """Write reward and all cost_real channels for every video frame."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            (
                "time_step",
                "reward",
                *COST_REAL_COLUMNS,
                "cost_real_max",
                "unsafe",
            )
        )
        for time_step, reward, components in zip(time_steps, rewards, cost_real):
            maximum = float(np.max(components))
            writer.writerow(
                [
                    int(time_step),
                    f"{reward:.10g}",
                    *(f"{value:.10g}" for value in components),
                    f"{maximum:.10g}",
                    int(maximum >= 0.0),
                ]
            )


def print_statistics(
    time_steps: np.ndarray,
    rewards: np.ndarray,
    cost_real: np.ndarray,
) -> None:
    """Print episode reward and the location of the worst geometric cost."""
    maximum_index = np.unravel_index(np.argmax(cost_real), cost_real.shape)
    frame_index, component_index = maximum_index
    per_step_maximum = np.max(cost_real, axis=1)
    unsafe_steps = per_step_maximum >= 0.0
    print(
        "Reward: total={:.6f}, mean={:.6f}, min={:.6f}, max={:.6f}".format(
            rewards.sum(), rewards.mean(), rewards.min(), rewards.max()
        )
    )
    print(
        "Maximum cost_real: {:.6f} at step {} ({})".format(
            cost_real[maximum_index],
            int(time_steps[frame_index]),
            COST_REAL_COLUMNS[component_index],
        )
    )
    print(
        "Unsafe steps: {}/{} ({:.3f}%)".format(
            int(unsafe_steps.sum()),
            len(unsafe_steps),
            100.0 * unsafe_steps.mean(),
        )
    )
def render_video(args: argparse.Namespace) -> None:
    agent_times, agent_states = load_state_csv(args.agent_csv)
    obstacle0_times, obstacle0_states = load_state_csv(args.obstacle0_csv)
    obstacle1_times, obstacle1_states = load_state_csv(args.obstacle1_csv)
    goal_times, goal_states = load_state_csv(args.goal_csv)
    validate_timelines(
        agent_times, obstacle0_times, obstacle1_times, goal_times
    )

    if args.fps <= 0.0:
        raise ValueError("--fps must be positive.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")

    params = (
        MVEIntersection_LowSpeed_ISSf_CBF_Dynamic_WestEnter_NewScaling_Preview.PARAMS
    )
    ego_bounding_box = np.asarray(params["ego_bb_size"], dtype=np.float64)
    obstacle_bounding_box = np.asarray(
        params["obst_bb_size"], dtype=np.float64
    )
    ego_rear_offset = float(params["ego_lr"])
    obstacle_rear_offset = float(params["obst_lr"])
    turn_half = float(params["intersection_radius"])
    main_half = float(params["main_road_half_width"])
    auxiliary_half = float(params["auxiliary_road_half_width"])

    view_half = float(args.view_half)
    if view_half <= max(turn_half, main_half, auxiliary_half):
        raise ValueError("--view-half is too small for the intersection geometry.")

    figure, axis = plt.subplots(1, 1, figsize=(10, 10), dpi=args.dpi)
    axis.set_xlim(-view_half, view_half)
    axis.set_ylim(-view_half, view_half)
    axis.set_aspect("equal")
    axis.set_xlabel("x / m")
    axis.set_ylabel("y / m")
    draw_intersection(
        axis, view_half, turn_half, main_half, auxiliary_half
    )

    # Match the environment renderer: the recorded goal at every time step is
    # shown as a small green point, reconstructing the followed reference path
    # without regenerating or extending it.
    axis.scatter(
        goal_states[:, 0],
        goal_states[:, 1],
        color="#2fdd00",
        zorder=7,
        s=5,
        alpha=1.0,
        marker=".",
    )

    ego_artists = add_vehicle(
        axis,
        agent_states[0],
        ego_bounding_box,
        ego_rear_offset,
        "#0068ff",
        6,
    )
    obstacle0_artists = add_vehicle(
        axis,
        obstacle0_states[0],
        obstacle_bounding_box,
        obstacle_rear_offset,
        "#8a0000",
        5,
    )
    obstacle1_artists = add_vehicle(
        axis,
        obstacle1_states[0],
        obstacle_bounding_box,
        obstacle_rear_offset,
        "#8a0000",
        5,
    )

    def update(frame: int):
        update_vehicle(
            *ego_artists,
            agent_states[frame],
            ego_bounding_box,
            ego_rear_offset,
        )
        update_vehicle(
            *obstacle0_artists,
            obstacle0_states[frame],
            obstacle_bounding_box,
            obstacle_rear_offset,
        )
        update_vehicle(
            *obstacle1_artists,
            obstacle1_states[frame],
            obstacle_bounding_box,
            obstacle_rear_offset,
        )
        return (
            *ego_artists,
            *obstacle0_artists,
            *obstacle1_artists,
        )

    animation = FuncAnimation(
        figure,
        update,
        frames=agent_states.shape[0],
        interval=1000.0 / args.fps,
        blit=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=args.fps)
    try:
        animation.save(args.output, writer=writer, dpi=args.dpi)
    finally:
        plt.close(figure)

    statistics_output = args.statistics_output
    if statistics_output is None:
        statistics_output = args.output.with_name(
            f"{args.output.stem}_statistics.csv"
        )
    rewards, cost_real = compute_statistics(
        agent_states,
        obstacle0_states,
        obstacle1_states,
        goal_states,
        params,
    )
    write_statistics_csv(
        statistics_output, agent_times, rewards, cost_real
    )
    print(
        f"Rendered {agent_states.shape[0]} frames to {args.output.resolve()}"
    )
    print(f"Wrote per-step statistics to {statistics_output.resolve()}")
    print_statistics(agent_times, rewards, cost_real)


def main() -> None:
    render_video(parse_args())


if __name__ == "__main__":
    main()
