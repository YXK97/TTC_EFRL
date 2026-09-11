"""Render a straight-road rollout and reproduce offline safety statistics.

The four input CSV files contain the recorded low-speed state convention::

    [rear_x, rear_y, heading_x, heading_y, speed, steering]

The script deliberately computes only the geometric ``cost_real`` channels.
It does not reconstruct actions or evaluate any CBF/ISSf-CBF expression.
"""

import argparse
import csv
import pathlib
import sys
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import FancyArrow, Rectangle
import numpy as np


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from defmarl.env.mve_lowspeed_ISSf_CBF_dynamic_preview import (
    MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF_Dynamic_Preview,
)
from defmarl.env.utils import process_lane_marks
from defmarl.utils.scaling_lowspeed import (
    scaling_calc_parameterized,
    scaling_calc_unbounded_bound,
)


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "video_csv" / "straight"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "video" / "straight"
DEFAULT_STEM = "0831-1654_epi00"
DEFAULT_AGENT_CSV = DEFAULT_INPUT_DIR / f"{DEFAULT_STEM}_agent00_states.csv"
DEFAULT_OBSTACLE0_CSV = DEFAULT_INPUT_DIR / f"{DEFAULT_STEM}_obst00_states.csv"
DEFAULT_OBSTACLE1_CSV = DEFAULT_INPUT_DIR / f"{DEFAULT_STEM}_obst01_states.csv"
DEFAULT_GOAL_CSV = DEFAULT_INPUT_DIR / f"{DEFAULT_STEM}_goal00_states.csv"
DEFAULT_OUTPUT_VIDEO = DEFAULT_OUTPUT_DIR / f"{DEFAULT_STEM}_straight.mp4"

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
    "cost_real_bound_y_low",
    "cost_real_bound_y_high",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a straight-road rollout from state CSV files and export "
            "per-step reward/geometric cost statistics."
        )
    )
    parser.add_argument("--agent-csv", type=pathlib.Path, default=DEFAULT_AGENT_CSV)
    parser.add_argument(
        "--obstacle0-csv", type=pathlib.Path, default=DEFAULT_OBSTACLE0_CSV
    )
    parser.add_argument(
        "--obstacle1-csv", type=pathlib.Path, default=DEFAULT_OBSTACLE1_CSV
    )
    parser.add_argument("--goal-csv", type=pathlib.Path, default=DEFAULT_GOAL_CSV)
    parser.add_argument(
        "--output", type=pathlib.Path, default=DEFAULT_OUTPUT_VIDEO,
        help="Output MP4 path.",
    )
    parser.add_argument(
        "--statistics-output", type=pathlib.Path, default=None,
        help=(
            "Output statistics CSV. By default it is written beside the video "
            "as <video_stem>_statistics.csv."
        ),
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args()


def load_state_csv(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load semantic state columns, with support for historical s0...s5 CSVs."""
    if not path.is_file():
        raise FileNotFoundError(f"State CSV does not exist: {path}")

    table = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if table.size == 0:
        raise ValueError(f"State CSV is empty: {path}")
    table = np.atleast_1d(table)
    available = set(table.dtype.names or ())
    if set(STATE_COLUMNS).issubset(available):
        columns = STATE_COLUMNS
    elif set(LEGACY_STATE_COLUMNS).issubset(available):
        columns = LEGACY_STATE_COLUMNS
    else:
        raise ValueError(
            f"{path} must contain time_step and either {STATE_COLUMNS} "
            f"or {LEGACY_STATE_COLUMNS}."
        )
    if "time_step" not in available:
        raise ValueError(f"{path} is missing column: time_step")

    time_steps = np.asarray(table["time_step"], dtype=np.int64)
    states = np.column_stack(
        [np.asarray(table[column], dtype=np.float64) for column in columns]
    )
    if not np.all(np.isfinite(states)):
        raise ValueError(f"State CSV contains NaN or Inf: {path}")
    heading_norms = np.linalg.norm(states[:, 2:4], axis=1)
    if np.any(heading_norms <= 1e-8):
        row = int(np.flatnonzero(heading_norms <= 1e-8)[0])
        raise ValueError(f"State CSV has a zero heading at row {row}: {path}")
    return time_steps, states


def validate_timelines(*timelines: np.ndarray) -> None:
    """Reject misaligned inputs instead of silently truncating a trajectory."""
    if not all(np.array_equal(timelines[0], timeline) for timeline in timelines[1:]):
        raise ValueError("The four CSV files must have identical time_step sequences.")


def normalize_heading(state: np.ndarray) -> np.ndarray:
    heading = state[2:4]
    return heading / np.linalg.norm(heading)


def geometric_center(state: np.ndarray, rear_offset: float) -> np.ndarray:
    return state[:2] + rear_offset * normalize_heading(state)


def add_vehicle(
    axis,
    state: np.ndarray,
    bounding_box: np.ndarray,
    rear_offset: float,
    color: str,
    zorder: int,
):
    """Create the same oriented rectangle and arrow as the env renderer."""
    heading = normalize_heading(state)
    center = geometric_center(state, rear_offset)
    angle = np.degrees(np.arctan2(heading[1], heading[0]))
    arrow_length = np.linalg.norm(bounding_box) / 2.0
    rectangle = Rectangle(
        center - bounding_box / 2.0,
        width=bounding_box[0],
        height=bounding_box[1],
        angle=angle,
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
    heading = normalize_heading(state)
    center = geometric_center(state, rear_offset)
    angle = np.degrees(np.arctan2(heading[1], heading[0]))
    arrow_length = np.linalg.norm(bounding_box) / 2.0
    arrow.set_data(
        x=center[0],
        y=center[1],
        dx=heading[0] * arrow_length,
        dy=heading[1] * arrow_length,
    )
    rectangle.set_xy(center - bounding_box / 2.0)
    rectangle.set_angle(angle)


def compute_statistics(
    agent_states: np.ndarray,
    obstacle0_states: np.ndarray,
    obstacle1_states: np.ndarray,
    goal_states: np.ndarray,
    params: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reproduce get_reward and geometric cost_real without ISSf-CBF."""
    # The environment's W is diagonal and its pre-static penalty is currently
    # zero. Keeping the general expression here preserves exact state weights.
    error = agent_states - goal_states
    reward_weights = np.asarray(
        [2e-5, 2e-5, 0.0, 0.0, 1e-4, 0.0], dtype=np.float64
    )
    rewards = -np.sqrt(np.sum(error * error * reward_weights, axis=1))
    pre_static_penalty = float(params.get("pre_static_penalty", 0.0))
    rewards -= pre_static_penalty * (
        agent_states[:, 0] <= obstacle0_states[:, 0]
    )

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

    alphas = jax.vmap(obstacle_alphas)(agents, obstacles)
    alphas = jnp.nan_to_num(alphas, nan=0.0, posinf=1e6, neginf=0.0)
    obstacle_cost_real = jnp.max(1.0 - alphas, axis=1)

    y_low = float(params["default_state_range"][2])
    y_high = float(params["default_state_range"][3])
    lower_A = jnp.array([[0.0, 1.0]], dtype=jnp.float32)
    lower_b = jnp.array([y_low], dtype=jnp.float32)
    upper_A = jnp.array([[0.0, -1.0]], dtype=jnp.float32)
    upper_b = jnp.array([-y_high], dtype=jnp.float32)
    boundary_alpha = jax.vmap(
        scaling_calc_unbounded_bound, in_axes=(0, None, None, None, None)
    )
    lower_alpha = boundary_alpha(agents, ego_bb, ego_lr, lower_A, lower_b)
    upper_alpha = boundary_alpha(agents, ego_bb, ego_lr, upper_A, upper_b)
    lower_alpha = jnp.nan_to_num(lower_alpha, nan=0.0, posinf=1e6, neginf=0.0)
    upper_alpha = jnp.nan_to_num(upper_alpha, nan=0.0, posinf=1e6, neginf=0.0)

    # With one controlled ego there is no ego-to-ego collision pair.
    agent_cost_real = -jnp.ones(agents.shape[0], dtype=jnp.float32) * 3.0
    cost_real = jnp.stack(
        [
            agent_cost_real,
            obstacle_cost_real,
            1.0 - lower_alpha,
            1.0 - upper_alpha,
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
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "time_step",
        "reward",
        *COST_REAL_COLUMNS,
        "cost_real_max",
        "unsafe",
    )
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(header)
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


def print_statistics(rewards: np.ndarray, cost_real: np.ndarray) -> None:
    maximum = np.max(cost_real, axis=1)
    unsafe = maximum >= 0.0
    print("Reward statistics:")
    print(
        "  sum={:.6f}, mean={:.6f}, min={:.6f}, max={:.6f}".format(
            rewards.sum(), rewards.mean(), rewards.min(), rewards.max()
        )
    )
    print("Cost-real statistics (negative is safe):")
    for index, name in enumerate(COST_REAL_COLUMNS):
        values = cost_real[:, index]
        print(
            "  {}: mean={:.6f}, min={:.6f}, max={:.6f}".format(
                name, values.mean(), values.min(), values.max()
            )
        )
    print(
        "  overall: mean(max channels)={:.6f}, max={:.6f}, "
        "unsafe_steps={}/{} ({:.3f}%)".format(
            maximum.mean(),
            maximum.max(),
            int(unsafe.sum()),
            len(unsafe),
            100.0 * unsafe.mean(),
        )
    )


def render_video(args: argparse.Namespace) -> None:
    agent_times, agent_states = load_state_csv(args.agent_csv)
    obstacle0_times, obstacle0_states = load_state_csv(args.obstacle0_csv)
    obstacle1_times, obstacle1_states = load_state_csv(args.obstacle1_csv)
    goal_times, goal_states = load_state_csv(args.goal_csv)
    validate_timelines(agent_times, obstacle0_times, obstacle1_times, goal_times)
    if args.fps <= 0.0:
        raise ValueError("--fps must be positive.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")

    params = MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF_Dynamic_Preview.PARAMS
    ego_bb = np.asarray(params["ego_bb_size"], dtype=np.float64)
    obst_bb = np.asarray(params["obst_bb_size"], dtype=np.float64)
    ego_lr = float(params["ego_lr"])
    obst_lr = float(params["obst_lr"])
    state_range = np.asarray(params["default_state_range"], dtype=np.float64)
    x_low, x_high, y_low, y_high = state_range[:4]

    # Match mve_lowspeed_base.render_video, including its equal metric scale,
    # road-edge lines, center lane divider, colors, and goal-point cloud.
    figure_height = (y_high + 3.0 - (y_low - 3.0)) * 20.0 / (
        x_high + 3.0 - (x_low - 3.0)
    ) + 4.0
    figure, axis = plt.subplots(1, 1, figsize=(30, figure_height), dpi=args.dpi)
    axis.set_xlim(x_low, x_high)
    axis.set_ylim(y_low - 3.0, y_high + 3.0)
    axis.set_aspect("equal")
    axis.set_xlabel("x / m")
    axis.set_ylabel("y / m")
    edge_lines, lane_lines = process_lane_marks(
        params["default_state_range"][2:4], params["lane_width"]
    )
    axis.axhline(float(edge_lines[0]), linewidth=1.5, color="b")
    axis.axhline(float(edge_lines[1]), linewidth=1.5, color="b")
    if lane_lines is not None:
        for lane_y in lane_lines:
            axis.axhline(float(lane_y), linewidth=1.0, color="b", linestyle="--")
    axis.scatter(
        goal_states[:, 0],
        goal_states[:, 1],
        color="#2fdd00",
        zorder=7,
        s=5,
        alpha=1.0,
        marker=".",
    )

    ego_artists = add_vehicle(axis, agent_states[0], ego_bb, ego_lr, "#0068ff", 6)
    obstacle0_artists = add_vehicle(
        axis, obstacle0_states[0], obst_bb, obst_lr, "#8a0000", 5
    )
    obstacle1_artists = add_vehicle(
        axis, obstacle1_states[0], obst_bb, obst_lr, "#8a0000", 5
    )

    def update(frame: int):
        update_vehicle(*ego_artists, agent_states[frame], ego_bb, ego_lr)
        update_vehicle(
            *obstacle0_artists, obstacle0_states[frame], obst_bb, obst_lr
        )
        update_vehicle(
            *obstacle1_artists, obstacle1_states[frame], obst_bb, obst_lr
        )
        return (*ego_artists, *obstacle0_artists, *obstacle1_artists)

    animation = FuncAnimation(
        figure,
        update,
        frames=agent_states.shape[0],
        interval=1000.0 / args.fps,
        blit=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        animation.save(
            args.output, writer=FFMpegWriter(fps=args.fps), dpi=args.dpi
        )
    finally:
        plt.close(figure)

    statistics_output = args.statistics_output
    if statistics_output is None:
        statistics_output = args.output.with_name(
            f"{args.output.stem}_statistics.csv"
        )
    rewards, cost_real = compute_statistics(
        agent_states, obstacle0_states, obstacle1_states, goal_states, params
    )
    write_statistics_csv(statistics_output, agent_times, rewards, cost_real)
    print(f"Rendered {agent_states.shape[0]} frames to {args.output.resolve()}")
    print(f"Wrote per-step statistics to {statistics_output.resolve()}")
    print_statistics(rewards, cost_real)


def main() -> None:
    render_video(parse_args())


if __name__ == "__main__":
    main()
