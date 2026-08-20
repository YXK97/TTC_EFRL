"""Low-speed asymmetric-intersection environment with dynamic obstacles.

This module combines the six-state kinematic bicycle model used by the other
low-speed environments with an ISSf-CBF cost for an intersection.  Road bounds
are modeled directly as four convex corner polygons; no virtual boundary
vehicles are created.
"""

import functools as ft
import pathlib
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from typing_extensions import override

from .designed_scene_gen_intersection_split_dynamic import (
    AUX_LANE_WIDTH,
    MAIN_LANE_CENTERS,
    MAIN_LANE_WIDTH,
    ROAD_HALF,
    TURN_HALF,
    gen_scene_randomly_split_dynamic,
)
from .mve_lowspeed_CBF_dynamic import MVELaneChangeAndOverTake_LowSpeed_CBF_Dynamic
from defmarl.utils.graph import GraphsTuple
from defmarl.trainer.data import Rollout
from defmarl.utils.scaling_lowspeed import (
    compute_intersections,
    heading_rot_matrix,
    rear_to_center,
    safe_norm,
    scaling_calc,
)
from defmarl.utils.typing import Action, Array, Cost, ObstState, State
from defmarl.utils.utils import (
    find_closest_goal_indices,
    gen_i_j_pairs,
    save_anim,
    tree_index,
)


class MVEIntersectionLowSpeedDynamicState(NamedTuple):
    """Graph environment state including the moving vehicle's speed controller."""

    agent: State
    goal: State
    obstacle: State
    dynamic_obstacle_accel: Array
    dynamic_obstacle_target_speed: Array

    @property
    def n_agent(self) -> int:
        return self.agent.shape[0]


@jax.jit
def intersection_corner_halfspaces(
    main_half_width: Array,
    auxiliary_half_width: Array,
    turn_half: Array,
) -> Tuple[Array, Array]:
    """Return ``A, b`` for the four convex off-road corner regions.

    The horizontal main road has the y half-width ``main_half_width`` and the
    vertical auxiliary road has the x half-width ``auxiliary_half_width``.  Each
    diagonal joins ``(-turn_half, -main_half_width)`` and
    ``(-auxiliary_half_width, -turn_half)`` in the southwest corner, followed by
    reflections for the other three corners.
    """
    diagonal_x = turn_half - main_half_width
    diagonal_y = turn_half - auxiliary_half_width
    diagonal_b = main_half_width * auxiliary_half_width - turn_half ** 2

    A = jnp.array(
        [
            # Southwest: left of the auxiliary road, below the main road.
            [[1.0, 0.0], [0.0, 1.0], [diagonal_x, diagonal_y]],
            # Southeast: reflection of southwest across the y axis.
            [[-1.0, 0.0], [0.0, 1.0], [-diagonal_x, diagonal_y]],
            # Northeast: reflection across both axes.
            [[-1.0, 0.0], [0.0, -1.0], [-diagonal_x, -diagonal_y]],
            # Northwest: reflection of southwest across the x axis.
            [[1.0, 0.0], [0.0, -1.0], [diagonal_x, -diagonal_y]],
        ],
        dtype=jnp.float32,
    )
    b = jnp.array(
        [
            [-auxiliary_half_width, -main_half_width, diagonal_b],
            [-auxiliary_half_width, -main_half_width, diagonal_b],
            [-auxiliary_half_width, -main_half_width, diagonal_b],
            [-auxiliary_half_width, -main_half_width, diagonal_b],
        ],
        dtype=jnp.float32,
    )
    return A, b


@jax.jit
def scaling_calc_lowspeed_convex_bound(
    state: State,
    A: Array,
    b: Array,
    bb_size: Array,
    rear_to_center_offset: Array,
) -> Array:
    """Compute vehicle-to-convex-polygon scaling in low-speed coordinates.

    ``state[:2]`` is the rear-axle center, so the scaling origin first moves to
    the body center.  Alpha has the same convention as vehicle scaling:
    ``alpha > 1`` is separated, ``alpha == 1`` touches, and ``alpha < 1``
    penetrates the forbidden polygon.
    """
    center = rear_to_center(state, rear_to_center_offset)
    rotation = heading_rot_matrix(state)
    local_vertices = jnp.array(
        [
            [bb_size[0] / 2.0, bb_size[1] / 2.0],
            [bb_size[0] / 2.0, -bb_size[1] / 2.0],
            [-bb_size[0] / 2.0, bb_size[1] / 2.0],
            [-bb_size[0] / 2.0, -bb_size[1] / 2.0],
        ]
    )
    vertices = center + local_vertices @ rotation.T

    intersections = jax.vmap(
        compute_intersections, in_axes=(None, 0, None, None, None)
    )(center, vertices, A, b, center + 1e8)
    intersection_distances = safe_norm(intersections - center, axis=-1)
    vertex_distances = safe_norm(vertices - center, axis=-1)[:, None]
    scaling = jnp.min(intersection_distances / vertex_distances)

    # When the scaling center is already inside the forbidden polygon, alpha
    # must collapse towards zero rather than reporting the next outward edge.
    center_outside_measure = jnp.max(A @ center - b)
    center_outside_weight = jax.nn.sigmoid(1e6 * center_outside_measure)
    return center_outside_weight * scaling


@jax.jit
def scaling_calc_intersection_bounds_lowspeed(
    state: State,
    bb_size: Array,
    rear_to_center_offset: Array,
    main_half_width: Array,
    auxiliary_half_width: Array,
    turn_half: Array,
) -> Array:
    """Return the most dangerous scaling factor among four corner polygons."""
    all_A, all_b = intersection_corner_halfspaces(
        main_half_width, auxiliary_half_width, turn_half
    )
    alphas = jax.vmap(
        scaling_calc_lowspeed_convex_bound,
        in_axes=(None, 0, 0, None, None),
    )(state, all_A, all_b, bb_size, rear_to_center_offset)
    return jnp.min(alphas)


class MVEIntersection_LowSpeed_ISSf_CBF_Dynamic(
    MVELaneChangeAndOverTake_LowSpeed_CBF_Dynamic
):
    """Low-speed intersection using ego-only ISSf-CBF derivatives.

    Dynamic obstacles move in ``step``, but each CBF evaluation intentionally
    treats their current pose as constant.  Relative obstacle kinematics can be
    added later without changing the scene or graph contracts.
    """

    PARAMS = MVELaneChangeAndOverTake_LowSpeed_CBF_Dynamic.PARAMS.copy()
    PARAMS.update(
        {
            "default_state_range": jnp.array(
                [-ROAD_HALF-5, ROAD_HALF+5, -ROAD_HALF-5, ROAD_HALF+5], dtype=jnp.float32
            ),
            # +/-50 m is only the initialization window.  The road and reference
            # continue beyond it, so retain enough rollout room for 256 low-speed
            # steps without clipping ego at the nominal scene boundary.
            "rollout_state_range": jnp.array(
                [-ROAD_HALF - 200.0, ROAD_HALF + 200.0, -ROAD_HALF - 200.0, ROAD_HALF + 200.0],
                dtype=jnp.float32,
            ),
            "comm_radius": 100.0,
            "lane_width": MAIN_LANE_WIDTH,
            "lane_centers": MAIN_LANE_CENTERS,
            # With two lanes in total, one lane width is also the road's
            # centerline-to-curb half-width.  Keep the geometric meaning explicit
            # so this remains clear if the lane count changes later.
            "main_road_half_width": MAIN_LANE_WIDTH,
            "auxiliary_road_half_width": AUX_LANE_WIDTH,
            "intersection_radius": TURN_HALF,
            "obst_bb_size": jnp.array([4.0, 2.0], dtype=jnp.float32),
            "v_min": 1.0 / 3.6,
            "v_max": 40.0 / 3.6,
            "gamma": 100.0,
            "issf_epsilon_0": 1.0,
            "issf_epsilon_rate": 1.0,
            "issf_epsilon_min": 100.0,
        }
    )
    PARAMS.update(
        {
            "obst_radius": jnp.linalg.norm(PARAMS["obst_bb_size"] / 2.0),
        }
    )

    def __init__(
        self,
        num_agents: int,
        area_size: Optional[float] = None,
        max_step: int = 256,
        max_travel: Optional[float] = None,
        dt: float = 0.05,
        reward_min: float = -17.0,
        reward_max: float = 0.5,
        params: dict = None,
    ):
        area_size = self.PARAMS["rollout_state_range"][:4] if area_size is None else area_size
        params = self.PARAMS if params is None else params
        super().__init__(
            num_agents,
            area_size,
            max_step,
            max_travel,
            dt,
            reward_min,
            reward_max,
            params,
        )

    @override
    @property
    def n_cost(self) -> int:
        return 3

    @override
    @property
    def cost_components(self) -> Tuple[str, ...]:
        return "agent collisions", "obs collisions", "bound collisions"

    def _generate_scene(self, key: Array):
        """Generate reset data; fixed-entry subclasses override only this hook."""
        return gen_scene_randomly_split_dynamic(
            key,
            self.num_agents,
            self.num_goals,
            self.params["default_state_range"][:2],
            self.params["default_state_range"][2:4],
            self.params["lane_width"],
            self.params["lane_centers"],
        )

    @override
    def reset(self, key: Array) -> Tuple[GraphsTuple, Array]:
        (
            agents,
            obstacles,
            all_goals,
            all_derivatives,
            dynamic_accel,
            dynamic_target_speed,
        ) = self._generate_scene(key)
        self.all_goals = all_goals
        self.all_dsYddts = all_derivatives
        goal_indices = find_closest_goal_indices(
            self._observable(agents), self._observable(all_goals)
        )
        agent_indices = jnp.arange(agents.shape[0])
        goals = all_goals[agent_indices, goal_indices, :]
        derivatives = all_derivatives[agent_indices, goal_indices, :]
        self.num_obsts = obstacles.shape[0]
        env_state = MVEIntersectionLowSpeedDynamicState(
            agents,
            goals,
            obstacles,
            dynamic_accel,
            dynamic_target_speed,
        )
        return self.get_graph(env_state), derivatives

    @override
    def obst_step_euler(
        self,
        obstacle_states: ObstState,
        dynamic_accel: Array,
        dynamic_target_speed: Array,
    ) -> ObstState:
        """Keep obstacle 0 static and move obstacle 1 towards its target speed."""
        assert obstacle_states.shape == (2, self.state_dim)
        static_obstacle = obstacle_states[0]
        dynamic_obstacle = obstacle_states[1]
        heading = dynamic_obstacle[2:4] / jnp.maximum(
            jnp.linalg.norm(dynamic_obstacle[2:4]), 1e-6
        )
        speed = dynamic_obstacle[4]

        # Clipping the signed speed error produces gradual acceleration and
        # gradual deceleration without overshooting the sampled target speed.
        speed_error = dynamic_target_speed - speed
        speed_delta = jnp.clip(
            speed_error,
            -dynamic_accel * self.dt,
            dynamic_accel * self.dt,
        )
        speed_next = jnp.clip(speed + speed_delta, 0.0, self.params["v_max"])
        speed_mid = 0.5 * (speed + speed_next)
        position_next = dynamic_obstacle[:2] + speed_mid * heading * self.dt
        dynamic_obstacle_next = (
            dynamic_obstacle.at[:2]
            .set(position_next)
            .at[2:4]
            .set(heading)
            .at[4]
            .set(speed_next)
        )
        return jnp.stack([static_obstacle, dynamic_obstacle_next], axis=0)

    @override
    def step(self, graph: GraphsTuple, action: Action, get_eval_info: bool = False):
        del get_eval_info
        env_state = graph.env_states
        action = self.transform_action(action)
        next_agents = self.agent_step_euler(env_state.agent, action)
        next_obstacles = self.obst_step_euler(
            env_state.obstacle,
            env_state.dynamic_obstacle_accel,
            env_state.dynamic_obstacle_target_speed,
        )
        next_goals, next_derivatives = self.goal_dsYddt_step(next_agents)
        next_env_state = MVEIntersectionLowSpeedDynamicState(
            next_agents,
            next_goals,
            next_obstacles,
            env_state.dynamic_obstacle_accel,
            env_state.dynamic_obstacle_target_speed,
        )
        reward = self.get_reward(graph, action)
        cost, cost_real = self.get_cost(graph, action)
        return (
            self.get_graph(next_env_state),
            next_derivatives,
            reward,
            cost,
            cost_real,
            jnp.array(False),
            {},
        )

    def _intersection_alpha(self, state: State) -> Array:
        """Scaling factor shared by real collision, CBF, and ISSf terms."""
        return scaling_calc_intersection_bounds_lowspeed(
            state,
            self.params["ego_bb_size"],
            self.params["ego_lr"],
            self.params["main_road_half_width"],
            self.params["auxiliary_road_half_width"],
            self.params["intersection_radius"],
        )

    def _issf_constraint(self, alpha_fn, state: State, steering: Array) -> Tuple[Array, Array]:
        """Evaluate one ego-only ISSf-CBF constraint for a supplied alpha."""
        threshold = self.params["alpha_thresh"]
        gamma = self.params["gamma"]

        def alpha_from_pose(pose):
            full_state = jnp.array(
                [pose[0], pose[1], pose[2], pose[3], state[4], state[5]]
            )
            return alpha_fn(full_state)

        pose = state[:4]
        alpha, alpha_grad = jax.value_and_grad(alpha_from_pose)(pose)
        alpha = jnp.nan_to_num(alpha, nan=0.0, posinf=1e6, neginf=0.0)
        alpha_grad = jnp.nan_to_num(alpha_grad, nan=0.0, posinf=0.0, neginf=0.0)
        heading = pose[2:4] / jnp.maximum(jnp.linalg.norm(pose[2:4]), 1e-6)
        angular_speed = state[4] / self.params["ego_L"] * jnp.tan(steering)
        pose_dot = jnp.array(
            [
                state[4] * heading[0],
                state[4] * heading[1],
                -heading[1] * angular_speed,
                heading[0] * angular_speed,
            ]
        )
        h = alpha - threshold
        h_dot = jnp.dot(alpha_grad, pose_dot)

        # The uncertain steering channel is identical to the existing low-speed
        # ISSf implementation.  Longitudinal acceleration is intentionally not
        # included because this alpha has relative degree greater than one in it.
        steering_channel = jnp.array(
            [
                0.0,
                0.0,
                -heading[1] * state[4] / self.params["ego_L"],
                heading[0] * state[4] / self.params["ego_L"],
            ]
        )
        g_dot = jnp.dot(alpha_grad, steering_channel)
        epsilon = self.params["issf_epsilon_min"] + self.params[
            "issf_epsilon_0"
        ] * jax.nn.softplus(self.params["issf_epsilon_rate"] * h)
        young_penalty = jnp.square(g_dot) / epsilon
        residual = h_dot / gamma + h - young_penalty / gamma
        cost = jnp.nan_to_num(-residual, nan=10.0, posinf=10.0, neginf=-3.0)
        return cost, 1.0 - alpha

    @override
    def get_cost(self, graph: GraphsTuple, action: Action) -> Tuple[Cost, Cost]:
        """Compute obstacle and polygon-boundary ISSf-CBF costs.

        Obstacle poses are closed over by ``obstacle_alpha``.  Consequently JAX
        differentiates only with respect to ego pose, matching the requested
        instantaneous-stationary approximation for dynamic obstacles.
        """
        num_agents = graph.env_states.agent.shape[0]
        num_obstacles = graph.env_states.obstacle.shape[0]
        steering = self._filter_delta(graph.env_states.agent[:, 5], action[:, 1])

        # Multi-ego collision CBF remains disabled, consistently with the current
        # low-speed ISSf environments.  This project normally trains one ego.
        agent_cost = -jnp.ones((num_agents,), dtype=jnp.float32) * 3.0
        agent_cost_real = -jnp.ones((num_agents,), dtype=jnp.float32) * 3.0

        if num_obstacles == 0:
            obstacle_cost = -jnp.ones((num_agents,), dtype=jnp.float32) * 3.0
            obstacle_cost_real = -jnp.ones((num_agents,), dtype=jnp.float32) * 3.0
        else:
            i_pairs, j_pairs = gen_i_j_pairs(num_agents, num_obstacles)

            def between(state, obstacle, steering_value):
                def obstacle_alpha(ego_state):
                    return scaling_calc(
                        ego_state,
                        obstacle,
                        self.params["ego_bb_size"],
                        self.params["ego_lr"],
                        self.params["obst_bb_size"],
                        self.params["obst_lr"],
                    )

                return self._issf_constraint(obstacle_alpha, state, steering_value)

            pair_costs, pair_reals = jax.vmap(between)(
                graph.env_states.agent[i_pairs],
                graph.env_states.obstacle[j_pairs],
                steering[i_pairs],
            )
            obstacle_cost = jnp.max(
                pair_costs.reshape((num_agents, num_obstacles)), axis=1
            )
            obstacle_cost_real = jnp.max(
                pair_reals.reshape((num_agents, num_obstacles)), axis=1
            )

        bound_cost, bound_cost_real = jax.vmap(
            lambda state, steering_value: self._issf_constraint(
                self._intersection_alpha, state, steering_value
            )
        )(graph.env_states.agent, steering)

        cost = jnp.stack([agent_cost, obstacle_cost, bound_cost], axis=1)
        cost_real = jnp.stack(
            [agent_cost_real, obstacle_cost_real, bound_cost_real], axis=1
        )
        cost = jnp.where(cost <= 0.0, cost, cost + 1.0)
        return jnp.clip(cost, a_min=-10.0, a_max=10.0), cost_real

    def _scaling_cost_intersection(self, graph: GraphsTuple) -> Tuple[Cost, Cost]:
        """Action-independent alpha costs used by ``unsafe_mask``."""
        threshold = self.params["alpha_thresh"]
        agents = graph.env_states.agent
        obstacles = graph.env_states.obstacle
        num_agents = agents.shape[0]
        num_obstacles = obstacles.shape[0]
        agent_cost = -jnp.ones((num_agents,), dtype=jnp.float32)
        agent_real = -jnp.ones((num_agents,), dtype=jnp.float32)

        i_pairs, j_pairs = gen_i_j_pairs(num_agents, num_obstacles)
        obstacle_alphas = jax.vmap(
            scaling_calc, in_axes=(0, 0, None, None, None, None)
        )(
            agents[i_pairs],
            obstacles[j_pairs],
            self.params["ego_bb_size"],
            self.params["ego_lr"],
            self.params["obst_bb_size"],
            self.params["obst_lr"],
        )
        obstacle_alphas = obstacle_alphas.reshape((num_agents, num_obstacles))
        obstacle_cost = jnp.max(threshold - obstacle_alphas, axis=1)
        obstacle_real = jnp.max(1.0 - obstacle_alphas, axis=1)
        bound_alpha = jax.vmap(self._intersection_alpha)(agents)
        bound_cost = threshold - bound_alpha
        bound_real = 1.0 - bound_alpha
        return (
            jnp.stack([agent_cost, obstacle_cost, bound_cost], axis=1),
            jnp.stack([agent_real, obstacle_real, bound_real], axis=1),
        )

    @override
    @ft.partial(jax.jit, static_argnums=(0,))
    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        _, cost_real = self._scaling_cost_intersection(graph)
        return jnp.any(cost_real >= 0.0, axis=-1)

    @override
    def render_video(
        self,
        rollout: Rollout,
        video_path: pathlib.Path,
        Ta_is_unsafe=None,
        n_goals: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Render the same asymmetric road polygons used by the CBF.

        Keeping the drawing coordinates derived from environment parameters is
        useful during inspection: a rendered curb cannot silently disagree with
        the half-space boundary used for training.
        """
        del n_goals, kwargs
        road_half = float(ROAD_HALF)
        turn_half = float(self.params["intersection_radius"])
        main_half = float(self.params["main_road_half_width"])
        auxiliary_half = float(self.params["auxiliary_road_half_width"])
        trajectory = rollout.graph

        # Expand the viewport when rollout entities have travelled beyond the
        # nominal 100 x 100 m initialization window.  Forbidden regions and road
        # markings are then drawn to the new viewport edge, visually expressing
        # the same unbounded half-spaces used by the CBF.
        visible_positions = np.concatenate(
            [
                np.asarray(trajectory.env_states.agent[..., :2]).reshape(-1, 2),
                np.asarray(trajectory.env_states.obstacle[..., :2]).reshape(-1, 2),
                np.asarray(trajectory.env_states.goal[..., :2]).reshape(-1, 2),
            ],
            axis=0,
        )
        max_visible_coordinate = float(np.max(np.abs(visible_positions)))
        view_half = road_half + 5.0

        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=120)
        ax.set_xlim(-view_half, view_half)
        ax.set_ylim(-view_half, view_half)
        ax.set_aspect("equal")
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")

        # These vertices close only at the current viewport edge.  The actual
        # CBF polygons are unbounded and are never clipped at ROAD_HALF.
        corner_polygons = [
            [(-view_half, -view_half), (-view_half, -main_half), (-turn_half, -main_half),
             (-auxiliary_half, -turn_half), (-auxiliary_half, -view_half)],
            [(auxiliary_half, -view_half), (auxiliary_half, -turn_half), (turn_half, -main_half),
             (view_half, -main_half), (view_half, -view_half)],
            [(auxiliary_half, turn_half), (turn_half, main_half), (view_half, main_half),
             (view_half, view_half), (auxiliary_half, view_half)],
            [(-view_half, main_half), (-turn_half, main_half), (-auxiliary_half, turn_half),
             (-auxiliary_half, view_half), (-view_half, view_half)],
        ]
        for polygon in corner_polygons:
            ax.fill(
                *zip(*polygon),
                facecolor="#e6e6e6",
                edgecolor="#666666",
                linewidth=1.0,
                zorder=0,
            )

        road_color = "#1f4e79"
        dash_style = (0, (7, 7))
        # Main-road curbs and its two-lane center separator.
        for y in (-main_half, main_half):
            ax.plot([-view_half, -turn_half], [y, y], color=road_color, linewidth=1.4)
            ax.plot([turn_half, view_half], [y, y], color=road_color, linewidth=1.4)
        ax.plot([-view_half, -turn_half], [0.0, 0.0], color=road_color, linestyle=dash_style)
        ax.plot([turn_half, view_half], [0.0, 0.0], color=road_color, linestyle=dash_style)

        # Auxiliary-road curbs and its two-lane center separator.
        for x in (-auxiliary_half, auxiliary_half):
            ax.plot([x, x], [-view_half, -turn_half], color=road_color, linewidth=1.4)
            ax.plot([x, x], [turn_half, view_half], color=road_color, linewidth=1.4)
        ax.plot([0.0, 0.0], [-view_half, -turn_half], color=road_color, linestyle=dash_style)
        ax.plot([0.0, 0.0], [turn_half, view_half], color=road_color, linestyle=dash_style)

        # Match mve_lowspeed_base.py: render exactly the reference points that
        # ego recorded during rollout.  No trajectory regeneration, interpolation,
        # or tangent extension is performed here.
        recorded_goals = np.asarray(trajectory.env_states.goal[:, :, :2])
        ax.scatter(
            recorded_goals[:, :, 0].reshape(-1),
            recorded_goals[:, :, 1].reshape(-1),
            color="#2fdd00",
            zorder=7,
            s=5,
            alpha=1.0,
            marker=".",
        )

        graph0 = tree_index(trajectory, 0)
        agent_arrows, agent_rects = self._plot_pose(
            ax,
            graph0.env_states.agent,
            self.params["ego_bb_size"],
            self.params["ego_lr"],
            "#0068ff",
            6,
        )
        obstacle_arrows, obstacle_rects = self._plot_pose(
            ax,
            graph0.env_states.obstacle,
            self.params["obst_bb_size"],
            self.params["obst_lr"],
            "#8a0000",
            5,
        )

        def edge_segments(graph):
            positions = np.asarray(graph.states[:, :2])
            edge_index = np.stack(
                [np.asarray(graph.senders), np.asarray(graph.receivers)], axis=0
            )
            padding_id = int(np.asarray(graph.n_node)) - 1
            edge_index = edge_index[:, ~np.any(edge_index == padding_id, axis=0)]
            if edge_index.shape[1] == 0:
                return np.zeros((0, 2, 2), dtype=np.float32)
            return np.stack(
                [positions[edge_index[0]], positions[edge_index[1]]], axis=1
            )

        edge_collection = LineCollection(
            edge_segments(graph0), colors="0.2", linewidths=1.0, alpha=0.35, zorder=3
        )
        ax.add_collection(edge_collection)

        # Keep the per-step training signals on the left, matching the other
        # low-speed environments.  Agent-agent cost is intentionally omitted:
        # this intersection task asks to inspect obstacle and boundary costs.
        cost_text = ax.text(
            0.02,
            1.00,
            "",
            transform=ax.transAxes,
            va="bottom",
            size=14,
            color="k",
        )
        # Runtime status belongs on the opposite side so it cannot overlap the
        # multi-line cost/reward block.
        step_text = ax.text(
            0.99,
            1.10,
            "step=0",
            transform=ax.transAxes,
            va="bottom",
            ha="right",
            size=14,
            color="k",
        )
        unsafe_text = ax.text(
            0.99,
            1.03,
            "unsafe=[]",
            transform=ax.transAxes,
            va="bottom",
            ha="right",
            size=14,
            color="k",
        )

        def update_pose(arrows, rectangles, states, bb_size, rear_offset):
            headings = np.asarray(self._normalize_heading(states[:, 2:4]))
            centers = np.asarray(states[:, :2] + rear_offset * self._normalize_heading(states[:, 2:4]))
            angles = np.arctan2(headings[:, 1], headings[:, 0]) * 180.0 / np.pi
            bb = np.asarray(bb_size)
            radius = float(jnp.linalg.norm(jnp.asarray(bb_size)))
            for idx in range(states.shape[0]):
                arrows[idx].set_data(
                    x=centers[idx, 0],
                    y=centers[idx, 1],
                    dx=headings[idx, 0] * radius / 2.0,
                    dy=headings[idx, 1] * radius / 2.0,
                )
                rectangles[idx].set_xy(
                    (centers[idx, 0] - bb[0] / 2.0, centers[idx, 1] - bb[1] / 2.0)
                )
                rectangles[idx].set_angle(float(angles[idx]))

        def update(frame: int):
            graph = tree_index(trajectory, frame)
            update_pose(
                agent_arrows, agent_rects, graph.env_states.agent,
                self.params["ego_bb_size"], self.params["ego_lr"]
            )
            update_pose(
                obstacle_arrows, obstacle_rects, graph.env_states.obstacle,
                self.params["obst_bb_size"], self.params["obst_lr"]
            )
            edge_collection.set_segments(edge_segments(graph))

            # rollout.costs has shape [step, agent, cost_component].  Display
            # the worst agent value for each requested component, as in
            # mve_lowspeed_base.py.
            if frame < len(rollout.costs):
                frame_costs = np.asarray(rollout.costs[frame])
                obstacle_cost = float(np.max(frame_costs[:, 1]))
                boundary_cost = float(np.max(frame_costs[:, 2]))
                reward = float(np.asarray(rollout.rewards[frame]))
                cost_text.set_text(
                    "Cost:\n"
                    f"    {self.cost_components[1]}: {obstacle_cost:5.4f}\n"
                    f"    {self.cost_components[2]}: {boundary_cost:5.4f}\n"
                    f"Reward: {reward:5.4f}"
                )

            unsafe_text.set_text("unsafe=[]")
            if Ta_is_unsafe is not None and frame < len(Ta_is_unsafe):
                unsafe_text.set_text(
                    f"unsafe={np.where(Ta_is_unsafe[frame])[0]}"
                )
            step_text.set_text(f"step={frame}")
            return [
                edge_collection,
                *agent_arrows,
                *agent_rects,
                *obstacle_arrows,
                *obstacle_rects,
                cost_text,
                step_text,
                unsafe_text,
            ]

        animation = FuncAnimation(
            fig, update, frames=len(trajectory.n_node), interval=1000.0 / 30.0, blit=True
        )
        try:
            save_anim(animation, video_path)
        finally:
            plt.close(fig)


# A compact alias is useful in Python code, while the underscored class name
# matches the repository's existing environment naming convention.
MVEIntersectionLowSpeedISSfCBFDynamic = MVEIntersection_LowSpeed_ISSf_CBF_Dynamic
