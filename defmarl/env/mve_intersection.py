import pathlib
import jax
import jax.numpy as jnp
import functools as ft
import numpy as np

from typing import Optional, Tuple, List
from typing_extensions import override
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrow

from .mve import MVE, MVEEnvState, MVEEnvGraphsTuple
from .mve_lanechangeANDovertake import MVELaneChangeAndOverTake
from .designed_scene_gen_intersection import gen_scene_randomly, gen_handmade_scene
from defmarl.trainer.data import Rollout
from defmarl.utils.graph import GraphsTuple
from defmarl.utils.typing import Array, AgentState, ObstState, Cost
from defmarl.utils.utils import find_closest_goal_indices, tree_index, MutablePatchCollection, save_anim
from ..utils.scaling import scaling_calc
from ..utils.scaling_intersection import scaling_calc_intersection_bounds


INF = jnp.inf


class MVEIntersection(MVELaneChangeAndOverTake):
    """Intersection turning environment.

    This class keeps the graph/state/action contract of MVELaneChangeAndOverTake,
    but replaces the straight-road assumptions that are unsafe for intersection
    turns: scene generation, rollout bounds, obstacle motion, and x/y boundary
    costs.
    """

    PARAMS = dict(MVELaneChangeAndOverTake.PARAMS)
    PARAMS.update({
        "default_state_range": jnp.array([
            -100., 100., -100., 100.,
            -INF, INF, -INF, INF,
            -180., 180., -INF, INF,
            -INF, INF, -INF, INF,
        ]),
        "rollout_state_range": jnp.array([
            -110., 110., -110., 110.,
            -150., 150., -150., 150.,
            -180., 180., -360., 360.,
            -INF, INF, -INF, INF,
        ]),
        "rollout_state_b_range": jnp.array([
            -INF, INF, -INF, INF,
            30., 100., -INF, INF,
            -INF, INF, -INF, INF,
            -INF, INF, -INF, INF,
        ]),
        "lane_width": 3.,
        "intersection_radius": 17.5,
        "alpha_thresh": 1.05,
    })
    PARAMS.update({
        "lane_centers": jnp.array([-3., 0., 3.], dtype=jnp.float32),
    })

    def __init__(self,
                 num_agents: int,
                 area_size: Optional[float] = None,
                 max_step: int = 256,
                 max_travel: Optional[float] = None,
                 dt: float = 0.05,
                 reward_min: float = -17.,
                 reward_max: float = 0.5,
                 params: dict = None):
        area_size = MVEIntersection.PARAMS["rollout_state_range"][:4] if area_size is None else area_size
        params = MVEIntersection.PARAMS if params is None else params
        super().__init__(num_agents, area_size, max_step, max_travel, dt, reward_min, reward_max, params)

    @override
    @property
    def cost_components(self) -> Tuple[str, ...]:
        return (
            "agent collisions",
            "obs collisions",
            "bound collisions",
        )

    @override
    @property
    def n_cost(self) -> int:
        return 3

    @override
    def reset(self, key: Array) -> Tuple[GraphsTuple, jnp.ndarray]:
        xrange = self.params["default_state_range"][:2]
        yrange = self.params["default_state_range"][2:4]
        lane_width = self.params["lane_width"]
        lane_centers = self.params["lane_centers"]

        agents, obsts, all_goals, all_dsYddts = gen_scene_randomly(
            key, self.num_agents, self.num_goals, xrange, yrange, lane_width, lane_centers
        )
        # agents, obsts, all_goals, all_dsYddts = gen_handmade_scene(
        #     key, self.num_agents, self.num_goals, xrange, yrange, lane_width, lane_centers
        # )
        self.all_goals = all_goals
        self.all_dsYddts = all_dsYddts
        goals_init_indices = find_closest_goal_indices(agents, all_goals)
        agents_indices = jnp.arange(agents.shape[0])
        goals = all_goals[agents_indices, goals_init_indices, :]
        dsYddts = all_dsYddts[agents_indices, goals_init_indices, :]
        env_state = MVEEnvState(agents, goals, obsts)
        self.num_obsts = obsts.shape[0]
        return self.get_graph(env_state), dsYddts

    @override
    def obst_step_euler(self, o_obst_states: ObstState) -> ObstState:
        num_obsts = o_obst_states.shape[0]
        assert o_obst_states.shape == (num_obsts, self.state_dim)
        o_x = o_obst_states[:, 0]
        o_y = o_obst_states[:, 1]
        o_vx = o_obst_states[:, 2]
        o_vy = o_obst_states[:, 3]
        o_obst_states_new = o_obst_states.at[:, 0].set(o_x + o_vx / 3.6 * self.dt)
        o_obst_states_new = o_obst_states_new.at[:, 1].set(o_y + o_vy / 3.6 * self.dt)
        return o_obst_states_new

    def _bound_cost(self, agent_states: AgentState) -> Tuple[Array, Array]:
        thresh = self.params["alpha_thresh"]
        alpha = jax.vmap(scaling_calc_intersection_bounds)(agent_states)
        return thresh - alpha, 1.0 - alpha

    @override
    def get_cost(self, graph: MVEEnvGraphsTuple) -> Tuple[Cost, Cost]:
        thresh = self.params["alpha_thresh"]
        num_agents = graph.env_states.agent.shape[0]
        num_obsts = graph.env_states.obstacle.shape[0]
        agent_states = graph.type_states(type_idx=self.AGENT, n_type=num_agents)
        # state: x y vx vy theta dthetadt bw bh
        convert_vec_s = jnp.array([1, 1, 3.6, 3.6, 180/jnp.pi, 180/jnp.pi, 1, 1], dtype=jnp.float32)
        # scaling 系列函数内部使用弧度计算旋转矩阵，而环境 state 中 theta/dtheta 的单位是 degree。
        # 如果不转换，东/南/北方向车辆的包围盒姿态会被算歪，靠外侧车道初始时也可能误判为接近 bound。
        agent_states_metric = agent_states / convert_vec_s

        if num_agents == 1:
            a_agent_cost = -jnp.ones((num_agents,), dtype=jnp.float32)
            a_agent_cost_real = -jnp.ones((num_agents,), dtype=jnp.float32)
        else:
            a_agent_cost = -jnp.ones((num_agents,), dtype=jnp.float32)
            a_agent_cost_real = -jnp.ones((num_agents,), dtype=jnp.float32)

        if num_obsts == 0:
            a_obst_cost = -jnp.ones((num_agents,), dtype=jnp.float32)
            a_obst_cost_real = -jnp.ones((num_agents,), dtype=jnp.float32)
        else:
            obstacle_states = graph.type_states(type_idx=self.OBST, n_type=num_obsts)
            obstacle_states_metric = obstacle_states / convert_vec_s
            i_grid, j_grid = jnp.meshgrid(jnp.arange(num_agents), jnp.arange(num_obsts), indexing="ij")
            state_i_pairs = agent_states_metric[i_grid.flatten(), :]
            state_j_pairs = obstacle_states_metric[j_grid.flatten(), :]
            alpha_pairs = jax.vmap(scaling_calc, in_axes=(0, 0))(state_i_pairs, state_j_pairs)
            alpha_matrix = alpha_pairs.reshape((num_agents, num_obsts))
            a_obst_cost = jnp.max(thresh - alpha_matrix, axis=1)
            a_obst_cost_real = jnp.max(1.0 - alpha_matrix, axis=1)

        a_bound_cost, a_bound_cost_real = self._bound_cost(agent_states_metric)

        cost = jnp.stack([
            a_agent_cost,
            a_obst_cost,
            a_bound_cost,
        ], axis=1)
        cost_real = jnp.stack([
            a_agent_cost_real,
            a_obst_cost_real,
            a_bound_cost_real,
        ], axis=1)
        assert cost.shape == (num_agents, self.n_cost)
        assert cost_real.shape == (num_agents, self.n_cost)

        eps = 1.0
        cost = jnp.where(cost <= 0.0, cost, cost + eps)
        cost = jnp.clip(cost, a_min=-3.0)
        return cost, cost_real

    @override
    @ft.partial(jax.jit, static_argnums=(0,))
    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        _, cost_real = self.get_cost(graph)
        return jnp.any(cost_real >= 0.0, axis=-1)

    def render_video(
            self,
            rollout: Rollout,
            video_path: pathlib.Path,
            Ta_is_unsafe=None,
            viz_opts: Optional[dict] = None,
            n_goals: Optional[int] = None,
            **kwargs
    ) -> None:
        """渲染十字路口场景。

        这里不能复用 lanechange 的画法：lanechange 只画一组平行车道线，
        而十字路口需要画南/北/东/西四个入口道路和四个角落障碍区域。
        """
        T_goal_states = jax.vmap(lambda x: x.type_states(type_idx=MVE.GOAL, n_type=self.num_agents))(rollout.graph)
        ref_goals = np.asarray(T_goal_states[:, :, :2])
        n_goals = self.num_agents if n_goals is None else n_goals
        if viz_opts is None:
            viz_opts = {}

        ax: Axes
        xlim = np.array([-110.0, 110.0])
        ylim = np.array([-110.0, 110.0])
        fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=120)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.set(aspect="equal")
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        plt.axis("on")

        road_half = 100.0
        turn_half = self.params["intersection_radius"]
        lane_half_width = 4.5
        lane_sep = 1.5

        # 四个角落障碍区域，与 scaling_intersection.py 中的四个 bound 定义一致。
        corner_color = "#e6e6e6"
        corner_edge_color = "#666666"
        corner_polys = [
            [(-road_half, -road_half), (-road_half, -lane_half_width), (-turn_half, -lane_half_width),
             (-lane_half_width, -turn_half), (-lane_half_width, -road_half)],
            [(lane_half_width, -road_half), (lane_half_width, -turn_half), (turn_half, -lane_half_width),
             (road_half, -lane_half_width), (road_half, -road_half)],
            [(lane_half_width, turn_half), (turn_half, lane_half_width), (road_half, lane_half_width),
             (road_half, road_half), (lane_half_width, road_half)],
            [(-road_half, lane_half_width), (-turn_half, lane_half_width), (-lane_half_width, turn_half),
             (-lane_half_width, road_half), (-road_half, road_half)],
        ]
        for poly in corner_polys:
            ax.fill(*zip(*poly), facecolor=corner_color, edgecolor=corner_edge_color, linewidth=1.2, zorder=0)

        # 道路外边界和车道虚线。中心 29m x 29m 区域留空，表示路口区域。
        road_color = "#1f4e79"
        dash_style = (0, (8, 8))
        for x in [-lane_half_width, lane_half_width]:
            ax.plot([x, x], [-road_half, -turn_half], color=road_color, linewidth=1.5, zorder=1)
            ax.plot([x, x], [turn_half, road_half], color=road_color, linewidth=1.5, zorder=1)
        for x in [-lane_sep, lane_sep]:
            ax.plot([x, x], [-road_half, -turn_half], color=road_color, linewidth=1.0,
                    linestyle=dash_style, zorder=1)
            ax.plot([x, x], [turn_half, road_half], color=road_color, linewidth=1.0,
                    linestyle=dash_style, zorder=1)
        for y in [-lane_half_width, lane_half_width]:
            ax.plot([-road_half, -turn_half], [y, y], color=road_color, linewidth=1.5, zorder=1)
            ax.plot([turn_half, road_half], [y, y], color=road_color, linewidth=1.5, zorder=1)
        for y in [-lane_sep, lane_sep]:
            ax.plot([-road_half, -turn_half], [y, y], color=road_color, linewidth=1.0,
                    linestyle=dash_style, zorder=1)
            ax.plot([turn_half, road_half], [y, y], color=road_color, linewidth=1.0,
                    linestyle=dash_style, zorder=1)

        ax.axhline(0.0, color="#999999", linewidth=0.5, alpha=0.5, zorder=1)
        ax.axvline(0.0, color="#999999", linewidth=0.5, alpha=0.5, zorder=1)
        ax.text(0.0, -106.0, "South road", ha="center", va="center", fontsize=12)
        ax.text(0.0, 106.0, "North road", ha="center", va="center", fontsize=12)
        ax.text(-106.0, 0.0, "West road", ha="center", va="center", fontsize=12)
        ax.text(106.0, 0.0, "East road", ha="center", va="center", fontsize=12)

        T_graph = rollout.graph
        graph0 = tree_index(T_graph, 0)

        agent_color = "#0068ff"
        goal_color = "#2fdd00"
        obst_color = "#8a0000"
        edge_goal_color = goal_color

        obsts_state = graph0.type_nodes(type_idx=MVE.OBST, n_type=self.num_obsts)
        obsts_pos = obsts_state[:, :2]
        obsts_theta = obsts_state[:, 4]
        obsts_bb_size = obsts_state[:, 6:8]
        obsts_radius = jnp.linalg.norm(obsts_bb_size, axis=1)
        plot_obsts_arrow = [FancyArrow(x=obsts_pos[i, 0], y=obsts_pos[i, 1],
                                       dx=jnp.cos(obsts_theta[i] * jnp.pi / 180) * obsts_radius[i] / 2,
                                       dy=jnp.sin(obsts_theta[i] * jnp.pi / 180) * obsts_radius[i] / 2,
                                       length_includes_head=True, width=0.3,
                                       color=obst_color, alpha=1.0)
                            for i in range(len(obsts_theta))]
        plot_obsts_rec = [plt.Rectangle(xy=tuple(obsts_pos[i, :] - obsts_bb_size[i, :] / 2),
                                        width=obsts_bb_size[i, 0], height=obsts_bb_size[i, 1],
                                        angle=obsts_theta[i], rotation_point="center",
                                        color=obst_color, linewidth=0.0, alpha=0.6)
                          for i in range(len(obsts_theta))]
        col_obsts = MutablePatchCollection(plot_obsts_arrow + plot_obsts_rec, match_original=True, zorder=5)
        ax.add_collection(col_obsts)

        agents_state = graph0.type_states(type_idx=MVE.AGENT, n_type=self.num_agents)
        agents_pos = agents_state[:, :2]
        agents_theta = agents_state[:, 4]
        agents_bb_size = agents_state[:, 6:8]
        agents_radius = jnp.linalg.norm(agents_bb_size, axis=1)
        arrow_width_scale = jnp.maximum(jnp.mean(obsts_radius), 1.0)
        plot_agents_arrow = [FancyArrow(x=agents_pos[i, 0], y=agents_pos[i, 1],
                                        dx=jnp.cos(agents_theta[i] * jnp.pi / 180) * agents_radius[i] / 2,
                                        dy=jnp.sin(agents_theta[i] * jnp.pi / 180) * agents_radius[i] / 2,
                                        width=agents_radius[i] / arrow_width_scale * 0.3,
                                        length_includes_head=True, alpha=1.0, color=agent_color)
                             for i in range(self.num_agents)]
        plot_agents_rec = [plt.Rectangle(xy=tuple(agents_pos[i, :] - agents_bb_size[i, :] / 2),
                                         width=agents_bb_size[i, 0], height=agents_bb_size[i, 1],
                                         angle=agents_theta[i], rotation_point="center",
                                         color=agent_color, linewidth=0.0, alpha=0.6)
                           for i in range(self.num_agents)]
        col_agents = MutablePatchCollection(plot_agents_arrow + plot_agents_rec, match_original=True, zorder=6)
        ax.add_collection(col_agents)

        all_ref_xs = ref_goals[:, :, 0].reshape(-1)
        all_ref_ys = ref_goals[:, :, 1].reshape(-1)
        ax.scatter(all_ref_xs, all_ref_ys, color=goal_color, zorder=4, s=8, alpha=0.9, marker=".")

        all_pos = graph0.states[:, :2]
        edge_index = np.stack([graph0.senders, graph0.receivers], axis=0)
        is_pad = np.any(edge_index == self.num_agents + n_goals + self.num_obsts, axis=0)
        e_edge_index = edge_index[:, ~is_pad]
        e_start, e_end = all_pos[e_edge_index[0, :]], all_pos[e_edge_index[1, :]]
        e_lines = np.stack([e_start, e_end], axis=1)
        e_is_goal = (self.num_agents <= graph0.senders) & (graph0.senders < self.num_agents + n_goals)
        e_is_goal = e_is_goal[~is_pad]
        e_colors = [edge_goal_color if e_is_goal[ii] else "0.2" for ii in range(len(e_start))]
        col_edges = LineCollection(e_lines, colors=e_colors, linewidths=2, alpha=0.5, zorder=3)
        ax.add_collection(col_edges)

        text_font_opts = dict(size=12, color="k", family="sans-serif", weight="normal", transform=ax.transAxes)
        cost_text = ax.text(0.02, 1.00, "Cost: 1.0\nReward: 1.0", va="bottom", **text_font_opts)
        safe_text = []
        if Ta_is_unsafe is not None:
            safe_text = [ax.text(0.99, 1.00, "Unsafe: {}", va="bottom", ha="right", **text_font_opts)]
        kk_text = ax.text(0.99, 1.04, "kk=0", va="bottom", ha="right", **text_font_opts)
        if rollout.zs is not None:
            z_text = ax.text(0.5, 1.04, "z: []", va="bottom", ha="center", **text_font_opts)

        label_font_opts = dict(size=14, color="k", family="sans-serif", weight="normal",
                               ha="center", va="center", transform=ax.transData,
                               clip_on=True, zorder=8, alpha=0.0)
        agent_labels = [ax.text(float(agents_pos[ii, 0]), float(agents_pos[ii, 1]), f"{ii}", **label_font_opts)
                        for ii in range(self.num_agents)]

        if "Vh" in viz_opts:
            Vh_text = ax.text(0.99, 0.99, "Vh: []", va="top", ha="right", **text_font_opts)

        def init_fn() -> List[plt.Artist]:
            return [col_obsts, col_agents, col_edges, *agent_labels, cost_text, *safe_text, kk_text]

        def update(kk: int) -> List[plt.Artist]:
            graph = tree_index(T_graph, kk)
            n_pos_t = graph.states[:-1, :2]
            n_theta_t = graph.states[:-1, 4]
            n_bb_size_t = graph.nodes[:-1, 6:8]
            n_radius = jnp.linalg.norm(n_bb_size_t, axis=1)

            for ii in range(self.num_agents):
                plot_agents_arrow[ii].set_data(
                    x=n_pos_t[ii, 0], y=n_pos_t[ii, 1],
                    dx=jnp.cos(n_theta_t[ii] * jnp.pi / 180) * n_radius[ii] / 2,
                    dy=jnp.sin(n_theta_t[ii] * jnp.pi / 180) * n_radius[ii] / 2,
                )
                plot_agents_rec[ii].set_xy(xy=tuple(n_pos_t[ii, :] - n_bb_size_t[ii, :] / 2))
                plot_agents_rec[ii].set_angle(angle=n_theta_t[ii])
                agent_labels[ii].set_position(n_pos_t[ii, :])

            for ii in range(self.num_obsts):
                obst_node_idx = self.num_agents + n_goals + ii
                plot_obsts_arrow[ii].set_data(
                    x=n_pos_t[obst_node_idx, 0],
                    y=n_pos_t[obst_node_idx, 1],
                    dx=jnp.cos(n_theta_t[obst_node_idx] * jnp.pi / 180) * n_radius[obst_node_idx] / 2,
                    dy=jnp.sin(n_theta_t[obst_node_idx] * jnp.pi / 180) * n_radius[obst_node_idx] / 2,
                )
                plot_obsts_rec[ii].set_xy(
                    xy=tuple(n_pos_t[obst_node_idx, :] - n_bb_size_t[obst_node_idx, :] / 2)
                )
                plot_obsts_rec[ii].set_angle(angle=n_theta_t[obst_node_idx])

            e_edge_index_t = np.stack([graph.senders, graph.receivers], axis=0)
            is_pad_t = np.any(e_edge_index_t == self.num_agents + n_goals + self.num_obsts, axis=0)
            e_edge_index_t = e_edge_index_t[:, ~is_pad_t]
            e_start_t, e_end_t = n_pos_t[e_edge_index_t[0, :]], n_pos_t[e_edge_index_t[1, :]]
            e_is_goal_t = (self.num_agents <= graph.senders) & (graph.senders < self.num_agents + n_goals)
            e_is_goal_t = e_is_goal_t[~is_pad_t]
            e_colors_t = [edge_goal_color if e_is_goal_t[ii] else "0.2" for ii in range(len(e_start_t))]
            e_lines_t = np.stack([e_start_t, e_end_t], axis=1)
            col_edges.set_segments(e_lines_t)
            col_edges.set_colors(e_colors_t)

            if kk < len(rollout.costs):
                all_costs = ""
                for i_cost in range(rollout.costs[kk].shape[1]):
                    all_costs += f"    {self.cost_components[i_cost]}: {rollout.costs[kk][:, i_cost].max():5.4f}\n"
                all_costs = all_costs[:-2]
                cost_text.set_text(f"Cost:\n{all_costs}\nReward: {rollout.rewards[kk]:5.4f}")
            else:
                cost_text.set_text("")
            if Ta_is_unsafe is not None:
                if kk < len(Ta_is_unsafe):
                    unsafe_idx = np.where(Ta_is_unsafe[kk])[0]
                    safe_text[0].set_text("Unsafe: {}".format(unsafe_idx))
                else:
                    safe_text[0].set_text("Unsafe: {}")

            kk_text.set_text("kk={:04}".format(kk))
            if rollout.zs is not None:
                z_text.set_text(f"z: {rollout.zs[kk]}")
            if "Vh" in viz_opts:
                Vh_text.set_text(f"Vh: {viz_opts['Vh'][kk]}")

            return [col_obsts, col_agents, col_edges, *agent_labels, cost_text, *safe_text, kk_text]

        fps = 30.0
        mspf = 1_000 / fps
        ani = FuncAnimation(fig, update, frames=len(T_graph.n_node), init_func=init_fn, interval=mspf, blit=True)
        save_anim(ani, video_path)
