import jax
import jax.numpy as jnp
import jax.random as jr

from abc import ABC, abstractmethod
from typing import Tuple

from defmarl.utils.typing import AgentState, Array, ObstState, PathRefs, PRNGKey, State
from defmarl.utils.utils import const_f, quintic_polynomial_f, three_sec_f


def make_state(x: Array, y: Array, theta_rad: Array, v_mps: Array, delta_rad: Array = 0.0) -> State:
    heading_x = jnp.cos(theta_rad)
    heading_y = jnp.sin(theta_rad)
    return jnp.stack([x, y, heading_x, heading_y, v_mps, jnp.asarray(delta_rad, dtype=jnp.float32)])


def calc_quintic_eff_from_heading(starts: AgentState, terminals: AgentState):
    zeros = jnp.zeros((starts.shape[0],), dtype=jnp.float32)

    def A_b_create_and_solve(start, terminal):
        x0 = start[0]
        x1 = terminal[0]
        A = jnp.array([[1, x0, x0 ** 2, x0 ** 3, x0 ** 4, x0 ** 5],
                       [0, 1, 2 * x0, 3 * x0 ** 2, 4 * x0 ** 3, 5 * x0 ** 4],
                       [0, 0, 2, 6 * x0, 12 * x0 ** 2, 20 * x0 ** 3],
                       [1, x1, x1 ** 2, x1 ** 3, x1 ** 4, x1 ** 5],
                       [0, 1, 2 * x1, 3 * x1 ** 2, 4 * x1 ** 3, 5 * x1 ** 4],
                       [0, 0, 2, 6 * x1, 12 * x1 ** 2, 20 * x1 ** 3]])
        y0 = start[1]
        y1 = terminal[1]
        slope0 = start[3] / jnp.maximum(start[2], 1e-6)
        slope1 = terminal[3] / jnp.maximum(terminal[2], 1e-6)
        b = jnp.array([y0, slope0, 0, y1, slope1, 0])
        return jnp.linalg.solve(A, b)

    coeffs_f = jax.vmap(A_b_create_and_solve, in_axes=(0, 0))(starts, terminals)
    coeffs_df = jnp.stack(
        [coeffs_f[:, 1], 2 * coeffs_f[:, 2], 3 * coeffs_f[:, 3], 4 * coeffs_f[:, 4], 5 * coeffs_f[:, 5], zeros],
        axis=1,
    )
    coeffs_ddf = jnp.stack(
        [2 * coeffs_f[:, 2], 6 * coeffs_f[:, 3], 12 * coeffs_f[:, 4], 20 * coeffs_f[:, 5], zeros, zeros],
        axis=1,
    )
    coeffs_dddf = jnp.stack(
        [6 * coeffs_f[:, 3], 24 * coeffs_f[:, 4], 60 * coeffs_f[:, 5], zeros, zeros, zeros],
        axis=1,
    )
    return coeffs_f, coeffs_df, coeffs_ddf, coeffs_dddf


def generate_lanechange_path_points(xrange: Array,
                                    num_agents: int,
                                    num_points: int,
                                    S_start_state: AgentState,
                                    S_terminal_state: AgentState,
                                    points_interval: float = 0.1) -> Tuple[PathRefs, jnp.ndarray]:
    """生成由水平直线-五次多项式曲线-水平直线组成的6维参考轨迹点。"""
    one6_patheffs_f, one6_patheffs_df, one6_patheffs_ddf, one6_patheffs_dddf = \
        calc_quintic_eff_from_heading(S_start_state[None, :], S_terminal_state[None, :])
    quintic_f = quintic_polynomial_f(one6_patheffs_f)
    quintic_df = quintic_polynomial_f(one6_patheffs_df)
    quintic_ddf = quintic_polynomial_f(one6_patheffs_ddf)
    quintic_dddf = quintic_polynomial_f(one6_patheffs_dddf)

    zeros = jnp.zeros((1, 1), dtype=jnp.float32)
    const_f_ystart = const_f(S_start_state[1][None, None])
    const_f_yterminal = const_f(S_terminal_state[1][None, None])
    const_f_zeros = const_f(zeros)

    poly_sec_f = three_sec_f(const_f_ystart, quintic_f, const_f_yterminal,
                             S_start_state[0][None, None], S_terminal_state[0][None, None])
    poly_sec_df = three_sec_f(const_f_zeros, quintic_df, const_f_zeros,
                              S_start_state[0][None, None], S_terminal_state[0][None, None])
    poly_sec_ddf = three_sec_f(const_f_zeros, quintic_ddf, const_f_zeros,
                               S_start_state[0][None, None], S_terminal_state[0][None, None])
    poly_sec_dddf = three_sec_f(const_f_zeros, quintic_dddf, const_f_zeros,
                                S_start_state[0][None, None], S_terminal_state[0][None, None])

    onen_xs = jnp.linspace(
        start=xrange[0][None],
        stop=xrange[0][None] + (num_points + 1) * points_interval,
        num=num_points,
        dtype=jnp.float32,
    ).T
    onen_ys = poly_sec_f(onen_xs)
    onen_dys = poly_sec_df(onen_xs)
    onen_ddys = poly_sec_ddf(onen_xs)
    onen_dddys = poly_sec_dddf(onen_xs)
    onen_thetas_rad = jnp.arctan(onen_dys)
    onen_heading_x = jnp.cos(onen_thetas_rad)
    onen_heading_y = jnp.sin(onen_thetas_rad)
    onen_vs_mps = jnp.repeat(S_terminal_state[4][None, None], onen_thetas_rad.shape[1], axis=1)
    onen_zeros = jnp.zeros_like(onen_xs)

    onenS_goals = jnp.stack(
        [onen_xs, onen_ys, onen_heading_x, onen_heading_y, onen_vs_mps, onen_zeros],
        axis=2,
    )
    anS_goals = jnp.repeat(onenS_goals, num_agents, axis=0)

    onen_vxs_mps = onen_vs_mps * jnp.cos(onen_thetas_rad)
    onen_dYddt = onen_vxs_mps * onen_dys
    onen_ddYddt = onen_vxs_mps ** 2 * onen_ddys
    onen_dddYddt = onen_vxs_mps ** 3 * onen_dddys
    onen4_dsYddts = jnp.stack([onen_ys, onen_dYddt, onen_ddYddt, onen_dddYddt], axis=2)
    an4_dsYddts = jnp.repeat(onen4_dsYddts, num_agents, axis=0)
    return anS_goals, an4_dsYddts


def generate_horizontal_path_points(xrange: Array,
                                    num_agents: int,
                                    num_points: int,
                                    start_y: jnp.ndarray,
                                    terminal_v: jnp.ndarray,
                                    points_interval: float = 0.1) -> Tuple[PathRefs, jnp.ndarray]:
    """生成6维水平直线参考轨迹点。"""
    assert start_y.shape == terminal_v.shape == ()
    onen_xs = jnp.linspace(
        start=xrange[0][None],
        stop=xrange[0][None] + (num_points + 1) * points_interval,
        num=num_points,
        dtype=jnp.float32,
    ).T
    onen_ys = jnp.repeat(start_y[None, None], num_points, axis=1)
    onen_heading_x = jnp.ones_like(onen_ys)
    onen_heading_y = jnp.zeros_like(onen_ys)
    onen_vs_mps = jnp.repeat(terminal_v[None, None], num_points, axis=1)
    onen_zeros = jnp.zeros_like(onen_xs)

    onenS_goals = jnp.stack(
        [onen_xs, onen_ys, onen_heading_x, onen_heading_y, onen_vs_mps, onen_zeros],
        axis=2,
    )
    anS_goals = jnp.repeat(onenS_goals, num_agents, axis=0)

    onen4_dsYddts = jnp.stack([onen_ys, onen_zeros, onen_zeros, onen_zeros], axis=2)
    an4_dsYddts = jnp.repeat(onen4_dsYddts, num_agents, axis=0)
    return anS_goals, an4_dsYddts


class LaneChangeANDOvertakeScene(ABC):
    """两车道lanechange/overtake split场景基类。"""

    def __init__(self, key: PRNGKey, num_agents: int, num_ref_points: int, xrange: Array, yrange: Array,
                 lane_width: float, lane_centers: Array):
        self.key = key
        self.num_agents = num_agents
        self._num_ref_points = num_ref_points
        self._xrange = xrange
        self._yrange = yrange
        self._lane_width = lane_width
        self._lane_centers = lane_centers

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def num_ref_points(self) -> int:
        return self._num_ref_points

    @property
    def state_dim(self) -> int:
        return 6

    @property
    def num_lanes(self) -> int:
        assert self.lane_centers.shape[0] == 2, "本场景仅支持双车道配置！"
        return 2

    @property
    def lane_centers(self) -> Array:
        return self._lane_centers

    @property
    def xrange(self) -> Array:
        return self._xrange

    @property
    def yrange(self) -> Array:
        return self._yrange

    @property
    def num_moving_obsts(self) -> int:
        return 0

    @property
    def num_static_obsts(self) -> int:
        return 1

    @property
    def num_obsts(self) -> int:
        return self.num_moving_obsts + self.num_static_obsts

    @abstractmethod
    def make(self) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
        pass


class LaneChangeMiddleStaticEdgeFastMoving2lane_Start(LaneChangeANDOvertakeScene):
    """变道场景-从原点开始以速度0接近静态障碍车。"""

    def __init__(self, key: PRNGKey, num_agents: int, num_ref_points: int, xrange: Array, yrange: Array,
                 lane_width: float, lane_centers: Array):
        super(LaneChangeMiddleStaticEdgeFastMoving2lane_Start, self).__init__(
            key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers
        )

    @property
    def name(self) -> str:
        return "lanechange_scene_with_middle_static_obstacle_and_edge_2_lane_start"

    def make(self) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
        start_x_key, terminal_x_key, start_y_key, start_terminal_v_key, agent_x_key, agent_y_key, agent_v_key, \
            sobst_x_key, sobst_y_key, sobst_theta_key, mobst_x_key, mobst_y_key, mobst_vx_key = jr.split(self.key, 13)

        # 生成轨迹
        start_x = jr.uniform(start_x_key, shape=(), dtype=jnp.float32, minval=self.xrange[0],
                             maxval=(self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0])
        terminal_x = jr.uniform(terminal_x_key, shape=(), dtype=jnp.float32,
                                minval=2 * (self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0],
                                maxval=self.xrange[1])
        start_lane = jr.choice(start_y_key, jnp.array([0, 1], dtype=jnp.int32), shape=())
        terminal_lane = 1 - start_lane
        start_y = self.lane_centers[start_lane]
        terminal_y = self.lane_centers[terminal_lane]
        start_theta = terminal_theta = jnp.array(0.0, dtype=jnp.float32)
        start_v = terminal_v = jr.uniform(start_terminal_v_key, shape=(), dtype=jnp.float32, minval=20, maxval=40) / 3.6
        S_start_state = make_state(start_x, start_y, start_theta, start_v)
        S_terminal_state = make_state(terminal_x, terminal_y, terminal_theta, terminal_v)
        anS_goals, an4_dsYddts = generate_lanechange_path_points(
            self.xrange, self.num_agents, self.num_ref_points, S_start_state, S_terminal_state
        )

        # 生成静态障碍物，x坐标位于[xs_min, xs_max]区间随机，y坐标在两条车道中随机选择
        xs_min = self.xrange[0] + 35.0
        xs_max = self.xrange[1] - 15.0
        sobst_x = jr.uniform(sobst_x_key, shape=(), dtype=jnp.float32, minval=xs_min, maxval=xs_max)
        sobst_lane = jr.choice(sobst_y_key, jnp.array([0, 1], dtype=jnp.int32), shape=())
        sobst_y = self.lane_centers[sobst_lane]
        sobst_theta = jr.uniform(sobst_theta_key, shape=(), dtype=jnp.float32, minval=0, maxval=0)
        S_sobst_state = make_state(sobst_x, sobst_y, sobst_theta, jnp.array(0.0, dtype=jnp.float32))
        oS_obst_state = S_sobst_state[None]

        # 生成初始agent，接近起点，y坐标跟参考轨迹起点车道一致
        agent_x = jr.uniform(agent_x_key, shape=(), dtype=jnp.float32, minval=self.xrange[0], maxval=start_x)
        a_agent_x = jnp.repeat(agent_x[None], self.num_agents, axis=0)
        a_agent_y = jnp.repeat(start_y[None], self.num_agents, axis=0) + jr.uniform(
            agent_y_key, shape=(self.num_agents,), dtype=jnp.float32, minval=-0.1, maxval=0.1
        )
        a_agent_hx = jnp.ones((self.num_agents,), dtype=jnp.float32)
        a_agent_hy = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        a_agent_v = jnp.zeros((self.num_agents,), dtype=jnp.float32) # 初始速度固定为0
        a_agent_delta = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        aS_agent_state = jnp.stack([a_agent_x, a_agent_y, a_agent_hx, a_agent_hy, a_agent_v, a_agent_delta], axis=1)

        '''
        # 生成动态障碍物，y坐标位于变道目标车道坐标，可稍有浮动，vx可随机选择一个较大值，x坐标需要计算agent恰好变道到目标车道时的时间
        t = (terminal_x - agent_x) / terminal_vx
        mobst_y = jr.uniform(mobst_y_key, shape=(), dtype=jnp.float32, minval=-0.5, maxval=0.5) + terminal_y
        mobst_vx = jr.uniform(mobst_vx_key, shape=(), dtype=jnp.float32, minval=90, maxval=120) # 90 ~ 120 km/h
        mobst_x = terminal_x - t * mobst_vx - 5 + \
                  jr.uniform(mobst_x_key, shape=(), dtype=jnp.float32, minval=-2, maxval=2)
        S_mobst_state = jnp.concatenate([mobst_x[None], mobst_y[None], mobst_vx[None], Sm3_other0], axis=0)
        oS_obst_state = jnp.stack([S_sobst_state, S_mobst_state], axis=0)
        '''
        return aS_agent_state, oS_obst_state, anS_goals, an4_dsYddts


class LaneChangeMiddleStaticEdgeFastMoving2lane_Approach(LaneChangeANDOvertakeScene):
    """变道场景-远距离接近静态障碍车。"""

    def __init__(self, key: PRNGKey, num_agents: int, num_ref_points: int, xrange: Array, yrange: Array,
                 lane_width: float, lane_centers: Array):
        super(LaneChangeMiddleStaticEdgeFastMoving2lane_Approach, self).__init__(
            key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers
        )

    @property
    def name(self) -> str:
        return "lanechange_scene_with_middle_static_obstacle_and_edge_2_lane_approach"

    def make(self) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
        start_x_key, terminal_x_key, start_y_key, start_terminal_v_key, agent_x_key, agent_y_key, agent_v_key, \
            sobst_x_key, sobst_y_key, sobst_theta_key, mobst_x_key, mobst_y_key, mobst_vx_key = jr.split(self.key, 13)

        # 生成轨迹
        start_x = jr.uniform(start_x_key, shape=(), dtype=jnp.float32, minval=self.xrange[0],
                             maxval=(self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0])
        terminal_x = jr.uniform(terminal_x_key, shape=(), dtype=jnp.float32,
                                minval=2 * (self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0],
                                maxval=self.xrange[1])
        start_lane = jr.choice(start_y_key, jnp.array([0, 1], dtype=jnp.int32), shape=())
        terminal_lane = 1 - start_lane
        start_y = self.lane_centers[start_lane]
        terminal_y = self.lane_centers[terminal_lane]
        start_theta = terminal_theta = jnp.array(0.0, dtype=jnp.float32)
        start_v = terminal_v = jr.uniform(start_terminal_v_key, shape=(), dtype=jnp.float32, minval=20, maxval=40) / 3.6
        S_start_state = make_state(start_x, start_y, start_theta, start_v)
        S_terminal_state = make_state(terminal_x, terminal_y, terminal_theta, terminal_v)
        anS_goals, an4_dsYddts = generate_lanechange_path_points(
            self.xrange, self.num_agents, self.num_ref_points, S_start_state, S_terminal_state
        )

        # 生成静态障碍物，x坐标位于[xs_min, xs_max]区间随机，y坐标在两条车道中随机选择
        xs_min = self.xrange[0] + 35.0
        xs_max = self.xrange[1] - 15.0
        sobst_x = jr.uniform(sobst_x_key, shape=(), dtype=jnp.float32, minval=xs_min, maxval=xs_max)
        sobst_lane = jr.choice(sobst_y_key, jnp.array([0, 1], dtype=jnp.int32), shape=())
        sobst_y = self.lane_centers[sobst_lane]
        sobst_theta = jr.uniform(sobst_theta_key, shape=(), dtype=jnp.float32, minval=0, maxval=0)
        S_sobst_state = make_state(sobst_x, sobst_y, sobst_theta, jnp.array(0.0, dtype=jnp.float32))
        oS_obst_state = S_sobst_state[None]

        # 生成初始agent，远距离接近时位于障碍车后方，y坐标跟参考轨迹起点车道一致
        agent_x = sobst_x - jr.uniform(agent_x_key, shape=(), dtype=jnp.float32, minval=18.0, maxval=32.0)
        a_agent_x = jnp.repeat(agent_x[None], self.num_agents, axis=0)
        a_agent_y = jnp.repeat(start_y[None], self.num_agents, axis=0) + jr.uniform(
            agent_y_key, shape=(self.num_agents,), dtype=jnp.float32, minval=-0.1, maxval=0.1
        )
        a_agent_hx = jnp.ones((self.num_agents,), dtype=jnp.float32)
        a_agent_hy = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        a_agent_v = jr.uniform(agent_v_key, shape=(self.num_agents,), dtype=jnp.float32,
                               minval=0.0, maxval=40.0) / 3.6
        a_agent_delta = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        aS_agent_state = jnp.stack([a_agent_x, a_agent_y, a_agent_hx, a_agent_hy, a_agent_v, a_agent_delta], axis=1)

        '''
        # 生成动态障碍物，y坐标位于变道目标车道坐标，可稍有浮动，vx可随机选择一个较大值，x坐标需要计算agent恰好变道到目标车道时的时间
        t = (terminal_x - agent_x) / terminal_vx
        mobst_y = jr.uniform(mobst_y_key, shape=(), dtype=jnp.float32, minval=-0.5, maxval=0.5) + terminal_y
        mobst_vx = jr.uniform(mobst_vx_key, shape=(), dtype=jnp.float32, minval=90, maxval=120) # 90 ~ 120 km/h
        mobst_x = terminal_x - t * mobst_vx - 5 + \
                  jr.uniform(mobst_x_key, shape=(), dtype=jnp.float32, minval=-2, maxval=2)
        S_mobst_state = jnp.concatenate([mobst_x[None], mobst_y[None], mobst_vx[None], Sm3_other0], axis=0)
        oS_obst_state = jnp.stack([S_sobst_state, S_mobst_state], axis=0)
        '''
        return aS_agent_state, oS_obst_state, anS_goals, an4_dsYddts


class LaneChangeMiddleStaticEdgeFastMoving2lane_Side(LaneChangeANDOvertakeScene):
    """变道场景-并排/绕行中。"""

    def __init__(self, key: PRNGKey, num_agents: int, num_ref_points: int, xrange: Array, yrange: Array,
                 lane_width: float, lane_centers: Array):
        super(LaneChangeMiddleStaticEdgeFastMoving2lane_Side, self).__init__(
            key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers
        )

    @property
    def name(self) -> str:
        return "lanechange_scene_with_middle_static_obstacle_and_edge_2_lane_side"

    def make(self) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
        start_x_key, terminal_x_key, start_y_key, start_terminal_v_key, agent_x_key, agent_y_key, agent_v_key, \
            sobst_x_key, sobst_y_key, sobst_theta_key, mobst_x_key, mobst_y_key, mobst_vx_key = jr.split(self.key, 13)

        # 生成轨迹
        start_x = jr.uniform(start_x_key, shape=(), dtype=jnp.float32, minval=self.xrange[0],
                             maxval=(self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0])
        terminal_x = jr.uniform(terminal_x_key, shape=(), dtype=jnp.float32,
                                minval=2 * (self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0],
                                maxval=self.xrange[1])
        start_lane = jr.choice(start_y_key, jnp.array([0, 1], dtype=jnp.int32), shape=())
        terminal_lane = 1 - start_lane
        start_y = self.lane_centers[start_lane]
        terminal_y = self.lane_centers[terminal_lane]
        start_theta = terminal_theta = jnp.array(0.0, dtype=jnp.float32)
        start_v = terminal_v = jr.uniform(start_terminal_v_key, shape=(), dtype=jnp.float32, minval=20, maxval=40) / 3.6
        S_start_state = make_state(start_x, start_y, start_theta, start_v)
        S_terminal_state = make_state(terminal_x, terminal_y, terminal_theta, terminal_v)
        anS_goals, an4_dsYddts = generate_lanechange_path_points(
            self.xrange, self.num_agents, self.num_ref_points, S_start_state, S_terminal_state
        )

        # 生成静态障碍物，x坐标位于[xs_min, xs_max]区间随机，y坐标在两条车道中随机选择
        xs_min = self.xrange[0] + 35.0
        xs_max = self.xrange[1] - 15.0
        sobst_x = jr.uniform(sobst_x_key, shape=(), dtype=jnp.float32, minval=xs_min, maxval=xs_max)
        sobst_lane = jr.choice(sobst_y_key, jnp.array([0, 1], dtype=jnp.int32), shape=())
        sobst_y = self.lane_centers[sobst_lane]
        side_y = self.lane_centers[1 - sobst_lane]
        sobst_theta = jr.uniform(sobst_theta_key, shape=(), dtype=jnp.float32, minval=0, maxval=0)
        S_sobst_state = make_state(sobst_x, sobst_y, sobst_theta, jnp.array(0.0, dtype=jnp.float32))
        oS_obst_state = S_sobst_state[None]

        # 生成初始agent，并排/绕行中时x坐标接近障碍车，y坐标位于障碍车另一条车道
        agent_x = sobst_x + jr.uniform(agent_x_key, shape=(), dtype=jnp.float32, minval=-4.0, maxval=4.0)
        a_agent_x = jnp.repeat(agent_x[None], self.num_agents, axis=0)
        a_agent_y = jnp.repeat(side_y[None], self.num_agents, axis=0) + jr.uniform(
            agent_y_key, shape=(self.num_agents,), dtype=jnp.float32, minval=-0.1, maxval=0.1
        )
        a_agent_hx = jnp.ones((self.num_agents,), dtype=jnp.float32)
        a_agent_hy = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        a_agent_v = jr.uniform(agent_v_key, shape=(self.num_agents,), dtype=jnp.float32,
                               minval=0.0, maxval=40.0) / 3.6
        a_agent_delta = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        aS_agent_state = jnp.stack([a_agent_x, a_agent_y, a_agent_hx, a_agent_hy, a_agent_v, a_agent_delta], axis=1)

        '''
        # 生成动态障碍物，y坐标位于变道目标车道坐标，可稍有浮动，vx可随机选择一个较大值，x坐标需要计算agent恰好变道到目标车道时的时间
        t = (terminal_x - agent_x) / terminal_vx
        mobst_y = jr.uniform(mobst_y_key, shape=(), dtype=jnp.float32, minval=-0.5, maxval=0.5) + terminal_y
        mobst_vx = jr.uniform(mobst_vx_key, shape=(), dtype=jnp.float32, minval=90, maxval=120) # 90 ~ 120 km/h
        mobst_x = terminal_x - t * mobst_vx - 5 + \
                  jr.uniform(mobst_x_key, shape=(), dtype=jnp.float32, minval=-2, maxval=2)
        S_mobst_state = jnp.concatenate([mobst_x[None], mobst_y[None], mobst_vx[None], Sm3_other0], axis=0)
        oS_obst_state = jnp.stack([S_sobst_state, S_mobst_state], axis=0)
        '''
        return aS_agent_state, oS_obst_state, anS_goals, an4_dsYddts


class LaneChangeMiddleStaticEdgeFastMoving2lane_Passed(LaneChangeANDOvertakeScene):
    """变道场景-已经过障碍但还未变道到目标车道。"""

    def __init__(self, key: PRNGKey, num_agents: int, num_ref_points: int, xrange: Array, yrange: Array,
                 lane_width: float, lane_centers: Array):
        super(LaneChangeMiddleStaticEdgeFastMoving2lane_Passed, self).__init__(
            key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers
        )

    @property
    def name(self) -> str:
        return "lanechange_scene_with_middle_static_obstacle_and_edge_2_lane_passed"

    def make(self) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
        start_x_key, terminal_x_key, start_y_key, start_terminal_v_key, agent_x_key, agent_y_key, agent_v_key, \
            sobst_x_key, sobst_y_key, sobst_theta_key, mobst_x_key, mobst_y_key, mobst_vx_key = jr.split(self.key, 13)

        # 生成轨迹
        start_x = jr.uniform(start_x_key, shape=(), dtype=jnp.float32, minval=self.xrange[0],
                             maxval=(self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0])
        terminal_x = jr.uniform(terminal_x_key, shape=(), dtype=jnp.float32,
                                minval=2 * (self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0],
                                maxval=self.xrange[1])
        start_lane = jr.choice(start_y_key, jnp.array([0, 1], dtype=jnp.int32), shape=())
        terminal_lane = 1 - start_lane
        start_y = self.lane_centers[start_lane]
        terminal_y = self.lane_centers[terminal_lane]
        start_theta = terminal_theta = jnp.array(0.0, dtype=jnp.float32)
        start_v = terminal_v = jr.uniform(start_terminal_v_key, shape=(), dtype=jnp.float32, minval=20, maxval=40) / 3.6
        S_start_state = make_state(start_x, start_y, start_theta, start_v)
        S_terminal_state = make_state(terminal_x, terminal_y, terminal_theta, terminal_v)
        anS_goals, an4_dsYddts = generate_lanechange_path_points(
            self.xrange, self.num_agents, self.num_ref_points, S_start_state, S_terminal_state
        )

        # 生成静态障碍物，x坐标位于[xs_min, xs_max]区间随机，y坐标在两条车道中随机选择
        xs_min = self.xrange[0] + 35.0
        xs_max = self.xrange[1] - 15.0
        sobst_x = jr.uniform(sobst_x_key, shape=(), dtype=jnp.float32, minval=xs_min, maxval=xs_max)
        sobst_lane = jr.choice(sobst_y_key, jnp.array([0, 1], dtype=jnp.int32), shape=())
        sobst_y = self.lane_centers[sobst_lane]
        sobst_theta = jr.uniform(sobst_theta_key, shape=(), dtype=jnp.float32, minval=0, maxval=0)
        S_sobst_state = make_state(sobst_x, sobst_y, sobst_theta, jnp.array(0.0, dtype=jnp.float32))
        oS_obst_state = S_sobst_state[None]

        # 生成初始agent，通过障碍车后，y坐标在两条车道中心线之间连续取值
        agent_y_center_key, agent_y_noise_key = jr.split(agent_y_key, 2)
        agent_x = sobst_x + jr.uniform(agent_x_key, shape=(), dtype=jnp.float32, minval=8.0, maxval=18.0)
        y_low = jnp.minimum(self.lane_centers[0], self.lane_centers[1])
        y_high = jnp.maximum(self.lane_centers[0], self.lane_centers[1])
        agent_y = jr.uniform(agent_y_center_key, shape=(), dtype=jnp.float32, minval=y_low, maxval=y_high)
        a_agent_x = jnp.repeat(agent_x[None], self.num_agents, axis=0)
        a_agent_y = jnp.repeat(agent_y[None], self.num_agents, axis=0) + jr.uniform(
            agent_y_noise_key, shape=(self.num_agents,), dtype=jnp.float32, minval=-0.1, maxval=0.1
        )
        a_agent_hx = jnp.ones((self.num_agents,), dtype=jnp.float32)
        a_agent_hy = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        a_agent_v = jr.uniform(agent_v_key, shape=(self.num_agents,), dtype=jnp.float32,
                               minval=0.0, maxval=40.0) / 3.6
        a_agent_delta = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        aS_agent_state = jnp.stack([a_agent_x, a_agent_y, a_agent_hx, a_agent_hy, a_agent_v, a_agent_delta], axis=1)

        '''
        # 生成动态障碍物，y坐标位于变道目标车道坐标，可稍有浮动，vx可随机选择一个较大值，x坐标需要计算agent恰好变道到目标车道时的时间
        t = (terminal_x - agent_x) / terminal_vx
        mobst_y = jr.uniform(mobst_y_key, shape=(), dtype=jnp.float32, minval=-0.5, maxval=0.5) + terminal_y
        mobst_vx = jr.uniform(mobst_vx_key, shape=(), dtype=jnp.float32, minval=90, maxval=120) # 90 ~ 120 km/h
        mobst_x = terminal_x - t * mobst_vx - 5 + \
                  jr.uniform(mobst_x_key, shape=(), dtype=jnp.float32, minval=-2, maxval=2)
        S_mobst_state = jnp.concatenate([mobst_x[None], mobst_y[None], mobst_vx[None], Sm3_other0], axis=0)
        oS_obst_state = jnp.stack([S_sobst_state, S_mobst_state], axis=0)
        '''
        return aS_agent_state, oS_obst_state, anS_goals, an4_dsYddts


class LaneChangeMiddleStaticEdgeFastMoving2lane_Done(LaneChangeANDOvertakeScene):
    """变道场景-已经越过障碍且已经变道到目标车道。"""

    def __init__(self, key: PRNGKey, num_agents: int, num_ref_points: int, xrange: Array, yrange: Array,
                 lane_width: float, lane_centers: Array):
        super(LaneChangeMiddleStaticEdgeFastMoving2lane_Done, self).__init__(
            key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers
        )

    @property
    def name(self) -> str:
        return "lanechange_scene_with_middle_static_obstacle_and_edge_2_lane_done"

    def make(self) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
        start_x_key, terminal_x_key, start_y_key, start_terminal_v_key, agent_x_key, agent_y_key, agent_v_key, \
            sobst_x_key, sobst_y_key, sobst_theta_key, mobst_x_key, mobst_y_key, mobst_vx_key = jr.split(self.key, 13)

        # 生成轨迹
        start_x = jr.uniform(start_x_key, shape=(), dtype=jnp.float32, minval=self.xrange[0],
                             maxval=(self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0])
        terminal_x = jr.uniform(terminal_x_key, shape=(), dtype=jnp.float32,
                                minval=2 * (self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0],
                                maxval=self.xrange[1])
        start_lane = jr.choice(start_y_key, jnp.array([0, 1], dtype=jnp.int32), shape=())
        terminal_lane = 1 - start_lane
        start_y = self.lane_centers[start_lane]
        terminal_y = self.lane_centers[terminal_lane]
        start_theta = terminal_theta = jnp.array(0.0, dtype=jnp.float32)
        start_v = terminal_v = jr.uniform(start_terminal_v_key, shape=(), dtype=jnp.float32, minval=20, maxval=40) / 3.6
        S_start_state = make_state(start_x, start_y, start_theta, start_v)
        S_terminal_state = make_state(terminal_x, terminal_y, terminal_theta, terminal_v)
        anS_goals, an4_dsYddts = generate_lanechange_path_points(
            self.xrange, self.num_agents, self.num_ref_points, S_start_state, S_terminal_state
        )

        # 生成静态障碍物，x坐标位于[xs_min, xs_max]区间随机，y坐标在两条车道中随机选择
        xs_min = self.xrange[0] + 35.0
        xs_max = self.xrange[1] - 15.0
        sobst_x = jr.uniform(sobst_x_key, shape=(), dtype=jnp.float32, minval=xs_min, maxval=xs_max)
        sobst_lane = jr.choice(sobst_y_key, jnp.array([0, 1], dtype=jnp.int32), shape=())
        sobst_y = self.lane_centers[sobst_lane]
        sobst_theta = jr.uniform(sobst_theta_key, shape=(), dtype=jnp.float32, minval=0, maxval=0)
        S_sobst_state = make_state(sobst_x, sobst_y, sobst_theta, jnp.array(0.0, dtype=jnp.float32))
        oS_obst_state = S_sobst_state[None]

        # 生成初始agent，越过障碍且完成变道时位于目标车道
        agent_x = sobst_x + jr.uniform(agent_x_key, shape=(), dtype=jnp.float32, minval=18.0, maxval=32.0)
        a_agent_x = jnp.repeat(agent_x[None], self.num_agents, axis=0)
        a_agent_y = jnp.repeat(terminal_y[None], self.num_agents, axis=0) + jr.uniform(
            agent_y_key, shape=(self.num_agents,), dtype=jnp.float32, minval=-0.1, maxval=0.1
        )
        a_agent_hx = jnp.ones((self.num_agents,), dtype=jnp.float32)
        a_agent_hy = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        a_agent_v = jr.uniform(agent_v_key, shape=(self.num_agents,), dtype=jnp.float32,
                               minval=0.0, maxval=40.0) / 3.6
        a_agent_delta = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        aS_agent_state = jnp.stack([a_agent_x, a_agent_y, a_agent_hx, a_agent_hy, a_agent_v, a_agent_delta], axis=1)

        '''
        # 生成动态障碍物，y坐标位于变道目标车道坐标，可稍有浮动，vx可随机选择一个较大值，x坐标需要计算agent恰好变道到目标车道时的时间
        t = (terminal_x - agent_x) / terminal_vx
        mobst_y = jr.uniform(mobst_y_key, shape=(), dtype=jnp.float32, minval=-0.5, maxval=0.5) + terminal_y
        mobst_vx = jr.uniform(mobst_vx_key, shape=(), dtype=jnp.float32, minval=90, maxval=120) # 90 ~ 120 km/h
        mobst_x = terminal_x - t * mobst_vx - 5 + \
                  jr.uniform(mobst_x_key, shape=(), dtype=jnp.float32, minval=-2, maxval=2)
        S_mobst_state = jnp.concatenate([mobst_x[None], mobst_y[None], mobst_vx[None], Sm3_other0], axis=0)
        oS_obst_state = jnp.stack([S_sobst_state, S_mobst_state], axis=0)
        '''
        return aS_agent_state, oS_obst_state, anS_goals, an4_dsYddts


class OvertakeEdgeStaticMiddle2lane_Start(LaneChangeANDOvertakeScene):
    """超车场景-从原点附近以速度0开始接近静态障碍车。"""

    def __init__(self, key: PRNGKey, num_agents: int, num_ref_points: int, xrange: Array, yrange: Array,
                 lane_width: float, lane_centers: Array):
        super(OvertakeEdgeStaticMiddle2lane_Start, self).__init__(
            key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers
        )

    @property
    def name(self) -> str:
        return "overtake_scene_with_edge_static_obstacle_and_middle_2_lane_start"

    def make(self) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
        start_x_key, terminal_x_key, start_y_key, start_terminal_v_key, agent_x_key, agent_y_key, agent_v_key, \
            sobst_x_key, sobst_y_key, sobst_theta_key, mobst_x_key, mobst_y_key, mobst_vx_key = jr.split(self.key, 13)

        # 生成轨迹
        start_x = jr.uniform(start_x_key, shape=(), dtype=jnp.float32, minval=self.xrange[0],
                             maxval=(self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0])
        terminal_x = jr.uniform(terminal_x_key, shape=(), dtype=jnp.float32,
                                minval=2 * (self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0],
                                maxval=self.xrange[1])
        del terminal_x
        start_y = terminal_y = jr.choice(start_y_key, self.lane_centers[jnp.array([0, 1])], shape=())
        terminal_v = jr.uniform(start_terminal_v_key, shape=(), dtype=jnp.float32, minval=20, maxval=40) / 3.6
        anS_goals, an4_dsYddts = generate_horizontal_path_points(
            self.xrange, self.num_agents, self.num_ref_points, start_y, terminal_v
        )

        # 生成静态障碍物，x坐标位于[xs_min, xs_max]区间随机，y坐标在两条车道中随机选择
        xs_min = self.xrange[0] + 35.0
        xs_max = self.xrange[1] - 15.0
        sobst_x = jr.uniform(sobst_x_key, shape=(), dtype=jnp.float32, minval=xs_min, maxval=xs_max)
        sobst_lane = jr.choice(sobst_y_key, jnp.array([0, 1], dtype=jnp.int32), shape=())
        sobst_y = self.lane_centers[sobst_lane]
        sobst_theta = jr.uniform(sobst_theta_key, shape=(), dtype=jnp.float32, minval=0, maxval=0)
        S_sobst_state = make_state(sobst_x, sobst_y, sobst_theta, jnp.array(0.0, dtype=jnp.float32))
        oS_obst_state = S_sobst_state[None]

        # 生成初始agent，远距离接近时位于障碍车后方，y坐标跟参考轨迹起点车道一致
        agent_x = jr.uniform(agent_x_key, shape=(), dtype=jnp.float32, minval=self.xrange[0], maxval=start_x)
        a_agent_x = jnp.repeat(agent_x[None], self.num_agents, axis=0)
        a_agent_y = jnp.repeat(start_y[None], self.num_agents, axis=0) + jr.uniform(
            agent_y_key, shape=(self.num_agents,), dtype=jnp.float32, minval=-0.1, maxval=0.1
        )
        a_agent_hx = jnp.ones((self.num_agents,), dtype=jnp.float32)
        a_agent_hy = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        a_agent_v = jnp.zeros((self.num_agents,), dtype=jnp.float32) # 初始速度固定为0
        a_agent_delta = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        aS_agent_state = jnp.stack([a_agent_x, a_agent_y, a_agent_hx, a_agent_hy, a_agent_v, a_agent_delta], axis=1)

        '''
        # 生成动态障碍物，y坐标位于中间车道，可稍有浮动，vx可随机选择一个较大值，x坐标需要计算agent恰好到静态障碍物时时动态障碍物位于
        # agent附近
        t = (sobst_x - agent_x) / terminal_vx
        mobst_y = jr.uniform(mobst_y_key, shape=(), dtype=jnp.float32, minval=-0.5, maxval=0.5) + self.lane_centers[1]
        mobst_vx = jr.uniform(mobst_vx_key, shape=(), dtype=jnp.float32, minval=90, maxval=120)  # 90 ~ 120 km/h
        mobst_x = sobst_x - t * mobst_vx  + \
                  jr.uniform(mobst_x_key, shape=(), dtype=jnp.float32, minval=5, maxval=15)
        S_mobst_state = jnp.concatenate([mobst_x[None], mobst_y[None], mobst_vx[None], Sm3_other0], axis=0)
        oS_obst_state = jnp.stack([S_sobst_state, S_mobst_state], axis=0)
        '''
        return aS_agent_state, oS_obst_state, anS_goals, an4_dsYddts


class OvertakeEdgeStaticMiddle2lane_Approach(LaneChangeANDOvertakeScene):
    """超车场景-远距离接近静态障碍车。"""

    def __init__(self, key: PRNGKey, num_agents: int, num_ref_points: int, xrange: Array, yrange: Array,
                 lane_width: float, lane_centers: Array):
        super(OvertakeEdgeStaticMiddle2lane_Approach, self).__init__(
            key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers
        )

    @property
    def name(self) -> str:
        return "overtake_scene_with_edge_static_obstacle_and_middle_2_lane_approach"

    def make(self) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
        start_x_key, terminal_x_key, start_y_key, start_terminal_v_key, agent_x_key, agent_y_key, agent_v_key, \
            sobst_x_key, sobst_y_key, sobst_theta_key, mobst_x_key, mobst_y_key, mobst_vx_key = jr.split(self.key, 13)

        # 生成轨迹
        start_x = jr.uniform(start_x_key, shape=(), dtype=jnp.float32, minval=self.xrange[0],
                             maxval=(self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0])
        terminal_x = jr.uniform(terminal_x_key, shape=(), dtype=jnp.float32,
                                minval=2 * (self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0],
                                maxval=self.xrange[1])
        del start_x, terminal_x
        start_y = terminal_y = jr.choice(start_y_key, self.lane_centers[jnp.array([0, 1])], shape=())
        terminal_v = jr.uniform(start_terminal_v_key, shape=(), dtype=jnp.float32, minval=20, maxval=40) / 3.6
        anS_goals, an4_dsYddts = generate_horizontal_path_points(
            self.xrange, self.num_agents, self.num_ref_points, start_y, terminal_v
        )

        # 生成静态障碍物，x坐标位于[xs_min, xs_max]区间随机，y坐标在两条车道中随机选择
        xs_min = self.xrange[0] + 35.0
        xs_max = self.xrange[1] - 15.0
        sobst_x = jr.uniform(sobst_x_key, shape=(), dtype=jnp.float32, minval=xs_min, maxval=xs_max)
        sobst_lane = jr.choice(sobst_y_key, jnp.array([0, 1], dtype=jnp.int32), shape=())
        sobst_y = self.lane_centers[sobst_lane]
        sobst_theta = jr.uniform(sobst_theta_key, shape=(), dtype=jnp.float32, minval=0, maxval=0)
        S_sobst_state = make_state(sobst_x, sobst_y, sobst_theta, jnp.array(0.0, dtype=jnp.float32))
        oS_obst_state = S_sobst_state[None]

        # 生成初始agent，远距离接近时位于障碍车后方，y坐标跟参考轨迹起点车道一致
        agent_x = sobst_x - jr.uniform(agent_x_key, shape=(), dtype=jnp.float32, minval=18.0, maxval=32.0)
        a_agent_x = jnp.repeat(agent_x[None], self.num_agents, axis=0)
        a_agent_y = jnp.repeat(start_y[None], self.num_agents, axis=0) + jr.uniform(
            agent_y_key, shape=(self.num_agents,), dtype=jnp.float32, minval=-0.1, maxval=0.1
        )
        a_agent_hx = jnp.ones((self.num_agents,), dtype=jnp.float32)
        a_agent_hy = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        a_agent_v = jr.uniform(agent_v_key, shape=(self.num_agents,), dtype=jnp.float32,
                               minval=0.0, maxval=40.0) / 3.6
        a_agent_delta = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        aS_agent_state = jnp.stack([a_agent_x, a_agent_y, a_agent_hx, a_agent_hy, a_agent_v, a_agent_delta], axis=1)

        '''
        # 生成动态障碍物，y坐标位于中间车道，可稍有浮动，vx可随机选择一个较大值，x坐标需要计算agent恰好到静态障碍物时时动态障碍物位于
        # agent附近
        t = (sobst_x - agent_x) / terminal_vx
        mobst_y = jr.uniform(mobst_y_key, shape=(), dtype=jnp.float32, minval=-0.5, maxval=0.5) + self.lane_centers[1]
        mobst_vx = jr.uniform(mobst_vx_key, shape=(), dtype=jnp.float32, minval=90, maxval=120)  # 90 ~ 120 km/h
        mobst_x = sobst_x - t * mobst_vx  + \
                  jr.uniform(mobst_x_key, shape=(), dtype=jnp.float32, minval=5, maxval=15)
        S_mobst_state = jnp.concatenate([mobst_x[None], mobst_y[None], mobst_vx[None], Sm3_other0], axis=0)
        oS_obst_state = jnp.stack([S_sobst_state, S_mobst_state], axis=0)
        '''
        return aS_agent_state, oS_obst_state, anS_goals, an4_dsYddts


class OvertakeEdgeStaticMiddle2lane_Side(LaneChangeANDOvertakeScene):
    """超车场景-并排/绕行中。"""

    def __init__(self, key: PRNGKey, num_agents: int, num_ref_points: int, xrange: Array, yrange: Array,
                 lane_width: float, lane_centers: Array):
        super(OvertakeEdgeStaticMiddle2lane_Side, self).__init__(
            key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers
        )

    @property
    def name(self) -> str:
        return "overtake_scene_with_edge_static_obstacle_and_middle_2_lane_side"

    def make(self) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
        start_x_key, terminal_x_key, start_y_key, start_terminal_v_key, agent_x_key, agent_y_key, agent_v_key, \
            sobst_x_key, sobst_y_key, sobst_theta_key, mobst_x_key, mobst_y_key, mobst_vx_key = jr.split(self.key, 13)

        # 生成轨迹
        start_x = jr.uniform(start_x_key, shape=(), dtype=jnp.float32, minval=self.xrange[0],
                             maxval=(self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0])
        terminal_x = jr.uniform(terminal_x_key, shape=(), dtype=jnp.float32,
                                minval=2 * (self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0],
                                maxval=self.xrange[1])
        del start_x, terminal_x
        start_y = terminal_y = jr.choice(start_y_key, self.lane_centers[jnp.array([0, 1])], shape=())
        terminal_v = jr.uniform(start_terminal_v_key, shape=(), dtype=jnp.float32, minval=20, maxval=40) / 3.6
        anS_goals, an4_dsYddts = generate_horizontal_path_points(
            self.xrange, self.num_agents, self.num_ref_points, start_y, terminal_v
        )

        # 生成静态障碍物，x坐标位于[xs_min, xs_max]区间随机，y坐标在两条车道中随机选择
        xs_min = self.xrange[0] + 35.0
        xs_max = self.xrange[1] - 15.0
        sobst_x = jr.uniform(sobst_x_key, shape=(), dtype=jnp.float32, minval=xs_min, maxval=xs_max)
        sobst_lane = jr.choice(sobst_y_key, jnp.array([0, 1], dtype=jnp.int32), shape=())
        sobst_y = self.lane_centers[sobst_lane]
        side_y = self.lane_centers[1 - sobst_lane]
        sobst_theta = jr.uniform(sobst_theta_key, shape=(), dtype=jnp.float32, minval=0, maxval=0)
        S_sobst_state = make_state(sobst_x, sobst_y, sobst_theta, jnp.array(0.0, dtype=jnp.float32))
        oS_obst_state = S_sobst_state[None]

        # 生成初始agent，并排/绕行中时x坐标接近障碍车，y坐标位于障碍车另一条车道
        agent_x = sobst_x + jr.uniform(agent_x_key, shape=(), dtype=jnp.float32, minval=-4.0, maxval=4.0)
        a_agent_x = jnp.repeat(agent_x[None], self.num_agents, axis=0)
        a_agent_y = jnp.repeat(side_y[None], self.num_agents, axis=0) + jr.uniform(
            agent_y_key, shape=(self.num_agents,), dtype=jnp.float32, minval=-0.1, maxval=0.1
        )
        a_agent_hx = jnp.ones((self.num_agents,), dtype=jnp.float32)
        a_agent_hy = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        a_agent_v = jr.uniform(agent_v_key, shape=(self.num_agents,), dtype=jnp.float32,
                               minval=0.0, maxval=40.0) / 3.6
        a_agent_delta = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        aS_agent_state = jnp.stack([a_agent_x, a_agent_y, a_agent_hx, a_agent_hy, a_agent_v, a_agent_delta], axis=1)

        '''
        # 生成动态障碍物，y坐标位于中间车道，可稍有浮动，vx可随机选择一个较大值，x坐标需要计算agent恰好到静态障碍物时时动态障碍物位于
        # agent附近
        t = (sobst_x - agent_x) / terminal_vx
        mobst_y = jr.uniform(mobst_y_key, shape=(), dtype=jnp.float32, minval=-0.5, maxval=0.5) + self.lane_centers[1]
        mobst_vx = jr.uniform(mobst_vx_key, shape=(), dtype=jnp.float32, minval=90, maxval=120)  # 90 ~ 120 km/h
        mobst_x = sobst_x - t * mobst_vx  + \
                  jr.uniform(mobst_x_key, shape=(), dtype=jnp.float32, minval=5, maxval=15)
        S_mobst_state = jnp.concatenate([mobst_x[None], mobst_y[None], mobst_vx[None], Sm3_other0], axis=0)
        oS_obst_state = jnp.stack([S_sobst_state, S_mobst_state], axis=0)
        '''
        return aS_agent_state, oS_obst_state, anS_goals, an4_dsYddts


class OvertakeEdgeStaticMiddle2lane_Passed(LaneChangeANDOvertakeScene):
    """超车场景-已经过障碍但还未变道到目标车道。"""

    def __init__(self, key: PRNGKey, num_agents: int, num_ref_points: int, xrange: Array, yrange: Array,
                 lane_width: float, lane_centers: Array):
        super(OvertakeEdgeStaticMiddle2lane_Passed, self).__init__(
            key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers
        )

    @property
    def name(self) -> str:
        return "overtake_scene_with_edge_static_obstacle_and_middle_2_lane_passed"

    def make(self) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
        start_x_key, terminal_x_key, start_y_key, start_terminal_v_key, agent_x_key, agent_y_key, agent_v_key, \
            sobst_x_key, sobst_y_key, sobst_theta_key, mobst_x_key, mobst_y_key, mobst_vx_key = jr.split(self.key, 13)

        # 生成轨迹
        start_x = jr.uniform(start_x_key, shape=(), dtype=jnp.float32, minval=self.xrange[0],
                             maxval=(self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0])
        terminal_x = jr.uniform(terminal_x_key, shape=(), dtype=jnp.float32,
                                minval=2 * (self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0],
                                maxval=self.xrange[1])
        del start_x, terminal_x
        start_y = terminal_y = jr.choice(start_y_key, self.lane_centers[jnp.array([0, 1])], shape=())
        terminal_v = jr.uniform(start_terminal_v_key, shape=(), dtype=jnp.float32, minval=20, maxval=40) / 3.6
        anS_goals, an4_dsYddts = generate_horizontal_path_points(
            self.xrange, self.num_agents, self.num_ref_points, start_y, terminal_v
        )

        # 生成静态障碍物，x坐标位于[xs_min, xs_max]区间随机，y坐标在两条车道中随机选择
        xs_min = self.xrange[0] + 35.0
        xs_max = self.xrange[1] - 15.0
        sobst_x = jr.uniform(sobst_x_key, shape=(), dtype=jnp.float32, minval=xs_min, maxval=xs_max)
        sobst_lane = jr.choice(sobst_y_key, jnp.array([0, 1], dtype=jnp.int32), shape=())
        sobst_y = self.lane_centers[sobst_lane]
        sobst_theta = jr.uniform(sobst_theta_key, shape=(), dtype=jnp.float32, minval=0, maxval=0)
        S_sobst_state = make_state(sobst_x, sobst_y, sobst_theta, jnp.array(0.0, dtype=jnp.float32))
        oS_obst_state = S_sobst_state[None]

        # 生成初始agent，通过障碍车后，y坐标在两条车道中心线之间连续取值
        agent_y_center_key, agent_y_noise_key = jr.split(agent_y_key, 2)
        agent_x = sobst_x + jr.uniform(agent_x_key, shape=(), dtype=jnp.float32, minval=8.0, maxval=18.0)
        y_low = jnp.minimum(self.lane_centers[0], self.lane_centers[1])
        y_high = jnp.maximum(self.lane_centers[0], self.lane_centers[1])
        agent_y = jr.uniform(agent_y_center_key, shape=(), dtype=jnp.float32, minval=y_low, maxval=y_high)
        a_agent_x = jnp.repeat(agent_x[None], self.num_agents, axis=0)
        a_agent_y = jnp.repeat(agent_y[None], self.num_agents, axis=0) + jr.uniform(
            agent_y_noise_key, shape=(self.num_agents,), dtype=jnp.float32, minval=-0.1, maxval=0.1
        )
        a_agent_hx = jnp.ones((self.num_agents,), dtype=jnp.float32)
        a_agent_hy = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        a_agent_v = jr.uniform(agent_v_key, shape=(self.num_agents,), dtype=jnp.float32,
                               minval=0.0, maxval=40.0) / 3.6
        a_agent_delta = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        aS_agent_state = jnp.stack([a_agent_x, a_agent_y, a_agent_hx, a_agent_hy, a_agent_v, a_agent_delta], axis=1)

        '''
        # 生成动态障碍物，y坐标位于中间车道，可稍有浮动，vx可随机选择一个较大值，x坐标需要计算agent恰好到静态障碍物时时动态障碍物位于
        # agent附近
        t = (sobst_x - agent_x) / terminal_vx
        mobst_y = jr.uniform(mobst_y_key, shape=(), dtype=jnp.float32, minval=-0.5, maxval=0.5) + self.lane_centers[1]
        mobst_vx = jr.uniform(mobst_vx_key, shape=(), dtype=jnp.float32, minval=90, maxval=120)  # 90 ~ 120 km/h
        mobst_x = sobst_x - t * mobst_vx  + \
                  jr.uniform(mobst_x_key, shape=(), dtype=jnp.float32, minval=5, maxval=15)
        S_mobst_state = jnp.concatenate([mobst_x[None], mobst_y[None], mobst_vx[None], Sm3_other0], axis=0)
        oS_obst_state = jnp.stack([S_sobst_state, S_mobst_state], axis=0)
        '''
        return aS_agent_state, oS_obst_state, anS_goals, an4_dsYddts


class OvertakeEdgeStaticMiddle2lane_Done(LaneChangeANDOvertakeScene):
    """超车场景-已经越过障碍且已经回到目标车道。"""

    def __init__(self, key: PRNGKey, num_agents: int, num_ref_points: int, xrange: Array, yrange: Array,
                 lane_width: float, lane_centers: Array):
        super(OvertakeEdgeStaticMiddle2lane_Done, self).__init__(
            key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers
        )

    @property
    def name(self) -> str:
        return "overtake_scene_with_edge_static_obstacle_and_middle_2_lane_done"

    def make(self) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
        start_x_key, terminal_x_key, start_y_key, start_terminal_v_key, agent_x_key, agent_y_key, agent_v_key, \
            sobst_x_key, sobst_y_key, sobst_theta_key, mobst_x_key, mobst_y_key, mobst_vx_key = jr.split(self.key, 13)

        # 生成轨迹
        start_x = jr.uniform(start_x_key, shape=(), dtype=jnp.float32, minval=self.xrange[0],
                             maxval=(self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0])
        terminal_x = jr.uniform(terminal_x_key, shape=(), dtype=jnp.float32,
                                minval=2 * (self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0],
                                maxval=self.xrange[1])
        del start_x, terminal_x
        start_y = terminal_y = jr.choice(start_y_key, self.lane_centers[jnp.array([0, 1])], shape=())
        terminal_v = jr.uniform(start_terminal_v_key, shape=(), dtype=jnp.float32, minval=20, maxval=40) / 3.6
        anS_goals, an4_dsYddts = generate_horizontal_path_points(
            self.xrange, self.num_agents, self.num_ref_points, start_y, terminal_v
        )

        # 生成静态障碍物，x坐标位于[xs_min, xs_max]区间随机，y坐标在两条车道中随机选择
        xs_min = self.xrange[0] + 35.0
        xs_max = self.xrange[1] - 15.0
        sobst_x = jr.uniform(sobst_x_key, shape=(), dtype=jnp.float32, minval=xs_min, maxval=xs_max)
        sobst_lane = jr.choice(sobst_y_key, jnp.array([0, 1], dtype=jnp.int32), shape=())
        sobst_y = self.lane_centers[sobst_lane]
        sobst_theta = jr.uniform(sobst_theta_key, shape=(), dtype=jnp.float32, minval=0, maxval=0)
        S_sobst_state = make_state(sobst_x, sobst_y, sobst_theta, jnp.array(0.0, dtype=jnp.float32))
        oS_obst_state = S_sobst_state[None]

        # 生成初始agent，越过障碍且回到参考轨迹车道
        agent_x = sobst_x + jr.uniform(agent_x_key, shape=(), dtype=jnp.float32, minval=18.0, maxval=32.0)
        a_agent_x = jnp.repeat(agent_x[None], self.num_agents, axis=0)
        a_agent_y = jnp.repeat(terminal_y[None], self.num_agents, axis=0) + jr.uniform(
            agent_y_key, shape=(self.num_agents,), dtype=jnp.float32, minval=-0.1, maxval=0.1
        )
        a_agent_hx = jnp.ones((self.num_agents,), dtype=jnp.float32)
        a_agent_hy = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        a_agent_v = jr.uniform(agent_v_key, shape=(self.num_agents,), dtype=jnp.float32,
                               minval=0.0, maxval=40.0) / 3.6
        a_agent_delta = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        aS_agent_state = jnp.stack([a_agent_x, a_agent_y, a_agent_hx, a_agent_hy, a_agent_v, a_agent_delta], axis=1)

        '''
        # 生成动态障碍物，y坐标位于中间车道，可稍有浮动，vx可随机选择一个较大值，x坐标需要计算agent恰好到静态障碍物时时动态障碍物位于
        # agent附近
        t = (sobst_x - agent_x) / terminal_vx
        mobst_y = jr.uniform(mobst_y_key, shape=(), dtype=jnp.float32, minval=-0.5, maxval=0.5) + self.lane_centers[1]
        mobst_vx = jr.uniform(mobst_vx_key, shape=(), dtype=jnp.float32, minval=90, maxval=120)  # 90 ~ 120 km/h
        mobst_x = sobst_x - t * mobst_vx  + \
                  jr.uniform(mobst_x_key, shape=(), dtype=jnp.float32, minval=5, maxval=15)
        S_mobst_state = jnp.concatenate([mobst_x[None], mobst_y[None], mobst_vx[None], Sm3_other0], axis=0)
        oS_obst_state = jnp.stack([S_sobst_state, S_mobst_state], axis=0)
        '''
        return aS_agent_state, oS_obst_state, anS_goals, an4_dsYddts


def gen_scene_randomly_split(key: PRNGKey, num_agents: int, num_ref_points: int, xrange: Array, yrange: Array,
                       lane_width: float, lane_centers: Array) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
    choose_key, scene_key = jr.split(key, 2)
    scene_list = [
        LaneChangeMiddleStaticEdgeFastMoving2lane_Start(
            scene_key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers
        ).make,
        LaneChangeMiddleStaticEdgeFastMoving2lane_Approach(
            scene_key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers
        ).make,
        LaneChangeMiddleStaticEdgeFastMoving2lane_Side(
            scene_key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers
        ).make,
        LaneChangeMiddleStaticEdgeFastMoving2lane_Passed(
            scene_key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers
        ).make,
        LaneChangeMiddleStaticEdgeFastMoving2lane_Done(
            scene_key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers
        ).make,
        OvertakeEdgeStaticMiddle2lane_Start(
            scene_key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers
        ).make,
        OvertakeEdgeStaticMiddle2lane_Approach(
            scene_key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers
        ).make,
        OvertakeEdgeStaticMiddle2lane_Side(
            scene_key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers
        ).make,
        OvertakeEdgeStaticMiddle2lane_Passed(
            scene_key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers
        ).make,
        OvertakeEdgeStaticMiddle2lane_Done(
            scene_key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers
        ).make,
    ]
    probs = jnp.array([0.075, 0.175, 0.1, 0.1, 0.050, 0.075, 0.175, 0.1, 0.1, 0.050])
    # probs = jnp.array([0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0]) # for eval
    choose_id = jr.choice(choose_key, len(scene_list), p=probs)
    aS_agent_state, oS_obst_state, anS_goals, an4_dsYddts = jax.lax.switch(choose_id, scene_list)

    return aS_agent_state, oS_obst_state, anS_goals, an4_dsYddts
