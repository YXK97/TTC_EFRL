import jax
import jax.numpy as jnp
import jax.random as jr

from abc import ABC, abstractmethod, abstractproperty
from typing import Tuple

from defmarl.utils.typing import PRNGKey, State, AgentState, ObstState, Array, PathRefs
from defmarl.utils.utils import const_f, quintic_polynomial_f, three_sec_f


def heading_from_theta(theta_rad: Array) -> Tuple[Array, Array]:
    return jnp.cos(theta_rad), jnp.sin(theta_rad)


def make_state(x: Array, y: Array, theta_rad: Array, v_mps: Array, delta_rad: Array = 0.0) -> State:
    hx, hy = heading_from_theta(theta_rad)
    return jnp.stack([x, y, hx, hy, v_mps, jnp.asarray(delta_rad, dtype=jnp.float32)])


def calc_quintic_eff_from_heading(starts: AgentState, terminals: AgentState):
    zeros = jnp.zeros((starts.shape[0],), dtype=jnp.float32)

    def A_b_create_and_solve(start, terminal):
        x0 = start[0]
        x1 = terminal[0]
        A = jnp.array([[1, x0, x0**2,   x0**3,    x0**4,    x0**5],
                       [0,  1,  2*x0, 3*x0**2,  4*x0**3,  5*x0**4],
                       [0,  0,     2,    6*x0, 12*x0**2, 20*x0**3],
                       [1, x1, x1**2,   x1**3,    x1**4,    x1**5],
                       [0,  1,  2*x1, 3*x1**2,  4*x1**3,  5*x1**4],
                       [0,  0,     2,    6*x1, 12*x1**2, 20*x1**3]])
        y0 = start[1]
        y1 = terminal[1]
        slope0 = start[3] / jnp.maximum(start[2], 1e-6)
        slope1 = terminal[3] / jnp.maximum(terminal[2], 1e-6)
        b = jnp.array([y0, slope0, 0, y1, slope1, 0])
        return jnp.linalg.solve(A, b)

    coeffs_f = jax.vmap(A_b_create_and_solve, in_axes=(0, 0))(starts, terminals)
    coeffs_df = jnp.stack([coeffs_f[:, 1], 2*coeffs_f[:, 2], 3*coeffs_f[:, 3], 4*coeffs_f[:, 4], 5*coeffs_f[:, 5], zeros], axis=1)
    coeffs_ddf = jnp.stack([2*coeffs_f[:, 2], 6*coeffs_f[:, 3], 12*coeffs_f[:, 4], 20*coeffs_f[:, 5], zeros, zeros], axis=1)
    coeffs_dddf = jnp.stack([6*coeffs_f[:, 3], 24*coeffs_f[:, 4], 60*coeffs_f[:, 5], zeros, zeros, zeros], axis=1)
    return coeffs_f, coeffs_df, coeffs_ddf, coeffs_dddf

def generate_lanechange_path_points(xrange: Array,
                                    num_agents: int,
                                    num_points: int,
                                    S_start_state: AgentState,
                                    S_terminal_state: AgentState,
                                    points_interval: int=0.1) -> Tuple[PathRefs, jnp.ndarray]:
    """生成由水平直线-五次多项式曲线-水平直线组成的分段参考轨迹点，默认每0.1m生成一个参考点，共生成3200个"""
    # 生成中间的五次多项式
    one6_patheffs_f, one6_patheffs_df, one6_patheffs_ddf, one6_patheffs_dddf = \
        calc_quintic_eff_from_heading(S_start_state[None, :], S_terminal_state[None, :])
    quintic_f = quintic_polynomial_f(one6_patheffs_f)
    quintic_df = quintic_polynomial_f(one6_patheffs_df)
    quintic_ddf = quintic_polynomial_f(one6_patheffs_ddf)
    quintic_dddf = quintic_polynomial_f(one6_patheffs_dddf)
    # 构建三个值的常数函数
    zeros = jnp.zeros((1, 1), dtype=jnp.float32)
    const_f_ystart = const_f(S_start_state[1][None, None])
    const_f_yterminal = const_f(S_terminal_state[1][None, None])
    const_f_zeros = const_f(zeros)
    # 构建中间为五次多项式的分段函数
    poly_sec_f = three_sec_f(const_f_ystart, quintic_f, const_f_yterminal,
                             S_start_state[0][None, None], S_terminal_state[0][None, None])
    poly_sec_df = three_sec_f(const_f_zeros, quintic_df, const_f_zeros,
                              S_start_state[0][None, None], S_terminal_state[0][None, None])
    poly_sec_ddf = three_sec_f(const_f_zeros, quintic_ddf, const_f_zeros,
                               S_start_state[0][None, None], S_terminal_state[0][None, None])
    poly_sec_dddf = three_sec_f(const_f_zeros, quintic_dddf, const_f_zeros,
                                S_start_state[0][None, None], S_terminal_state[0][None, None])
    # 构建路径点
    onen_xs = jnp.linspace(start=xrange[0][None], stop=xrange[0][None] + (num_points + 1) * points_interval, num=num_points,
                           dtype=jnp.float32).T
    onen_ys = poly_sec_f(onen_xs)
    onen_dys = poly_sec_df(onen_xs)
    onen_ddys = poly_sec_ddf(onen_xs)
    onen_dddys = poly_sec_dddf(onen_xs)
    onen_thetas_rad = jnp.arctan(onen_dys)
    onen_heading_x = jnp.cos(onen_thetas_rad)
    onen_heading_y = jnp.sin(onen_thetas_rad)
    # state: x y heading_x heading_y v(m/s) delta(rad)
    onen_vs_mps = jnp.repeat(S_terminal_state[4][None, None], onen_thetas_rad.shape[1], axis=1)
    onen_zeros = jnp.zeros_like(onen_xs)

    onenS_goals = jnp.stack([onen_xs, onen_ys, onen_heading_x, onen_heading_y, onen_vs_mps, onen_zeros],
                            axis=2)
    anS_goals = jnp.repeat(onenS_goals, num_agents, axis=0)

    # 计算dsYddts
    onen_vxs_mps = onen_vs_mps * jnp.cos(onen_thetas_rad)
    onen_dYddt = onen_vxs_mps * onen_dys
    onen_ddYddt = onen_vxs_mps**2 * onen_ddys
    onen_dddYddt = onen_vxs_mps**3 * onen_dddys
    onen4_dsYddts = jnp.stack([onen_ys, onen_dYddt, onen_ddYddt, onen_dddYddt], axis=2)
    an4_dsYddts = jnp.repeat(onen4_dsYddts, num_agents, axis=0)
    return anS_goals, an4_dsYddts

def generate_horizontal_path_points(xrange: Array,
                                    num_agents: int,
                                    num_points: int,
                                    start_y: jnp.ndarray, # shape应为()
                                    terminal_v: jnp.ndarray, # shape应为()
                                    points_interval: int = 0.1) -> Tuple[PathRefs, jnp.ndarray]:
    """生成由水平直线参考轨迹点，默认每0.1m生成一个参考点，共生成3200个"""
    assert start_y.shape == terminal_v.shape == ()
    # 构建路径点
    onen_xs = jnp.linspace(start=xrange[0][None], stop=xrange[0][None] + (num_points + 1) * points_interval, num=num_points,
                           dtype=jnp.float32).T
    onen_ys = jnp.repeat(start_y[None, None], num_points, axis=1)
    onen_heading_x = jnp.ones_like(onen_ys)
    onen_heading_y = jnp.zeros_like(onen_ys)
    # state: x y heading_x heading_y v(m/s) delta(rad)
    onen_vs_mps = jnp.repeat(terminal_v[None, None], num_points, axis=1)
    onen_zeros = jnp.zeros_like(onen_xs)

    onenS_goals = jnp.stack([onen_xs, onen_ys, onen_heading_x, onen_heading_y, onen_vs_mps, onen_zeros],
                            axis=2)
    anS_goals = jnp.repeat(onenS_goals, num_agents, axis=0)

    # 计算dsYddts
    onen4_dsYddts = jnp.stack([onen_ys, onen_zeros, onen_zeros, onen_zeros], axis=2)
    an4_dsYddts = jnp.repeat(onen4_dsYddts, num_agents, axis=0)
    return anS_goals, an4_dsYddts


class SceneBase(ABC):
    """用于生成一些相对固定的基础场景"""

    def __init__(self, key:PRNGKey, num_agents:int):
        self.key = key
        self.num_agents = num_agents

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def num_ref_points(self) -> int:
        pass

    @property
    @abstractmethod
    def state_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def num_lanes(self) -> int:
        pass

    @property
    @abstractmethod
    def lane_centers(self) -> Array:
        pass

    @property
    @abstractmethod
    def xrange(self) -> Array:
        pass

    @property
    @abstractmethod
    def yrange(self) -> Array:
        pass

    @property
    @abstractmethod
    def num_moving_obsts(self) -> int:
        pass

    @property
    @abstractmethod
    def num_static_obsts(self) -> int:
        pass

    @property
    def num_obsts(self) -> int:
        return self.num_moving_obsts + self.num_static_obsts

    @abstractmethod
    def make(self) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
        pass


class LaneChangeANDOvertakeScene(SceneBase, ABC):
    """专用于lanechangeANDovertake任务的场景，需要指定x和y的范围以及车道宽度，所有agent共享同一条轨迹，但起点和初始状态不一定一样"""

    def __init__(self, key:PRNGKey, num_agents:int, num_ref_points:int, xrange:Array, yrange:Array, lane_width:float,
                 lane_centers:Array):
        super(LaneChangeANDOvertakeScene, self).__init__(key, num_agents)
        self._xrange = xrange
        self._yrange = yrange
        self._num_ref_points = num_ref_points
        self._lane_width = lane_width
        self._lane_centers = lane_centers

    @property
    def num_ref_points(self) -> int:
        return self._num_ref_points

    @property
    def state_dim(self) -> int:
        return 6 # state: x y heading_x heading_y v(m/s) delta(rad)

    @property
    def num_lanes(self):
        return self._lane_centers.shape[0]

    @property
    def lane_centers(self):
        return self._lane_centers

    @property
    def xrange(self):
        return self._xrange

    @property
    def yrange(self):
        return self._yrange

class LaneChangeMiddleStaticEdgeFastMoving2lane(LaneChangeANDOvertakeScene):
    """此场景只可适用于双车道场景，1个静态障碍物，静态障碍物放在车道中间，但是位置随机；
    agent由第1/2车道向第2/1车道变道，使得agent在差不多到第3/1车道时与动态障碍物发生碰撞，图例如下：
    ==================================================
    1                           ---------------> reference path
    ------------------------/-------------------------
    2     ego  ■ --------/    ♦  static obstacle
    ==================================================
    """

    def __init__(self, key: PRNGKey, num_agents: int, num_ref_points:int, xrange:Array, yrange:Array, lane_width:float,
                 lane_centers:Array):
        super(LaneChangeMiddleStaticEdgeFastMoving2lane, self).__init__(key, num_agents, num_ref_points, xrange, yrange,
                                                                   lane_width, lane_centers)

    @property
    def name(self) -> str:
        return 'lanechange_scene_with_middle_static_obstacle_and_edge_2_lane'

    @property
    def num_lanes(self) -> int:
        assert self.lane_centers.shape[0] == 2, '本场景支持双车道配置！'
        return 2

    @property
    def num_moving_obsts(self) -> int:
        return 0

    @property
    def num_static_obsts(self) -> int:
        return 1

    def make(self) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
        start_x_key, terminal_x_key, start_y_key, start_terminal_v_key, agent_x_key, agent_y_key, agent_v_key, \
            sobst_y_key, sobst_theta_key, mobst_x_key, mobst_y_key, mobst_vx_key = jr.split(self.key, 12)

        # 生成轨迹
        start_x = jr.uniform(start_x_key, shape=(), dtype=jnp.float32, minval=self.xrange[0],
                             maxval=(self.xrange[1]-self.xrange[0])/3+self.xrange[0])
        terminal_x = jr.uniform(terminal_x_key, shape=(), dtype=jnp.float32,
                                minval=2*(self.xrange[1]-self.xrange[0])/3+self.xrange[0],
                                maxval=self.xrange[1])

        start_lane = jr.choice(start_y_key, jnp.array([0, 1], dtype=jnp.int32), shape=())
        terminal_lane = 1 - start_lane
        start_y = self.lane_centers[start_lane]
        terminal_y = self.lane_centers[terminal_lane]
        start_theta = terminal_theta = jnp.array([0.0])[0]
        start_v = terminal_v = jr.uniform(start_terminal_v_key, shape=(), dtype=jnp.float32,
                                          minval=10, maxval=30) / 3.6
        S_start_state = make_state(start_x, start_y, start_theta, start_v)
        S_terminal_state = make_state(terminal_x, terminal_y, terminal_theta, terminal_v)
        anS_goals, an4_dsYddts = generate_lanechange_path_points(self.xrange, self.num_agents, self.num_ref_points,
                                                                 S_start_state, S_terminal_state)

        # 生成初始agent，x坐标都一样，y坐标可略有上下波动，vx可不一样，其它的都是0
        agent_x = jr.uniform(agent_x_key, shape=(), dtype=jnp.float32, minval=self.xrange[0], maxval=start_x)
        a_agent_x = jnp.repeat(agent_x[None], self.num_agents, axis=0) # 变道前同一x
        a_agent_y = jnp.repeat(start_y[None], self.num_agents, axis=0) + jr.uniform(agent_y_key, shape=(self.num_agents,),
                                                                                    dtype=jnp.float32, minval=-0.1, maxval=0.1)
        a_agent_hx = jnp.repeat(jnp.cos(start_theta)[None], self.num_agents, axis=0)
        a_agent_hy = jnp.repeat(jnp.sin(start_theta)[None], self.num_agents, axis=0)
        a_agent_v = jr.uniform(agent_v_key, shape=(self.num_agents,), dtype=jnp.float32,
                                minval=10, maxval=30) / 3.6
        a_agent_delta = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        aS_agent_state = jnp.stack([a_agent_x, a_agent_y, a_agent_hx, a_agent_hy, a_agent_v, a_agent_delta], axis=1)

        # 生成静态障碍物，x坐标位于变道后2/3区间随机，y坐标随机选择车道1/2，航向角可选-180°到180°之间
        sobst_x = 2*(self.xrange[1]-self.xrange[0])/3 + self.xrange[0]
        sobst_y = jr.choice(sobst_y_key, self.lane_centers[jnp.array([0,1])], shape=())
        # sobst_y = jr.uniform(sobst_y_key, shape=(), dtype=jnp.float32, minval=-0., maxval=0.) + sobst_y_key
        sobst_theta = jr.uniform(sobst_theta_key, shape=(), dtype=jnp.float32, minval=0, maxval=0)
        S_sobst_state = make_state(sobst_x, sobst_y, sobst_theta, jnp.array(0.0))
        oS_obst_state = S_sobst_state[None]
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

class OvertakeEdgeStaticMiddle2lane(LaneChangeANDOvertakeScene):
    """此场景只可适用于双车道场景，1个静态障碍物，静态障碍物放在第1/2车道，
    agent沿第1/2车道直行，静态障碍物在后2/3处随机车道1/2，图例如下：
    ==================================================
    1
    --------------------------------------------------
    2     ego  ■ --------♦  static obstacle--> reference path
    ==================================================
    """

    def __init__(self, key: PRNGKey, num_agents: int, num_ref_points:int, xrange:Array, yrange:Array, lane_width:float,
                 lane_centers:Array):
        super(OvertakeEdgeStaticMiddle2lane, self).__init__(key, num_agents, num_ref_points, xrange, yrange,
                                                                 lane_width, lane_centers)

    @property
    def name(self) -> str:
        return 'overtake_scene_with_edge_static_obstacle_and_middle_fast_moving_obstacle'

    @property
    def num_lanes(self) -> int:
        assert self.lane_centers.shape[0] == 2, '本场景仅支持双车道配置！'
        return 2

    @property
    def num_moving_obsts(self) -> int:
        return 0

    @property
    def num_static_obsts(self) -> int:
        return 1

    def make(self) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
        start_x_key, terminal_x_key, start_y_key, start_terminal_v_key, agent_x_key, agent_y_key, agent_v_key, \
            sobst_y_key, sobst_theta_key, mobst_x_key, mobst_y_key, mobst_vx_key = jr.split(self.key, 12)

        # 生成轨迹
        start_x = jr.uniform(start_x_key, shape=(), dtype=jnp.float32, minval=self.xrange[0],
                             maxval=(self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0])
        terminal_x = jr.uniform(terminal_x_key, shape=(), dtype=jnp.float32,
                                minval=2 * (self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0],
                                maxval=self.xrange[1])
        start_theta = terminal_theta = jnp.array([0.0])[0]
        start_y = terminal_y = jr.choice(start_y_key, self.lane_centers[jnp.array([0, 1])], shape=())
        terminal_v = jr.uniform(start_terminal_v_key, shape=(), dtype=jnp.float32,
                                 minval=10, maxval=30) / 3.6
        anS_goals, an4_dsYddts = generate_horizontal_path_points(self.xrange, self.num_agents, self.num_ref_points,
                                                                 start_y, terminal_v)

        # 生成初始agent，x坐标都一样，y坐标可略有上下波动，vx可不一样，其它的都是0
        agent_x = jr.uniform(agent_x_key, shape=(), dtype=jnp.float32, minval=self.xrange[0], maxval=start_x)
        a_agent_x = jnp.repeat(agent_x[None], self.num_agents, axis=0)  # 变道前同一x
        a_agent_y = jnp.repeat(start_y[None], self.num_agents, axis=0) \
                    + jr.uniform(agent_y_key, shape=(self.num_agents,), dtype=jnp.float32, minval=-0.1, maxval=0.1)
        a_agent_hx = jnp.repeat(jnp.cos(start_theta)[None], self.num_agents, axis=0)
        a_agent_hy = jnp.repeat(jnp.sin(start_theta)[None], self.num_agents, axis=0)
        a_agent_v = jr.uniform(agent_v_key, shape=(self.num_agents,), dtype=jnp.float32,
                                minval=10, maxval=30) / 3.6
        a_agent_delta = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        aS_agent_state = jnp.stack([a_agent_x, a_agent_y, a_agent_hx, a_agent_hy, a_agent_v, a_agent_delta], axis=1)

        # 生成静态障碍物，位于start和terminal中间，y坐标可以略微上下浮动，航向角可选-180°到180°之间
        sobst_x = 2*(self.xrange[1]-self.xrange[0])/3 + self.xrange[0]
        sobst_y = jr.choice(start_y_key, self.lane_centers[jnp.array([0,1])], shape=())
        # sobst_y = jr.uniform(sobst_y_key, shape=(), dtype=jnp.float32, minval=-0., maxval=0.) + sobst_y_key
        sobst_theta = jr.uniform(sobst_theta_key, shape=(), dtype=jnp.float32, minval=0, maxval=0)
        S_sobst_state = make_state(sobst_x, sobst_y, sobst_theta, jnp.array(0.0))
        oS_obst_state = S_sobst_state[None]
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

class HandMadeSceneLaneChange2LaneFastMoving(LaneChangeANDOvertakeScene):
    """
    【双车道版本】车道宽度 = 2.5m
    静态障碍物：放在车道1
    ==================================================
    1     ego  ■ --------\        ♦ static obstacle
    ------------------------\-----------------------
    2                          \-------------> reference path
    ==================================================
    """

    def __init__(self, key: PRNGKey, num_agents: int, num_ref_points:int, xrange:Array, yrange:Array, lane_width:float,
                 lane_centers:Array):
        super().__init__(key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers)

    @property
    def name(self) -> str:
        return 'handmade_lanechange_2lane'

    @property
    def num_lanes(self) -> int:
        # 双车道场景
        assert self.lane_centers.shape[0] == 2, f'本场景仅支持双车道配置！当前车道数：{self.lane_centers.shape[0]}'
        return 2

    @property
    def num_moving_obsts(self) -> int:
        return 0

    @property
    def num_static_obsts(self) -> int:
        return 1

    def make(self) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
        start_x = jnp.array([10.])[0]
        terminal_x = jnp.array([40.])[0]

        start_y = self.lane_centers[1]
        terminal_y = self.lane_centers[0]
        start_v = terminal_v = jnp.array([20.0])[0] / 3.6
        start_theta = terminal_theta = jnp.array([0.0])[0]
        S_start_state = make_state(start_x, start_y, start_theta, start_v)
        S_terminal_state = make_state(terminal_x, terminal_y, terminal_theta, terminal_v)

        anS_goals, an4_dsYddts = generate_lanechange_path_points(
            self.xrange, self.num_agents, self.num_ref_points,
            S_start_state, S_terminal_state
        )


        agent_x = jnp.array([0.])[0]
        agent_v = jnp.array([0.0])[0]
        a_agent_x = jnp.repeat(agent_x[None], self.num_agents, axis=0)
        a_agent_y = jnp.repeat(start_y[None], self.num_agents, axis=0)
        a_agent_hx = jnp.repeat(jnp.cos(start_theta)[None], self.num_agents, axis=0)
        a_agent_hy = jnp.repeat(jnp.sin(start_theta)[None], self.num_agents, axis=0)
        a_agent_v = jnp.repeat(agent_v[None], self.num_agents, axis=0)
        a_agent_delta = jnp.zeros((self.num_agents,), dtype=jnp.float32)

        aS_agent_state = jnp.stack([a_agent_x, a_agent_y, a_agent_hx, a_agent_hy, a_agent_v, a_agent_delta], axis=1)

        sobst_x = (start_x + terminal_x) / 2 + 15
        sobst_y = terminal_y
        S_sobst_state = make_state(sobst_x, sobst_y, jnp.array(0.0), jnp.array(0.0))
        oS_obst_state = S_sobst_state[None]

        return aS_agent_state, oS_obst_state, anS_goals, an4_dsYddts

def gen_scene_randomly(key: PRNGKey, num_agents: int, num_ref_points: int, xrange: Array, yrange: Array,
                       lane_width: float, lane_centers: Array) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
    choose_key, scene_key = jr.split(key, 2)
    scene_list = [LaneChangeMiddleStaticEdgeFastMoving2lane(scene_key, num_agents, num_ref_points, xrange, yrange,
                                                       lane_width, lane_centers).make,
                  OvertakeEdgeStaticMiddle2lane(scene_key, num_agents, num_ref_points, xrange, yrange,
                                                     lane_width, lane_centers).make]
    choose_id = jr.choice(choose_key, len(scene_list))
    aS_agent_state, oS_obst_state, anS_goals, an4_dsYddts = jax.lax.switch(choose_id, scene_list)

    return aS_agent_state, oS_obst_state, anS_goals, an4_dsYddts

def gen_handmade_scene(key: PRNGKey, num_agents: int, num_ref_points: int, xrange: Array, yrange: Array,
                       lane_width: float, lane_centers: Array) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
    choose_key, scene_key = jr.split(key, 2)
    scene_list = [
        HandMadeSceneLaneChange2LaneFastMoving(scene_key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers).make, # Scenario I
    ]
    choose_id = jr.choice(choose_key, len(scene_list))
    aS_agent_state, oS_obst_state, anS_goals, an4_dsYddts = jax.lax.switch(choose_id, scene_list)

    return aS_agent_state, oS_obst_state, anS_goals, an4_dsYddts
