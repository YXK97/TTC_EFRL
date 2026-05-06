import jax
import jax.numpy as jnp
import jax.random as jr

from abc import ABC, abstractmethod, abstractproperty
from typing import Tuple

from defmarl.utils.typing import PRNGKey, State, AgentState, ObstState, Array, PathRefs
from defmarl.utils.utils import calc_linear_eff, calc_quintic_eff_lowspeed, const_f, linear_f, quintic_polynomial_f, three_sec_f

def generate_lanechange_path_points(xrange: Array,
                                    num_agents: int,
                                    num_points: int,
                                    S_start_state: AgentState,
                                    S_terminal_state: AgentState,
                                    points_interval: int=0.1) -> Tuple[PathRefs, jnp.ndarray]:
    """生成由水平直线-五次多项式曲线-水平直线组成的分段参考轨迹点，默认每0.1m生成一个参考点，共生成3200个"""
    # 生成中间的五次多项式
    one6_patheffs_f, one6_patheffs_df, one6_patheffs_ddf, one6_patheffs_dddf = \
        calc_quintic_eff_lowspeed(S_start_state[None,:], S_terminal_state[None,:])
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
    onen_thetas_deg = onen_thetas_rad * 180 / jnp.pi
    # state: x y θ v δ bw bh lr
    onen_vs_kmph = jnp.repeat(S_terminal_state[3][None, None], onen_thetas_rad.shape[1], axis=1)
    onen_zeros = jnp.zeros_like(onen_xs)

    onenS_goals = jnp.stack([onen_xs, onen_ys, onen_thetas_deg, onen_vs_kmph, onen_zeros, onen_zeros, onen_zeros, onen_zeros],
                            axis=2)
    anS_goals = jnp.repeat(onenS_goals, num_agents, axis=0)

    # 计算dsYddts
    onen_vs_mps = onen_vs_kmph / 3.6
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
                                    terminal_vx: jnp.ndarray, # shape应为()
                                    points_interval: int = 0.1) -> Tuple[PathRefs, jnp.ndarray]:
    """生成由水平直线参考轨迹点，默认每0.1m生成一个参考点，共生成3200个"""
    assert start_y.shape == terminal_vx.shape == ()
    # 构建路径点
    onen_xs = jnp.linspace(start=xrange[0][None], stop=xrange[0][None] + (num_points + 1) * points_interval, num=num_points,
                           dtype=jnp.float32).T
    onen_ys = jnp.repeat(start_y[None, None], num_points, axis=1)
    onen_thetas_rad = jnp.zeros_like(onen_ys)
    onen_thetas_deg = jnp.zeros_like(onen_ys)
    # state: x y vx vy θ dθdt bw bh
    onen_vs_kmph = jnp.repeat(terminal_vx[None, None], num_points, axis=1)
    onen_vxs_kmph = onen_vs_kmph
    onen_vys_kmph = jnp.zeros_like(onen_ys)
    onen_dthetas_degps = jnp.zeros_like(onen_ys)
    onen_zeros = jnp.zeros_like(onen_xs)

    onenS_goals = jnp.stack([onen_xs, onen_ys, onen_vxs_kmph, onen_vys_kmph, onen_thetas_deg, onen_dthetas_degps, onen_zeros, onen_zeros],
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
        return 8 # state: x y θ v δ bw bh lr

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
        start_v = terminal_v = jnp.array([20])[0]
        start_theta = terminal_theta = jnp.array([0.])[0]
        Sm4_other0 = jnp.zeros((self.state_dim - 4,), dtype=jnp.float32)
        S_start_state = jnp.concatenate([start_x[None], start_y[None], start_theta[None], start_v[None], Sm4_other0])
        S_terminal_state = jnp.concatenate([terminal_x[None], terminal_y[None], terminal_theta[None], terminal_v[None], Sm4_other0])

        anS_goals, an4_dsYddts = generate_lanechange_path_points(
            self.xrange, self.num_agents, self.num_ref_points,
            S_start_state, S_terminal_state
        )


        agent_x = jnp.array([0.])[0]
        agent_v = jnp.array([0])[0]
        a_agent_x = jnp.repeat(agent_x[None], self.num_agents, axis=0)
        a_agent_y = jnp.repeat(start_y[None], self.num_agents, axis=0)
        a_agent_theta = jnp.repeat(start_theta[None], self.num_agents, axis=0)
        a_agent_v = jnp.repeat(agent_v[None], self.num_agents, axis=0)
        aSm4_other0 = jnp.repeat(Sm4_other0[None, :], self.num_agents, axis=0)

        aS_agent_state = jnp.concatenate([
            a_agent_x[:, None], a_agent_y[:, None], a_agent_theta[:,None], a_agent_v[:, None], aSm4_other0
        ], axis=1)

        sobst_x = (start_x + terminal_x) / 2 + 15
        sobst_y = terminal_y
        S_sobst_state = jnp.stack([sobst_x, sobst_y, 0., 0., 0., 0., 0., 0.])
        oS_obst_state = S_sobst_state[None]

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