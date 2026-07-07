import jax
import jax.numpy as jnp
import jax.random as jr

from abc import ABC, abstractmethod
from typing import Tuple

from defmarl.utils.typing import PRNGKey, AgentState, ObstState, Array, PathRefs


ROAD_HALF = 100.0
TURN_HALF = 17.5
POINT_INTERVAL = 0.1
LANE_CENTERS = jnp.array([-3.0, 0.0, 3.0], dtype=jnp.float32)

# road: 0 南路, 1 东路, 2 北路, 3 西路
# ROAD_DIRS 表示车辆在该道路上“朝路口方向/离开路口方向”的单位前进方向，由世界坐标系下的单位向量表示。
# 例如南路车辆朝北进入路口，所以方向为 (0, 1)；东路车辆朝西进入路口，所以方向为 (-1, 0)。
# 道路上的位置统一写成：xy = 前进方向 * longitudinal + 右法向 * lane_offset。
# longitudinal < 0 表示在起始道路上、尚未进入转向区；longitudinal > 0 表示在终止道路上、驶离转向区。
ROAD_DIRS = jnp.array([
    [0.0, 1.0],
    [-1.0, 0.0],
    [0.0, -1.0],
    [1.0, 0.0],
], dtype=jnp.float32)
ROAD_THETAS_DEG = jnp.array([90.0, 180.0, -90.0, 0.0], dtype=jnp.float32)


def _right_normal(direction: Array) -> Array:
    """返回车辆前进方向的右侧法向，用来把车道编号映射成横向偏移。"""
    return jnp.array([direction[1], -direction[0]], dtype=jnp.float32)


def _heading_to_world_velocity(speed_kmph: Array, theta_deg: Array) -> Tuple[Array, Array]:
    theta_rad = theta_deg * jnp.pi / 180.0
    return speed_kmph * jnp.cos(theta_rad), speed_kmph * jnp.sin(theta_rad)


def _state_xy_speed_theta(xy: Array, speed_kmph: Array, theta_deg: Array,
                          dtheta_degps: Array = 0.0) -> Array:
    vx, vy = _heading_to_world_velocity(speed_kmph, theta_deg)
    return jnp.stack([
        xy[0],
        xy[1],
        vx,
        vy,
        theta_deg,
        jnp.asarray(dtheta_degps, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
    ])


def _other_lane_idx(lane_idx: Array, choose_key: PRNGKey) -> Array:
    """从三车道中选择一个不同于 forbidden lane 的车道。"""
    return (lane_idx + 1 + jr.randint(choose_key, shape=(), minval=0, maxval=2)) % 3


def _road_point(road_idx: Array, longitudinal: Array, lane_offset: Array) -> Array:
    """根据道路编号、纵向坐标和车道横向偏移生成世界坐标系下的点。"""
    direction = ROAD_DIRS[road_idx]
    normal = _right_normal(direction)
    longitudinal = jnp.asarray(longitudinal, dtype=jnp.float32)
    lane_offset = jnp.asarray(lane_offset, dtype=jnp.float32)
    return longitudinal[..., None] * direction + lane_offset[..., None] * normal


def _speed_profile(path_s: Array, approach_len: Array, arc_len: Array, exit_len: Array,
                   decel_len: Array, gen_speed: Array, turn_speed: Array) -> Array:
    """沿参考路径生成目标纵向速度。

    路径被分成四段：
    1. 起始道路生成区：保持 gen_speed。
    2. 起始道路减速区：从 gen_speed 线性降到 turn_speed。
    3. 转向区圆弧：保持 turn_speed。
    4. 终止道路减速区：从 turn_speed 线性升到 gen_speed。
    """
    start_gen_len = approach_len - decel_len
    arc_start = approach_len
    exit_start = approach_len + arc_len
    exit_decel_end = exit_start + decel_len

    start_decel_ratio = jnp.clip((path_s - start_gen_len) / decel_len, 0.0, 1.0)
    exit_accel_ratio = jnp.clip((path_s - exit_start) / decel_len, 0.0, 1.0)
    start_decel_speed = gen_speed + (turn_speed - gen_speed) * start_decel_ratio
    exit_accel_speed = turn_speed + (gen_speed - turn_speed) * exit_accel_ratio

    return jnp.where(
        path_s < start_gen_len,
        gen_speed,
        jnp.where(
            path_s < arc_start,
            start_decel_speed,
            jnp.where(
                path_s < exit_start,
                turn_speed,
                jnp.where(path_s < exit_decel_end, exit_accel_speed, gen_speed),
            ),
        ),
    )


def generate_turn_path_points(num_agents: int,
                              num_points: int,
                              start_road_idx: Array,
                              turn_sign: Array,
                              lane_offset: Array,
                              decel_len: Array,
                              gen_speed_kmph: Array,
                              turn_speed_kmph: Array,
                              points_interval: float = POINT_INTERVAL) -> Tuple[PathRefs, jnp.ndarray]:
    """按四向三车道十字路口生成左/右转参考点。

    参考点状态定义保持为 (x, y, vx, vy, theta, dtheta/dt, bw, bh)，其中速度为世界坐标系 km/h。
    """
    s = jnp.linspace(
        start=jnp.array(0.0, dtype=jnp.float32),
        stop=jnp.array((num_points - 1) * points_interval, dtype=jnp.float32),
        num=num_points,
        dtype=jnp.float32,
    )
    theta0_deg = ROAD_THETAS_DEG[start_road_idx]
    theta0_rad = theta0_deg * jnp.pi / 180.0
    turn_radius = TURN_HALF + turn_sign * lane_offset
    turn_radius = jnp.maximum(turn_radius, 1.0)
    approach_len = ROAD_HALF - TURN_HALF
    arc_len = jnp.pi * turn_radius / 2.0
    exit_len = ROAD_HALF - TURN_HALF
    arc_start = approach_len
    exit_start = approach_len + arc_len
    path_total = approach_len + arc_len + exit_len

    s_path = jnp.minimum(s, path_total)
    approach_s = jnp.minimum(s_path, approach_len)
    arc_s = jnp.clip(s_path - arc_start, 0.0, arc_len)
    exit_s = jnp.clip(s_path - exit_start, 0.0, exit_len)

    # 起始道路：从区域边界 -100m 沿道路前进到转向区边界 -TURN_HALF。
    start_longitudinal = -ROAD_HALF + approach_s
    approach_xy = _road_point(start_road_idx, start_longitudinal, lane_offset)

    # 转向区：从起始道路终点出发，用四分之一圆弧连接到终止道路起点。
    # turn_sign = 1 表示左转，turn_sign = -1 表示右转。
    # 半径根据车道偏移修正，使同一车道号在转向前后保持一致。
    phi = arc_s / turn_radius
    theta_arc_rad = theta0_rad + turn_sign * phi
    arc_delta = turn_sign * turn_radius * jnp.stack([
        jnp.sin(theta_arc_rad) - jnp.sin(theta0_rad),
        -jnp.cos(theta_arc_rad) + jnp.cos(theta0_rad),
    ], axis=1)
    arc_start_xy = _road_point(start_road_idx, -TURN_HALF, lane_offset)
    arc_xy = arc_start_xy[None, :] + arc_delta

    # 驶离段不能直接用 terminal_road_idx 对应的 ROAD_DIRS。
    # ROAD_DIRS 描述的是各道路驶向路口的方向；而转弯完成后车辆是沿新的航向驶离路口。
    terminal_longitudinal = TURN_HALF + exit_s

    in_approach = s_path < arc_start
    in_arc = (s_path >= arc_start) & (s_path < exit_start)

    terminal_theta_deg = theta0_deg + turn_sign * 90.0
    terminal_theta_deg = (terminal_theta_deg + 180.0) % 360.0 - 180.0
    terminal_theta_rad = terminal_theta_deg * jnp.pi / 180.0
    terminal_dir = jnp.stack([jnp.cos(terminal_theta_rad), jnp.sin(terminal_theta_rad)])
    terminal_normal = _right_normal(terminal_dir)
    exit_xy = terminal_longitudinal[:, None] * terminal_dir + lane_offset * terminal_normal

    xs = jnp.where(in_approach, approach_xy[:, 0], jnp.where(in_arc, arc_xy[:, 0], exit_xy[:, 0]))
    ys = jnp.where(in_approach, approach_xy[:, 1], jnp.where(in_arc, arc_xy[:, 1], exit_xy[:, 1]))

    theta_deg = jnp.where(
        in_approach,
        theta0_deg,
        jnp.where(in_arc, theta_arc_rad * 180.0 / jnp.pi, terminal_theta_deg),
    )
    speed_kmph = _speed_profile(
        s_path, approach_len, arc_len, exit_len, decel_len, gen_speed_kmph, turn_speed_kmph
    )
    vx_kmph, vy_kmph = _heading_to_world_velocity(speed_kmph, theta_deg)
    speed_mps = speed_kmph / 3.6
    dtheta_degps = jnp.where(
        in_arc,
        turn_sign * speed_mps / turn_radius * 180.0 / jnp.pi,
        0.0,
    )
    zeros = jnp.zeros_like(xs)

    onenS_goals = jnp.stack([xs, ys, vx_kmph, vy_kmph, theta_deg, dtheta_degps, zeros, zeros], axis=1)[None, :, :]
    anS_goals = jnp.repeat(onenS_goals, num_agents, axis=0)

    # 兼容原 UFTSTC 横向控制接口。与 designed_scene_gen.py 保持一致：
    # 对可写成 y=f(x) 的路段，使用 [y, dy/dx*vx, d2y/dx2*vx^2, d3y/dx3*vx^3]。
    cos_theta = jnp.cos(theta_arc_rad)
    safe_cos_theta = jnp.where(
        jnp.abs(cos_theta) < 1e-3,
        jnp.sign(cos_theta + 1e-6) * 1e-3,
        cos_theta,
    )
    curvature = turn_sign / turn_radius
    dys = jnp.where(in_arc, jnp.tan(theta_arc_rad), 0.0)
    ddys = jnp.where(in_arc, curvature / safe_cos_theta ** 3, 0.0)
    dddys = jnp.where(in_arc, 3.0 * curvature ** 2 * jnp.sin(theta_arc_rad) / safe_cos_theta ** 5, 0.0)
    vxs_mps = vx_kmph / 3.6
    dYdt = vxs_mps * dys
    ddYdt = vxs_mps ** 2 * ddys
    dddYdt = vxs_mps ** 3 * dddys
    one4_dsYddts = jnp.stack([ys, dYdt, ddYdt, dddYdt], axis=1)[None, :, :]
    an4_dsYddts = jnp.repeat(one4_dsYddts, num_agents, axis=0)
    return anS_goals, an4_dsYddts


def generate_straight_path_points(num_agents: int,
                                  num_points: int,
                                  start_road_idx: Array,
                                  lane_offset: Array,
                                  decel_len: Array,
                                  gen_speed_kmph: Array,
                                  crossing_speed_kmph: Array,
                                  points_interval: float = POINT_INTERVAL) -> Tuple[PathRefs, jnp.ndarray]:
    """按四向三车道十字路口生成直行参考点。

    直行轨迹从起始道路边界沿同一车道中心线穿过路口，再从对向道路驶出。
    转向区在直行场景中退化为一段长度为 2 * TURN_HALF 的低速直线段。
    """
    s = jnp.linspace(
        start=jnp.array(0.0, dtype=jnp.float32),
        stop=jnp.array((num_points - 1) * points_interval, dtype=jnp.float32),
        num=num_points,
        dtype=jnp.float32,
    )
    theta_deg = ROAD_THETAS_DEG[start_road_idx]
    approach_len = ROAD_HALF - TURN_HALF
    crossing_len = 2.0 * TURN_HALF
    exit_len = ROAD_HALF - TURN_HALF
    path_total = approach_len + crossing_len + exit_len

    s_path = jnp.minimum(s, path_total)
    longitudinal = -ROAD_HALF + s_path
    xy = _road_point(start_road_idx, longitudinal, lane_offset)

    speed_kmph = _speed_profile(
        s_path, approach_len, crossing_len, exit_len, decel_len, gen_speed_kmph, crossing_speed_kmph
    )
    vx_kmph, vy_kmph = _heading_to_world_velocity(speed_kmph, theta_deg)
    zeros = jnp.zeros_like(s_path)
    theta_refs = jnp.repeat(theta_deg[None], num_points, axis=0)

    onenS_goals = jnp.stack([
        xy[:, 0], xy[:, 1], vx_kmph, vy_kmph, theta_refs, zeros, zeros, zeros
    ], axis=1)[None, :, :]
    anS_goals = jnp.repeat(onenS_goals, num_agents, axis=0)

    # 兼容原横向控制接口。直行场景不产生转向加速度和 jerk，只记录世界 y 方向速度分量。
    vy_mps = vy_kmph / 3.6
    one4_dsYddts = jnp.stack([xy[:, 1], vy_mps, zeros, zeros], axis=1)[None, :, :]
    an4_dsYddts = jnp.repeat(one4_dsYddts, num_agents, axis=0)
    return anS_goals, an4_dsYddts


class IntersectionSceneBase(ABC):
    def __init__(self, key: PRNGKey, num_agents: int, num_ref_points: int,
                 xrange: Array, yrange: Array, lane_width: float, lane_centers: Array):
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
        return 8

    @property
    def num_moving_obsts(self) -> int:
        return 1

    @property
    def num_static_obsts(self) -> int:
        return 1

    @abstractmethod
    def make(self) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
        pass


class IntersectionTurnScene(IntersectionSceneBase):
    """十字路口转向类基础场景。

    只处理转向，不处理直行。agent 从西路生成区出发，沿同一车道号左转或右转。
    目标轨迹在道路区域沿车道中心，在转向区以圆弧连接，速度由生成区高速逐渐降至转向区低速，
    再在终止道路减速区逐渐升回生成区速度。
    """

    dynamic_mode = 0
    turn_sign = 1

    @property
    def name(self) -> str:
        return "intersection_turn_scene"

    def _make_dynamic_obstacle(self, keys, start_road_idx, lane_idx, lane_offset,
                               agent_r, agent_speed, decel_len) -> Array:
        road_choice_key, lane_key, speed_key, jitter_key = keys
        mode = jnp.array(self.dynamic_mode, dtype=jnp.int32)
        side = jr.choice(road_choice_key, jnp.array([-1, 1], dtype=jnp.int32), shape=())

        # mode == 0：动态障碍车从与 agent 初始道路垂直的道路驶入，例如 agent 在南路，则障碍车在东路或西路。
        # mode == 1：动态障碍车从与 agent 初始道路平行的道路驶入，例如 agent 在南路，则障碍车在南路或北路。
        perpendicular_road = (start_road_idx + side) % 4
        parallel_road = jnp.where(side > 0, start_road_idx, (start_road_idx + 2) % 4)
        obst_road_idx = jnp.where(mode == 0, perpendicular_road, parallel_road)

        # 垂直动态障碍物：三条车道任意选一条。
        random_lane_idx = jr.randint(lane_key, shape=(), minval=0, maxval=3)
        # 平行动动态障碍物：
        # 1. 与 agent 同向时，不能选择 agent 初始车道。
        # 2. 与 agent 反向时，不能选择与 agent 初始车道物理位置相对的车道。
        #    三车道索引为 0/1/2，反向道路中与 lane_idx 对应同一条物理车道的是 2-lane_idx。
        same_direction = obst_road_idx == start_road_idx
        forbidden_parallel_lane_idx = jnp.where(same_direction, lane_idx, 2 - lane_idx)
        other_lane_idx = _other_lane_idx(forbidden_parallel_lane_idx, lane_key)
        obst_lane_idx = jnp.where(mode == 0, random_lane_idx, other_lane_idx)
        obst_lane_offset = LANE_CENTERS[obst_lane_idx]

        direction = ROAD_DIRS[obst_road_idx]
        theta_deg = ROAD_THETAS_DEG[obst_road_idx]
        speed = jr.uniform(speed_key, shape=(), dtype=jnp.float32, minval=20.0, maxval=100.0)

        # 粗略估计 agent 从当前位置到转向区入口的时间，再反推动态障碍车初始位置。
        # 这样 agent 到达转向区时，动态障碍车大概率也在冲突点附近，形成干扰。
        dist_to_turn = -TURN_HALF - agent_r
        avg_speed_mps = ((agent_speed + 35.0) * 0.5) / 3.6
        t_to_turn = dist_to_turn / jnp.maximum(avg_speed_mps, 0.1)

        # 垂直模式的冲突点取障碍车道路穿过路口中心附近的位置。
        # 平行模式取动态障碍车自己道路、自己车道的转向区入口；
        # 这样车道选择会真实反映到障碍车轨迹上，而不是被 agent 初始车道覆盖。
        conflict_xy = jnp.where(
            mode == 0,
            _road_point(obst_road_idx, 0.0, obst_lane_offset),
            _road_point(obst_road_idx, -TURN_HALF, obst_lane_offset),
        )
        # 加大纵向扰动，使动态障碍车既可能提前到达，也可能滞后到达，甚至已经驶过冲突点。
        longitudinal_jitter = jr.uniform(jitter_key, shape=(), dtype=jnp.float32, minval=-80.0, maxval=80.0)
        xy = conflict_xy - direction * (speed / 3.6 * t_to_turn + longitudinal_jitter)
        return _state_xy_speed_theta(xy, speed, theta_deg)

    def make(self) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
        (start_road_key, turn_key, lane_key, decel_key, gen_speed_key, turn_speed_key,
         agent_r_key, agent_lat_key, agent_speed_key, agent_theta_key, sobst_idx_key,
         sobst_pos_key, sobst_theta_key, dyn_road_key, dyn_lane_key, dyn_speed_key,
         dyn_jitter_key) = jr.split(self.key, 17)
        del start_road_key, turn_key, turn_speed_key, agent_theta_key

        start_road_idx = jnp.array(3, dtype=jnp.int32)
        turn_sign = jnp.array(self.turn_sign, dtype=jnp.int32)
        lane_idx = jr.randint(lane_key, shape=(), minval=0, maxval=3)
        lane_offset = LANE_CENTERS[lane_idx]

        # 减速区长度在场景生成时确定，范围为 40m 到 60m。
        decel_len = jr.uniform(decel_key, shape=(), dtype=jnp.float32, minval=40.0, maxval=60.0)
        gen_speed = jr.uniform(gen_speed_key, shape=(), dtype=jnp.float32, minval=60.0, maxval=90.0)
        turn_speed = jnp.array(40.0, dtype=jnp.float32)
        anS_goals, an4_dsYddts = generate_turn_path_points(
            self.num_agents,
            self.num_ref_points,
            start_road_idx,
            turn_sign,
            lane_offset,
            decel_len,
            gen_speed,
            turn_speed,
        )

        gen_zone_high = -TURN_HALF - decel_len
        a_agent_r = jr.uniform(
            agent_r_key, shape=(self.num_agents,), dtype=jnp.float32,
            minval=-ROAD_HALF, maxval=gen_zone_high,
        )
        a_agent_lat = jr.uniform(
            agent_lat_key, shape=(self.num_agents,), dtype=jnp.float32,
            minval=-0.5, maxval=0.5,
        )
        a_agent_speed = jr.uniform(
            agent_speed_key, shape=(self.num_agents,), dtype=jnp.float32,
            minval=60.0, maxval=90.0,
        )
        a_agent_theta = jnp.repeat(ROAD_THETAS_DEG[start_road_idx][None], self.num_agents, axis=0)
        start_dir = ROAD_DIRS[start_road_idx]
        start_normal = _right_normal(start_dir)
        a2_agent_xy = start_dir[None, :] * a_agent_r[:, None] + start_normal[None, :] * (
            lane_offset + a_agent_lat
        )[:, None]
        a_agent_vx, a_agent_vy = _heading_to_world_velocity(a_agent_speed, a_agent_theta)
        a_zeros = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        aS_agent_state = jnp.stack([
            a2_agent_xy[:, 0], a2_agent_xy[:, 1], a_agent_vx, a_agent_vy,
            a_agent_theta, a_zeros, a_zeros, a_zeros,
        ], axis=1)

        # 静态障碍车集中放在转向区域附近，而不是参考路径远端。
        # s≈ROAD_HALF 时处于路口中心附近，这里向前后各取一段范围。
        sobst_min_idx = min(max(0, int((ROAD_HALF - TURN_HALF - 10.0) / POINT_INTERVAL)), self.num_ref_points - 2)
        sobst_max_idx = min(max(sobst_min_idx + 1, int((ROAD_HALF + TURN_HALF + 20.0) / POINT_INTERVAL)),
                            self.num_ref_points - 1)
        sobst_idx = jr.randint(sobst_idx_key, shape=(), minval=sobst_min_idx, maxval=sobst_max_idx)
        S_sobst_ref = anS_goals[0, sobst_idx, :]
        sobst_offset = jr.uniform(sobst_pos_key, shape=(2,), dtype=jnp.float32, minval=-1.0, maxval=1.0)
        sobst_theta = S_sobst_ref[4] + jr.uniform(
            sobst_theta_key, shape=(), dtype=jnp.float32, minval=-5.0, maxval=5.0
        )
        S_sobst_state = jnp.stack([
            S_sobst_ref[0] + sobst_offset[0],
            S_sobst_ref[1] + sobst_offset[1],
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(0.0, dtype=jnp.float32),
            sobst_theta,
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(0.0, dtype=jnp.float32),
        ])

        S_mobst_state = self._make_dynamic_obstacle(
            (dyn_road_key, dyn_lane_key, dyn_speed_key, dyn_jitter_key),
            start_road_idx,
            lane_idx,
            lane_offset,
            a_agent_r[0],
            a_agent_speed[0],
            decel_len,
        )
        oS_obst_state = jnp.stack([S_sobst_state, S_mobst_state], axis=0)
        return aS_agent_state, oS_obst_state, anS_goals, an4_dsYddts


class IntersectionTurnPerpendicularDynamicScene(IntersectionTurnScene):
    """左转场景：agent 从西路左转，动态障碍车方向与 agent 初始道路方向垂直。

    图例：
    ==================================================
                 北路
                  |
                  ↑ reference / arc
    西路 ego ■ ---+
    -----------□□------------  dynamic obstacle
                  |
                  |
                 南路
    ==================================================
    """

    dynamic_mode = 0
    turn_sign = 1

    @property
    def name(self) -> str:
        return "intersection_left_turn_from_west_with_perpendicular_dynamic_obstacle"


class IntersectionTurnParallelDynamicScene(IntersectionTurnScene):
    """右转场景：agent 从西路右转，动态障碍车方向与 agent 初始道路方向平行。

    图例：
    ==================================================
                 北路
                  |
                  |
    西路 ego ■ ---+---------- 东路
                  |
                  ↓ reference / arc
                 南路
    ==================================================
    """

    dynamic_mode = 1
    turn_sign = -1

    @property
    def name(self) -> str:
        return "intersection_right_turn_from_west_with_parallel_dynamic_obstacle"


class IntersectionStraightPerpendicularDynamicScene(IntersectionSceneBase):
    """直行场景：agent 从西路直行穿过路口，动态障碍车从垂直道路驶入。

    图例：
    ==================================================
                 北路
                  |
                  |
    西路 ego ■ ----+-----> 东路 / reference path
                  |
                  ♦ static obstacle
                  |
                 南路
    ==================================================
    """

    @property
    def name(self) -> str:
        return "intersection_straight_with_perpendicular_dynamic_obstacle"

    def _make_dynamic_obstacle(self, keys, start_road_idx, lane_offset,
                               agent_r, agent_speed) -> Array:
        road_choice_key, lane_key, speed_key, jitter_key = keys
        side = jr.choice(road_choice_key, jnp.array([-1, 1], dtype=jnp.int32), shape=())
        obst_road_idx = (start_road_idx + side) % 4
        obst_lane_idx = jr.randint(lane_key, shape=(), minval=0, maxval=3)
        obst_lane_offset = LANE_CENTERS[obst_lane_idx]

        agent_direction = ROAD_DIRS[start_road_idx]
        agent_normal = _right_normal(agent_direction)
        obst_direction = ROAD_DIRS[obst_road_idx]
        obst_normal = _right_normal(obst_direction)
        theta_deg = ROAD_THETAS_DEG[obst_road_idx]
        speed = jr.uniform(speed_key, shape=(), dtype=jnp.float32, minval=20.0, maxval=100.0)

        # agent 从生成区到转向区入口的预计时间，用平均速度粗略估算。
        # 直行场景中低速区目标速度为 30~40km/h，这里沿用 35km/h 作为估计中值。
        dist_to_turn = -TURN_HALF - agent_r
        avg_speed_mps = ((agent_speed + 35.0) * 0.5) / 3.6
        t_to_turn = dist_to_turn / jnp.maximum(avg_speed_mps, 0.1)

        # 垂直道路与 agent 车道中心线的交点作为冲突点。
        # 例如 agent 从南路直行时，agent_normal * lane_offset 决定 x，
        # obstacle_normal * obst_lane_offset 决定 y，两者相加得到两条车道中心线的交点。
        conflict_xy = agent_normal * lane_offset + obst_normal * obst_lane_offset
        # 加大纵向扰动，使动态障碍车覆盖提前、滞后和无明显干扰的情况。
        longitudinal_jitter = jr.uniform(jitter_key, shape=(), dtype=jnp.float32, minval=-80.0, maxval=80.0)
        xy = conflict_xy - obst_direction * (speed / 3.6 * t_to_turn + longitudinal_jitter)
        return _state_xy_speed_theta(xy, speed, theta_deg)

    def make(self) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
        (start_road_key, lane_key, decel_key, gen_speed_key, crossing_speed_key,
         agent_r_key, agent_lat_key, agent_speed_key, agent_theta_key, sobst_idx_key,
         sobst_pos_key, sobst_theta_key, dyn_road_key, dyn_lane_key, dyn_speed_key,
         dyn_jitter_key) = jr.split(self.key, 16)
        del start_road_key, crossing_speed_key, agent_theta_key

        start_road_idx = jnp.array(3, dtype=jnp.int32)
        lane_idx = jr.randint(lane_key, shape=(), minval=0, maxval=3)
        lane_offset = LANE_CENTERS[lane_idx]

        # 减速区长度在场景生成时确定，范围为 40m 到 60m。
        decel_len = jr.uniform(decel_key, shape=(), dtype=jnp.float32, minval=40.0, maxval=60.0)
        gen_speed = jr.uniform(gen_speed_key, shape=(), dtype=jnp.float32, minval=60.0, maxval=90.0)
        crossing_speed = jnp.array(40.0, dtype=jnp.float32)
        anS_goals, an4_dsYddts = generate_straight_path_points(
            self.num_agents,
            self.num_ref_points,
            start_road_idx,
            lane_offset,
            decel_len,
            gen_speed,
            crossing_speed,
        )

        gen_zone_high = -TURN_HALF - decel_len
        a_agent_r = jr.uniform(
            agent_r_key, shape=(self.num_agents,), dtype=jnp.float32,
            minval=-ROAD_HALF, maxval=gen_zone_high,
        )
        a_agent_lat = jr.uniform(
            agent_lat_key, shape=(self.num_agents,), dtype=jnp.float32,
            minval=-0.5, maxval=0.5,
        )
        a_agent_speed = jr.uniform(
            agent_speed_key, shape=(self.num_agents,), dtype=jnp.float32,
            minval=60.0, maxval=90.0,
        )
        a_agent_theta = jnp.repeat(ROAD_THETAS_DEG[start_road_idx][None], self.num_agents, axis=0)
        start_dir = ROAD_DIRS[start_road_idx]
        start_normal = _right_normal(start_dir)
        a2_agent_xy = start_dir[None, :] * a_agent_r[:, None] + start_normal[None, :] * (
            lane_offset + a_agent_lat
        )[:, None]
        a_agent_vx, a_agent_vy = _heading_to_world_velocity(a_agent_speed, a_agent_theta)
        a_zeros = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        aS_agent_state = jnp.stack([
            a2_agent_xy[:, 0], a2_agent_xy[:, 1], a_agent_vx, a_agent_vy,
            a_agent_theta, a_zeros, a_zeros, a_zeros,
        ], axis=1)

        # 静态障碍车从参考路径上随机选点放置，更多集中在路口中心区域附近。
        sobst_min_idx = min(
            max(0, int((ROAD_HALF - TURN_HALF - 10.0) / POINT_INTERVAL)),
            self.num_ref_points - 2,
        )
        sobst_max_idx = min(
            max(sobst_min_idx + 1, int((ROAD_HALF + TURN_HALF + 10.0) / POINT_INTERVAL)),
            self.num_ref_points - 1,
        )
        sobst_idx = jr.randint(sobst_idx_key, shape=(), minval=sobst_min_idx, maxval=sobst_max_idx)
        S_sobst_ref = anS_goals[0, sobst_idx, :]
        sobst_offset = jr.uniform(sobst_pos_key, shape=(2,), dtype=jnp.float32, minval=-1.0, maxval=1.0)
        sobst_theta = S_sobst_ref[4] + jr.uniform(
            sobst_theta_key, shape=(), dtype=jnp.float32, minval=-5.0, maxval=5.0
        )
        S_sobst_state = jnp.stack([
            S_sobst_ref[0] + sobst_offset[0],
            S_sobst_ref[1] + sobst_offset[1],
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(0.0, dtype=jnp.float32),
            sobst_theta,
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(0.0, dtype=jnp.float32),
        ])

        S_mobst_state = self._make_dynamic_obstacle(
            (dyn_road_key, dyn_lane_key, dyn_speed_key, dyn_jitter_key),
            start_road_idx,
            lane_offset,
            a_agent_r[0],
            a_agent_speed[0],
        )
        oS_obst_state = jnp.stack([S_sobst_state, S_mobst_state], axis=0)
        return aS_agent_state, oS_obst_state, anS_goals, an4_dsYddts


def _fixed_agent_states(num_agents: int, start_road_idx: Array, longitudinal: Array,
                        lane_offset: Array, speed_kmph: Array, theta_delta_deg: Array = 0.0) -> Array:
    """生成 handmade 场景中固定初始位置和速度的 agent 状态。"""
    direction = ROAD_DIRS[start_road_idx]
    normal = _right_normal(direction)
    xy = direction * longitudinal + normal * lane_offset
    theta_deg = ROAD_THETAS_DEG[start_road_idx] + theta_delta_deg
    vx, vy = _heading_to_world_velocity(speed_kmph, theta_deg)
    oneS_agent_state = jnp.stack([
        xy[0],
        xy[1],
        vx,
        vy,
        jnp.asarray(theta_deg, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
    ])[None, :]
    return jnp.repeat(oneS_agent_state, num_agents, axis=0)


def _fixed_static_obstacle_from_ref(anS_goals: Array, ref_idx: int,
                                    xy_offset: Array, theta_delta_deg: Array = 0.0) -> Array:
    """从参考轨迹取一个点放置静态障碍车。"""
    idx = min(max(0, ref_idx), anS_goals.shape[1] - 1)
    S_ref = anS_goals[0, idx, :]
    return jnp.stack([
        S_ref[0] + xy_offset[0],
        S_ref[1] + xy_offset[1],
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        S_ref[4] + theta_delta_deg,
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
    ])


def _fixed_dynamic_obstacle_to_conflict(obst_road_idx: Array, obst_lane_offset: Array,
                                        speed_kmph: Array, conflict_xy: Array,
                                        t_to_conflict: Array,
                                        longitudinal_shift: Array = 0.0) -> Array:
    """按给定冲突点和预计到达时间，反推匀速动态障碍车的初始位置。"""
    direction = ROAD_DIRS[obst_road_idx]
    theta_deg = ROAD_THETAS_DEG[obst_road_idx]
    # obst_lane_offset 参数保留在接口中，便于调用处直接看出障碍车车道选择。
    _ = obst_lane_offset
    xy = conflict_xy - direction * (speed_kmph / 3.6 * t_to_conflict + longitudinal_shift)
    return _state_xy_speed_theta(xy, speed_kmph, theta_deg)


class HandMadeIntersectionLeftTurnWestNorthUFTSTC(IntersectionSceneBase):
    """确定左转场景：agent 从西路南侧第 0 车道左转到北路。

    agent 初始速度 80km/h；生成区/驶离区参考速度 90km/h，转向区 40km/h。
    静态障碍车固定在西路靠近转向区的过渡区，动态障碍车从东路北侧第 0 车道驶向西路。
    """

    @property
    def name(self) -> str:
        return "handmade_uftstc_left_turn_west_to_north"

    def make(self) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
        start_road_idx = jnp.array(3, dtype=jnp.int32)
        turn_sign = jnp.array(1, dtype=jnp.int32)
        lane_offset = jnp.array(3.0, dtype=jnp.float32)
        decel_len = jnp.array(50.0, dtype=jnp.float32)
        gen_speed = jnp.array(90.0, dtype=jnp.float32)
        turn_speed = jnp.array(40.0, dtype=jnp.float32)
        anS_goals, an4_dsYddts = generate_turn_path_points(
            self.num_agents, self.num_ref_points, start_road_idx, turn_sign,
            lane_offset, decel_len, gen_speed, turn_speed
        )

        agent_r = jnp.array(-90.0, dtype=jnp.float32)
        agent_speed = jnp.array(80.0, dtype=jnp.float32)
        aS_agent_state = _fixed_agent_states(self.num_agents, start_road_idx, agent_r, lane_offset, agent_speed)

        sobst_xy = _road_point(start_road_idx, jnp.array(-35.0, dtype=jnp.float32), lane_offset)
        S_sobst_state = _state_xy_speed_theta(
            sobst_xy,
            jnp.array(0.0, dtype=jnp.float32),
            ROAD_THETAS_DEG[start_road_idx],
        )

        avg_speed_mps = ((agent_speed + turn_speed) * 0.5) / 3.6
        t_to_turn = (-TURN_HALF - agent_r) / avg_speed_mps
        obst_road_idx = jnp.array(1, dtype=jnp.int32)
        obst_lane_offset = jnp.array(3.0, dtype=jnp.float32)
        conflict_xy = _road_point(obst_road_idx, jnp.array(0.0, dtype=jnp.float32), obst_lane_offset)
        S_mobst_state = _fixed_dynamic_obstacle_to_conflict(
            obst_road_idx,
            obst_lane_offset,
            jnp.array(60.0, dtype=jnp.float32),
            conflict_xy,
            t_to_turn,
        )
        oS_obst_state = jnp.stack([S_sobst_state, S_mobst_state], axis=0)
        return aS_agent_state, oS_obst_state, anS_goals, an4_dsYddts


class HandMadeIntersectionStraightWestEastUFTSTC(IntersectionSceneBase):
    """确定直行场景：agent 从西路南侧第 0 车道直行到东路。

    agent 初始速度 80km/h；生成区/驶离区参考速度 90km/h，路口区 40km/h。
    静态障碍车固定在西路靠近转向区的过渡区，动态障碍车从南路中间车道驶向北路。
    """

    @property
    def name(self) -> str:
        return "handmade_uftstc_straight_west_to_east"

    def make(self) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
        start_road_idx = jnp.array(3, dtype=jnp.int32)
        lane_offset = jnp.array(3.0, dtype=jnp.float32)
        decel_len = jnp.array(50.0, dtype=jnp.float32)
        gen_speed = jnp.array(90.0, dtype=jnp.float32)
        crossing_speed = jnp.array(40.0, dtype=jnp.float32)
        anS_goals, an4_dsYddts = generate_straight_path_points(
            self.num_agents, self.num_ref_points, start_road_idx, lane_offset,
            decel_len, gen_speed, crossing_speed
        )

        agent_r = jnp.array(-90.0, dtype=jnp.float32)
        agent_speed = jnp.array(80.0, dtype=jnp.float32)
        aS_agent_state = _fixed_agent_states(self.num_agents, start_road_idx, agent_r, lane_offset, agent_speed)

        sobst_xy = _road_point(start_road_idx, jnp.array(-35.0, dtype=jnp.float32), lane_offset)
        S_sobst_state = _state_xy_speed_theta(
            sobst_xy,
            jnp.array(0.0, dtype=jnp.float32),
            ROAD_THETAS_DEG[start_road_idx],
        )

        avg_speed_mps = ((agent_speed + crossing_speed) * 0.5) / 3.6
        t_to_turn = (-TURN_HALF - agent_r) / avg_speed_mps
        obst_road_idx = jnp.array(0, dtype=jnp.int32)
        obst_lane_offset = jnp.array(0.0, dtype=jnp.float32)
        conflict_xy = _road_point(obst_road_idx, jnp.array(0.0, dtype=jnp.float32), obst_lane_offset)
        S_mobst_state = _fixed_dynamic_obstacle_to_conflict(
            obst_road_idx,
            obst_lane_offset,
            jnp.array(60.0, dtype=jnp.float32),
            conflict_xy,
            t_to_turn,
        )
        oS_obst_state = jnp.stack([S_sobst_state, S_mobst_state], axis=0)
        return aS_agent_state, oS_obst_state, anS_goals, an4_dsYddts


def gen_scene_randomly(key: PRNGKey, num_agents: int, num_ref_points: int, xrange: Array, yrange: Array,
                       lane_width: float, lane_centers: Array) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
    choose_key, scene_key = jr.split(key, 2)
    scene_list = [
        IntersectionTurnPerpendicularDynamicScene(
            scene_key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers
        ).make,
        IntersectionTurnParallelDynamicScene(
            scene_key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers
        ).make,
        IntersectionStraightPerpendicularDynamicScene(
            scene_key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers
        ).make,
    ]
    choose_id = jr.choice(choose_key, len(scene_list))
    return jax.lax.switch(choose_id, scene_list)


def gen_handmade_scene(key: PRNGKey, num_agents: int, num_ref_points: int, xrange: Array, yrange: Array,
                       lane_width: float, lane_centers: Array) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
    choose_key, scene_key = jr.split(key, 2)
    scene_list = [
        HandMadeIntersectionLeftTurnWestNorthUFTSTC(
            scene_key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers
        ).make,
        HandMadeIntersectionStraightWestEastUFTSTC(
            scene_key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers
        ).make,
    ]
    choose_id = jr.choice(choose_key, len(scene_list))
    return jax.lax.switch(choose_id, scene_list)
