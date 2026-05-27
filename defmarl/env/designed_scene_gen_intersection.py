import jax
import jax.numpy as jnp
import jax.random as jr

from abc import ABC, abstractmethod
from typing import Tuple

from defmarl.utils.typing import PRNGKey, AgentState, ObstState, Array, PathRefs


ROAD_HALF = 100.0
TURN_HALF = 14.5
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


def _road_point(road_idx: Array, longitudinal: Array, lane_offset: Array) -> Array:
    """根据道路编号、纵向坐标和车道横向偏移生成世界坐标系下的点。"""
    direction = ROAD_DIRS[road_idx]
    normal = _right_normal(direction)
    return direction * longitudinal + normal * lane_offset


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
    start_dir = ROAD_DIRS[start_road_idx]
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

    # 起始道路：从区域边界 -100m 沿道路前进到转向区边界 -14.5m。
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

    # 左转/右转后的终止道路编号。
    # 这里道路编号按“南、东、北、西”逆时针排列，因此左转是 road_idx - 1，右转是 road_idx + 1。
    terminal_road_idx = (start_road_idx - turn_sign.astype(jnp.int32)) % 4
    terminal_longitudinal = TURN_HALF + exit_s
    exit_xy = _road_point(terminal_road_idx, terminal_longitudinal, lane_offset)

    in_approach = s_path < arc_start
    in_arc = (s_path >= arc_start) & (s_path < exit_start)
    xs = jnp.where(in_approach, approach_xy[:, 0], jnp.where(in_arc, arc_xy[:, 0], exit_xy[:, 0]))
    ys = jnp.where(in_approach, approach_xy[:, 1], jnp.where(in_arc, arc_xy[:, 1], exit_xy[:, 1]))

    terminal_theta_deg = theta0_deg + turn_sign * 90.0
    terminal_theta_deg = (terminal_theta_deg + 180.0) % 360.0 - 180.0
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

    # 兼容原横向控制接口。交叉口中这里记录世界 y 方向的参考量及其近似导数。
    vy_mps = vy_kmph / 3.6
    ay = jnp.where(in_arc, turn_sign * speed_mps ** 2 * jnp.cos(theta_arc_rad) / turn_radius, 0.0)
    jy = jnp.zeros_like(ay)
    one4_dsYddts = jnp.stack([ys, vy_mps, ay, jy], axis=1)[None, :, :]
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

    只处理转向，不处理直行。agent 从随机起始道路的生成区出发，沿同一车道号左转或右转。
    目标轨迹在道路区域沿车道中心，在转向区以圆弧连接，速度由生成区高速逐渐降至转向区低速，
    再在终止道路减速区逐渐升回生成区速度。
    """

    dynamic_mode = 0

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
        # 平行动动态障碍物：只允许选 agent 初始车道以外的两条车道。
        # lane_idx 是 agent 初始车道，randint(0, 2) 只会生成 0 或 1；
        # 因此 other_lane_idx 只可能是 lane_idx + 1 或 lane_idx + 2，再对 3 取模，
        # 结果一定不等于 lane_idx。
        # 举例：agent 在 0 号车道，则障碍车只能在 1 或 2 号车道。
        other_lane_idx = (lane_idx + 1 + jr.randint(lane_key, shape=(), minval=0, maxval=2)) % 3
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

        # 垂直模式的冲突点取障碍车道路穿过路口中心附近的位置；
        # 平行模式的冲突点取 agent 起始道路进入转向区的位置。
        conflict_xy = jnp.where(
            mode == 0,
            _road_point(obst_road_idx, 0.0, obst_lane_offset),
            _road_point(start_road_idx, -TURN_HALF, lane_offset),
        )
        longitudinal_jitter = jr.uniform(jitter_key, shape=(), dtype=jnp.float32, minval=-10.0, maxval=10.0)
        xy = conflict_xy - direction * (speed / 3.6 * t_to_turn + longitudinal_jitter)
        xy = jnp.clip(xy, -ROAD_HALF + 2.0, ROAD_HALF - 2.0)
        return _state_xy_speed_theta(xy, speed, theta_deg)

    def make(self) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
        (start_road_key, turn_key, lane_key, decel_key, gen_speed_key, turn_speed_key,
         agent_r_key, agent_lat_key, agent_speed_key, agent_theta_key, sobst_idx_key,
         sobst_pos_key, sobst_theta_key, dyn_road_key, dyn_lane_key, dyn_speed_key,
         dyn_jitter_key) = jr.split(self.key, 17)

        start_road_idx = jr.randint(start_road_key, shape=(), minval=0, maxval=4)
        turn_sign = jr.choice(turn_key, jnp.array([-1, 1], dtype=jnp.int32), shape=())
        lane_idx = jr.randint(lane_key, shape=(), minval=0, maxval=3)
        lane_offset = LANE_CENTERS[lane_idx]

        # 减速区长度在场景生成时确定，范围为 40m 到 60m。
        decel_len = jr.uniform(decel_key, shape=(), dtype=jnp.float32, minval=40.0, maxval=60.0)
        gen_speed = jr.uniform(gen_speed_key, shape=(), dtype=jnp.float32, minval=60.0, maxval=90.0)
        turn_speed = jr.uniform(turn_speed_key, shape=(), dtype=jnp.float32, minval=30.0, maxval=40.0)
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
        a_agent_theta = ROAD_THETAS_DEG[start_road_idx] + jr.uniform(
            agent_theta_key, shape=(self.num_agents,), dtype=jnp.float32,
            minval=-5.0, maxval=5.0,
        )
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

        # 静态障碍车从参考路径后段随机取一点放置，避开 agent 生成区。
        # 速度设为 0，朝向沿参考点方向再加少量扰动。
        sobst_min_idx = min(900, max(0, self.num_ref_points // 4))
        sobst_idx = jr.randint(sobst_idx_key, shape=(), minval=sobst_min_idx, maxval=self.num_ref_points - 1)
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
    """转向场景一：动态障碍车方向与 agent 初始道路方向垂直。

    图例以 agent 从南路左转到西路为例，实际生成时起始道路、左右转和车道均随机：
    ==================================================
                 北路
                  |
                  |
    西路 <----- arc/reference
    -----------□□------------  dynamic obstacle
                  |
                  |
                ego ■
                 南路
    ==================================================
    """

    dynamic_mode = 0

    @property
    def name(self) -> str:
        return "intersection_turn_with_perpendicular_dynamic_obstacle"


class IntersectionTurnParallelDynamicScene(IntersectionTurnScene):
    """转向场景二：动态障碍车方向与 agent 初始道路方向平行。

    图例以 agent 从南路右转到东路为例，动态障碍车在非 agent 初始车道直行穿过路口：
    ==================================================
                 北路
                  □  dynamic obstacle
                  |
                  |
    西路 ----------+---------- 东路 / reference
                  |
                  |
             ego ■
                 南路
    ==================================================
    """

    dynamic_mode = 1

    @property
    def name(self) -> str:
        return "intersection_turn_with_parallel_dynamic_obstacle"


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
    ]
    choose_id = jr.choice(choose_key, len(scene_list))
    return jax.lax.switch(choose_id, scene_list)


def gen_handmade_scene(key: PRNGKey, num_agents: int, num_ref_points: int, xrange: Array, yrange: Array,
                       lane_width: float, lane_centers: Array) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
    return IntersectionTurnPerpendicularDynamicScene(
        key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers
    ).make()
