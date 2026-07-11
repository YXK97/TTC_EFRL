import jax
import jax.numpy as jnp
import jax.random as jr

from typing import Tuple

from defmarl.env.designed_scene_gen_two_lane import (
    LaneChangeANDOvertakeScene,
    generate_horizontal_path_points,
    generate_lanechange_path_points,
    make_state,
)
from defmarl.utils.typing import AgentState, Array, ObstState, PathRefs, PRNGKey


def _repeat_agent_state(
    num_agents: int,
    agent_x: Array,
    agent_y: Array,
    agent_theta: Array,
    agent_v: Array,
    agent_y_key: PRNGKey,
    agent_v_key: PRNGKey,
    y_noise: float = 0.1,
) -> AgentState:
    a_agent_x = jnp.repeat(agent_x[None], num_agents, axis=0)
    a_agent_y = jnp.repeat(agent_y[None], num_agents, axis=0) + jr.uniform(
        agent_y_key, shape=(num_agents,), dtype=jnp.float32, minval=-y_noise, maxval=y_noise
    )
    a_agent_hx = jnp.repeat(jnp.cos(agent_theta)[None], num_agents, axis=0)
    a_agent_hy = jnp.repeat(jnp.sin(agent_theta)[None], num_agents, axis=0)
    a_agent_v = jnp.clip(
        agent_v + jr.uniform(agent_v_key, shape=(num_agents,), dtype=jnp.float32, minval=-0.5, maxval=0.5),
        5.0,
        40.0 / 3.6,
    )
    a_agent_delta = jnp.zeros((num_agents,), dtype=jnp.float32)
    return jnp.stack([a_agent_x, a_agent_y, a_agent_hx, a_agent_hy, a_agent_v, a_agent_delta], axis=1)


def _sample_static_obstacle_x(key: PRNGKey, xrange: Array) -> Array:
    xs_min = xrange[0] + 35.0
    xs_max = xrange[1] - 35.0
    return jr.uniform(key, shape=(), dtype=jnp.float32, minval=xs_min, maxval=xs_max)


def _sample_static_obstacle_y(key: PRNGKey, lane_centers: Array) -> Tuple[Array, Array]:
    sobst_lane = jr.choice(key, jnp.array([0, 1], dtype=jnp.int32), shape=())
    sobst_y = lane_centers[sobst_lane]
    side_y = lane_centers[1 - sobst_lane]
    return sobst_y, side_y


def _sample_between_lanes(key: PRNGKey, lane_centers: Array) -> Array:
    y_low = jnp.minimum(lane_centers[0], lane_centers[1])
    y_high = jnp.maximum(lane_centers[0], lane_centers[1])
    return jr.uniform(key, shape=(), dtype=jnp.float32, minval=y_low, maxval=y_high)


class SplitLaneChangeMiddleStaticEdgeFastMoving2laneBase(LaneChangeANDOvertakeScene):
    """LaneChange split scene base. Concrete subclasses only decide ego phase."""

    @property
    def name(self) -> str:
        return "lanechange_scene_with_middle_static_obstacle_and_edge_2_lane_split"

    @property
    def num_lanes(self) -> int:
        assert self.lane_centers.shape[0] == 2, "本场景支持双车道配置！"
        return 2

    @property
    def num_moving_obsts(self) -> int:
        return 0

    @property
    def num_static_obsts(self) -> int:
        return 1

    def sample_agent_pose(self, key: PRNGKey, xs: Array, ys: Array, side_y: Array, start_y: Array, terminal_y: Array) -> Tuple[Array, Array]:
        raise NotImplementedError

    def make(self) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
        start_x_key, terminal_x_key, start_y_key, start_terminal_v_key, agent_pose_key, agent_y_key, agent_v_key, \
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
        start_theta = terminal_theta = jnp.array([0.0])[0]
        start_v = terminal_v = jr.uniform(start_terminal_v_key, shape=(), dtype=jnp.float32,
                                          minval=20, maxval=40) / 3.6
        S_start_state = make_state(start_x, start_y, start_theta, start_v)
        S_terminal_state = make_state(terminal_x, terminal_y, terminal_theta, terminal_v)
        anS_goals, an4_dsYddts = generate_lanechange_path_points(self.xrange, self.num_agents, self.num_ref_points,
                                                                 S_start_state, S_terminal_state)

        # 生成静态障碍物，x坐标位于[xs_min, xs_max]区间随机，y坐标随机选择车道1/2，航向角可选-180°到180°之间
        sobst_x = _sample_static_obstacle_x(sobst_x_key, self.xrange)
        sobst_y, side_y = _sample_static_obstacle_y(sobst_y_key, self.lane_centers)
        sobst_theta = jr.uniform(sobst_theta_key, shape=(), dtype=jnp.float32, minval=0, maxval=0)
        S_sobst_state = make_state(sobst_x, sobst_y, sobst_theta, jnp.array(0.0))
        oS_obst_state = S_sobst_state[None]

        # 生成初始agent，x坐标都一样，y坐标可略有上下波动，vx可不一样，其它的都是0
        agent_x, agent_y = self.sample_agent_pose(agent_pose_key, sobst_x, sobst_y, side_y, start_y, terminal_y)
        aS_agent_state = _repeat_agent_state(self.num_agents, agent_x, agent_y, start_theta, start_v,
                                             agent_y_key, agent_v_key)
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


class LaneChangeMiddleStaticEdgeFastMoving2lane_Approach(SplitLaneChangeMiddleStaticEdgeFastMoving2laneBase):
    def sample_agent_pose(self, key: PRNGKey, xs: Array, ys: Array, side_y: Array, start_y: Array, terminal_y: Array) -> Tuple[Array, Array]:
        del ys, side_y, terminal_y
        agent_x = xs - jr.uniform(key, shape=(), dtype=jnp.float32, minval=18.0, maxval=32.0)
        return agent_x, start_y


class LaneChangeMiddleStaticEdgeFastMoving2lane_Side(SplitLaneChangeMiddleStaticEdgeFastMoving2laneBase):
    def sample_agent_pose(self, key: PRNGKey, xs: Array, ys: Array, side_y: Array, start_y: Array, terminal_y: Array) -> Tuple[Array, Array]:
        del ys, start_y, terminal_y
        agent_x = xs + jr.uniform(key, shape=(), dtype=jnp.float32, minval=-4.0, maxval=4.0)
        return agent_x, side_y


class LaneChangeMiddleStaticEdgeFastMoving2lane_Passed(SplitLaneChangeMiddleStaticEdgeFastMoving2laneBase):
    def sample_agent_pose(self, key: PRNGKey, xs: Array, ys: Array, side_y: Array, start_y: Array, terminal_y: Array) -> Tuple[Array, Array]:
        del ys, side_y, start_y, terminal_y
        x_key, y_key = jr.split(key, 2)
        agent_x = xs + jr.uniform(x_key, shape=(), dtype=jnp.float32, minval=8.0, maxval=18.0)
        agent_y = _sample_between_lanes(y_key, self.lane_centers)
        return agent_x, agent_y


class LaneChangeMiddleStaticEdgeFastMoving2lane_Done(SplitLaneChangeMiddleStaticEdgeFastMoving2laneBase):
    def sample_agent_pose(self, key: PRNGKey, xs: Array, ys: Array, side_y: Array, start_y: Array, terminal_y: Array) -> Tuple[Array, Array]:
        del ys, side_y, start_y
        agent_x = xs + jr.uniform(key, shape=(), dtype=jnp.float32, minval=18.0, maxval=32.0)
        return agent_x, terminal_y


class SplitOvertakeEdgeStaticMiddle2laneBase(LaneChangeANDOvertakeScene):
    """Overtake split scene base. Concrete subclasses only decide ego phase."""

    @property
    def name(self) -> str:
        return "overtake_scene_with_edge_static_obstacle_and_middle_fast_moving_obstacle_split"

    @property
    def num_lanes(self) -> int:
        assert self.lane_centers.shape[0] == 2, "本场景仅支持双车道配置！"
        return 2

    @property
    def num_moving_obsts(self) -> int:
        return 0

    @property
    def num_static_obsts(self) -> int:
        return 1

    def sample_agent_pose(self, key: PRNGKey, xs: Array, ys: Array, side_y: Array, start_y: Array, terminal_y: Array) -> Tuple[Array, Array]:
        raise NotImplementedError

    def make(self) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
        start_x_key, terminal_x_key, start_y_key, start_terminal_v_key, agent_pose_key, agent_y_key, agent_v_key, \
            sobst_x_key, sobst_y_key, sobst_theta_key, mobst_x_key, mobst_y_key, mobst_vx_key = jr.split(self.key, 13)

        # 生成轨迹
        start_x = jr.uniform(start_x_key, shape=(), dtype=jnp.float32, minval=self.xrange[0],
                             maxval=(self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0])
        terminal_x = jr.uniform(terminal_x_key, shape=(), dtype=jnp.float32,
                                minval=2 * (self.xrange[1] - self.xrange[0]) / 3 + self.xrange[0],
                                maxval=self.xrange[1])
        del terminal_x
        start_theta = terminal_theta = jnp.array([0.0])[0]
        del terminal_theta
        start_y = terminal_y = jr.choice(start_y_key, self.lane_centers[jnp.array([0, 1])], shape=())
        terminal_v = jr.uniform(start_terminal_v_key, shape=(), dtype=jnp.float32,
                                minval=20, maxval=40) / 3.6
        anS_goals, an4_dsYddts = generate_horizontal_path_points(self.xrange, self.num_agents, self.num_ref_points,
                                                                 start_y, terminal_v)

        # 生成静态障碍物，位于[xs_min, xs_max]区间随机，y坐标随机选择车道1/2，航向角可选-180°到180°之间
        sobst_x = _sample_static_obstacle_x(sobst_x_key, self.xrange)
        sobst_y, side_y = _sample_static_obstacle_y(sobst_y_key, self.lane_centers)
        sobst_theta = jr.uniform(sobst_theta_key, shape=(), dtype=jnp.float32, minval=0, maxval=0)
        S_sobst_state = make_state(sobst_x, sobst_y, sobst_theta, jnp.array(0.0))
        oS_obst_state = S_sobst_state[None]

        # 生成初始agent，x坐标都一样，y坐标可略有上下波动，vx可不一样，其它的都是0
        agent_x, agent_y = self.sample_agent_pose(agent_pose_key, sobst_x, sobst_y, side_y, start_y, terminal_y)
        aS_agent_state = _repeat_agent_state(self.num_agents, agent_x, agent_y, start_theta, terminal_v,
                                             agent_y_key, agent_v_key)
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


class OvertakeEdgeStaticMiddle2lane_Approach(SplitOvertakeEdgeStaticMiddle2laneBase):
    def sample_agent_pose(self, key: PRNGKey, xs: Array, ys: Array, side_y: Array, start_y: Array, terminal_y: Array) -> Tuple[Array, Array]:
        del ys, side_y, terminal_y
        agent_x = xs - jr.uniform(key, shape=(), dtype=jnp.float32, minval=18.0, maxval=32.0)
        return agent_x, start_y


class OvertakeEdgeStaticMiddle2lane_Side(SplitOvertakeEdgeStaticMiddle2laneBase):
    def sample_agent_pose(self, key: PRNGKey, xs: Array, ys: Array, side_y: Array, start_y: Array, terminal_y: Array) -> Tuple[Array, Array]:
        del ys, start_y, terminal_y
        agent_x = xs + jr.uniform(key, shape=(), dtype=jnp.float32, minval=-4.0, maxval=4.0)
        return agent_x, side_y


class OvertakeEdgeStaticMiddle2lane_Passed(SplitOvertakeEdgeStaticMiddle2laneBase):
    def sample_agent_pose(self, key: PRNGKey, xs: Array, ys: Array, side_y: Array, start_y: Array, terminal_y: Array) -> Tuple[Array, Array]:
        del ys, side_y, start_y, terminal_y
        x_key, y_key = jr.split(key, 2)
        agent_x = xs + jr.uniform(x_key, shape=(), dtype=jnp.float32, minval=8.0, maxval=18.0)
        agent_y = _sample_between_lanes(y_key, self.lane_centers)
        return agent_x, agent_y


class OvertakeEdgeStaticMiddle2lane_Done(SplitOvertakeEdgeStaticMiddle2laneBase):
    def sample_agent_pose(self, key: PRNGKey, xs: Array, ys: Array, side_y: Array, start_y: Array, terminal_y: Array) -> Tuple[Array, Array]:
        del ys, side_y, start_y
        agent_x = xs + jr.uniform(key, shape=(), dtype=jnp.float32, minval=18.0, maxval=32.0)
        return agent_x, terminal_y


def gen_scene_randomly(key: PRNGKey, num_agents: int, num_ref_points: int, xrange: Array, yrange: Array,
                       lane_width: float, lane_centers: Array) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
    choose_key, scene_key = jr.split(key, 2)
    scene_list = [
        LaneChangeMiddleStaticEdgeFastMoving2lane_Approach(scene_key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers).make,
        LaneChangeMiddleStaticEdgeFastMoving2lane_Side(scene_key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers).make,
        LaneChangeMiddleStaticEdgeFastMoving2lane_Passed(scene_key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers).make,
        LaneChangeMiddleStaticEdgeFastMoving2lane_Done(scene_key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers).make,
        OvertakeEdgeStaticMiddle2lane_Approach(scene_key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers).make,
        OvertakeEdgeStaticMiddle2lane_Side(scene_key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers).make,
        OvertakeEdgeStaticMiddle2lane_Passed(scene_key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers).make,
        OvertakeEdgeStaticMiddle2lane_Done(scene_key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers).make,
    ]
    probs = jnp.array([0.175, 0.150, 0.125, 0.050, 0.175, 0.150, 0.125, 0.050])
    choose_id = jr.choice(choose_key, len(scene_list), p=probs)
    aS_agent_state, oS_obst_state, anS_goals, an4_dsYddts = jax.lax.switch(choose_id, scene_list)

    return aS_agent_state, oS_obst_state, anS_goals, an4_dsYddts


def gen_scene_randomly_split(key: PRNGKey, num_agents: int, num_ref_points: int, xrange: Array, yrange: Array,
                             lane_width: float, lane_centers: Array) -> Tuple[AgentState, ObstState, PathRefs, jnp.ndarray]:
    return gen_scene_randomly(key, num_agents, num_ref_points, xrange, yrange, lane_width, lane_centers)
