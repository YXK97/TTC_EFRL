import pathlib
import jax
import jax.random as jr
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
from .designed_scene_gen_delta_state import gen_scene_randomly
from .utils import process_lane_centers, process_lane_marks, relative_state_delta_state
from defmarl.trainer.data import Rollout_NormedGraph
from defmarl.utils.graph import EdgeBlock, GetGraph, GraphsTuple
from defmarl.utils.typing import Action, Reward, Cost, Array, State, AgentState, ObstState, Done, Info
from defmarl.utils.utils import tree_index, MutablePatchCollection, save_anim, calc_2d_rot_matrix, \
    find_closest_goal_indices, gen_i_j_pairs, gen_i_j_pairs_no_identical, normalize_angle
from ..utils.scaling_delta_state import scaling_calc, scaling_calc_bound

INF = jnp.inf

class MVENormedGraph_CBF(MVE):
    """该任务使用agent位姿和预设轨迹的偏移量、加减速度和方向盘转角的大小作为的reward的度量，
    scaling factor作为cost的度量，每个agent分配一个goal并规划出一条轨迹（五次多项式），
    环境为四车道，障碍车均沿车道作匀速直线运动"""

    PARAMS = {
        # 宝骏E300参数，只有bb和m是准的，其它的都是估计的
        "ego_lf": 0.8475, # m，假设质心位于几何中心
        "ego_lr": 0.9025, # m，假设质心位于几何中心
        "ego_bb_size": jnp.array([2.625, 1.647]), # bounding box的[width, height] m
        "ego_m": 940., # kg
        "ego_Iz": 752.25333, # kg*m^2，假设质心位于几何中心
        "ego_Cf": 47850., # N/rad
        "ego_Cr": 46510., # N/rad00
        "comm_radius": 100,
        "obst_bb_size": jnp.array([4., 2.]), # bounding box的[width, height] m

        # [x_l, x_h, y_l, y_h, vx_b_l, vx_b_h, vy_b_l, vy_b_h, θ_l, θ_h, dθdt_l, dθdt_h, \
        # delta_r_l, delta_r_h, bbw_l, bbw_h, bbh_l, bbh_h
        # 单位：x,y,bbw,bbh: m  vx_b,vy_b: km/h,  θ: °, dθdt: °/s,  δ_r: °
        # 这里速度为车身坐标系下的速度
        "default_state_range": jnp.array([-100., 100., -4.5, 4.5, -INF, INF, -INF, INF, -180., 180., -INF, INF,
        -3., 3., -INF, INF, -INF, INF]), # 默认范围，用于指示正常工作的状态范围
        "rollout_state_range": jnp.array([-200., 150., -10., 10., 30., 100., -INF, INF, -180., 180., -INF, INF,
        -3., 3., -INF, INF, -INF, INF]), # rollout过程中的限制，强制约束
        "agent_init_state_range": jnp.array([-100., -50., -3., 3., -INF, INF, -INF, INF, -180., 180., -INF, INF,
        0., 0., -INF, INF, -INF, INF]), # 用于agent初始化的状态范围
        "terminal_state_range": jnp.array([50., 100., -3., 3., -INF, INF, -INF, INF, -180., 180., -INF, INF,
        0., 0., -INF, INF, -INF, INF]), # 随机生成terminal时的状态范围

        "lane_width": 3, # 车道宽度，m
        "v_bias": 5, # 可允许的速度偏移量
        "alpha_thresh": 0.01,
        "gamma1": 3.,
        "gamma2": 3.
    }
    PARAMS.update({
        "ego_radius": jnp.linalg.norm(PARAMS["ego_bb_size"]/2), # m
        "ego_L": PARAMS["ego_lf"]+PARAMS["ego_lr"], # m
        "lane_centers": process_lane_centers(PARAMS["default_state_range"][2:4], PARAMS["lane_width"]), # 车道中心线y坐标 m
    })
    if "obst_bb_size" in PARAMS and PARAMS["obst_bb_size"].shape == (2,):
        PARAMS.update({"obst_radius": jnp.linalg.norm(PARAMS["obst_bb_size"]/2)})
    # PARAMS.update({"n_obsts": PARAMS["lane_centers"].shape[0]}) # 本环境每根车道一辆障碍车
    assert PARAMS["terminal_state_range"][0] - PARAMS["agent_init_state_range"][1] >= 100

    def __init__(self,
                 num_agents: int,
                 area_size: Optional[float] = None,
                 max_step: int = 256,
                 max_travel: Optional[float] = None,
                 dt: float = 0.05,
                 reward_min: float = -17.,
                 reward_max: float = 0.5,
                 params: dict = None
                 ):
        area_size = MVENormedGraph_CBF.PARAMS["rollout_state_range"][:4] if area_size is None else area_size
        params = MVENormedGraph_CBF.PARAMS if params is None else params
        super(MVENormedGraph_CBF, self).__init__(num_agents, area_size, max_step, max_travel, dt, reward_min, reward_max, params)
        self.all_goals = jnp.zeros((num_agents, self.num_goals, self.state_dim))  # 参考点初始化
        self.all_dsYddts = jnp.zeros((num_agents, self.num_goals, 4)) # 轨迹的y方向偏移量与偏移量导数初始化
        self.num_obsts = 0 # 初始化

    @override
    @property
    def state_dim(self) -> int:
        return 9 # x y vx_b vy_b θ dθ/dt δ_r bw bh, 除了速度和前轮转角是车身坐标系下的，其余均为世界坐标系下的值

    @override
    @property
    def node_dim(self) -> int:
        return 12  # state_dim(9)  indicator(3): agent: 001, goal: 010, obstacle: 100, pad: 00-1

    @override
    @property
    def edge_dim(self) -> int:
        return 9 # Δstate: Δx, Δy, Δvx, Δvy, Δθ, Δdθ/dt, Δδ_r, Δbw, Δbh

    @override
    @property
    def action_dim(self) -> int:
        return 2  # a：车辆纵向加速度（m/s^2） δ：前轮转角（逆时针为正，°）

    @override
    @property
    def n_cost(self) -> int:
        return 4 # agent间碰撞(1) + agent-obstacle碰撞(1) + agent超出y轴范围(高+低，2)

    @override
    @property
    def cost_components(self) -> Tuple[str, ...]:
        return "agent collisions", "obs collisions", "bound exceeds y low", "bound exceeds y high"

    @property
    def num_goals(self) -> int:
        return 4800 # 每个agent参考轨迹点的数量

    @override
    def reset(self, key: Array) -> Tuple[GraphsTuple, GraphsTuple]:
        """使用场景类别生成函数进行agent、goal和obstacle的生成"""
        c_ycs = self.params["lane_centers"]
        xrange = self.params["default_state_range"][:2]
        yrange = self.params["default_state_range"][2:4]
        lanewidth = self.params["lane_width"]
        agents, obsts, all_goals, _ = gen_scene_randomly(key, self.num_agents, self.num_goals, xrange, yrange, lanewidth, c_ycs)
        # agents, obsts, all_goals, _ = gen_handmade_scene(key, self.num_agents, self.num_goals, xrange, yrange, lanewidth, c_ycs)
        self.all_goals = all_goals
        goals_init_indices = find_closest_goal_indices(agents, all_goals)
        agents_indices = jnp.arange(agents.shape[0])
        goals = all_goals[agents_indices, goals_init_indices, :]
        env_state = MVEEnvState(agents, goals, obsts)

        self.num_obsts = obsts.shape[0]

        graph, normed_graph = self.get_graph(env_state)
        return graph, normed_graph

    @override
    def agent_step_euler(self, aS_agent_states: AgentState, ad_action: Action) -> AgentState:
        """对agent，使用3-DOF自行车动力学模型"""
        assert ad_action.shape == (self.num_agents, self.action_dim)
        assert aS_agent_states.shape == (self.num_agents, self.state_dim)
        convert_vec_s = jnp.array([1, 1, 3.6, 3.6, 180/jnp.pi, 180/jnp.pi, 180/jnp.pi]) # eg. km/h / convert_vec -> m/s
        convert_vec_a = jnp.array([1, 180/jnp.pi]) # m/s²不变，° / convert_vec_a -> rad

        def delta_low_pass_filter(delta_desire, delta_real, gamma=0.2) -> float:
            # delta_desire: 目标前轮偏角 (rad)
            # delta_real: 上一帧的实际前轮偏角 (rad)
            # gamma: 滤波系数，默认 0.2
            delta_real_new = gamma * delta_desire + (1.0 - gamma) * delta_real
            return delta_real_new

        def delta_rate_limiter(delta_desire, delta_real, max_steer_rate=0.5) -> float:
            # delta_desire: 目标前轮偏角 (rad)
            # delta_real: 上一帧的实际前轮偏角 (rad)
            # max_steer_rate: 前轮最大转向角速度 (rad/s)，默认 0.5 rad/s (约 28 deg/s)
            # 输出物理限制下的下一时刻前轮转角 (rad)
            max_change = max_steer_rate * self.dt
            desired_change = delta_desire - delta_real
            actual_change = jnp.clip(desired_change, -max_change, max_change)
            delta_real_new = delta_real + actual_change
            return delta_real_new


        # 参数提取
        as_S = aS_agent_states[:, :-2] # x, y, vx_b, vy_b, θ, dθ/dt, δ_r
        as_S_metric = as_S / convert_vec_s # km/h->m/s, degree->rad, degree/s->rad/s
        ad_action_metric = ad_action / convert_vec_a
        a_delta_metric = ad_action_metric[:, 1]
        a_vx_b_metric = as_S_metric[:, 2] # m/s
        a_vy_b_metric = as_S_metric[:, 3] # m/s
        a_theta_metric = as_S_metric[:, 4] # rad
        a_dthetadt_metric = as_S_metric[:, 5] # rad/s

        a_delta_r_metric = as_S_metric[:, 6] # rad
        a_delta_smoothed_metric = jax.vmap(delta_low_pass_filter, in_axes=(0, 0))(a_delta_metric, a_delta_r_metric)
        a_delta_r_new_metric = jax.vmap(delta_rate_limiter, in_axes=(0, 0))(a_delta_smoothed_metric, a_delta_r_metric)
        ad_action_r_metric = ad_action_metric.at[:, 1].set(a_delta_r_new_metric)
        a_delta_r_new = a_delta_r_new_metric * 180/jnp.pi

        a_ones = jnp.ones((self.num_agents,), dtype=jnp.float32)
        m = self.params["ego_m"] # kg
        lf = self.params["ego_lf"] # m
        lr = self.params["ego_lr"] # m
        Iz = self.params["ego_Iz"] # kg*m^2
        Cf = 2*self.params["ego_Cf"] # N/rad，自行车模型需要将轮胎的侧偏刚度×2以代表轴刚度
        Cr = 2*self.params["ego_Cr"] # N/rad

        # 车辆3自由度control affine(小转向角近似)动力学模型 状态更新
        as_f = jnp.stack([a_vx_b_metric * jnp.cos(a_theta_metric) - a_vy_b_metric * jnp.sin(a_theta_metric),
                          a_vx_b_metric * jnp.sin(a_theta_metric) + a_vy_b_metric * jnp.cos(a_theta_metric),
                          a_vy_b_metric * a_dthetadt_metric,
                          -a_vx_b_metric * a_dthetadt_metric - (Cf + Cr) * a_vy_b_metric / (m * a_vx_b_metric) + \
                          (Cr * lr - Cf * lf) * a_dthetadt_metric / (m * a_vx_b_metric),
                          a_dthetadt_metric,
                          (Cr * lr - Cf * lf) * a_vy_b_metric / (Iz * a_vx_b_metric) - \
                          (Cf * (lf ** 2) + Cr * (lr ** 2)) * a_dthetadt_metric / (Iz * a_vx_b_metric)], axis=1)
        asd_g = jnp.zeros((self.num_agents, 6, self.action_dim), dtype=jnp.float32)
        asd_g = asd_g.at[:, 2, 0].set(a_ones)
        asd_g = asd_g.at[:, 2, 1].set(Cf*(a_vy_b_metric+lf*a_dthetadt_metric)/(m*a_vx_b_metric))
        asd_g = asd_g.at[:, 3, 1].set(a_ones*Cf/m)
        asd_g = asd_g.at[:, 5, 1].set(a_ones*Cf*lf/Iz)
        as_dS_metric = (as_f + jnp.einsum('asd, ad -> as', asd_g, ad_action_r_metric)) * self.dt
        asm1_S_new_unclip_metric_no_delta_r = as_S_metric[:, :6] + as_dS_metric
        as_S_new_unclip_metric = jnp.concatenate([asm1_S_new_unclip_metric_no_delta_r, a_delta_r_new_metric[:, None]], axis=1)
        as_S_new_unclip = as_S_new_unclip_metric * convert_vec_s # 公制单位转换为任务单位
        as_S_new_unclip = as_S_new_unclip.at[:, 4].set(normalize_angle(as_S_new_unclip[:, 4]))  # θ限制在[-180, 180]°
        aS_S_new_unclip = aS_agent_states.at[:, :-2].set(as_S_new_unclip)
        assert aS_S_new_unclip.shape == (self.num_agents, self.state_dim)
        aS_S_new = self.clip_state(aS_S_new_unclip)
        return aS_S_new

    def obst_step_euler(self, o_obst_states: ObstState) -> ObstState:
        """障碍车作匀速直线运动"""
        num_obsts = o_obst_states.shape[0]
        assert o_obst_states.shape == (num_obsts, self.state_dim)

        # 匀速直线运动模型
        o_x = o_obst_states[:, 0]
        o_y = o_obst_states[:, 1]
        o_vx_b = o_obst_states[:, 2] / 3.6
        o_vy_b = o_obst_states[:, 3] / 3.6
        o_theta = o_obst_states[:, 4] * jnp.pi / 180
        o_obst_states_new_no_y = o_obst_states.at[:, 0].set(
            o_x + (o_vx_b * jnp.cos(o_theta) - o_vy_b * jnp.sin(o_theta)) * self.dt)
        o_obst_states_new = o_obst_states_new_no_y.at[:, 1].set(
            o_y + (o_vx_b * jnp.sin(o_theta) + o_vy_b * jnp.cos(o_theta)) * self.dt)

        assert o_obst_states_new.shape == (num_obsts, self.state_dim)
        return o_obst_states_new

    def goal_step(self, aS_agent_states_new: AgentState) -> State:
        """根据下一步的agent位置，寻找相应的距离最近的目标点作为参考"""
        a_goals_indices = find_closest_goal_indices(aS_agent_states_new, self.all_goals)
        a_agents_indices = jnp.arange(aS_agent_states_new.shape[0])
        aS_goal_states = self.all_goals[a_agents_indices, a_goals_indices, :]

        return aS_goal_states

    @override
    def step(
            self, graph: MVEEnvGraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[MVEEnvGraphsTuple, MVEEnvGraphsTuple, Reward, Cost, Cost, Done, Info]:
        # get information from graph
        agent_states = graph.type_states(type_idx=MVE.AGENT, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=MVE.GOAL, n_type=self.num_agents) # debug
        obst_states = graph.type_states(type_idx=MVE.OBST, n_type=self.num_obsts)
        next_obst_states = self.obst_step_euler(obst_states)

        # calculate next graph
        action = self.transform_action(action)
        next_agent_states = self.agent_step_euler(agent_states, action)
        next_goal_states = self.goal_step(next_agent_states)
        next_env_state = MVEEnvState(next_agent_states, next_goal_states, next_obst_states)
        info = {}

        # the episode ends when reaching max_episode_steps
        done = jnp.array(False)

        # calculate reward and cost
        reward = self.get_reward(graph, action)
        cost, cost_real = self.get_cost(graph, action)
        '''
        # debug
        jax.debug.print("============================= \n"
                        "old_states: \n"
                        "agent={old_agent_states} \n"
                        "goal={old_goal_states} \n"
                        "obstacle={old_obstacle_states} \n"
                        "\n"
                        "action={action} \n"
                        "\n"
                        "new_states: \n"
                        "agent={new_agent_states} \n"
                        "goal={new_goal_states} \n"
                        "obstacle={new_obstacle_states} \n"
                        "============================= \n"
                        ,
                        old_agent_states = agent_states,
                        old_goal_states = goal_states,
                        old_obstacle_states = obst_states,
                        action=action,
                        new_agent_states = next_agent_states,
                        new_goal_states = next_goal_states,
                        new_obstacle_states = next_obst_states)
        '''

        next_graph, next_normed_graph = self.get_graph(next_env_state)

        return next_graph, next_normed_graph, reward, cost, cost_real, done, info

    def get_reward(self, graph: MVEEnvGraphsTuple, ad_action: Action) -> Reward:
        num_agents = graph.env_states.agent.shape[0]
        num_goals = graph.env_states.goal.shape[0]
        assert num_agents == num_goals

        aS_agents_states = graph.type_states(type_idx=MVE.AGENT, n_type=num_agents)
        aS_goals_states = graph.type_states(type_idx=MVE.GOAL, n_type=num_goals)
        # state: x, y, vx_b, vy_b, θ, dθ/dt, bw, bh
        # 参数提取
        a2_goal_pos_m = aS_goals_states[:, :2]
        a2_goal_v_b_kmph = aS_goals_states[:, 2:4]
        a2_goal_vx_b_kmph = a2_goal_v_b_kmph[:, 0]
        a_goal_theta_deg = aS_goals_states[:, 4]
        a2_agent_pos_m = aS_agents_states[:, :2]
        a2_agent_v_b_kmph = aS_agents_states[:, 2:4]
        a2_agent_vx_b_kmph = a2_agent_v_b_kmph[:, 0]
        a_agent_theta_deg = aS_agents_states[:, 4]

        # 待比较的state
        a4_goals = jnp.concatenate([a2_goal_pos_m, a2_goal_vx_b_kmph[:, None], a_goal_theta_deg[:, None]], axis=1)
        a4_agents = jnp.concatenate([a2_agent_pos_m, a2_agent_vx_b_kmph[:, None], a_agent_theta_deg[:, None]], axis=1)
        a4_e = a4_agents - a4_goals

        # 权重矩阵
        W = jnp.diag(jnp.array([1e-4, 1e-4, 2.5e-7, 1e-8]))

        reward = -jnp.sqrt(jnp.einsum('ai, ij, ja -> a', a4_e, W, a4_e.transpose())).mean()

        # 动作惩罚
        reward -= (ad_action[:, 0]**2).mean() * 0.00005
        reward -= (ad_action[:, 1]**2).mean() * 0.0001

        return reward

    def get_cost(self, graph: MVEEnvGraphsTuple, ad_action: Action) -> Tuple[Cost, Cost]:
        """使用与scaling factor有关的CBF作为评价指标H，H<thresh安全，H>=thresh不安全"""
        thresh = self.params["alpha_thresh"]
        num_agents = graph.env_states.agent.shape[0]
        num_obsts = graph.env_states.obstacle.shape[0]

        def get_dynamics_fv_gv(s1):
            """计算速度状态的漂移向量场 f_v 和控制矩阵 g_v"""
            # 参数提取
            m = self.params["ego_m"]  # kg
            lf = self.params["ego_lf"]  # m
            lr = self.params["ego_lr"]  # m
            Iz = self.params["ego_Iz"]  # kg*m^2
            Cf = 2 * self.params["ego_Cf"]  # N/rad，自行车模型需要将轮胎的侧偏刚度×2以代表轴刚度
            Cr = 2 * self.params["ego_Cr"]  # N/rad
            vx_b = s1[2] / 3.6 # km/h -> m/s
            vy_b = s1[3] / 3.6 # km/h -> m/s
            dtheta = s1[5] * jnp.pi/180  # deg/s -> rad/s

            # 计算 f_v(X)
            fv1 = vy_b * dtheta
            fv2 = -vx_b * dtheta - (Cf + Cr) / (m * vx_b) * vy_b + (Cr * lr + Cf * lf) / (m * vx_b) * dtheta
            fv3 = (Cr * lr - Cf * lf) / (Iz * vx_b) * vy_b - (Cf * lf ** 2 + Cr * lr ** 2) / (Iz * vx_b) * dtheta
            fv = jnp.stack([fv1, fv2, fv3])

            # 计算 g_v(X)
            gv11 = 1.0
            gv12 = Cf * (vy_b + lf * dtheta) / (m * vx_b)
            gv21 = 0.0
            gv22 = Cf / m
            gv31 = 0.0
            gv32 = Cf * lf / Iz
            gv = jnp.array([
                [gv11, gv12],
                [gv21, gv22],
                [gv31, gv32]
            ])

            return fv, gv

        def get_hocbf_constraints_optimized_between_states(s1, s2):
            gamma1 = self.params["gamma1"]
            gamma2 = self.params["gamma2"]
            # 向公制单位转换
            convert_vec_s = jnp.array([1, 1, 3.6, 3.6, 180 / jnp.pi, 180 / jnp.pi, 180/jnp.pi, 1, 1])  # eg. km/h / convert_vec -> m/s
            s1_metric = s1 / convert_vec_s
            s2_metric = s2 / convert_vec_s
            # 拆解状态变量
            p = jnp.array([s1_metric[0], s1_metric[1], s1_metric[4]])
            s1_rest_metric = jnp.array([s1_metric[2], s1_metric[3], s1_metric[5], s1_metric[6], s1_metric[7]])
            Vb = jnp.array([s1_metric[2], s1_metric[3], s1_metric[5]])  # [vx, vy, dtheta]
            theta = s1_metric[4]
            dtheta = s1_metric[5]

            # 运动学矩阵及时间演化
            R_k = jnp.array([
                [jnp.cos(theta), -jnp.sin(theta), 0.],
                [jnp.sin(theta), jnp.cos(theta), 0.],
                [0., 0., 1.]
            ])
            R_bar_k = jnp.array([
                [-jnp.sin(theta), -jnp.cos(theta), 0.],
                [jnp.cos(theta), -jnp.sin(theta), 0.],
                [0., 0., 0.]
            ])
            p_dot = R_k @ Vb

            # 定义纯函数并执行单次 Forward-over-Reverse AD
            def alpha_fn(p_in):
                s1_full = jnp.array(
                    [p_in[0], p_in[1], s1_rest_metric[0], s1_rest_metric[1], p_in[2], s1_rest_metric[2], s1_rest_metric[3], s1_rest_metric[4]])
                return scaling_calc(s1_full, s2_metric)

            # 构建同时返回 value 和 grad 的函数
            vg_fn = jax.value_and_grad(alpha_fn)

            # 一次 JVP 调用，拿到 4 个需要的值
            primal_out, tangent_out = jax.jvp(vg_fn, (p,), (p_dot,))
            alpha = primal_out[0]  # \alpha^*(p_k)
            grad_p = primal_out[1]  # \nabla_p \alpha^*(p_k)
            alpha_dot = tangent_out[0]  # \dot{\alpha}^* (其实就是 grad_p @ p_dot，JAX 顺手帮我们算了)
            grad_p_dot = tangent_out[1]  # \frac{d}{dt}(\nabla_p \alpha^*(p_k))

            # 提取动力学矩阵 fv, gv，注意这里的s1不能是公制单位
            fv, gv = get_dynamics_fv_gv(s1)

            grad_p_R = grad_p @ R_k # 提取公共项 \nabla_p \alpha^*(p_k) R_k
            A_cbf = grad_p_R @ gv # A_cbf = \nabla_p \alpha^*(p_k) R_k g_v(X_k)
            # term_ddot_alpha = (\frac{d}{dt}\nabla_p \alpha^* R_k + \nabla_p \alpha^* \bar{R}_k \dot{\theta}) V_{b,k} + \nabla_p \alpha^* R_k f_v
            term_ddot_alpha = (grad_p_dot @ R_k + grad_p @ (R_bar_k * dtheta)) @ Vb + grad_p_R @ fv
            # 最终约束形式为 A_cbf * u >= b_cbf
            b_cbf = - term_ddot_alpha - (gamma1 + gamma2) * alpha_dot - gamma1 * gamma2 * (alpha - 1.0)

            return A_cbf, b_cbf, alpha

        def get_hocbf_constraints_optimized_between_state_and_bound(s1, A, b):
            gamma1 = self.params["gamma1"]
            gamma2 = self.params["gamma2"]
            # 向公制单位转换
            convert_vec_s = jnp.array([1, 1, 3.6, 3.6, 180 / jnp.pi, 180 / jnp.pi, 180/jnp.pi, 1, 1])  # eg. km/h / convert_vec -> m/s
            s1_metric = s1 / convert_vec_s
            # 拆解状态变量
            p = jnp.array([s1_metric[0], s1_metric[1], s1_metric[4]])
            s1_rest_metric = jnp.array([s1_metric[2], s1_metric[3], s1_metric[5], s1_metric[6], s1_metric[7]])
            Vb = jnp.array([s1_metric[2], s1_metric[3], s1_metric[5]])  # [vx, vy, dtheta]
            theta = s1_metric[4]
            dtheta = s1_metric[5]

            # 运动学矩阵及时间演化
            R_k = jnp.array([
                [jnp.cos(theta), -jnp.sin(theta), 0.],
                [jnp.sin(theta), jnp.cos(theta), 0.],
                [0., 0., 1.]
            ])
            R_bar_k = jnp.array([
                [-jnp.sin(theta), -jnp.cos(theta), 0.],
                [jnp.cos(theta), -jnp.sin(theta), 0.],
                [0., 0., 0.]
            ])
            p_dot = R_k @ Vb

            # 定义纯函数并执行单次 Forward-over-Reverse AD
            def alpha_fn_2(p_in):
                s1_full = jnp.array(
                    [p_in[0], p_in[1], s1_rest_metric[0], s1_rest_metric[1], p_in[2], s1_rest_metric[2],
                     s1_rest_metric[3], s1_rest_metric[4]])
                return scaling_calc_bound(s1_full, A, b)

            # 构建同时返回 value 和 grad 的函数
            vg_fn = jax.value_and_grad(alpha_fn_2)

            # 一次 JVP 调用，拿到 4 个需要的值
            primal_out, tangent_out = jax.jvp(vg_fn, (p,), (p_dot,))
            alpha = primal_out[0]  # \alpha^*(p_k)
            grad_p = primal_out[1]  # \nabla_p \alpha^*(p_k)
            alpha_dot = tangent_out[0]  # \dot{\alpha}^* (其实就是 grad_p @ p_dot，JAX 顺手帮我们算了)
            grad_p_dot = tangent_out[1]  # \frac{d}{dt}(\nabla_p \alpha^*(p_k))

            # 提取动力学矩阵 fv, gv，注意这里的s1不能是公制单位
            fv, gv = get_dynamics_fv_gv(s1)

            grad_p_R = grad_p @ R_k  # 提取公共项 \nabla_p \alpha^*(p_k) R_k
            A_cbf = grad_p_R @ gv  # A_cbf = \nabla_p \alpha^*(p_k) R_k g_v(X_k)
            # term_ddot_alpha = (\frac{d}{dt}\nabla_p \alpha^* R_k + \nabla_p \alpha^* \bar{R}_k \dot{\theta}) V_{b,k} + \nabla_p \alpha^* R_k f_v
            term_ddot_alpha = (grad_p_dot @ R_k + grad_p @ (R_bar_k * dtheta)) @ Vb + grad_p_R @ fv
            # 最终约束形式为 A_cbf * u >= b_cbf
            b_cbf = - term_ddot_alpha - (gamma1 + gamma2) * alpha_dot - gamma1 * gamma2 * (alpha - 1.0)

            """
            # debug
            jax.debug.print("=============== \n"
                            "agent_state = {agent_state} \n"
                            "alpha* = {alpha_star} \n"
                            "alpha_dot = {alpha_dot} \n"
                            "alpha_ddot = {alpha_ddot} \n"
                            "=============== \n",
                            agent_state=s1,
                            alpha_star=alpha,
                            alpha_dot=alpha_dot,
                            alpha_ddot=term_ddot_alpha)
            """
            return A_cbf, b_cbf, alpha

        agent_states = graph.type_states(type_idx=MVE.AGENT, n_type=num_agents)
        convert_vec_a = jnp.array([1, 180 / jnp.pi])
        ad_action_wrong_delta_r_metric = ad_action / convert_vec_a # m/s²不变，° / convert_vec_a -> rad
        a_delta_r = agent_states[:, 6] * jnp.pi / 180
        ad_action_r_metric = ad_action_wrong_delta_r_metric.at[:, 1].set(a_delta_r)

        # 这一环境不考虑ego车辆之间的碰撞（test时只有1个agent）
        a_agent_cost = -jnp.ones((num_agents,), dtype=jnp.float32) * 3
        a_agent_cost_real = -jnp.ones((num_agents,), dtype=jnp.float32) * 3

        # agent 和 obst 之间的scaling factor
        if num_obsts == 0:
            a_obst_cost = -jnp.ones((num_agents,), dtype=jnp.float32)
            a_obst_cost_real = -jnp.ones((num_agents,), dtype=jnp.float32)
        else:
            obstacle_states = graph.type_states(type_idx=MVE.OBST, n_type=num_obsts)
            i_pairs, j_pairs = gen_i_j_pairs(num_agents, num_obsts)
            state_i_pairs = agent_states[i_pairs, :]
            state_j_pairs = obstacle_states[j_pairs, :]

            A_cbf_pairs, b_cbf_pairs, alpha_pairs = jax.vmap(get_hocbf_constraints_optimized_between_states, in_axes=(0, 0))(state_i_pairs, state_j_pairs)
            A_cbf_matrix = A_cbf_pairs.reshape((num_agents, num_obsts, self.action_dim))
            b_cbf_matrix = b_cbf_pairs.reshape((num_agents, num_obsts))
            alpha_matrix = alpha_pairs.reshape((num_agents, num_obsts))

            Au_cbf_matrix = jnp.einsum("aod, ad -> ao", A_cbf_matrix, ad_action_r_metric)
            cbf_result_matrix = b_cbf_matrix + thresh - Au_cbf_matrix

            a_obst_cost = jnp.max(cbf_result_matrix, axis=1)
            a_obst_cost_real = jnp.max(1-alpha_matrix, axis=1) # α*>1 表示真实安全
        # a_obst_cost = -jnp.ones((num_agents,), dtype=jnp.float32) # debug

        # agent 和 bound 之间的scaling factor，只对y方向有约束
        state_range = self.params["default_state_range"]
        yl = state_range[2]
        A_yl = jnp.array([[0., 1.]])
        b_yl = jnp.array([yl])
        A_cbf_yl, b_cbf_yl, alpha_yl = jax.vmap(get_hocbf_constraints_optimized_between_state_and_bound,
                                                         in_axes=(0, None, None))(agent_states, A_yl, b_yl)
        Au_cbf_yl = jnp.einsum("ad, ad -> a", A_cbf_yl, ad_action_r_metric)
        cbf_result_yl = b_cbf_yl + thresh - Au_cbf_yl
        a_bound_yl_cost = cbf_result_yl
        a_bound_yl_cost_real = 1 - alpha_yl # α*>1 表示真实安全

        yh = state_range[3]
        A_yh = jnp.array([[0., -1.]])
        b_yh = jnp.array([-yh])
        A_cbf_yh, b_cbf_yh, alpha_yh = jax.vmap(get_hocbf_constraints_optimized_between_state_and_bound,
                                                in_axes=(0, None, None))(agent_states, A_yh, b_yh)
        Au_cbf_yh = jnp.einsum("ad, ad -> a", A_cbf_yh, ad_action_r_metric)
        cbf_result_yh = b_cbf_yh + thresh - Au_cbf_yh
        a_bound_yh_cost = cbf_result_yh
        a_bound_yh_cost_real = 1 - alpha_yh # α*>1 表示真实安全

        # a_bound_yl_cost = -jnp.ones((num_agents,), dtype=jnp.float32) # debug
        # a_bound_yh_cost = -jnp.ones((num_agents,), dtype=jnp.float32) # debug

        cost = jnp.stack([a_agent_cost, a_obst_cost, a_bound_yl_cost, a_bound_yh_cost], axis=1)
        cost_real = jnp.stack([a_agent_cost_real, a_obst_cost_real,
                               a_bound_yl_cost_real, a_bound_yh_cost_real], axis=1)
        assert cost.shape == (num_agents, self.n_cost)
        assert cost_real.shape == (num_agents, self.n_cost)

        """
        # debug
        if num_obsts > 0:
            obst_states = graph.type_states(type_idx=MVE.OBST, n_type=num_obsts)
            jax.debug.print("======================= \n "
                                "agent_states={agent_states} \n "
                                "obst_states={obst_states} \n"
                                "cost={cost} \n"
                                "==================== \n ",
                                agent_states=agent_states,
                                obst_states=obst_states,
                                cost=cost,)
        else:
            jax.debug.print("======================= \n "
                                "agent_states={agent_states} \n "
                                "cost={cost} \n"
                                "==================== \n ",
                                agent_states=agent_states,
                                cost=cost)
        """

        # add margin and clip
        # eps = 1.
        # cost = jnp.where(cost <= 0.0, cost, cost + eps)
        cost = jnp.clip(cost, a_min=-3.0)

        return cost, cost_real

    @override
    def render_video(
            self,
            rollout: Rollout_NormedGraph,
            video_path: pathlib.Path,
            Ta_is_unsafe=None,
            viz_opts: Optional[dict] = None,
            n_goals: Optional[int] = None,
            **kwargs
    ) -> None:
        T_goal_states = jax.vmap(lambda x: x.type_states(type_idx=MVE.GOAL, n_type=self.num_agents))(rollout.graph)
        ref_goals = T_goal_states[:, :, :2]
        n_goals = self.num_agents if n_goals is None else n_goals

        ax: Axes
        xlim = self.params["rollout_state_range"][:2]
        ylim = self.params["default_state_range"][2:4]
        fig, ax = plt.subplots(1, 1, figsize=(30,
                                (ylim[1]+3-(ylim[0]-3))*20/(xlim[1]+3-(xlim[0]-3))+4)
                               , dpi=100)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0]-3, ylim[1]+3)
        ax.set(aspect="equal")
        plt.axis("on")
        if viz_opts is None:
            viz_opts = {}

        # 画车道线
        two_yms_bold, l_yms_scatter = process_lane_marks(self.params["default_state_range"][2:4], self.params["lane_width"])
        ax.axhline(y=two_yms_bold[0], linewidth=1.5, color='b')
        ax.axhline(y=two_yms_bold[1], linewidth=1.5, color='b')
        if l_yms_scatter is not None:
            for ym in l_yms_scatter:
                ax.axhline(y=ym, linewidth=1, color='b', linestyle='--')

        # plot the first frame
        T_graph = rollout.graph
        graph0 = tree_index(T_graph, 0)

        agent_color = "#0068ff"
        goal_color = "#2fdd00"
        obst_color = "#8a0000"
        edge_goal_color = goal_color

        # plot obstacles
        num_obsts = graph0.env_states.obstacle.shape[0]
        if num_obsts > 0:
            obsts_state = graph0.type_nodes(type_idx=MVE.OBST, n_type=self.num_obsts)
            # state: x, y, vx, vy, θ, dθ/dt, bw, bh
            obsts_pos = obsts_state[:, :2]
            obsts_theta = obsts_state[:, 4]
            obsts_bb_size = obsts_state[:, 7:9]
            obsts_radius = jnp.linalg.norm(obsts_bb_size, axis=1)
            plot_obsts_arrow = [FancyArrow(x=obsts_pos[i,0], y=obsts_pos[i,1],
                                           dx=jnp.cos(obsts_theta[i]*jnp.pi/180)*obsts_radius[i]/2,
                                           dy=jnp.sin(obsts_theta[i]*jnp.pi/180)*obsts_radius[i]/2,
                                           length_includes_head=True,
                                           width=0.3, color=obst_color, alpha=1.0) for i in range(len(obsts_theta))]
            plot_obsts_rec = [plt.Rectangle(xy=tuple(obsts_pos[i,:]-obsts_bb_size[i,:]/2),
                                            width=obsts_bb_size[i,0], height=obsts_bb_size[i,1],
                                            angle=obsts_theta[i], rotation_point='center',
                                                color=obst_color, linewidth=0.0, alpha=0.6) for i in range(len(obsts_theta))]
            col_obsts = MutablePatchCollection(plot_obsts_arrow+plot_obsts_rec, match_original=True, zorder=5)
            ax.add_collection(col_obsts)

        # plot agents
        agents_state = graph0.type_states(type_idx=MVE.AGENT, n_type=self.num_agents)
        # state: x, y, vx, vy, θ, dθ/dt, δ, bb_w, bb_h, a0 ... a5
        agents_pos = agents_state[:, :2]
        agents_theta = agents_state[:, 4]
        agents_bb_size = agents_state[:, 7:9]
        agents_radius = jnp.linalg.norm(agents_bb_size, axis=1)
        plot_agents_arrow = [FancyArrow(x=agents_pos[i, 0], y=agents_pos[i, 1],
                                        dx=jnp.cos(agents_theta[i] * jnp.pi / 180) * agents_radius[i]/2,
                                        dy=jnp.sin(agents_theta[i] * jnp.pi / 180) * agents_radius[i]/2,
                                        width=agents_radius[i] / jnp.mean(obsts_radius)*0.3,
                                        length_includes_head=True,
                                        alpha=1.0, color=agent_color) for i in range(self.num_agents)]
        plot_agents_rec = [plt.Rectangle(xy=tuple(agents_pos[i,:]-agents_bb_size[i,:]/2),
                                         width=agents_bb_size[i,0], height=agents_bb_size[i,1],
                                         angle=agents_theta[i], rotation_point='center',
                                         color=agent_color, linewidth=0.0, alpha=0.6) for i in range(self.num_agents)]
        col_agents = MutablePatchCollection(plot_agents_arrow+plot_agents_rec, match_original=True, zorder=6)
        ax.add_collection(col_agents)

        # plot reference points
        # state: x, y, vx, vy, θ, dθ/dt, bw,
        all_ref_xs = ref_goals[:, :, 0].reshape(-1)
        all_ref_ys = ref_goals[:, :, 1].reshape(-1)
        ax.scatter(all_ref_xs, all_ref_ys, color=goal_color, zorder=7, s=5, alpha=1.0, marker='.')

        # plot edges
        all_pos = graph0.states[:, :2]
        edge_index = np.stack([graph0.senders, graph0.receivers], axis=0)
        is_pad = np.any(edge_index == self.num_agents + n_goals + self.num_obsts, axis=0)
        e_edge_index = edge_index[:, ~is_pad]
        e_start, e_end = all_pos[e_edge_index[0, :]], all_pos[e_edge_index[1, :]]
        e_lines = np.stack([e_start, e_end], axis=1)  # (e, n_pts, dim)
        e_is_goal = (self.num_agents <= graph0.senders) & (graph0.senders < self.num_agents + n_goals)
        e_is_goal = e_is_goal[~is_pad]
        e_colors = [edge_goal_color if e_is_goal[ii] else "0.2" for ii in range(len(e_start))]
        col_edges = LineCollection(e_lines, colors=e_colors, linewidths=2, alpha=0.5, zorder=3)
        ax.add_collection(col_edges)

        # texts
        text_font_opts = dict(
            size=16,
            color="k",
            family="sans-serif",
            weight="normal",
            transform=ax.transAxes,
        )
        cost_text = ax.text(0.02, 1.00, "Cost: 1.0\nReward: 1.0", va="bottom", **text_font_opts)
        if Ta_is_unsafe is not None:
            safe_text = [ax.text(0.99, 1.00, "Unsafe: {}", va="bottom", ha="right", **text_font_opts)]
        kk_text = ax.text(0.99, 1.04, "kk=0", va="bottom", ha="right", **text_font_opts)
        if rollout.zs is not None:
            z_text = ax.text(0.5, 1.04, "z: []", va="bottom", ha="center", **text_font_opts)

        # add agent labels
        label_font_opts = dict(
            size=20,
            color="k",
            family="sans-serif",
            weight="normal",
            ha="center",
            va="center",
            transform=ax.transData,
            clip_on=True,
            zorder=8,
            alpha=0.
        )
        agent_labels = [ax.text(float(agents_pos[ii, 0]), float(agents_pos[ii, 1]), f"{ii}", **label_font_opts)
                        for ii in range(self.num_agents)]

        if "Vh" in viz_opts:
            Vh_text = ax.text(0.99, 0.99, "Vh: []", va="top", ha="right", **text_font_opts)

        # init function for animation
        def init_fn() -> List[plt.Artist]:
            return [col_obsts, col_agents, col_edges, *agent_labels, cost_text, *safe_text, kk_text]

        def update(kk: int) -> List[plt.Artist]:
            graph = tree_index(T_graph, kk)
            n_pos_t = graph.states[:-1, :2] # 最后一个node是padding，不要
            n_theta_t = graph.states[:-1, 4]
            n_bb_size_t = graph.nodes[:-1, 7:9]
            n_radius = jnp.linalg.norm(n_bb_size_t, axis=1)

            # update agents' positions and labels
            for ii in range(self.num_agents):
                plot_agents_arrow[ii].set_data(x=n_pos_t[ii, 0], y=n_pos_t[ii, 1],
                                               dx=jnp.cos(n_theta_t[ii]*jnp.pi/180)*n_radius[ii]/2,
                                               dy=jnp.sin(n_theta_t[ii]*jnp.pi/180)*n_radius[ii]/2)
                plot_agents_rec[ii].set_xy(xy=tuple(n_pos_t[ii, :]-n_bb_size_t[ii, :]/2))
                plot_agents_rec[ii].set_angle(angle=n_theta_t[ii])
                agent_labels[ii].set_position(n_pos_t[ii, :])

            # update obstacles' positions
            num_obsts = graph.env_states.obstacle.shape[0]
            if num_obsts > 0:
                for ii in range(self.num_obsts):
                     plot_obsts_arrow[ii].set_data(x=n_pos_t[self.num_agents+n_goals+ii, 0],
                                                    y=n_pos_t[self.num_agents+n_goals+ii, 1],
                                                    dx=jnp.cos(n_theta_t[self.num_agents+n_goals+ii]*jnp.pi/180)*n_radius[
                                                        self.num_agents+n_goals+ii]/2,
                                                    dy=jnp.sin(n_theta_t[self.num_agents+n_goals+ii]*jnp.pi/180)*n_radius[
                                                        self.num_agents+n_goals+ii]/2)
                     plot_obsts_rec[ii].set_xy(xy=tuple(n_pos_t[self.num_agents+n_goals+ii, :]-n_bb_size_t[self.num_agents+n_goals+ii, :]/2))
                     plot_obsts_rec[ii].set_angle(angle=n_theta_t[self.num_agents+n_goals+ii])

            # update edges
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

            # update cost and safe labels
            if kk < len(rollout.costs):
                all_costs = ""
                for i_cost in range(rollout.costs[kk].shape[1]):
                    all_costs += f"    {self.cost_components[i_cost]}: {rollout.costs[kk][:, i_cost].max():5.4f}\n"
                all_costs = all_costs[:-2]
                cost_text.set_text(f"Cost:\n{all_costs}\nReward: {rollout.rewards[kk]:5.4f}")
            else:
                cost_text.set_text("")
            if kk < len(Ta_is_unsafe):
                a_is_unsafe = Ta_is_unsafe[kk]
                unsafe_idx = np.where(a_is_unsafe)[0]
                safe_text[0].set_text("Unsafe: {}".format(unsafe_idx))
            else:
                safe_text[0].set_text("Unsafe: {}")

            kk_text.set_text("kk={:04}".format(kk))

            # Update the z text.
            if rollout.zs is not None:
                z_text.set_text(f"z: {rollout.zs[kk]}")

            if "Vh" in viz_opts:
                Vh_text.set_text(f"Vh: {viz_opts['Vh'][kk]}")

            return [col_obsts, col_agents, col_edges, *agent_labels, cost_text, *safe_text, kk_text]

        fps = 30.0
        spf = 1 / fps
        mspf = 1_000 * spf
        anim_T = len(T_graph.n_node)
        ani = FuncAnimation(fig, update, frames=anim_T, init_func=init_fn, interval=mspf, blit=True)
        save_anim(ani, video_path)

    def edge_blocks(self, state: MVEEnvState) -> List[EdgeBlock]:
        num_agents = state.agent.shape[0]
        num_goals = state.goal.shape[0]
        assert num_agents == num_goals
        num_obsts = state.obstacle.shape[0]

        agent_pos = state.agent[:, :2]
        id_agent = jnp.arange(num_agents)

        """
        # agent - agent connection
        agent_agent_edges = []
        if num_agents > 1:
            pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]
            dist = jnp.linalg.norm(pos_diff, axis=-1)
            dist += jnp.eye(dist.shape[1]) * (self.params["comm_radius"] + 1)
            agent_agent_mask = jnp.less(dist, self.params["comm_radius"])
            i_pairs, j_pairs = gen_i_j_pairs_no_identical(num_agents, num_agents)
            agent_state_i_pairs = state.agent[i_pairs, :]
            agent_state_j_pairs = state.agent[j_pairs, :]
            rel_state_pairs = jax.vmap(relative_state, in_axes=(0, 0))(agent_state_i_pairs, agent_state_j_pairs)
            rel_state = jnp.zeros((num_agents, num_agents, self.state_dim), dtype=jnp.float32) # 相对状态矩阵初始化
            rel_state = rel_state.at[i_pairs, j_pairs, :].set(rel_state_pairs)
            agent_agent_edges = [EdgeBlock(rel_state, agent_agent_mask, id_agent, id_agent)]
        """

        # agent - goal connection
        agent_goal_edges = []
        for i_agent in range(self.num_agents):
            agent_state_i = state.agent[i_agent]
            goal_state_i = state.goal[i_agent]
            rel_state = relative_state_delta_state(agent_state_i, goal_state_i)
            agent_goal_edges.append(EdgeBlock(rel_state[None, None, :], jnp.ones((1, 1)),
                                              jnp.array([i_agent]), jnp.array([i_agent + self.num_agents])))

        # agent - obstacle connection
        agent_obst_edges = []
        if num_obsts > 0:
            obs_pos = state.obstacle[:, :2]
            poss_diff = agent_pos[:, None, :] - obs_pos[None, :, :]
            dist = jnp.linalg.norm(poss_diff, axis=-1)
            agent_obs_mask = jnp.less(dist, self.params["comm_radius"])
            id_obs = jnp.arange(num_obsts) + num_agents * 2
            i_pairs, j_pairs = gen_i_j_pairs(num_agents, num_obsts)
            agent_state_i_pairs = state.agent[i_pairs, :]
            obst_state_j_pairs = state.obstacle[j_pairs, :]
            rel_state_pairs = jax.vmap(relative_state_delta_state, in_axes=(0, 0))(agent_state_i_pairs, obst_state_j_pairs)
            rel_state = rel_state_pairs.reshape((num_agents, num_obsts, self.state_dim))
            agent_obst_edges = [EdgeBlock(rel_state, agent_obs_mask, id_agent, id_obs)]

        # return agent_agent_edges + agent_goal_edges + agent_obst_edges

        """
        #debug
        jax.debug.print("=============================== \n"
                        "agent_goal_rel_state = {rel_state} \n"
                        "agent_goal_mask = {agent_goal_mask} \n"
                        "=============================== \n",
                        rel_state=rel_state,
                        agent_goal_mask=agent_goal_mask)
        """

        return agent_goal_edges + agent_obst_edges # 跟踪任务debug

    @override
    def get_graph(self, env_state: MVEEnvState, obst_as_ego: bool = False) -> Tuple[MVEEnvGraphsTuple, MVEEnvGraphsTuple]:
        del obst_as_ego # 手动生成场景，所以不需要这个参数了
        """输出一个原始图，一个归一化图"""
        num_agents = env_state.agent.shape[0]
        num_goals = env_state.goal.shape[0]
        num_obsts = env_state.obstacle.shape[0]
        assert num_agents > 0 and num_goals > 0, "至少需要设定agent和goal!"
        assert num_agents == num_goals, "每一个agent对应一个goal"

        # normalization information
        aS_agent_states = env_state.agent
        aS_goal_states = env_state.goal
        oS_obst_states = env_state.obstacle
        x_center = aS_agent_states[:, 0].mean()
        aS_agent_states_normed = aS_agent_states.at[:, 0].set(aS_agent_states[:, 0] - x_center)
        aS_goal_states_normed = aS_goal_states.at[:, 0].set(aS_goal_states[:, 0] - x_center)
        oS_obst_states_normed = oS_obst_states.at[:, 0].set(oS_obst_states[:, 0] - x_center)

        # node features
        node_feats = jnp.zeros((num_agents + num_goals + num_obsts, self.node_dim))
        normed_node_feats = jnp.zeros_like(node_feats)
        node_feats = node_feats.at[:num_agents, :self.state_dim].set(aS_agent_states)
        normed_node_feats = normed_node_feats.at[:num_agents, :self.state_dim].set(aS_agent_states_normed)
        node_feats = node_feats.at[num_agents: num_agents + num_goals, :self.state_dim].set(aS_goal_states)
        normed_node_feats = normed_node_feats.at[num_agents: num_agents + num_goals, :self.state_dim].set(aS_goal_states_normed)
        if num_obsts > 0:
            node_feats = node_feats.at[num_agents + num_goals:, :self.state_dim].set(env_state.obstacle)
            normed_node_feats = normed_node_feats.at[num_agents + num_goals:, :self.state_dim].set(oS_obst_states_normed)

        # bounding box长宽
        # state: x y vx vy θ dθdt bw bh
        node_feats = node_feats.at[:num_agents, 7:9].set(self.params["ego_bb_size"])
        normed_node_feats = normed_node_feats.at[:num_agents, 7:9].set(self.params["ego_bb_size"])
        if num_obsts > 0:
            node_feats = node_feats.at[num_agents + num_goals:, 7:9].set(self.params["obst_bb_size"])
            normed_node_feats = normed_node_feats.at[num_agents + num_goals:, 7:9].set(self.params["obst_bb_size"])

        # indicators
        node_feats = node_feats.at[:num_agents, -1].set(1.0)
        normed_node_feats = normed_node_feats.at[:num_agents, -1].set(1.0)
        node_feats = node_feats.at[num_agents: num_agents + num_goals, -2].set(1.0)
        normed_node_feats = normed_node_feats.at[num_agents: num_agents + num_goals, -2].set(1.0)
        if num_obsts > 0:
            node_feats = node_feats.at[num_agents + num_goals:, -3].set(1.0)
            normed_node_feats = normed_node_feats.at[num_agents + num_goals:, -3].set(1.0)

        # node type
        node_type = -jnp.ones((num_agents + num_goals + num_obsts), dtype=jnp.int32)
        node_type = node_type.at[:num_agents].set(MVE.AGENT)
        node_type = node_type.at[num_agents: num_agents + num_goals].set(MVE.GOAL)
        if num_obsts > 0:
            node_type = node_type.at[num_agents + num_goals:].set(MVE.OBST)

        # edges
        edge_blocks = self.edge_blocks(env_state)

        # create graph
        states = node_feats[:num_agents + num_goals, :-3]
        normed_states = normed_node_feats[:num_agents + num_goals, :-3]

        if num_obsts > 0:
            states = jnp.concatenate([states, node_feats[num_agents + num_goals:, :-3]], axis=0)
            normed_states = jnp.concatenate([normed_states, normed_node_feats[num_agents + num_goals:, :-3]], axis=0)
            new_env_state = MVEEnvState(node_feats[:num_agents, :-3],
                                        node_feats[num_agents: num_agents + num_goals, :-3],
                                        node_feats[num_agents + num_goals:, :-3])
            new_normed_env_state = MVEEnvState(normed_node_feats[:num_agents, :-3],
                                               normed_node_feats[num_agents: num_agents + num_goals, :-3],
                                               normed_node_feats[num_agents + num_goals:, :-3])
        else:
            new_env_state = MVEEnvState(node_feats[:num_agents, :-3],
                                        node_feats[num_agents: num_agents + num_goals, :-3],
                                        jnp.empty((0, self.state_dim)))
            new_normed_env_state = MVEEnvState(normed_node_feats[:num_agents, :-3],
                                               normed_node_feats[num_agents: num_agents + num_goals, :-3],
                                               jnp.empty((0, self.state_dim)))

        graph = GetGraph(node_feats, node_type, edge_blocks, new_env_state, states).to_padded()
        normed_graph = GetGraph(normed_node_feats, node_type, edge_blocks, new_normed_env_state, normed_states).to_padded()
        return graph, normed_graph

    @override
    def state_lim(self, state: Optional[State]) -> Tuple[State, State]:
        """世界坐标系下的状态约束"""
        lower_lim = self.params["rollout_state_range"][jnp.array([0, 2, 4, 6, 8, 10, 12, 14, 16])]
        upper_lim = self.params["rollout_state_range"][jnp.array([1, 3, 5, 7, 9, 11, 13, 15, 17])]
        return lower_lim, upper_lim

    @override
    def action_lim(self) -> Tuple[Action, Action]:
        lower_lim = jnp.array([-2., -3.])[None, :].repeat(self.num_agents, axis=0) # ax: m/s^2, δ: °
        upper_lim = jnp.array([2., 3.])[None, :].repeat(self.num_agents, axis=0)
        return lower_lim, upper_lim
