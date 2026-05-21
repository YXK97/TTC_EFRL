import pathlib
import jax
import jax.random as jr
import jax.numpy as jnp
import functools as ft
import numpy as np

import os

from typing import Optional, Tuple, List
from typing_extensions import override
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrow

from .mve import MVE, MVEEnvState, MVEEnvGraphsTuple
from .designed_scene_gen_two_lane import gen_handmade_scene, gen_scene_randomly
from .utils import process_lane_centers, process_lane_marks, relative_state
from defmarl.trainer.data import Rollout, Record
from defmarl.utils.graph import EdgeBlock, GetGraph, GraphsTuple
from defmarl.utils.typing import Action, Reward, Cost, Array, State, AgentState, ObstState, Done, Info
from defmarl.utils.utils import tree_index, MutablePatchCollection, save_anim, calc_2d_rot_matrix, \
    find_closest_goal_indices, gen_i_j_pairs, gen_i_j_pairs_no_identical, normalize_angle
from defmarl.utils.scaling_lowspeed import scaling_calc, scaling_calc_bound

INF = jnp.inf


class MVELaneChangeAndOverTake_LowSpeed_CBF(MVE):
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
        # "obst_bb_size": jnp.array([4., 2.]), # bounding box的[width, height] m
        "obst_bb_size": jnp.array([2.625,1.647]), # bounding box的[width, height] m
        "obst_lr":0.9025,

        # [x_l, x_h, y_l, y_h, θ_l, θ_h, v_l, v_h, δ_l, δ_h, bw_l, bw_h, bh_l, bh_h, lr_l, lr_h]
        # 单位：x,y,bw,bh,lr: m  v: km/h,  θ: °
        "rollout_state_range": jnp.array([-5., 150., -10., 10., -180., 180., 0., 30., -10., 10., 0., INF, 0., INF, 0., INF]),
        # "rollout_state_b_range": jnp.array([-INF, INF, -INF, INF, -180., 180., 30., 100., 0., INF, 0., INF, 0., INF]),
        "agent_init_state_range": jnp.array([-100., -50., -3., 3., -180., 180., -INF, INF, 0., INF, 0., INF, 0., INF]),
        "terminal_state_range": jnp.array([50., 100., -3., 3., -180., 180., -INF, INF, 0., INF, 0., INF, 0., INF]),
        "default_state_range": jnp.array([0., 100., -3., 3., -180., 180., -INF, INF, 0., INF, 0., INF, 0., INF]),

        "lane_width": 3, # 车道宽度，m
        "v_bias": 5, # 可允许的速度偏移量
        "alpha_thresh": 1.05, # alpha大于thresh时才判定为安全，用于避障时让agent离obst不要那么近
        "speed_filter_alpha": 0.5,   # 速度滤波系数
        "delta_filter_alpha": 0.5,   # 转角滤波系数
        "max_dv": 0.5,          # km/h
        "max_delta": 0.2,      # deg

        "gamma": 3., # 用于CBF
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
                 max_step: int = 512,
                 max_travel: Optional[float] = None,
                 dt: float = 0.05,
                 reward_min: float = -17.,
                 reward_max: float = 0.5,
                 params: dict = None
                 ):
        area_size = MVELaneChangeAndOverTake_LowSpeed_CBF.PARAMS["rollout_state_range"][:4] if area_size is None else area_size
        params = MVELaneChangeAndOverTake_LowSpeed_CBF.PARAMS if params is None else params
        super(MVELaneChangeAndOverTake_LowSpeed_CBF, self).__init__(num_agents, area_size, max_step, max_travel, dt, reward_min, reward_max, params)
        # assert self.params["n_obsts"] == MVELaneChangeAndOverTake.PARAMS["n_obsts"], "本环境只接受2个障碍物的设置！"
        self.all_goals = jnp.zeros((num_agents, self.num_goals, self.state_dim))  # 参考点初始化
        self.all_dsYddts = jnp.zeros((num_agents, self.num_goals, 4)) # 轨迹的y方向偏移量与偏移量导数初始化
        self.num_obsts = 0 # 初始化

    @override
    @property
    def state_dim(self) -> int:
        return 8 # x y θ v δ bw bh lr

    @override
    @property
    def node_dim(self) -> int:
        return 11  # state_dim(8)  indicator(3): agent: 001, goal: 010, obstacle: 100, pad: 00-1

    @override
    @property
    def edge_dim(self) -> int:
        return 8 # Δx, Δy, Δθ, Δv, Δδ, Δdbw, Δdbh, Δdlr

    @override
    @property
    def action_dim(self) -> int:
        return 1  # δ：前轮转角（逆时针为正，°）

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
    def reset(self, key: Array) -> Tuple[GraphsTuple, jnp.ndarray]:
        """使用场景类别生成函数进行agent、goal和obstacle的生成"""
        c_ycs = self.params["lane_centers"]
        xrange = self.params["default_state_range"][:2]
        yrange = self.params["default_state_range"][2:4]
        lanewidth = self.params["lane_width"]
        agents, obsts, all_goals, all_dsYddts = gen_scene_randomly(key, self.num_agents, self.num_goals, xrange, yrange, lanewidth, c_ycs)
        # agents, obsts, all_goals, all_dsYddts = gen_handmade_scene(key, self.num_agents, self.num_goals, xrange, yrange, lanewidth, c_ycs)
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
    def agent_step_euler(self, aS_agent_states, aS_goal_states, ad_action): #对agent，使用3-DOF自行车运动学模型,车辆中心在后轴中心
        x = aS_agent_states[:, 0]
        y = aS_agent_states[:, 1]
        theta_deg = aS_agent_states[:, 2]
        v_real_kmph = aS_agent_states[:, 3]
        delta_real_deg = aS_agent_states[:,4]
        bw = aS_agent_states[:, 5]
        bh = aS_agent_states[:, 6]
        lr = aS_agent_states[:, 7]
        v_goal_kmph = aS_goal_states[:, 3]
        delta_goal_deg = ad_action[:, 0]

        def filter_v_delta(
                v_goal_kmph: Array,
                delta_goal_deg: Array,
                v_real_kmph: Array,
                delta_real_deg: Array,
        ) -> Tuple[Array, Array]:

            alpha_v = self.params.get("speed_filter_alpha", 0.5)
            dv_max = self.params.get("max_dv", 1.0)
            v_m = alpha_v * v_real_kmph + (1.0 - alpha_v) * v_goal_kmph
            dv =v_m - v_real_kmph
            dv_clip = jnp.clip(dv, -dv_max, dv_max)
            v_f = v_real_kmph + dv_clip

            # 单步最大转角变化量，单位 degree
            alpha_delta = self.params.get("delta_filter_alpha", 0.8)
            delta_max = self.params.get("max_delta", 1.0)
            delta_m = alpha_delta * delta_real_deg + (1.0 - alpha_delta) * delta_goal_deg
            ddelta = delta_m - delta_real_deg
            delta_clip = jnp.clip(ddelta, -delta_max, delta_max)
            delta_f = delta_real_deg + delta_clip

            return v_f, delta_f

        # 添加两个滤波，提取速度和转角，进行限制变化幅度
        v_kmph ,delta_deg =filter_v_delta(
            v_goal_kmph=v_goal_kmph,
            delta_goal_deg=delta_goal_deg,
            v_real_kmph=v_real_kmph,
            delta_real_deg=delta_real_deg,
        )

        theta = theta_deg * jnp.pi / 180.0
        delta = delta_deg * jnp.pi / 180.0
        v = v_kmph / 3.6

        L = self.params["ego_lf"] + self.params["ego_lr"]

        #状态更新
        x_new = x + v * jnp.cos(theta) * self.dt
        y_new = y + v * jnp.sin(theta) * self.dt
        theta_new = theta + v / L * jnp.tan(delta) * self.dt

        theta_new_deg = normalize_angle(theta_new * 180.0 / jnp.pi) #theta限制到-180°到180°
        aS_new = jnp.stack([x_new, y_new, theta_new_deg, v_kmph, delta_deg, bw, bh, lr], axis=1)
        return self.clip_state(aS_new)

    def obst_step_euler(self, o_obst_states: ObstState) -> ObstState:
        """障碍车作匀速直线运动"""
        num_obsts = o_obst_states.shape[0]
        assert o_obst_states.shape == (num_obsts, self.state_dim)

        # 匀速直线运动模型
        o_x = o_obst_states[:, 0]
        o_v = o_obst_states[:, 3]
        o_obst_states_new = o_obst_states.at[:, 0].set(o_x + o_v/3.6*self.dt)

        assert o_obst_states_new.shape == (num_obsts, self.state_dim)
        return o_obst_states_new

    def goal_dsYddt_step(self, aS_agent_states_new: AgentState) -> Tuple[State, jnp.ndarray]:
        """根据下一步的agent位置，寻找相应的距离最近的目标点作为参考"""
        a_goals_indices = find_closest_goal_indices(aS_agent_states_new, self.all_goals)
        a_agents_indices = jnp.arange(aS_agent_states_new.shape[0])
        aS_goal_states = self.all_goals[a_agents_indices, a_goals_indices, :]
        a4_dsYddts = self.all_dsYddts[a_agents_indices, a_goals_indices, :]

        return aS_goal_states, a4_dsYddts

    @override
    def step(
            self, graph: MVEEnvGraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[MVEEnvGraphsTuple, jnp.ndarray, Reward, Cost, Cost, Done, Info]:
        # get information from graph
        agent_states = graph.type_states(type_idx=MVE.AGENT, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=MVE.GOAL, n_type=self.num_agents)
        obst_states = graph.type_states(type_idx=MVE.OBST, n_type=self.num_obsts)
        next_obst_states = self.obst_step_euler(obst_states)

        # calculate next graph
        action = self.transform_action(action)
        next_agent_states = self.agent_step_euler(agent_states, goal_states, action)
        next_goal_states, next_dsYddts = self.goal_dsYddt_step(next_agent_states)
        next_env_state = MVEEnvState(next_agent_states, next_goal_states, next_obst_states)
        info = {}

        # the episode ends when reaching max_episode_steps
        done = jnp.array(False)

        # calculate reward and cost in this graph
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


        return self.get_graph(next_env_state), next_dsYddts, reward, cost, cost_real, done, info

    def get_reward(self, graph: MVEEnvGraphsTuple, ad_action: Action) -> Reward:
        num_agents = graph.env_states.agent.shape[0]
        num_goals = graph.env_states.goal.shape[0]
        assert num_agents == num_goals

        aS_agents_states = graph.type_states(type_idx=MVE.AGENT, n_type=num_agents)
        aS_goals_states = graph.type_states(type_idx=MVE.GOAL, n_type=num_goals)
        # state: x, y, θ, v, δ, bw, bh, lr

        a4_e = aS_agents_states[:, :3] - aS_goals_states[:, :3]

        # 权重矩阵
        W = jnp.diag(jnp.array([1e-4, 1e-4, 1e-8]))

        reward = -jnp.sqrt(jnp.einsum('ai, ij, ja -> a', a4_e, W, a4_e.transpose())).mean()

        # 动作惩罚
        reward -= (ad_action[:, 0]**2).mean() * 0.0001

        return reward

    def get_cost(self, graph: MVEEnvGraphsTuple, action: Action) -> Tuple[Cost, Cost]:
        """使用射线法计算的scaling factor：α为cost_real的评判指标，1-α<0安全，>=0不安全；
        对于cost，使用由α计算的CBF（H(α)）作为衡量标准，H(α)<0安全，>=0不安全"""
        thresh = self.params["alpha_thresh"]
        num_agents = graph.env_states.agent.shape[0]
        num_obsts = graph.env_states.obstacle.shape[0]

        def get_cbf_constraints_optimized_between_states(s1, s2, action):
            # 注意s1、s2和action应该用国际单位
            gamma = self.params["gamma"]
            L = self.params["ego_lf"] + self.params["ego_lr"]

            # state: x y θ v δ bw bh lr
            p = jnp.array([s1[0], s1[1], s1[2]])
            v = s1[3]
            delta_r = s1[4]
            delta = action[0]
            bw = s1[5]
            bh = s1[6]
            lr = s1[7]

            # 定义包装函数：输入为后轴位姿，计算 alpha
            def alpha_fn(p_in):
                # 这里的 p_in 是 [x, y, theta]
                s1_full = jnp.array([p_in[0], p_in[1], p_in[2], v, delta_r, bw, bh, lr])
                return scaling_calc(s1_full, s2)

            # 求解 alpha 及一阶次梯度 (JAX会自动处理从CG到后轴的链式求导)
            alpha, grad_p = jax.value_and_grad(alpha_fn)(p)

            # 构建一阶运动学向量场 f_c 与 g_c
            fc = jnp.array([v * jnp.cos(p[2]), v * jnp.sin(p[2]), 0.0])
            gc = jnp.array([0.0, 0.0, v / L])  # 控制输入仅为 delta，所以 gc 是 3x1 的向量

            # CBF 条件: \dot{h} >= -gamma*h，即-(Lfh + Lgh*u + gamma*h) <= 0
            # 这里h>0才表示安全，故h=alpha-thresh
            Lfh = jnp.dot(grad_p, fc)
            Lgh = jnp.dot(grad_p, gc)
            cost = -(Lfh + Lgh*delta + gamma*(alpha - thresh))

            # 真实安全判断条件：h <= 0
            # 这里h=1-alpha
            cost_real = 1 - alpha

            return cost, cost_real

        def get_cbf_constraints_optimized_between_state_and_bound(s1, A, b, action):
            # 注意s1和action应该是国际单位
            gamma = self.params["gamma"]
            L = self.params["ego_lf"] + self.params["ego_lr"]

            # state: x y θ v δ bw bh lr
            p = jnp.array([s1[0], s1[1], s1[2]])
            v = s1[3]
            delta_r = s1[4]
            delta = action[0]
            bw = s1[5]
            bh = s1[6]
            lr = s1[7]

            def alpha_fn_2(p_in):
                # 这里的 p_in 是 [x, y, theta]
                s1_full = jnp.array([p_in[0], p_in[1], p_in[2], v, delta_r, bw, bh, lr])
                return scaling_calc_bound(s1_full, A, b)

            alpha, grad_p = jax.value_and_grad(alpha_fn_2)(p)

            fc = jnp.array([v * jnp.cos(p[2]), v * jnp.sin(p[2]), 0.0])
            gc = jnp.array([0.0, 0.0, v / L])

            # CBF 条件: \dot{h} >= -gamma*h，即-(Lfh + Lgh*u + gamma*h) <= 0
            # 这里h>0才表示安全，故h=alpha-thresh
            Lfh = jnp.dot(grad_p, fc)
            Lgh = jnp.dot(grad_p, gc)
            cost = -(Lfh + Lgh * delta + gamma * (alpha - thresh))

            # 真实安全判断条件：h <= 0
            # 这里h=1-alpha
            cost_real = 1 - alpha

            return cost, cost_real

        # state: x y θ v δ bw bh lr
        convert_vec_s = jnp.array([1., 1., 180/jnp.pi, 3.6, 180/jnp.pi, 1., 1., 1.])
        agent_states = graph.type_states(type_idx=MVE.AGENT, n_type=num_agents)
        agent_states_metric = agent_states / convert_vec_s

        # action: δ
        convert_vec_a = jnp.array([180/jnp.pi])
        action_metric = action / convert_vec_a

        # agent之间的scaling factor
        """
        if num_agents == 1:
            a_agent_cost = -jnp.ones((1,), dtype=jnp.float32)
        else :
            i_pairs, j_pairs = gen_i_j_pairs_no_identical(num_agents, num_agents)
            state_i_pairs = agent_states[i_pairs, :]
            state_j_pairs = agent_states[j_pairs, :]
            alpha_pairs = jax.vmap(scaling_calc, in_axes=(0, 0))(state_i_pairs, state_j_pairs)
            alpha_matrix = jnp.full((num_agents, num_agents), INF)  # 初始化矩阵，填充无穷大
            alpha_matrix = alpha_matrix.at[i_pairs, j_pairs].set(alpha_pairs)
            # 每个agent对应的行取最大值（即与其他agent的最小α，α越小越不安全）
            a_agent_cost = jnp.max(thresh-alpha_matrix, axis=1)
            a_agent_cost_real = jnp.max(1-alpha_matrix, axis=1) # α*>1 表示真实安全
        """
        a_agent_cost = -jnp.ones((num_agents,), dtype=jnp.float32) * 3 # debug
        a_agent_cost_real = -jnp.ones((num_agents,), dtype=jnp.float32) * 3 # debug

        # agent 和 obst 之间的scaling factor
        if num_obsts == 0:
            a_obst_cost = -jnp.ones((num_agents,), dtype=jnp.float32) * 3
            a_obst_cost_real = -jnp.ones((num_agents,), dtype=jnp.float32) * 3
        else:
            obstacle_states = graph.type_states(type_idx=MVE.OBST, n_type=num_obsts)
            obstacle_states_metric = obstacle_states / convert_vec_s
            i_pairs, j_pairs = gen_i_j_pairs(num_agents, num_obsts)
            state_i_pairs = agent_states_metric[i_pairs, :]
            action_i_pairs = action_metric[i_pairs, :]
            state_j_pairs = obstacle_states_metric[j_pairs, :]

            cost_pairs, cost_real_pairs = jax.vmap(get_cbf_constraints_optimized_between_states, in_axes=(0, 0, 0))(
                state_i_pairs, state_j_pairs, action_i_pairs)
            cost_matrix = cost_pairs.reshape((num_agents, num_obsts))
            cost_real_matrix = cost_real_pairs.reshape((num_agents, num_obsts))

            a_obst_cost = jnp.max(cost_matrix, axis=1)
            a_obst_cost_real = jnp.max(cost_real_matrix, axis=1)
        # a_obst_cost = -jnp.ones((num_agents,), dtype=jnp.float32) # debug


        # agent 和 bound 之间的scaling factor，只对y方向有约束
        state_range = self.params["default_state_range"]
        yl = state_range[2]
        A_l = jnp.array([[0., 1.]])
        b_l = jnp.array([yl])
        a_bound_yl_cost, a_bound_yl_cost_real = jax.vmap(get_cbf_constraints_optimized_between_state_and_bound,
            in_axes=(0, None, None, 0))(agent_states_metric, A_l, b_l, action_metric)

        yh = state_range[3]
        A_h = jnp.array([[0., -1.]])
        b_h = jnp.array([-yh])
        a_bound_yh_cost, a_bound_yh_cost_real = jax.vmap(get_cbf_constraints_optimized_between_state_and_bound,
            in_axes=(0, None, None, 0))(agent_states_metric, A_h, b_h, action_metric)

        # a_bound_yl_cost = -jnp.ones((num_agents,), dtype=jnp.float32) * 3 # debug
        # a_bound_yh_cost = -jnp.ones((num_agents,), dtype=jnp.float32) * 3 # debug

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
        eps = 1.
        cost = jnp.where(cost <= 0.0, cost, cost + eps)
        cost = jnp.clip(cost, a_min=-3.0)

        return cost, cost_real

    @override
    def render_video(
            self,
            rollout: Rollout,
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
        obsts_state = graph0.type_nodes(type_idx=MVE.OBST, n_type=self.num_obsts)
        # state: x, y, θ, v, δ, bw, bh, lr
        obsts_pos_rear = obsts_state[:, :2]
        obsts_theta = obsts_state[:, 2]
        obsts_bb_size = obsts_state[:, 5:7]
        obsts_lr = obsts_state[:, 7]
        obsts_radius = jnp.linalg.norm(obsts_bb_size, axis=1)
        # TODO: 更改车辆中心为车体中心 xy
        obsts_pos_center_x = obsts_pos_rear[:, 0] + obsts_lr * jnp.cos(obsts_theta * jnp.pi / 180.0)
        obsts_pos_center_y = obsts_pos_rear[:, 1] + obsts_lr * jnp.sin(obsts_theta * jnp.pi / 180.0)
        obsts_pos = jnp.stack([obsts_pos_center_x, obsts_pos_center_y], axis=1)
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
        # state: x, y, θ, v, δ, bw, bh, lr  a0 ... a5
        agents_pos_rear = agents_state[:, :2] # 这是后轴
        agents_theta = agents_state[:, 2]     # 角度
        agents_bb_size = agents_state[:, 5:7]
        agents_lr = agents_state[:, 7]        # 提取 lr
        agents_radius = jnp.linalg.norm(agents_bb_size, axis=1)
        # 计算几何中心
        agents_pos_center_x = agents_pos_rear[:, 0] + agents_lr * jnp.cos(agents_theta * jnp.pi / 180.0)
        agents_pos_center_y = agents_pos_rear[:, 1] + agents_lr * jnp.sin(agents_theta * jnp.pi / 180.0)
        agents_pos = jnp.stack([agents_pos_center_x, agents_pos_center_y], axis=1)
        # mean_obsts_radius = jnp.mean(obsts_radius) if self.num_obsts > 0 else 1.0
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
        # state:  x, y, θ, v, δ, bw, bh, lr
        all_ref_xs = ref_goals[:, :, 0].reshape(-1)
        all_ref_ys = ref_goals[:, :, 1].reshape(-1)
        ax.scatter(all_ref_xs, all_ref_ys, color=goal_color, zorder=7, s=5, alpha=1.0, marker='.')

        # plot edges
        all_raw_pos = graph0.states[:, :2]
        all_theta = graph0.states[:, 2]
        all_lr = graph0.states[:, 7]

        # 判断：只有图节点前 num_agents 个是 agent，仅对它们转换中心，其他直接用 raw_pos
        is_agent = np.arange(len(all_raw_pos)) < self.num_agents
        all_pos_x = jnp.where(is_agent, all_raw_pos[:, 0] + all_lr * jnp.cos(all_theta * jnp.pi / 180.0), all_raw_pos[:, 0])
        all_pos_y = jnp.where(is_agent, all_raw_pos[:, 1] + all_lr * jnp.sin(all_theta * jnp.pi / 180.0), all_raw_pos[:, 1])
        all_pos = jnp.stack([all_pos_x, all_pos_y], axis=1)

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
            n_pos_t_raw = graph.states[:-1, :2] # 最后一个node是padding，不要
            n_theta_t = graph.states[:-1, 2]
            n_bb_size_t = graph.nodes[:-1, 5:7]
            n_lr_t = graph.states[:-1, 7]
            n_radius = jnp.linalg.norm(n_bb_size_t, axis=1)

            c_x_all = n_pos_t_raw[:, 0] + n_lr_t * np.cos(n_theta_t * np.pi / 180.0)
            c_y_all = n_pos_t_raw[:, 1] + n_lr_t * np.sin(n_theta_t * np.pi / 180.0)
            n_pos_t = np.stack([c_x_all, c_y_all], axis=1)

            # update agents' positions and labels
            for ii in range(self.num_agents):
                c_x, c_y = float(c_x_all[ii]), float(c_y_all[ii])
                plot_agents_arrow[ii].set_data(x=c_x, y=c_y,
                                               dx=jnp.cos(n_theta_t[ii] * jnp.pi / 180) * n_radius[ii] / 2,
                                               dy=jnp.sin(n_theta_t[ii] * jnp.pi / 180) * n_radius[ii] / 2)
                plot_agents_rec[ii].set_xy(xy=(c_x - n_bb_size_t[ii, 0] / 2, c_y - n_bb_size_t[ii, 1] / 2))
                plot_agents_rec[ii].set_angle(angle=n_theta_t[ii])
                agent_labels[ii].set_position((c_x, c_y))

            # update obstacles' positions
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
            rel_state = agent_state_i - goal_state_i
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
            rel_state_pairs = agent_state_i_pairs - obst_state_j_pairs
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
    def get_graph(self, env_state: MVEEnvState, obst_as_agent:bool = False) -> MVEEnvGraphsTuple:
        num_agents = env_state.agent.shape[0]
        num_goals = env_state.goal.shape[0]
        num_obsts = env_state.obstacle.shape[0]
        assert num_agents > 0 and num_goals > 0, "至少需要设定agent和goal!"
        assert num_agents == num_goals, "每一个agent对应一个goal"
        # node features
        # states
        node_feats = jnp.zeros((num_agents + num_goals + num_obsts, self.node_dim))
        node_feats = node_feats.at[:num_agents, :self.state_dim].set(env_state.agent)
        node_feats = node_feats.at[num_agents: num_agents + num_goals, :self.state_dim].set(env_state.goal)
        if num_obsts > 0:
            node_feats = node_feats.at[num_agents + num_goals:, :self.state_dim].set(env_state.obstacle)

        # bounding box 长宽和lr
        # state: x y θ v δ bw bh lr
        if obst_as_agent:
            node_feats = node_feats.at[:num_agents, 5:7].set(self.params["obst_bb_size"])
            node_feats = node_feats.at[:num_agents, 7].set(self.params["obst_lr"])
        else:
            node_feats = node_feats.at[:num_agents, 5:7].set(self.params["ego_bb_size"])
            node_feats = node_feats.at[:num_agents, 7].set(self.params["ego_lr"])
        if num_obsts > 0:
            node_feats = node_feats.at[num_agents + num_goals:, 5:7].set(self.params["obst_bb_size"])
            node_feats = node_feats.at[num_agents + num_goals:, 7].set(self.params["obst_lr"])

        # indicators
        node_feats = node_feats.at[:num_agents, -1].set(1.0)
        node_feats = node_feats.at[num_agents: num_agents + num_goals, -2].set(1.0)
        if num_obsts > 0:
            node_feats = node_feats.at[num_agents + num_goals:, -3].set(1.0)

        # node type
        node_type = -jnp.ones((num_agents + num_goals + num_obsts), dtype=jnp.int32)
        node_type = node_type.at[:num_agents].set(MVE.AGENT)
        node_type = node_type.at[num_agents: num_agents + num_goals].set(MVE.GOAL)
        if num_obsts > 0:
            node_type = node_type.at[num_agents + num_goals:].set(MVE.OBST)

        # edges
        edge_blocks = self.edge_blocks(env_state)

        # create graph
        states = jnp.concatenate([node_feats[:num_agents, :-3], node_feats[num_agents: num_agents + num_goals, :-3]],
                                 axis=0)
        if num_obsts > 0:
            states = jnp.concatenate([states, node_feats[num_agents + num_goals:, :-3]], axis=0)
            new_env_state = MVEEnvState(node_feats[:num_agents, :-3],
                                        node_feats[num_agents: num_agents + num_goals, :-3],
                                        node_feats[num_agents + num_goals:, :-3])
        else:
            new_env_state = MVEEnvState(node_feats[:num_agents, :-3],
                                        node_feats[num_agents: num_agents + num_goals, :-3],
                                        jnp.empty((0, self.state_dim)))
        return GetGraph(node_feats, node_type, edge_blocks, new_env_state, states).to_padded()

    @override
    def state_lim(self, state: Optional[State]) -> Tuple[State, State]:
        """世界坐标系下的状态约束"""
        lower_lim = self.params["rollout_state_range"][jnp.array([0, 2, 4, 6, 8, 10, 12, 14])]
        upper_lim = self.params["rollout_state_range"][jnp.array([1, 3, 5, 7, 9, 11, 13, 15])]
        return lower_lim, upper_lim


    @override
    def action_lim(self) -> Tuple[Action, Action]:
        lower_lim = jnp.array([-10.])[None, :].repeat(self.num_agents, axis=0) # δ: °
        upper_lim = jnp.array([10.])[None, :].repeat(self.num_agents, axis=0)
        return lower_lim, upper_lim

    @override
    @ft.partial(jax.jit, static_argnums=(0,))
    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        _, cost_real = self.get_cost(graph)
        return jnp.any(cost_real >= 0.0, axis=-1)

    def plot_agent_speed_from_rollout(self, rollout: Rollout, record: Record, save_path=None, use_body_frame=False):
        """
        绘制 agent 速度图和 Psid_metric（与 Psi 在同一个图中）
        :param rollout: 一个包含图数据的 Rollout 对象
        :param a_Psid_metric: Psid_metric 数据
        :param save_path: 如果传入路径，就保存为 png 文件，否则直接显示
        :param use_body_frame: 是否使用车身坐标系进行速度转换
        """
        T = len(rollout.graph.n_node)  # 时间步数
        A = self.num_agents  # 从类的实例获取 agent 数量
        vx_TA = np.zeros((T, A), dtype=np.float32)
        vy_TA = np.zeros((T, A), dtype=np.float32)
        x_T = np.zeros(T, dtype=np.float32)  # 世界坐标系下的 X 位置
        psi_T = np.zeros((T, A), dtype=np.float32)  # 每个 agent 的转角
        a_Psid_metric = record.Psid
        ao_BD= record.ao_BD
        a_deltaf=record.deltaf*7
        BD_lane=record.BD_lane
        a_Ye=record.a_Ye
        BD_weighted_sum=record.BD_weighted_sum
        # 遍历所有时间步，提取速度和位置信息
        for t in range(T):
            g = tree_index(rollout.graph, t)

            # 提取速度（vx 和 vy）
            vx = np.array(g.states[:A, 2])
            vy = np.array(g.states[:A, 3])

            # 提取位置（X 和 Psi）
            x = np.array(g.states[:A, 0])  # X 位置
            psi_deg = np.array(g.states[:A, 4])  # 转角 Psi (单位: 度)

            x_T[t] = np.mean(x)  # 取所有 agent 的平均位置作为该时间步的横坐标
            psi_T[t] = psi_deg  # 存储转角

            if use_body_frame:
                # 转换到车身坐标系
                theta = psi_deg * np.pi / 180.0
                c, s = np.cos(theta), np.sin(theta)
                vbx = c * vx + s * vy
                vby = -s * vx + c * vy
                vx, vy = vbx, vby

            # 存储速度
            vx_TA[t] = vx
            vy_TA[t] = vy

        # 计算总速度
        speed_TA = np.sqrt(vx_TA**2 + vy_TA**2)  # km/h
        time = x_T  # 使用世界坐标系下的 X 作为时间轴

        # 绘制图形
        fig, axes = plt.subplots(6, 1, figsize=(10, 10), sharex=True)

        for a in range(A):
            axes[0].plot(time, vx_TA[:, a], label=f"agent{a}")
        axes[0].set_ylabel("vx (km/h)")
        axes[0].legend(ncol=4, fontsize=8)

        for a in range(A):
            axes[1].plot(time, vy_TA[:, a], label=f"agent{a}")
        axes[1].set_ylabel("vy (km/h)")

        for a in range(A):
            axes[2].plot(time, a_Ye[:, a], label=f"a_Ye")
        axes[2].set_ylabel("a_Ye")

        # 绘制转角和 Psid_metric 图
        for a in range(A):
            axes[3].plot(time, psi_T[:, a], label=f"agent{a} - Psi", linestyle='--')
            axes[3].plot(time, a_Psid_metric[:, a], label=f"agent{a} - Psid_metric", linestyle='-')
        # axes[3].plot(time, YD_deta[:, a], label=f"agent{a} - YD_deta", linestyle='-.')
        axes[3].set_ylabel("Psi (degrees) / Psid_metric")
        axes[3].set_xlabel("World X Position (m)")

        for a in range(A):

            axes[4].plot(time, a_deltaf[:, a], label=f"deltaf", linestyle='-')
        # axes[3].plot(time, YD_deta[:, a], label=f"agent{a} - YD_deta", linestyle='-.')
        axes[4].set_ylabel("Psi (degrees) / Psid_metric")
        axes[4].set_xlabel("World X Position (m)")

        for a in range(A):
            #axes[4].plot(time, ao_BD[:, a], label=f"ao_BD")
            axes[5].plot(time, ao_BD[:, a], label=f"ao_BD",color='r')
            axes[5].plot(time, BD_lane[:,a], label=f"ao_BD",color='b')
            axes[5].plot(time, BD_weighted_sum[:,a], label=f"ao_BD",color='k')
        # axes[5].plot(time, a_Ye[:,a], label=f"a_ye")
        # axes[3].plot(time, YD_deta[:, a], label=f"agent{a} - YD_deta", linestyle='-.')
        axes[5].set_ylabel("Psi (degrees) / Psid_metric")
        axes[5].set_xlabel("World X Position (m)")

        title = "Agent speed (body frame)" if use_body_frame else "Agent speed (world frame)"
        fig.suptitle(title)
        fig.tight_layout()

        # 保存图像或展示
        if save_path is not None:
            plt.savefig(save_path, dpi=150)
            plt.close(fig)
        else:
            plt.show()