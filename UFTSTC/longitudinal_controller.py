import jax
import jax.numpy as jnp
import functools as ft
from typing import Tuple

from defmarl.env.mve import MVE
from defmarl.utils.utils import calc_2d_rot_matrix
from defmarl.utils.graph import GraphsTuple


class PIDController:
    """纵向速度跟踪PID控制器（输出为纵向加速度ax）"""

    def __init__(self, num_agents:int, kp:float, ki:float, kd:float, dt:float,
                 max_integral:float, min_integral:float,
                 max_ax:float, # m/s^2
                 min_ax:float  # m/s^2
                 ):

        # PID核心参数
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        # 积分项限制（抗积分饱和）
        self.max_integral = max_integral
        self.min_integral = min_integral

        # 输出纵向加速度限制（车辆物理约束）
        self.max_ax = max_ax
        self.min_ax = min_ax

        # 状态变量（用于离散化计算）
        self.num_agents = num_agents
        self.integral = jnp.zeros((self.num_agents,), dtype=jnp.float32) # 积分项累积值
        self.last_error = jnp.zeros((self.num_agents,), dtype=jnp.float32) # 上一次的偏差（用于微分项计算）


    @ft.partial(jax.jit, static_argnums=(0,))
    def pid_acceleration(self, current_error):
        current_error = current_error / 3.6 # km/h -> m/s

        proportional = self.kp * current_error

        self.integral += current_error * self.dt
        self.integral = jnp.clip(self.integral, self.min_integral, self.max_integral)
        integral = self.ki * self.integral

        derivative = self.kd * (current_error - self.last_error) / self.dt

        # 计算总控制量（输出加速度ax）
        ax = proportional + integral + derivative

        # 更新上一次偏差（为下一次微分项计算做准备）
        self.last_error = current_error

        return ax


    @ft.partial(jax.jit, static_argnums=(0,))
    def vx_error_in_body_fixed_coordinates(self, graph: GraphsTuple) -> jnp.ndarray:
        num_agents = graph.env_states.agent.shape[0]
        num_goals = graph.env_states.goal.shape[0]
        assert num_agents == num_goals

        aS_agents_states = graph.type_states(type_idx=MVE.AGENT, n_type=num_agents)
        aS_goals_states = graph.type_states(type_idx=MVE.GOAL, n_type=num_goals)
        # state: x, y, vx, vy, θ, dθ/dt, bw, bh
        # 参数提取
        a2_goal_v_kmph = aS_goals_states[:, 2:4]
        a_goal_theta_deg = aS_goals_states[:, 4]
        a2_agent_v_kmph = aS_agents_states[:, 2:4]
        a_agent_theta_deg = aS_agents_states[:, 4]

        # 旋转矩阵计算
        a22_Q_goal = jax.vmap(calc_2d_rot_matrix, in_axes=(0))(a_goal_theta_deg)
        a22_Q_agent = jax.vmap(calc_2d_rot_matrix, in_axes=(0))(a_agent_theta_deg)

        # 自车坐标系下的横纵向速度计算
        a2_goal_v_b_kmph = jnp.einsum('aij, ai -> aj', a22_Q_goal, a2_goal_v_kmph)
        a2_agent_v_b_kmph = jnp.einsum('aij, ai -> aj', a22_Q_agent, a2_agent_v_kmph)
        a_vx_b_error = a2_goal_v_b_kmph[:, 0] - a2_agent_v_b_kmph[:, 0] # km/h

        return a_vx_b_error


    @ft.partial(jax.jit, static_argnums=(0,))
    def clip_ax(self, ax:jnp.ndarray) -> jnp.ndarray:
        """处理单位为m/s^2"""
        ax = jnp.clip(ax, self.min_ax, self.max_ax)
        return ax


    @ft.partial(jax.jit, static_argnums=(0,))
    def normalize_ax(self, ax:jnp.ndarray) -> jnp.ndarray:
        """处理单位为m/s^2"""
        ax_center = (self.max_ax + self.min_ax)/2
        ax_half = self.max_ax - ax_center
        a_normalized_ax = (ax - ax_center) / ax_half
        return a_normalized_ax


    @ft.partial(jax.jit, static_argnums=(0,))
    def calc_ax(self, graph: GraphsTuple) -> Tuple[jnp.ndarray, jnp.ndarray]:
        a_vx_error = self.vx_error_in_body_fixed_coordinates(graph) # km/h
        a_ax = self.pid_acceleration(a_vx_error)
        a_ax_clip = self.clip_ax(a_ax) # m/s^2
        #a_ax_clip = jnp.zeros_like(a_ax_clip)
        a_ax_normalize = self.normalize_ax(a_ax_clip)
        return a_ax_normalize ,  a_ax_clip