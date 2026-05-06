import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import os
import functools as ft
import pickle
from typing import Optional, Tuple, Dict
from flax.training.train_state import TrainState

# 假设的基础类和工具函数导入
# from .module.root_finder import RootFinder
# from .utils import compute_dec_efocp_gae, compute_dec_efocp_V, val_to_optax_schedule

# ==========================================
# 1. JAX 版本的支撑超平面几何算子 (核心数学)
# ==========================================

@jax.jit
def jax_rot_mat(theta):
    """JAX 2D 旋转矩阵"""
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[c, -s], [s, c]])

@jax.jit
def compute_h12_jax(p1, p2, theta1, theta2, Q1, Q2, z12):
    """
    计算基于支撑超平面的安全距离 h12 (JAX 实现)
    p: (2,) 位置, theta: scalar 航向, Q: (2,2) 形状矩阵, z12: (2,) 支撑向量
    """
    R1 = jax_rot_mat(theta1)
    R2 = jax_rot_mat(theta2)

    Qbar1 = R1 @ Q1 @ R1.T
    Qbar2 = R2 @ Q2 @ R2.T
    Qbar1_inv = jnp.linalg.inv(Qbar1)

    dp = p2 - p1
    vz = Qbar1_inv @ z12
    n1 = jnp.linalg.norm(vz)
    vqz = Qbar2 @ vz
    n2 = jnp.linalg.norm(vqz)

    # 论文公式: h12 = (-n2 + dp^T * vz - 1.0) / n1
    h12 = (-n2 + jnp.dot(dp, vz) - 1.0) / (n1 + 1e-8)
    return h12

# ==========================================
# 2. 修改后的算法类
# ==========================================

class DefMARL_CBFs_Ellipsoid:
    def __init__(
            self,
            env,
            node_dim: int,
            edge_dim: int,
            action_dim: int,
            n_agents: int,
            Q_shapes: jnp.ndarray, # (n_agents, 2, 2) 各智能体形状
            gamma_h: float = 0.5,   # CBF 松弛系数
            lr_actor: float = 3e-4,
            lr_critic: float = 1e-3,
            **kwargs
    ):
        self.env = env
        self.n_agents = n_agents
        self.Q_shapes = Q_shapes # 预设的椭球形状
        self.gamma_h = gamma_h

        # 网络定义 (此处简化，实际应使用 Flax Module)
        # self.policy = ...
        # self.critic_Vl = ...
        # self.critic_Vh = ... (预测 h12 的网络)

        # 初始化支撑向量 z (n_agents, 2)
        self.init_z = jnp.array([1.0, 0.0]).repeat(n_agents, axis=0).reshape(n_agents, 2)

    def get_h_matrix(self, positions, headings, z_vectors):
        """
        批量计算所有智能体间的 h12 矩阵
        positions: (n, 2), headings: (n,), z_vectors: (n, 2)
        """
        def single_h(i, j):
            return compute_h12_jax(
                positions[i], positions[j],
                headings[i], headings[j],
                self.Q_shapes[i], self.Q_shapes[j],
                z_vectors[i]
            )

        # 使用 vmap 进行高效成对计算
        return jax.vmap(jax.vmap(single_h, in_axes=(None, 0)), in_axes=(0, None))(
            jnp.arange(self.n_agents), jnp.arange(self.n_agents)
        )

    def step(self, state, z_old, params, key):
        """
        执行一步，包含 z 的演化逻辑
        """
        # 1. 策略输出动作 (v, w) 和 z 的增量 uz
        # 这里 z 向量的更新遵循论文中的：z_new = z_old + dt * (I - zz^T) * uz
        action, log_pi, rnn_state = self.policy_apply(params['policy'], state, z_old, key)

        # 2. 环境 Step
        next_obs, reward, done, info = self.env.step(action)

        # 3. 更新 z (根据动作中的 uz 分量)
        uz = action[:, 3:5] # 假设动作后两位是控制支撑平面的 uz
        dt = 0.05
        proj_z = jnp.eye(2) - jnp.outer(z_old, z_old) # 简化表示，实际需对每个 agent vmap
        z_new = z_old + dt * (proj_z @ uz.T).T
        z_new = z_new / jnp.linalg.norm(z_new, axis=-1, keepdims=True)

        return next_obs, z_new, reward, done

    def compute_cbf_loss(self, rollout_data, params):
        """
        计算 Vh 网络的损失函数
        """
        # 获取 rollout 中的位置和角度数据
        pos = rollout_data.pos # (batch, T, n, 2)
        thetas = rollout_data.thetas # (batch, T, n)
        zs = rollout_data.zs # (batch, T, n, 2)

        # 实时计算每一帧的支撑超平面距离 h
        # h_vals: (batch, T, n_agents) 假设每个 agent 关注最近的障碍物
        h_vals = jax.vmap(jax.vmap(self.get_h_matrix))(pos, thetas, zs)

        # CBF 约束：h(t+1) >= (1 - gamma) * h(t)
        # 转化为 Reward/Penalty 项：
        # 如果 h < 0，产生巨大惩罚
        # 如果 h_dot + gamma*h < 0，产生安全约束违规

        # 这里的 Vh 网络学习目标是预测 h_vals
        # 损失函数参考原代码中的 Vh_train_state
        # ...
        return h_vals

    def update(self, rollouts):
        """
        更新循环
        """
        # 计算基于支撑超平面的优势函数 (GAE)
        # 此时的 costs 应该被替换为 -h_vals (因为 h 越小越危险，cost 越大)

        h_vals = self.compute_cbf_loss(rollouts, self.params)

        # 构造安全优势向量
        # bTah_Hk = (self.gamma_h - 1) * h_vals[:, :-1] + h_vals[:, 1:]

        # 调用原代码中的 update_inner 处理 PPO 梯度更新
        # self.update_inner(...)
        pass

# ==========================================
# 3. 集成到训练 Data Flow 中的具体实现
# ==========================================

def get_ellipsoid_costs(graph_nodes):
    """
    辅助函数：从图节点信息提取位置和角度，计算所有 agent 的安全距离
    graph_nodes: 包含 [x, y, theta, vx, vy, omega]
    """
    pos = graph_nodes[:, 0:2]
    theta = graph_nodes[:, 2]
    # 此处假设 z 向量存储在节点的额外维度中
    z_vecs = graph_nodes[:, 6:8]

    # 模拟上述 get_h_matrix 逻辑
    # 返回该时刻每个 agent 的最小 h12 (相对于所有邻居)
    # min_h = ...
    # return min_h
    pass

# 修改原代码中的 scan_value 函数以支持 h12
def scan_value_updated(self, rollout, init_rnn, params):
    # 1. 解析位置和支撑向量
    # 2. 调用 compute_h12_jax
    # 3. 计算 Vh(s) 预测当前的椭球距离
    # 4. 与真实的 h12 进行回归对比
    pass