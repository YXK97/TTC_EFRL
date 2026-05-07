import jax
import jax.numpy as jnp
import jax.nn as nn

from .typing import Array, Pos2d, State
from .utils import calc_2d_rot_matrix

# 全局数值稳定极小值（核心：防止除零/NaN）
EPS = 1e-6

def safe_norm(x, axis=-1):
    return jnp.sqrt(jnp.sum(x ** 2, axis=axis) + EPS)

@jax.jit
def compute_intersections(O: Pos2d, V: Pos2d, A, b, fill: Pos2d) -> Array:
    """计算输入两点的直线与Ax=b各边的交点"""
    # 解析输入直线系数（向量化扩展为(k,)形状，适配A的q条边）
    a0 = V[1] - O[1]
    b0 = O[0] - V[0]
    c0 = V[0] * O[1] - O[0] * V[1]
    max_abs = jnp.array([jnp.abs(a0), jnp.abs(b0), jnp.abs(c0)]).max()
    a0 = a0 / max_abs
    b0 = b0 / max_abs
    c0 = c0 / max_abs
    a0 = jnp.full_like(A[:, 0], a0)  # (k,)
    b0 = jnp.full_like(A[:, 0], b0)  # (k,)
    c0 = jnp.full_like(A[:, 0], c0)  # (k,)
    # 解析A*x = b的q条边的系数（每条边：ai x + bi y - b_i' = 0）
    ai = A[:, 0]  # (k,)，每条边的x系数
    bi = A[:, 1]  # (k,)，每条边的y系数
    ci = -b  # (k,)，每条边的常数项（ai x + bi y + ci = 0 → ci = -b_i'）
    # 向量化计算行列式D（判断平行/相交）
    D = a0 * bi - ai * b0  # (k,)，批量计算k个行列式
    # 向量化计算交点分子（避免除法提前引入NaN）
    x_num = b0 * ci - bi * c0  # (k,)，x坐标分子
    y_num = ai * c0 - a0 * ci  # (k,)，y坐标分子
    # 处理交点：D≠0时计算真实交点，D≈0时设为inf（平行/重合）
    x = jnp.where(jnp.abs(D) > EPS, x_num / D, fill[0])
    y = jnp.where(jnp.abs(D) > EPS, y_num / D, fill[1])
    # 后续处理
    cand = jnp.stack([x, y], axis=1)
    cand = filter_ray_direction(cand, O, V, fill)
    cand = filter_in_bound(cand, A, b, fill)

    return cand

@jax.jit
def filter_ray_direction(k2_intersections: Array, O: Pos2d, V: Pos2d, fill: Pos2d) -> Array:
    """筛选与射线方向一致”的交点（射线：从O指向V）"""
    # 射线方向向量
    dir_vec = V - O  # (2,)
    # 交点相对于r1的向量：inter_vec = 交点 - r1
    inter_vec = k2_intersections - O  # (k, 2)
    # 点积≥0（同向）：dir_vec · inter_vec ≥ 0
    k_same_dir = jnp.sum(dir_vec * inter_vec, axis=1) >= -EPS  # (k,)

    # 有效交点：同向，否则设为fill
    return jnp.where(k_same_dir[:, None], k2_intersections, fill)

@jax.jit
def filter_in_bound(k2_intersections: Array, A, b, fill: Pos2d) -> Array:
    """筛选满足Ax<=b的点"""
    def _single_in_bound(intersection: Pos2d, A, b) -> Array:
        val = jnp.dot(A, intersection) - b
        in_bound = jnp.max(val) <= EPS
        return in_bound
    k_in_bound = jax.vmap(_single_in_bound, in_axes=(0, None, None))(k2_intersections, A, b)

    return jnp.where(k_in_bound[:, None], k2_intersections, fill)


@jax.jit
def scaling_calc(s1: State, s2: State) -> Array:
    """计算agent和agent/obst的scaling factor"""
    # state: x y vx vy θ dθdt bb_w bb_h
    # 注意单位需要是 m m m/s m/s rad rad/s m m
    O1 = s1[:2]; O2 = s2[:2]
    # 计算 host 和 agent/obst 的顶点， host/agent/obst均为矩形
    Q1 = calc_2d_rot_matrix(s1[4])
    m_V = jnp.array([[s1[6] / 2, s1[7] / 2],
                    [s1[6] / 2, -s1[7] / 2],
                    [-s1[6] / 2, s1[7] / 2],
                    [-s1[6] / 2, -s1[7] / 2]])
    m_V = O1 + m_V @ Q1.T
    Q2 = calc_2d_rot_matrix(s2[4])
    n_P = jnp.array([[s2[6] / 2, s2[7] / 2],
                    [s2[6] / 2, -s2[7] / 2],
                    [-s2[6] / 2, s2[7] / 2],
                    [-s2[6] / 2, -s2[7] / 2]])
    n_P = O2 + n_P @ Q2.T

    # 计算S1和S2的A1 b1 A2 b2
    Ao = jnp.array([[ 1., 0.],  # x<=b/2
                    [-1., 0.],  # x>=-b/2
                    [ 0., 1.],  # y<=h/2
                    [ 0.,-1.]]) # y>=-h/2
    A1 = Ao @ Q1.T
    A2 = Ao @ Q2.T
    b1o = jnp.array([s1[6] / 2, s1[6] / 2, s1[7] / 2, s1[7] / 2])
    b2o = jnp.array([s2[6] / 2, s2[6] / 2, s2[7] / 2, s2[7] / 2])
    b1 = b1o + A1 @ O1
    b2 = b2o + A2 @ O2

    # host 向 自身顶点 发射射线
    mk2_intersections = jax.vmap(compute_intersections, in_axes=(None, 0, None, None, None))(
        O1, m_V, A2, b2, O1+1e8)
    mk_dist = safe_norm(mk2_intersections - O1, axis=-1)
    mk_dist0 = safe_norm(m_V - O1, axis=-1)[:, None].repeat(mk_dist.shape[1], axis=1)
    mk_scaling = mk_dist / mk_dist0
    # host 向 对方极点 发射射线
    nl2_intersections = jax.vmap(compute_intersections, in_axes=(None, 0, None, None, None))(
        O1, n_P, A1, b1, O1+1e-8)
    nl_dist = safe_norm(nl2_intersections - O1, axis=-1)
    nl_dist0 = safe_norm(n_P - O1, axis=-1)[:, None].repeat(nl_dist.shape[1], axis=1)
    nl_scaling = nl_dist0 / nl_dist
    scaling = jnp.array([mk_scaling.min(), nl_scaling.min()]).min()

    # 判断S1缩放中心是否在S2中
    in_bound = jnp.max(A2 @ O1 - b2)
    O_in_S2 = jax.nn.sigmoid(1e6 * in_bound)

    alpha = O_in_S2 * scaling
    return alpha

@jax.jit
def scaling_calc_bound(s: State, A: Array, b: Array) -> Array:
    """计算agent和bound的scaling factor，agent为有界多面凸集，bound为Ax<=b描述的单条直线分割的半空间，即A和b均为一行"""
    O = s[:2]
    # 计算host的顶点，host为矩形
    Q = calc_2d_rot_matrix(s[4])
    m_V = jnp.array([[ s[6]/2,  s[7]/2],
                     [ s[6]/2, -s[7]/2],
                     [-s[6]/2,  s[7]/2],
                     [-s[6]/2, -s[7]/2]])
    m_V = O + m_V @ Q.T

    # host向自身顶点发射射线
    mk2_intersections = jax.vmap(compute_intersections, in_axes=(None, 0, None, None, None))(
        O, m_V, A, b, O + 1e8)
    mk_dist = safe_norm(mk2_intersections - O, axis=-1)
    mk_dist0 = safe_norm(m_V - O, axis=-1)[:, None].repeat(mk_dist.shape[1], axis=1)
    scaling = (mk_dist / mk_dist0).min()

    # 判断S1缩放中心是否在bound中
    in_bound = jnp.max(A @ O - b)
    O_in_bound = jax.nn.sigmoid(1e6 * in_bound)

    alpha = O_in_bound * scaling
    return alpha

def compute_h_ij(p1, R1, Q1, p2, R2, Q2, z):
    """
    计算论文中的公式 (19)：支撑超平面到另一个椭球的解析距离 h_{ij}

    参数:
    p1, p2: (2,) 椭球 1 和 2 的中心位置
    R1, R2: (2, 2) 旋转矩阵 SO(2)
    Q1, Q2: (2, 2) 形状矩阵（对角阵，对角线元素为半轴长）
    z: (2,) 单位向量，指定了椭球 1 表面的支撑超平面切点映射
    """
    # 计算方向调整后的形状矩阵 (orientation-adjusted shape matrix)
    # Q_bar1 = R1 @ Q1 @ R1^T, 但实际上逆矩阵用得更多
    # 为了数值稳定性和计算效率，直接计算 Q_bar1 的逆: Q_bar1_inv = R1 @ Q1^{-1} @ R1^T
    inv_Q1 = jnp.diag(1.0 / jnp.diag(Q1))
    Q_bar1_inv = R1 @ inv_Q1 @ R1.T

    Q_bar2 = R2 @ Q2 @ R2.T

    # 核心中间变量 v = Q_bar1_inv @ z
    v = Q_bar1_inv @ z
    v_norm = jnp.linalg.norm(v)

    # 公式 (19) 的分子部分
    # term1 = - || \bar{Q}_2 \bar{Q}_1^{-1} z ||
    term1 = -jnp.linalg.norm(Q_bar2 @ v)
    # term2 = (p2 - p1)^T \bar{Q}_1^{-1} z
    term2 = jnp.dot(p2 - p1, v)

    # 计算距离 h_{ij}
    h = (term1 + term2 - 1.0) / v_norm

    return h

def optimize_supporting_hyperplane(p1, R1, Q1, p2, R2, Q2, num_steps=20, lr=0.1):
    """
    利用梯度上升寻找最优的 z 向量，使得 h_{ij} 最大化 (消除保守性, 等价于寻找真实最短距离)
    对标论文中的公式 (20) ~ (24)
    """
    # 初始化 z: 一个比较好的初值是从 p1 指向 p2 的方向
    dp = p2 - p1
    z_init = dp / (jnp.linalg.norm(dp) + 1e-8)

    def body_fn(i, z_val):
        # 1. 计算 h_ij 关于 z 的梯度: \partial h_{ij} / \partial z
        grad_h = jax.grad(compute_h_ij, argnums=6)(p1, R1, Q1, p2, R2, Q2, z_val)

        # 2. 梯度上升更新 z (等价于论文中切空间上的虚拟输入 u_z)
        z_new = z_val + lr * grad_h

        # 3. 投影回单位圆 (保证 ||z|| = 1)
        z_new = z_new / jnp.linalg.norm(z_new)
        return z_new

    # 使用 jax.lax.fori_loop 进行高效的图内循环迭代
    z_opt = jax.lax.fori_loop(0, num_steps, body_fn, z_init)

    # 计算最终收敛的准确距离
    exact_distance = compute_h_ij(p1, R1, Q1, p2, R2, Q2, z_opt)

    return exact_distance, z_opt

def calc_rsh_distance(s1: jnp.ndarray, s2: jnp.ndarray) -> jnp.ndarray:
    """
    主接口函数：计算两个车辆（椭球建模）之间的真实 RSH 安全距离
    假设 state s 的结构为: [x, y, v_x, v_y, theta, omega, length, width]
    """
    # 提取位置 p
    p1 = s1[:2]
    p2 = s2[:2]

    # 提取旋转矩阵 R
    R1 = calc_2d_rot_matrix(s1[4])
    R2 = calc_2d_rot_matrix(s2[4])

    # 提取车辆的 长 L 和 宽 W
    L1, W1 = s1[6], s1[7]
    L2, W2 = s2[6], s2[7]

    # 乘以 sqrt(2) 确保椭圆刚好包住矩形的四个角
    sqrt_2 = jnp.sqrt(2.0)

    Q1 = jnp.array([[(sqrt_2 / 2.0) * L1, 0.0],[0.0,                 (sqrt_2 / 2.0) * W1]
                    ])

    Q2 = jnp.array([[(sqrt_2 / 2.0) * L2, 0.0],[0.0,                 (sqrt_2 / 2.0) * W2]
                    ])
    # 提取形状矩阵 Q (椭球的长半轴和短半轴)
    # 矩形的长宽分别对应椭球的两个轴径 (length/2, width/2)
   # Q1 = jnp.array([[s1[6] / 2.0, 0.0],
        #            [0.0, s1[7] / 2.0]])
   # Q2 = jnp.array([[s2[6] / 2.0, 0.0],[0.0, s2[7] / 2.0]])

    # 计算最优支撑超平面距离 (迭代寻找真实距离)
    distance, optimal_z = optimize_supporting_hyperplane(p1, R1, Q1, p2, R2, Q2, num_steps=10, lr=0.2)

    return distance


def calc_rsh_distance_bound(s: jnp.ndarray,
                            n: jnp.ndarray,
                            b: jnp.ndarray) -> jnp.ndarray:
    """
    车辆（椭球建模）到半空间边界的 RSH 距离，论文 eq.(19) 对半平面边界的解析特化版。

    推导：当 "E_j" 退化为半空间 {q: nᵀq >= b[0]} 时，支撑超平面到边界的最大化距离
    在最优 z = n 处取到解析解：h = nᵀpᵢ - b[0] - ||Q̄ᵢ n||
    无需梯度迭代（∵ 边界为平面，最优 z 固定为 n）。

    Args:
        s:  agent state (state_dim,)，[x, y, vx, vy, θ(°), dθdt, bw(m), bh(m)]
        n:  单位法向量 (2,)，指向安全区域内部（如下边界取 [0., 1.]）
        b:  边界截距 (1,)，安全区域为 nᵀq >= b[0]

    Returns:
        h:  RSH 距离标量（米），h > 0 表示安全，与 calc_rsh_distance 量纲一致
    """
    p_i = s[:2]
    theta_rad = s[4] * jnp.pi / 180.0
    bw, bh = s[6], s[7]

    # 旋转矩阵 R_i (与 calc_rsh_distance 中一致)
    c, sin_ = jnp.cos(theta_rad), jnp.sin(theta_rad)
    R_i = jnp.array([[c, -sin_], [sin_, c]])

    # 半轴长（与 calc_rsh_distance 保持 √2/2 倍，使椭圆外接矩形四顶点）
    sqrt_2 = jnp.sqrt(2.0)
    q1 = (sqrt_2 / 2.0) * bw   # 长方向半轴
    q2 = (sqrt_2 / 2.0) * bh   # 宽方向半轴

    # ||Q̄_i n|| = sqrt((q1*(R_iᵀn)[0])² + (q2*(R_iᵀn)[1])²)
    # 其中 R_i^T = [[c, sin_],[-sin_, c]]
    r = R_i.T @ n   # shape (2,)，n 在车身坐标系下的分量
    norm_Q_bar_n = jnp.sqrt((q1 * r[0]) ** 2 + (q2 * r[1]) ** 2)

    # h = nᵀpᵢ - b[0] - ||Q̄ᵢ n||
    return jnp.dot(n, p_i) - b[0] - norm_Q_bar_n
