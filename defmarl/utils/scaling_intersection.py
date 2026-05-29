import jax
import jax.numpy as jnp

from typing import Tuple

from .typing import Array, State
from .utils import calc_2d_rot_matrix
from .scaling import EPS, compute_intersections, safe_norm


@jax.jit
def scaling_calc_convex_bound(s: State, A: Array, b: Array) -> Array:
    """计算 agent 与一个凸障碍区域的 scaling factor。

    这里的障碍区域由多条直线半平面共同定义：
        A @ [x, y] <= b

    与普通道路边界不同，十字路口四个角落障碍区不是单条边界线，
    而是由三条边界线围成的无界凸区域。alpha 越小表示越危险：
        alpha > 1：agent 矩形与该障碍区域有安全间隔
        alpha = 1：agent 矩形刚好接触障碍区域边界
        alpha < 1：agent 矩形已经侵入障碍区域
    """
    O = s[:2]

    # agent 的包围盒顶点。注意这里传入的 state 已经是 metric 单位：
    # x/y 为 m，theta 为 rad，包围盒长宽为 m。
    Q = calc_2d_rot_matrix(s[4])
    m_V = jnp.array([
        [s[6] / 2, s[7] / 2],
        [s[6] / 2, -s[7] / 2],
        [-s[6] / 2, s[7] / 2],
        [-s[6] / 2, -s[7] / 2],
    ])
    m_V = O + m_V @ Q.T

    # 从 agent 中心向自身四个顶点发射射线，计算射线第一次进入障碍区域边界的位置。
    # compute_intersections 会自动过滤掉不在射线方向上、或不满足 A x <= b 的交点。
    mk2_intersections = jax.vmap(compute_intersections, in_axes=(None, 0, None, None, None))(
        O, m_V, A, b, O + 1e8
    )
    mk_dist = safe_norm(mk2_intersections - O, axis=-1)
    mk_dist0 = safe_norm(m_V - O, axis=-1)[:, None].repeat(mk_dist.shape[1], axis=1)
    scaling = (mk_dist / mk_dist0).min()

    # 如果 agent 中心已经在障碍区域内部，则直接把 alpha 压到接近 0；
    # 如果中心在外部，则使用上面基于边界交点得到的缩放比例。
    in_bound = jnp.max(A @ O - b)
    O_in_bound = jax.nn.sigmoid(1e6 * in_bound)
    return O_in_bound * scaling


@jax.jit
def intersection_corner_bounds() -> Tuple[Array, Array]:
    """返回十字路口四个角落障碍区域的 A、b。

    每个区域均写成 A @ [x, y] <= b 的形式，顺序为：
        0. 西南障碍区域
        1. 东南障碍区域
        2. 东北障碍区域
        3. 西北障碍区域
    """
    A = jnp.array([
        # 西南：x <= -4.5, y <= -4.5, y <= -x - 22  -> x + y <= -22
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        # 东南：x >= 4.5, y <= -4.5, y <= x - 22   -> -x + y <= -22
        [[-1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]],
        # 东北：x >= 4.5, y >= 4.5, y >= -x + 22    -> -x - y <= -22
        [[-1.0, 0.0], [0.0, -1.0], [-1.0, -1.0]],
        # 西北：x <= -4.5, y >= 4.5, y >= x + 22    -> x - y <= -22
        [[1.0, 0.0], [0.0, -1.0], [1.0, -1.0]],
    ], dtype=jnp.float32)
    b = jnp.array([
        [-4.5, -4.5, -22.0],
        [-4.5, -4.5, -22.0],
        [-4.5, -4.5, -22.0],
        [-4.5, -4.5, -22.0],
    ], dtype=jnp.float32)
    return A, b


@jax.jit
def scaling_calc_intersection_bounds(s: State) -> Array:
    """计算 agent 相对四个十字路口角落障碍区域的最危险 scaling factor。"""
    A, b = intersection_corner_bounds()
    alphas = jax.vmap(scaling_calc_convex_bound, in_axes=(None, 0, 0))(s, A, b)
    return jnp.min(alphas)
