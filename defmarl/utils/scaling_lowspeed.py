import jax
import jax.numpy as jnp

from .typing import Array, Pos2d, State

# 全局数值稳定极小值（核心：防止除零/NaN）
EPS = 1e-6
RAY_VALIDITY_TOL = 1e-6

def safe_norm(x, axis=-1):
    return jnp.sqrt(jnp.sum(x ** 2, axis=axis) + EPS)


def safe_divide(numerator: Array, denominator: Array, eps: float = EPS) -> Array:
    safe_denominator = jnp.where(
        jnp.abs(denominator) > eps,
        denominator,
        jnp.where(denominator >= 0.0, eps, -eps),
    )
    return numerator / safe_denominator


@jax.jit
def ray_convex_entry_scaling(
    origin: Pos2d,
    target: Pos2d,
    A: Array,
    b: Array,
) -> Array:
    """Return the first ray parameter entering ``A @ x <= b``.

    The target is located at parameter one, so the entry parameter is directly
    the first ray-group scaling ratio.  Computing the feasible parameter
    interval avoids large fill coordinates and the discontinuous second pass
    that validates constructed Cartesian intersections.
    """
    normal_norms = jnp.maximum(jnp.linalg.norm(A, axis=1), EPS)
    normalized_A = A / normal_norms[:, None]
    normalized_b = b / normal_norms
    direction = target - origin
    denominator = normalized_A @ direction
    numerator = normalized_b - normalized_A @ origin
    nonparallel = jnp.abs(denominator) > EPS
    safe_denominator = jnp.where(nonparallel, denominator, 1.0)
    ratio = numerator / safe_denominator

    lower = jnp.max(jnp.where(denominator < -EPS, ratio, -jnp.inf))
    upper = jnp.min(jnp.where(denominator > EPS, ratio, jnp.inf))
    entry = jnp.maximum(lower, 0.0)
    parallel_feasible = jnp.all(
        jnp.where(nonparallel, True, numerator >= -RAY_VALIDITY_TOL)
    )
    interval_nonempty = (entry <= upper + RAY_VALIDITY_TOL) & (
        upper >= -RAY_VALIDITY_TOL
    )
    return jnp.where(parallel_feasible & interval_nonempty, entry, jnp.inf)


@jax.jit
def ray_rectangle_extreme_scaling(
    origin: Pos2d,
    extreme_point: Pos2d,
    rectangle_A: Array,
    rectangle_b: Array,
) -> Array:
    """Return ``||O G|| / ||O F||`` from the rectangle exit parameter."""
    normal_norms = jnp.maximum(jnp.linalg.norm(rectangle_A, axis=1), EPS)
    normalized_A = rectangle_A / normal_norms[:, None]
    normalized_b = rectangle_b / normal_norms
    direction = extreme_point - origin
    denominator = normalized_A @ direction
    numerator = normalized_b - normalized_A @ origin
    exits_forward = denominator > EPS
    safe_denominator = jnp.where(exits_forward, denominator, 1.0)
    exit_candidates = jnp.where(
        exits_forward, numerator / safe_denominator, jnp.inf
    )
    exit_parameter = jnp.min(exit_candidates)
    direction_valid = jnp.linalg.norm(direction) > EPS
    exit_valid = jnp.isfinite(exit_parameter) & (exit_parameter > EPS)
    return jnp.where(
        direction_valid & exit_valid,
        1.0 / jnp.maximum(exit_parameter, EPS),
        0.0,
    )

@jax.jit
def compute_intersections(O: Pos2d, V: Pos2d, A, b, fill: Pos2d) -> Array:
    """计算输入两点的直线与Ax=b各边的交点"""
    # 解析输入直线系数（向量化扩展为(k,)形状，适配A的q条边）
    a0 = V[1] - O[1]
    b0 = O[0] - V[0]
    c0 = V[0] * O[1] - O[0] * V[1]
    max_abs = jnp.maximum(jnp.array([jnp.abs(a0), jnp.abs(b0), jnp.abs(c0)]).max(), EPS)
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
    # jnp.where 会同时计算两个分支，因此除法本身也必须安全。
    x_raw = safe_divide(x_num, D)
    y_raw = safe_divide(y_num, D)
    x = jnp.where(jnp.abs(D) > EPS, x_raw, fill[0])
    y = jnp.where(jnp.abs(D) > EPS, y_raw, fill[1])
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
def normalize_heading(h: Array) -> Array:
    return h / jnp.maximum(jnp.linalg.norm(h), EPS)


@jax.jit
def heading_rot_matrix(s: State) -> Array:
    h = normalize_heading(s[2:4])
    return jnp.array([[h[0], -h[1]],
                      [h[1], h[0]]])


@jax.jit
def rear_to_center(s: State, lr: Array) -> Array:
    """
    state: x y heading_x heading_y v(m/s) delta(rad)
    x,y 是后轴中心
    返回车体几何中心坐标
    """
    x_rear = s[0]
    y_rear = s[1]
    h = normalize_heading(s[2:4])

    x_center = x_rear + lr * h[0]
    y_center = y_rear + lr * h[1]

    return jnp.array([x_center, y_center])

@jax.jit
def scaling_calc(s1: State, s2: State, bb1: Array, lr1: Array, bb2: Array, lr2: Array) -> Array:
    """计算agent和agent/obst的scaling factor"""
    # state: x y heading_x heading_y v(m/s) delta(rad)
    O1 = rear_to_center(s1, lr1)
    O2 = rear_to_center(s2, lr2)
    # 计算 host 和 agent/obst 的顶点， host/agent/obst均为矩形
    Q1 = heading_rot_matrix(s1)
    m_V = jnp.array([[bb1[0] / 2, bb1[1] / 2],
                     [bb1[0] / 2, -bb1[1] / 2],
                     [-bb1[0] / 2, bb1[1] / 2],
                     [-bb1[0] / 2, -bb1[1] / 2]])
    m_V = O1 + m_V @ Q1.T
    Q2 = heading_rot_matrix(s2)
    n_P = jnp.array([[bb2[0] / 2, bb2[1] / 2],
                     [bb2[0] / 2, -bb2[1] / 2],
                     [-bb2[0] / 2, bb2[1] / 2],
                     [-bb2[0] / 2, -bb2[1] / 2]])
    n_P = O2 + n_P @ Q2.T

    # 计算S1和S2的A1 b1 A2 b2
    Ao = jnp.array([[ 1., 0.],  # x<=b/2
                    [-1., 0.],  # x>=-b/2
                    [ 0., 1.],  # y<=h/2
                    [ 0.,-1.]]) # y>=-h/2
    A1 = Ao @ Q1.T
    A2 = Ao @ Q2.T
    b1o = jnp.array([bb1[0] / 2, bb1[0] / 2, bb1[1] / 2, bb1[1] / 2])
    b2o = jnp.array([bb2[0] / 2, bb2[0] / 2, bb2[1] / 2, bb2[1] / 2])
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
def scaling_calc_parameterized(
    s1: State,
    s2: State,
    bb1: Array,
    lr1: Array,
    bb2: Array,
    lr2: Array,
) -> Array:
    """Stable two-rectangle scaling using parameterized rays.

    This is geometrically equivalent to ``scaling_calc`` and retains the
    theoretical hard minimum over both ray groups.  It avoids explicitly
    constructing edge intersections, large fill points, and absolute-position
    dependent line normalization.
    """
    origin_1 = rear_to_center(s1, lr1)
    origin_2 = rear_to_center(s2, lr2)
    rotation_1 = heading_rot_matrix(s1)
    rotation_2 = heading_rot_matrix(s2)
    rectangle_template = jnp.array(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    )
    local_vertices_1 = jnp.array(
        [
            [bb1[0] / 2.0, bb1[1] / 2.0],
            [bb1[0] / 2.0, -bb1[1] / 2.0],
            [-bb1[0] / 2.0, bb1[1] / 2.0],
            [-bb1[0] / 2.0, -bb1[1] / 2.0],
        ]
    )
    local_vertices_2 = jnp.array(
        [
            [bb2[0] / 2.0, bb2[1] / 2.0],
            [bb2[0] / 2.0, -bb2[1] / 2.0],
            [-bb2[0] / 2.0, bb2[1] / 2.0],
            [-bb2[0] / 2.0, -bb2[1] / 2.0],
        ]
    )
    vertices_1 = origin_1 + local_vertices_1 @ rotation_1.T
    vertices_2 = origin_2 + local_vertices_2 @ rotation_2.T
    A1 = rectangle_template @ rotation_1.T
    A2 = rectangle_template @ rotation_2.T
    b1_local = jnp.array(
        [bb1[0] / 2.0, bb1[0] / 2.0, bb1[1] / 2.0, bb1[1] / 2.0]
    )
    b2_local = jnp.array(
        [bb2[0] / 2.0, bb2[0] / 2.0, bb2[1] / 2.0, bb2[1] / 2.0]
    )
    b1 = b1_local + A1 @ origin_1
    b2 = b2_local + A2 @ origin_2

    vertex_ray_scaling = jnp.min(
        jax.vmap(ray_convex_entry_scaling, in_axes=(None, 0, None, None))(
            origin_1, vertices_1, A2, b2
        )
    )
    extreme_ray_scaling = jnp.min(
        jax.vmap(
            ray_rectangle_extreme_scaling,
            in_axes=(None, 0, None, None),
        )(origin_1, vertices_2, A1, b1)
    )
    scaling = jnp.minimum(vertex_ray_scaling, extreme_ray_scaling)
    # ``ray_convex_entry_scaling`` returns zero when the scaling origin is
    # already inside P2.  The historical sigmoid(1e6 * gamma_0) gate is thus
    # redundant here and creates an artificial gradient spike at gamma_0=0.
    return jnp.nan_to_num(scaling, nan=0.0, posinf=1e6, neginf=0.0)


@jax.jit
def scaling_calc_unbounded_bound(
    state: State,
    bb_size: Array,
    rear_to_center_offset: Array,
    A: Array,
    b: Array,
) -> Array:
    """Compute scaling against an unbounded forbidden convex polygon.

    ``A @ x <= b`` describes the forbidden region.  A straight-road upper or
    lower boundary is a half-plane and has no finite extreme points.  Thus the
    ray-casting algorithm's ``n_g == 0`` branch applies: cast rays from the ego
    scaling origin through its four vertices, intersect them with the polygon,
    and take the minimum distance ratio.  No virtual rectangle, and therefore
    no artificial second group of corner rays, is introduced.
    """
    origin = rear_to_center(state, rear_to_center_offset)
    rotation = heading_rot_matrix(state)
    local_vertices = jnp.array(
        [
            [bb_size[0] / 2.0, bb_size[1] / 2.0],
            [bb_size[0] / 2.0, -bb_size[1] / 2.0],
            [-bb_size[0] / 2.0, bb_size[1] / 2.0],
            [-bb_size[0] / 2.0, -bb_size[1] / 2.0],
        ]
    )
    vertices = origin + local_vertices @ rotation.T
    vertex_ray_scaling = jnp.min(
        jax.vmap(ray_convex_entry_scaling, in_axes=(None, 0, None, None))(
            origin, vertices, A, b
        )
    )

    # The entry parameter is exactly zero when the origin is already inside
    # the forbidden half-plane, so no steep center-membership gate is needed.
    return jnp.nan_to_num(
        vertex_ray_scaling, nan=0.0, posinf=1e6, neginf=0.0
    )

@jax.jit
def scaling_calc_bound(s: State, bb_size: Array, lr: Array, A: Array, b: Array) -> Array:
    """计算agent和bound的scaling factor，agent为有界多面凸集，bound为Ax<=b描述的单条直线分割的半空间，即A和b均为一行"""
    O = rear_to_center(s, lr)
    # 计算host的顶点，host为矩形
    Q = heading_rot_matrix(s)
    m_V = jnp.array([[ bb_size[0]/2,  bb_size[1]/2],
                     [ bb_size[0]/2, -bb_size[1]/2],
                     [-bb_size[0]/2,  bb_size[1]/2],
                     [-bb_size[0]/2, -bb_size[1]/2]])
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
