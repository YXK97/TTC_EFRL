"""Intersection ISSf-CBF environment using parameterized ray scaling.

This module preserves the ray-casting scaling factor used by the original
environment.  It changes only how ray intersections are evaluated: substituting
``x(t) = origin + t * direction`` into the polygon halfspaces produces the
entry/exit parameters directly, avoiding constructed intersection coordinates,
large fill points, and a second geometric validity pass.
"""

import jax
import jax.numpy as jnp
from typing_extensions import override

from defmarl.utils.typing import Array, State

from .mve_intersection_lowspeed_ISSf_CBF_dynamic import (
    MVEIntersection_LowSpeed_ISSf_CBF_Dynamic,
    intersection_corner_extreme_points,
    intersection_corner_halfspaces,
)
from defmarl.utils.scaling_lowspeed import heading_rot_matrix, rear_to_center


RAY_EPS = 1e-6
RAY_VALIDITY_TOL = 1e-6


@jax.jit
def ray_convex_entry_scaling(
    origin: Array,
    target: Array,
    A: Array,
    b: Array,
) -> Array:
    """Return the first nonnegative ray parameter entering ``A x <= b``.

    ``target`` is at ray parameter one.  Consequently the returned entry
    parameter is exactly the first-group scaling ratio in the paper.  Rays that
    do not intersect the convex set return positive infinity and cannot become
    the hard-min candidate.
    """

    normal_norm = jnp.maximum(jnp.linalg.norm(A, axis=1), RAY_EPS)
    normalized_A = A / normal_norm[:, None]
    normalized_b = b / normal_norm
    direction = target - origin
    denominator = normalized_A @ direction
    numerator = normalized_b - normalized_A @ origin
    nonparallel = jnp.abs(denominator) > RAY_EPS
    safe_denominator = jnp.where(nonparallel, denominator, 1.0)
    ratio = numerator / safe_denominator

    lower = jnp.max(
        jnp.where(denominator < -RAY_EPS, ratio, -jnp.inf)
    )
    upper = jnp.min(
        jnp.where(denominator > RAY_EPS, ratio, jnp.inf)
    )
    entry = jnp.maximum(lower, 0.0)

    # A parallel constraint remains unchanged along the complete ray.  If the
    # origin violates it, this ray can never enter the polygon.
    parallel_feasible = jnp.all(
        jnp.where(nonparallel, True, numerator >= -RAY_VALIDITY_TOL)
    )
    interval_nonempty = (
        (entry <= upper + RAY_VALIDITY_TOL)
        & (upper >= -RAY_VALIDITY_TOL)
    )
    return jnp.where(parallel_feasible & interval_nonempty, entry, jnp.inf)


@jax.jit
def ray_rectangle_extreme_scaling(
    origin: Array,
    extreme_point: Array,
    rectangle_A: Array,
    rectangle_b: Array,
) -> Array:
    """Return the paper's extreme-point ratio without forming intersection F.

    The origin lies inside the ego rectangle.  If its ray towards ``G`` exits
    the rectangle at parameter ``t_exit``, then
    ``||O G|| / ||O F|| = 1 / t_exit``.
    """

    normal_norm = jnp.maximum(
        jnp.linalg.norm(rectangle_A, axis=1), RAY_EPS
    )
    normalized_A = rectangle_A / normal_norm[:, None]
    normalized_b = rectangle_b / normal_norm
    direction = extreme_point - origin
    denominator = normalized_A @ direction
    numerator = normalized_b - normalized_A @ origin
    nonparallel = denominator > RAY_EPS
    safe_denominator = jnp.where(nonparallel, denominator, 1.0)
    exit_candidates = jnp.where(
        nonparallel, numerator / safe_denominator, jnp.inf
    )
    exit_parameter = jnp.min(exit_candidates)
    direction_valid = jnp.linalg.norm(direction) > RAY_EPS
    exit_valid = jnp.isfinite(exit_parameter) & (exit_parameter > RAY_EPS)
    return jnp.where(
        direction_valid & exit_valid,
        1.0 / jnp.maximum(exit_parameter, RAY_EPS),
        0.0,
    )


@jax.jit
def parameterized_ray_scaling_convex_bound(
    state: State,
    A: Array,
    b: Array,
    extreme_points: Array,
    bb_size: Array,
    rear_to_center_offset: Array,
) -> Array:
    """Evaluate both ray groups while retaining the theoretical hard minimum."""

    center = rear_to_center(state, rear_to_center_offset)
    rotation = heading_rot_matrix(state)
    local_vertices = jnp.array(
        [
            [bb_size[0] / 2.0, bb_size[1] / 2.0],
            [bb_size[0] / 2.0, -bb_size[1] / 2.0],
            [-bb_size[0] / 2.0, bb_size[1] / 2.0],
            [-bb_size[0] / 2.0, -bb_size[1] / 2.0],
        ]
    )
    vertices = center + local_vertices @ rotation.T
    vertex_ray_scaling = jnp.min(
        jax.vmap(ray_convex_entry_scaling, in_axes=(None, 0, None, None))(
            center, vertices, A, b
        )
    )

    rectangle_template = jnp.array(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    )
    ego_A = rectangle_template @ rotation.T
    ego_b_local = jnp.array(
        [bb_size[0] / 2.0, bb_size[0] / 2.0, bb_size[1] / 2.0, bb_size[1] / 2.0]
    )
    ego_b = ego_b_local + ego_A @ center
    extreme_ray_scaling = jnp.min(
        jax.vmap(
            ray_rectangle_extreme_scaling,
            in_axes=(None, 0, None, None),
        )(center, extreme_points, ego_A, ego_b)
    )

    scaling = jnp.minimum(vertex_ray_scaling, extreme_ray_scaling)
    center_outside_measure = jnp.max(A @ center - b)
    center_outside_weight = jax.nn.sigmoid(1e6 * center_outside_measure)
    return center_outside_weight * scaling


@jax.jit
def scaling_calc_intersection_bounds_lowspeed_new(
    state: State,
    bb_size: Array,
    rear_to_center_offset: Array,
    main_half_width: Array,
    auxiliary_half_width: Array,
    turn_half: Array,
) -> Array:
    """Return the hard minimum parameterized scaling over four road corners."""

    all_A, all_b = intersection_corner_halfspaces(
        main_half_width, auxiliary_half_width, turn_half
    )
    all_extreme_points = intersection_corner_extreme_points(
        main_half_width, auxiliary_half_width, turn_half
    )
    alphas = jax.vmap(
        parameterized_ray_scaling_convex_bound,
        in_axes=(None, 0, 0, 0, None, None),
    )(
        state,
        all_A,
        all_b,
        all_extreme_points,
        bb_size,
        rear_to_center_offset,
    )
    return jnp.min(alphas)


class MVEIntersection_LowSpeed_ISSf_CBF_Dynamic_NewScaling(
    MVEIntersection_LowSpeed_ISSf_CBF_Dynamic
):
    """Original intersection environment with parameterized boundary rays."""

    @override
    def _intersection_alpha(self, state: State) -> Array:
        return scaling_calc_intersection_bounds_lowspeed_new(
            state,
            self.params["ego_bb_size"],
            self.params["ego_lr"],
            self.params["main_road_half_width"],
            self.params["auxiliary_road_half_width"],
            self.params["intersection_radius"],
        )


MVEIntersectionLowSpeedISSfCBFDynamicNewScaling = (
    MVEIntersection_LowSpeed_ISSf_CBF_Dynamic_NewScaling
)
