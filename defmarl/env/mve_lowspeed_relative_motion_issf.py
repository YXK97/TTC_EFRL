"""Shared relative-motion terms for low-speed dynamic ISSf-CBF environments."""

from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from defmarl.utils.issf_barrier import (
    compress_safe_barrier,
    safe_barrier_derivative,
)
from defmarl.utils.typing import Array, State


ScalingFunction = Callable[[State, State], Array]


def relative_obstacle_diagnostic_terms(
    env,
    scaling_fn: ScalingFunction,
    ego_state: State,
    obstacle_state: State,
    steering: Array,
) -> Tuple[Array, Array, Array, Array]:
    """Return alpha, ego gradient, relative h_dot, and ego ISSf channel.

    The obstacle updater in both target environments translates obstacle poses
    along a fixed heading.  Consequently obstacle acceleration changes future
    velocity but does not enter the first derivative of pose-based alpha.
    """

    def alpha_from_poses(ego_pose, obstacle_pose):
        ego_full = jnp.concatenate([ego_pose, ego_state[4:6]])
        obstacle_full = jnp.concatenate([obstacle_pose, obstacle_state[4:6]])
        return scaling_fn(ego_full, obstacle_full)

    alpha, (ego_alpha_grad, obstacle_alpha_grad) = jax.value_and_grad(
        alpha_from_poses, argnums=(0, 1)
    )(ego_state[:4], obstacle_state[:4])
    alpha = jnp.nan_to_num(alpha, nan=0.0, posinf=1e6, neginf=0.0)
    ego_alpha_grad = jnp.nan_to_num(
        ego_alpha_grad, nan=0.0, posinf=0.0, neginf=0.0
    )
    obstacle_alpha_grad = jnp.nan_to_num(
        obstacle_alpha_grad, nan=0.0, posinf=0.0, neginf=0.0
    )

    barrier_scale = safe_barrier_derivative(
        alpha,
        env.params["alpha_thresh"],
        env.params["issf_safe_barrier_kappa"],
    )
    ego_barrier_grad = barrier_scale * ego_alpha_grad
    obstacle_barrier_grad = barrier_scale * obstacle_alpha_grad

    ego_heading = ego_state[2:4] / jnp.maximum(
        jnp.linalg.norm(ego_state[2:4]), 1e-6
    )
    ego_angular_speed = (
        ego_state[4] / env.params["ego_L"] * jnp.tan(steering)
    )
    ego_pose_dot = jnp.array(
        [
            ego_state[4] * ego_heading[0],
            ego_state[4] * ego_heading[1],
            -ego_heading[1] * ego_angular_speed,
            ego_heading[0] * ego_angular_speed,
        ]
    )

    obstacle_heading = obstacle_state[2:4] / jnp.maximum(
        jnp.linalg.norm(obstacle_state[2:4]), 1e-6
    )
    # This matches obst_step_euler: obstacles translate but do not rotate.
    obstacle_pose_dot = jnp.array(
        [
            obstacle_state[4] * obstacle_heading[0],
            obstacle_state[4] * obstacle_heading[1],
            0.0,
            0.0,
        ]
    )

    ego_steering_channel = jnp.array(
        [
            0.0,
            0.0,
            -ego_heading[1] * ego_state[4] / env.params["ego_L"],
            ego_heading[0] * ego_state[4] / env.params["ego_L"],
        ]
    )
    relative_barrier_dot = (
        jnp.dot(ego_barrier_grad, ego_pose_dot)
        + jnp.dot(obstacle_barrier_grad, obstacle_pose_dot)
    )
    ego_steering_dot = jnp.dot(
        ego_barrier_grad, ego_steering_channel
    )
    return alpha, ego_alpha_grad, relative_barrier_dot, ego_steering_dot


def relative_obstacle_issf_constraint(
    env,
    scaling_fn: ScalingFunction,
    ego_state: State,
    obstacle_state: State,
    steering: Array,
) -> Tuple[Array, Array]:
    """Evaluate the existing ISSf formula with relative obstacle motion."""
    alpha, _, barrier_dot, steering_dot = relative_obstacle_diagnostic_terms(
        env, scaling_fn, ego_state, obstacle_state, steering
    )
    barrier = compress_safe_barrier(
        alpha,
        env.params["alpha_thresh"],
        env.params["issf_safe_barrier_kappa"],
    )
    epsilon = env.params["issf_epsilon_min"] + env.params[
        "issf_epsilon_0"
    ] * jax.nn.softplus(env.params["issf_epsilon_rate"] * barrier)
    young_penalty = jnp.square(steering_dot) / epsilon
    residual = (
        barrier_dot / env.params["gamma"]
        + barrier
        - young_penalty / env.params["gamma"]
    )
    cost = jnp.nan_to_num(
        -residual, nan=3.0, posinf=3.0, neginf=-3.0
    )
    return cost, 1.0 - alpha
