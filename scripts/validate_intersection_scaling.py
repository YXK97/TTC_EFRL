"""Compare explicit and parameterized intersection ray scaling numerically."""

import argparse
import pathlib
import sys

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from defmarl.env.mve_intersection_lowspeed_ISSf_CBF_dynamic import (
    MVEIntersection_LowSpeed_ISSf_CBF_Dynamic,
    scaling_calc_intersection_bounds_lowspeed,
)
from defmarl.env.mve_intersection_lowspeed_ISSf_CBF_dynamic_new_scaling import (
    scaling_calc_intersection_bounds_lowspeed_new,
)


def percentile_summary(values: np.ndarray) -> str:
    points = np.percentile(values, [50.0, 90.0, 99.0, 99.9, 100.0])
    return " ".join(
        f"p{label}={value:.6e}"
        for label, value in zip(("50", "90", "99", "99.9", "max"), points)
    )


def make_states(key, count: int) -> jnp.ndarray:
    """Mix global and road-focused samples to cover relevant geometry."""

    key_global, key_horizontal, key_vertical, key_angle = jr.split(key, 4)
    n_global = count // 3
    n_horizontal = count // 3
    n_vertical = count - n_global - n_horizontal
    global_xy = jr.uniform(
        key_global, (n_global, 2), minval=-48.0, maxval=48.0
    )
    horizontal_xy = jnp.stack(
        [
            jr.uniform(key_horizontal, (n_horizontal,), minval=-48.0, maxval=48.0),
            jr.uniform(
                jr.fold_in(key_horizontal, 1),
                (n_horizontal,),
                minval=-5.0,
                maxval=5.0,
            ),
        ],
        axis=1,
    )
    vertical_xy = jnp.stack(
        [
            jr.uniform(key_vertical, (n_vertical,), minval=-4.6, maxval=4.6),
            jr.uniform(
                jr.fold_in(key_vertical, 1),
                (n_vertical,),
                minval=-48.0,
                maxval=48.0,
            ),
        ],
        axis=1,
    )
    xy = jnp.concatenate([global_xy, horizontal_xy, vertical_xy], axis=0)
    angle = jr.uniform(key_angle, (count,), minval=-jnp.pi, maxval=jnp.pi)
    return jnp.concatenate(
        [
            xy,
            jnp.stack([jnp.cos(angle), jnp.sin(angle)], axis=1),
            jnp.full((count, 1), 3.0),
            jnp.zeros((count, 1)),
        ],
        axis=1,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()

    env = MVEIntersection_LowSpeed_ISSf_CBF_Dynamic(1, max_step=4)
    params = env.params
    scaling_args = (
        params["ego_bb_size"],
        params["ego_lr"],
        params["main_road_half_width"],
        params["auxiliary_road_half_width"],
        params["intersection_radius"],
    )

    def explicit(state):
        return scaling_calc_intersection_bounds_lowspeed(state, *scaling_args)

    def parameterized(state):
        return scaling_calc_intersection_bounds_lowspeed_new(state, *scaling_args)

    states = make_states(jr.PRNGKey(args.seed), args.samples)
    evaluate = jax.jit(
        jax.vmap(
            lambda state: (
                explicit(state),
                parameterized(state),
                jax.grad(explicit)(state)[:4],
                jax.grad(parameterized)(state)[:4],
            )
        )
    )
    explicit_alpha, new_alpha, explicit_grad, new_grad = map(
        np.asarray, evaluate(states)
    )
    effective = (
        np.isfinite(explicit_alpha)
        & np.isfinite(new_alpha)
        & (np.maximum(explicit_alpha, new_alpha) > 0.05)
        & (np.maximum(explicit_alpha, new_alpha) < 1e5)
    )

    alpha_error = np.abs(explicit_alpha - new_alpha)
    relative_error = alpha_error / np.maximum(
        np.maximum(np.abs(explicit_alpha), np.abs(new_alpha)), 1e-6
    )
    gradient_error = np.linalg.norm(explicit_grad - new_grad, axis=1)

    direction = jr.normal(jr.PRNGKey(args.seed + 1), (args.samples, 2))
    direction /= jnp.linalg.norm(direction, axis=1, keepdims=True)
    angle = jnp.arctan2(states[:, 3], states[:, 2])
    plus = (
        states.at[:, :2]
        .add(1e-5 * direction)
        .at[:, 2]
        .set(jnp.cos(angle + 1e-6))
        .at[:, 3]
        .set(jnp.sin(angle + 1e-6))
    )
    minus = (
        states.at[:, :2]
        .add(-1e-5 * direction)
        .at[:, 2]
        .set(jnp.cos(angle - 1e-6))
        .at[:, 3]
        .set(jnp.sin(angle - 1e-6))
    )
    perturb = jax.jit(
        jax.vmap(
            lambda state_plus, state_minus: (
                explicit(state_plus),
                explicit(state_minus),
                parameterized(state_plus),
                parameterized(state_minus),
                jax.grad(explicit)(state_plus)[:4],
                jax.grad(explicit)(state_minus)[:4],
                jax.grad(parameterized)(state_plus)[:4],
                jax.grad(parameterized)(state_minus)[:4],
            )
        )
    )
    ep, em, np_, nm, gep, gem, gnp, gnm = map(
        np.asarray, perturb(plus, minus)
    )
    explicit_alpha_jump = np.abs(ep - em)
    new_alpha_jump = np.abs(np_ - nm)
    explicit_gradient_jump = np.linalg.norm(gep - gem, axis=1)
    new_gradient_jump = np.linalg.norm(gnp - gnm, axis=1)

    print(f"seed={args.seed} samples={args.samples} effective={effective.sum()}")
    for name, values in (
        ("alpha_abs_error", alpha_error),
        ("alpha_relative_error", relative_error),
        ("gradient_error", gradient_error),
        ("explicit_alpha_perturbation", explicit_alpha_jump),
        ("new_alpha_perturbation", new_alpha_jump),
        ("explicit_gradient_perturbation", explicit_gradient_jump),
        ("new_gradient_perturbation", new_gradient_jump),
    ):
        print(f"{name}: {percentile_summary(values[effective])}")

    worst = np.where(effective)[0][np.argsort(alpha_error[effective])[-5:]]
    print("worst alpha comparisons:")
    for index in worst[::-1]:
        print(
            f"  state={np.asarray(states[index]).tolist()} "
            f"explicit={explicit_alpha[index]:.9e} "
            f"new={new_alpha[index]:.9e} "
            f"abs_error={alpha_error[index]:.9e}"
        )


if __name__ == "__main__":
    main()
