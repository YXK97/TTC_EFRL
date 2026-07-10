import argparse
import datetime
import functools as ft
import os
import pathlib
import sys

import ipdb
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UFTSTC.cvs_utils import dump_rollout_record_to_csv6
from UFTSTC.longitudinal_controller import PIDController
from UFTSTC_intersection.latral_controller_pid import UFTSTCIntersectionControllerPid
from UFTSTC_intersection.utils import (
    eval_rollout_uftstc,
    parse_pid_controller_args,
    parse_uftstc_controller_args,
)
from defmarl.utils.utils import parse_jax_array


def test(args):
    if args.visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_devices

    import jax
    import jax.numpy as jnp
    import jax.random as jr

    from defmarl.env import make_env
    from defmarl.utils.utils import jax_jit_np

    print(f"> Running UFTSTC_intersection/simtest.py {args}")
    print(f"> Using {jax.local_device_count()} devices")

    stamp_str = datetime.datetime.now().strftime("%m%d-%H%M")
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    if args.debug:
        jax.config.update("jax_disable_jit", True)
    np.random.seed(args.seed)

    env = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        max_step=args.max_step if args.max_step is not None else None,
        full_observation=args.full_observation,
        area_size=args.area_size,
    )
    env.params["scene_mode"] = args.scene

    lateral_controller_pid = UFTSTCIntersectionControllerPid(**parse_uftstc_controller_args(args, env))
    longitudinal_controller = PIDController(**parse_pid_controller_args(args, env))

    test_key = jr.PRNGKey(args.seed)
    test_keys = jr.split(test_key, 1_000)[: args.epi]
    test_keys = test_keys[args.offset:]
    rollout_fn = jax_jit_np(ft.partial(eval_rollout_uftstc, env, lateral_controller_pid, longitudinal_controller))

    rewards, costs, costs_real, is_unsafes, rates, rollouts, records = [], [], [], [], [], [], []
    for i_epi in range(args.epi):
        key_x0, _ = jr.split(test_keys[i_epi], 2)
        rollout, record = rollout_fn(key_x0)
        is_unsafes.append(jnp.any(rollout.costs_real >= 1e-6, axis=-1))

        dump_rollout_record_to_csv6(
            rollout,
            record,
            prefix=f"epi{i_epi:02d}",
            scene_tag=args.scene,
            uftstc_root=str(pathlib.Path.cwd() / "UFTSTC_intersection"),
        )

        epi_reward = rollout.rewards.sum()
        epi_cost = rollout.costs.max()
        epi_cost_real = rollout.costs_real.max()
        rewards.append(epi_reward)
        costs.append(epi_cost)
        costs_real.append(epi_cost_real)
        rollouts.append(rollout)
        records.append(record)

        safe_rate = 1 - is_unsafes[-1].max(axis=0).mean()
        print(f"epi: {i_epi}, reward: {epi_reward:.3f}, cost: {epi_cost:.3f}, cost_real: {epi_cost_real:.3f}, "
              f"safe rate: {safe_rate * 100:.3f}%")
        rates.append(np.array(safe_rate))

    is_unsafe = np.max(np.stack(is_unsafes), axis=1)
    safe_mean, safe_std = (1 - is_unsafe).mean(), (1 - is_unsafe).std()
    print(
        f"reward: {np.mean(rewards):.3f}, std: {np.std(rewards):.3f}, "
        f"cost: {np.mean(costs):.3f}, cost_real: {np.mean(costs_real):.3f}, "
        f"safe_rate: {safe_mean * 100:.3f}%, std: {safe_std * 100:.3f}%"
    )

    path = pathlib.Path(args.path)
    if args.no_video:
        return

    videos_dir = path / "videos"
    videos_dir.mkdir(exist_ok=True, parents=True)
    for ii, (rollout, Ta_is_unsafe) in enumerate(zip(rollouts, is_unsafes)):
        video_name = f"{args.scene}_n{args.num_agents}_epi{ii:02}_reward{rewards[ii]:.3f}_cost{costs[ii]:.3f}"
        video_path = videos_dir / f"{stamp_str}_{video_name}.mp4"
        env.render_video(rollout, video_path, Ta_is_unsafe, dpi=args.dpi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--scene", type=str, default="uftstc_left", choices=["uftstc_left", "uftstc_straight", "handmade", "random"])
    parser.add_argument("--switch-y", type=float, default=17.5)

    parser.add_argument("--Af", type=float, default=18)
    parser.add_argument("--r", type=float, default=15)
    parser.add_argument("--mu", type=float, default=30)
    parser.add_argument("--c", type=float, default=1.5)
    parser.add_argument("--k1", type=float, default=0.4)
    parser.add_argument("--k2", type=float, default=0.6)
    parser.add_argument("--k3", type=float, default=1.6)
    parser.add_argument("--k4", type=float, default=10)
    parser.add_argument("--v", type=float, default=0.6)
    parser.add_argument("--Delta1", type=float, default=3)
    parser.add_argument("--Delta2", type=float, default=50)
    parser.add_argument("--p-num", type=int, default=3)
    parser.add_argument("--p-den", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=7 / 9)

    parser.add_argument("--kp", type=float, default=2)
    parser.add_argument("--ki", type=float, default=0.3)
    parser.add_argument("--kd", type=float, default=0)
    parser.add_argument("--max-integral", type=float, default=50.0)
    parser.add_argument("--min-integral", type=float, default=-50.0)
    parser.add_argument("--kp_d", type=float, default=1.9)
    parser.add_argument("--ki_d", type=float, default=1)
    parser.add_argument("--kd_d", type=float, default=0.5)
    parser.add_argument("--max-integral_d", type=float, default=50.0)
    parser.add_argument("--min-integral_d", type=float, default=-50.0)

    parser.add_argument("--epi", type=int, default=1)
    parser.add_argument("--no-video", action="store_true", default=False)
    parser.add_argument("-n", "--num-agents", type=int, default=1)
    parser.add_argument("--obs", type=int, default=2)
    parser.add_argument("--env", type=str, default="MVEIntersection")
    parser.add_argument("--full-observation", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--area-size", type=parse_jax_array, default=None)
    parser.add_argument("--visible-devices", type=str, default=None)

    parser.add_argument("--y_min", type=float, default=-4.5)
    parser.add_argument("--y_max", type=float, default=4.5)
    parser.add_argument("--Af_lane", type=float, default=80)
    parser.add_argument("--r_lane", type=float, default=6.5)
    parser.add_argument("--mu_lane", type=float, default=10)
    parser.add_argument("--c_lane", type=float, default=1.5)
    parser.add_argument("--leak_near_lane", type=float, default=0.001)
    parser.add_argument("--leak_far_lane", type=float, default=0.01)

    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
