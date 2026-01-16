import argparse
import datetime
import functools as ft
import os
import pathlib
import ipdb
import numpy as np
import yaml

from UFTSTC.longitudinal_controller import PIDController
from UFTSTC.latral_controller import UFTSTCController
from utils import parse_uftstc_controller_args, parse_pid_controller_args, eval_rollout_uftstc
from defmarl.utils.utils import parse_jax_array


def test(args):
    if args.visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_devices

    import jax
    import jax.numpy as jnp
    import jax.random as jr

    from defmarl.algo import make_algo
    from defmarl.env import make_env
    from defmarl.trainer.data import Rollout
    from defmarl.trainer.utils import eval_rollout
    from defmarl.utils.utils import jax_jit_np, jax_vmap, parse_jax_array

    n_gpu = jax.local_device_count()
    print(f"> Running test.py {args}")
    print(f"> Using {n_gpu} devices")

    stamp_str = datetime.datetime.now().strftime("%m%d-%H%M")

    # set up environment variables and seed
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    if args.debug:
        jax.config.update("jax_disable_jit", True)
    np.random.seed(args.seed)


    # create environments
    env = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        max_step=args.max_step if args.max_step is not None else None,
        full_observation=args.full_observation,
        area_size=args.area_size,
    )

    args_lateral_controller = parse_uftstc_controller_args(args, env)
    lateral_controller = UFTSTCController(**args_lateral_controller)
    args_longitudinal_controller = parse_pid_controller_args(args, env)
    longitudinal_controller = PIDController(**args_longitudinal_controller)

    test_key = jr.PRNGKey(args.seed)
    test_keys = jr.split(test_key, 1_000)[: args.epi]
    test_keys = test_keys[args.offset:]

    rollout_fn = ft.partial(eval_rollout_uftstc,
                            env,
                            lateral_controller,
                            longitudinal_controller)
    rollout_fn = jax_jit_np(rollout_fn)
    is_unsafe_fn = jax_jit_np(jax_vmap(env.unsafe_mask))

    rewards = []
    costs = []
    costs_real = []
    is_unsafes = []
    rates = []
    rollouts = []

    for i_epi in range(args.epi):
        key_x0, _ = jr.split(test_keys[i_epi], 2)

        rollout: Rollout = rollout_fn(key_x0)
        is_unsafes.append(is_unsafe_fn(rollout.graph))

        epi_reward = rollout.rewards.sum()
        epi_cost = rollout.costs.max()
        epi_cost_real = rollout.costs_real.max()
        rewards.append(epi_reward)
        costs.append(epi_cost)
        costs_real.append(epi_cost_real)
        rollouts.append(rollout)

        safe_rate = 1 - is_unsafes[-1].max(axis=0).mean()
        print(f"epi: {i_epi}, reward: {epi_reward:.3f}, cost: {epi_cost:.3f}, cost_real: {epi_cost_real:.3f}, "
              f"safe rate: {safe_rate * 100:.3f}%")

        rates.append(np.array(safe_rate))

    is_unsafe = np.max(np.stack(is_unsafes), axis=1)
    safe_mean, safe_std = (1 - is_unsafe).mean(), (1 - is_unsafe).std()

    print(
        f"reward: {np.mean(rewards):.3f}, std: {np.std(rewards):.3f}, min/max reward: {np.min(rewards):.3f}/{np.max(rewards):.3f}, "
        f"cost: {np.mean(costs):.3f} min/max cost: {np.min(costs):.3f}/{np.max(costs):.3f}, "
        f"cost_real: {np.mean(costs_real):.3f} min/max cost_real: {np.min(costs_real):.3f}/{np.max(costs_real):.3f}, "
        f"safe_rate: {safe_mean * 100:.3f}%, std: {safe_std * 100:.3f}%"
    )

    path = args.path
    # save results
    if args.log:
        with open(os.path.join(path, "test_log.csv"), "a") as f:
            f.write(f"{env.num_agents},{args.epi},{env.max_episode_steps},"
                    f"{env.area_size},{env.params['n_obs']},"
                    f"{safe_mean * 100:.3f},{safe_std * 100:.3f}\n")

    # make video
    if args.no_video:
        return

    videos_dir = pathlib.Path(path) / "videos"
    videos_dir.mkdir(exist_ok=True, parents=True)
    for ii, (rollout, Ta_is_unsafe) in enumerate(zip(rollouts, is_unsafes)):
        safe_rate = rates[ii] * 100
        video_name = f"n{args.num_agents}_epi{ii:02}_reward{rewards[ii]:.3f}_cost{costs[ii]:.3f}_sr{safe_rate:.0f}"
        viz_opts = {}
        video_path = videos_dir / f"{stamp_str}_{video_name}.mp4"
        env.render_video(rollout, video_path, Ta_is_unsafe, viz_opts, dpi=args.dpi)


def main():
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument("--path", type=str, required=True, help="存放log和video的位置")

    # optional arguments
    # UFTSTC参数
    parser.add_argument("--Af", type=float, default=17)
    parser.add_argument("--r", type=float, default=3.9)
    parser.add_argument("--mu", type=float, default=10)
    parser.add_argument("--c", type=float, default=2)
    parser.add_argument("--k1", type=float, default=1e-5)
    parser.add_argument("--k2", type=float, default=2e-5)
    parser.add_argument("--k3", type=float, default=1e-1)
    parser.add_argument("--k4", type=float, default=1e-2)
    parser.add_argument("--v", type=float, default=0.5)
    parser.add_argument("--Delta1", type=float, default=3)
    parser.add_argument("--Delta2", type=float, default=50.)
    parser.add_argument("--p-num", type=int, default=3)
    parser.add_argument("--p-den", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=7/9)
    # PID参数
    parser.add_argument("--kp", type=float, default=1.2)
    parser.add_argument("--ki", type=float, default=0.1)
    parser.add_argument("--kd", type=float, default=0.3)
    parser.add_argument("--max-integral", type=float, default=50.)
    parser.add_argument("--min-integral", type=float, default=-50.)
    # 其他参数
    parser.add_argument("--epi", type=int, default=5)
    parser.add_argument("--no-video", action="store_true", default=False)
    parser.add_argument("-n", "--num-agents", type=int, default=1)
    parser.add_argument("--obs", type=int, default=2)
    parser.add_argument("--env", type=str, default='MVELaneChange')
    parser.add_argument("--full-observation", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--log", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--area-size", type=parse_jax_array, default=None,
                        help='输入jax数组，一维用逗号分隔（如10,20），二维用分号+逗号（如10,20;30,40）')
    parser.add_argument("--visible-devices", type=str, default=None)

    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
