import argparse
import datetime
import functools as ft
import os
import pathlib
import ipdb
import numpy as np
import yaml

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
    from defmarl.env.mve import MVE

    n_gpu = jax.local_device_count()
    print(f"> Running test.py {args}")
    print(f"> Using {n_gpu} devices")

    stamp_str = datetime.datetime.now().strftime("%m%d-%H%M")

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    if args.debug:
        jax.config.update("jax_disable_jit", True)
    np.random.seed(args.seed)

    if args.path is not None:
        with open(os.path.join(args.path, "config.yaml"), "r") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)

    num_agents = config.num_agents if args.num_agents is None else args.num_agents
    env = make_env(
        env_id=config.env if args.env is None else args.env,
        num_agents=num_agents,
        num_obs=config.obs if args.obs is None else args.obs,
        max_step=args.max_step,
        full_observation=args.full_observation,
        area_size=config.area_size if args.area_size is None else args.area_size,
        reward_min=config.reward_min if args.reward_min is None else args.reward_min,
        reward_max=config.reward_max if args.reward_max is None else args.reward_max
    )
    if args.scene_mode is not None:
        env.params["scene_mode"] = args.scene_mode

    path = args.path
    model_path = os.path.join(path, "models")
    if args.from_iter is None:
        models = os.listdir(model_path)
        from_iter = max([int(model) for model in models if model.isdigit()])
    else:
        from_iter = args.from_iter
    print("from_iter: ", from_iter)

    algo = make_algo(
        algo=config.algo,
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        cost_weight=config.cost_weight,
        actor_gnn_layers=config.gnn_layers,
        critic_gnn_layers=config.gnn_layers,
        Vh_gnn_layers=config.Vh_gnn_layers if hasattr(config, "Vh_gnn_layers") else 1,
        lr_actor=config.lr_actor,
        lr_cbf=config.lr_critic,
        max_grad_norm=2.0,
        seed=config.seed,
        use_rnn=config.use_rnn,
        rnn_layers=config.rnn_layers,
        use_lstm=config.use_lstm,
    )
    algo.load(model_path, from_iter)

    if args.stochastic:
        def act_fn(x, z, rnn_state, key):
            action, _, new_rnn_state = algo.step(x, z, rnn_state, key)
            return action, new_rnn_state
        act_fn = jax.jit(act_fn)
    else:
        act_fn = algo.act

    z_fn = algo.get_opt_z if hasattr(algo, "get_opt_z") else None
    if args.z is not None:
        if args.z == "min":
            z_fn = lambda graph, value_rnn_state: \
                (jnp.array([[-env.reward_max]]).repeat(env.num_agents, axis=0), value_rnn_state)
        elif args.z == "max":
            z_fn = lambda graph, value_rnn_state: \
                (jnp.array([[-env.reward_min]]).repeat(env.num_agents, axis=0), value_rnn_state)
        else:
            raise ValueError(f"Unknown z: {args.z}")

    act_fn = jax.jit(act_fn)
    init_rnn_state = algo.init_rnn_state
    init_Vh_rnn_state = algo.init_Vh_rnn_state if hasattr(algo, "init_Vh_rnn_state") else None

    test_key = jr.PRNGKey(args.seed)
    test_keys = jr.split(test_key, args.epi)
    test_keys = test_keys[args.offset:]

    rollout_fn = ft.partial(eval_rollout,
                            env,
                            act_fn,
                            init_rnn_state,
                            init_Vh_rnn_state=init_Vh_rnn_state,
                            z_fn=z_fn,
                            stochastic=args.stochastic)
    rollout_fn = jax_jit_np(rollout_fn)
    # is_unsafe_fn = jax_jit_np(jax_vmap(env.unsafe_mask))

    rewards = []
    costs = []
    costs_real = []
    is_unsafes = []
    rates = []
    rollouts = []

    for i_epi in range(args.epi):
        key_x0, _ = jr.split(test_keys[i_epi], 2)
        rollout: Rollout = rollout_fn(key_x0)

        if args.output_csv:
            T_graph = rollout.graph
            T = T_graph.states.shape[0]
            agent_states = []
            goal_states = []
            obst_states = []

            for t in range(T):
                graph_t = jax.tree_util.tree_map(lambda x: x[t], T_graph)
                agent_states.append(graph_t.type_states(type_idx=MVE.AGENT, n_type=env.num_agents))
                goal_states.append(graph_t.type_states(type_idx=MVE.GOAL, n_type=env.num_agents))
                obst_states.append(graph_t.type_states(type_idx=MVE.OBST, n_type=env.num_obsts))

            agent_states = jnp.stack(agent_states)
            goal_states = jnp.stack(goal_states)
            obst_states = jnp.stack(obst_states)

            state_dir = os.path.join(args.path, "state_csv")
            os.makedirs(state_dir, exist_ok=True)

            def infer_scene_info():
                reset_key = jr.split(key_x0, 2)[0]
                try:
                    if env.__class__.__name__ != "MVELaneChangeAndOverTake_LowSpeed_CBF":
                        raise RuntimeError("split scene metadata is only used for LowSpeed CBF")
                    from defmarl.env.designed_scene_gen_two_lane_split import gen_scene_randomly as gen_split_scene
                    agents0, obsts0, all_goals0, _ = gen_split_scene(
                        reset_key,
                        env.num_agents,
                        env.num_goals,
                        env.params["default_state_range"][:2],
                        env.params["default_state_range"][2:4],
                        env.params["lane_width"],
                        env.params["lane_centers"],
                    )
                except Exception:
                    agents0 = agent_states[0]
                    obsts0 = obst_states[0]
                    all_goals0 = goal_states.transpose((1, 0, 2))

                agent0 = np.asarray(agents0[0])
                obst0 = np.asarray(obsts0[0]) if np.asarray(obsts0).shape[0] > 0 else np.full((env.state_dim,), np.nan)
                goals0 = np.asarray(all_goals0[0])
                lane_centers = np.asarray(env.params["lane_centers"])
                goal_y_span = float(np.nanmax(goals0[:, 1]) - np.nanmin(goals0[:, 1]))
                scene_type = "lanechange" if goal_y_span > 0.5 else "overtake"
                dx_obst_agent = float(obst0[0] - agent0[0])
                agent_lane = int(np.nanargmin(np.abs(lane_centers - agent0[1]))) if np.isfinite(agent0[1]) else -1
                obst_lane = int(np.nanargmin(np.abs(lane_centers - obst0[1]))) if np.isfinite(obst0[1]) else -1
                terminal_goal_y = float(goals0[-1, 1])

                if dx_obst_agent > 6.0:
                    scene_phase = "Approach"
                elif abs(dx_obst_agent) <= 6.0:
                    scene_phase = "Side"
                elif abs(float(agent0[1]) - terminal_goal_y) <= 0.35:
                    scene_phase = "Done"
                else:
                    scene_phase = "Passed"

                return {
                    "episode": i_epi,
                    "scene_type": scene_type,
                    "scene_phase": scene_phase,
                    "agent_x": float(agent0[0]),
                    "agent_y": float(agent0[1]),
                    "agent_lane": agent_lane,
                    "obst_x": float(obst0[0]),
                    "obst_y": float(obst0[1]),
                    "obst_lane": obst_lane,
                    "dx_obst_minus_agent": dx_obst_agent,
                    "goal_y_start": float(goals0[0, 1]),
                    "goal_y_end": terminal_goal_y,
                    "goal_y_span": goal_y_span,
                }

            scene_info = infer_scene_info()
            scene_csv_path = os.path.join(state_dir, f"{stamp_str}_scene_info.csv")
            scene_header = list(scene_info.keys())
            scene_file_exists = os.path.exists(scene_csv_path)
            with open(scene_csv_path, "a") as f:
                if not scene_file_exists:
                    f.write(",".join(scene_header) + "\n")
                f.write(",".join(str(scene_info[k]) for k in scene_header) + "\n")
            print(f"scene信息保存位置: {scene_csv_path}, scene_type={scene_info['scene_type']}, phase={scene_info['scene_phase']}")

            def save_entity_csv(states, entity_name, entity_id):
                csv_path = os.path.join(state_dir, f"{stamp_str}_epi{i_epi:02d}_{entity_name}{entity_id:02d}_states.csv")
                state_dim = states.shape[-1]
                state_cols = [f"s{i}" for i in range(state_dim)]
                with open(csv_path, "w") as f:
                    f.write(",".join(["time_step"] + state_cols) + "\n")
                    for t in range(states.shape[0]):
                        values = ",".join(f"{float(v):.6f}" for v in states[t, entity_id])
                        f.write(f"{t},{values}\n")
                return csv_path

            csv_paths = []
            for agent_id in range(agent_states.shape[1]):
                csv_paths.append(save_entity_csv(agent_states, "agent", agent_id))
            for goal_id in range(goal_states.shape[1]):
                csv_paths.append(save_entity_csv(goal_states, "goal", goal_id))
            for obst_id in range(obst_states.shape[1]):
                csv_paths.append(save_entity_csv(obst_states, "obst", obst_id))
            for csv_path in csv_paths:
                print(f"csv保存位置: {csv_path}")
        
        is_unsafes.append(jnp.any(rollout.costs_real >= 1e-6, axis=-1))
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

    if args.log:
        with open(os.path.join(path, "test_log.csv"), "a") as f:
            f.write(f"{env.num_agents},{args.epi},{env.max_episode_steps},"
                    f"{env.area_size},{env.params['n_obs']},"
                    f"{safe_mean * 100:.3f},{safe_std * 100:.3f}\n")

    if args.no_video:
        return

    videos_dir = pathlib.Path(path) / "videos" / f"{from_iter}"
    videos_dir.mkdir(exist_ok=True, parents=True)
    for ii, (rollout, Ta_is_unsafe) in enumerate(zip(rollouts, is_unsafes)):
        safe_rate = rates[ii] * 100
        video_name = f"n{num_agents}_epi{ii:02}_reward{rewards[ii]:.3f}_cost{costs[ii]:.3f}_sr{safe_rate:.0f}"
        video_path = videos_dir / f"{stamp_str}_{video_name}.mp4"
        env.render_video(rollout, video_path, Ta_is_unsafe, dpi=args.dpi)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--reward_min", type=float, default=None)
    parser.add_argument("--reward_max", type=float, default=None)
    parser.add_argument("--epi", type=int, default=5)
    parser.add_argument("--no-video", action="store_true", default=False)
    parser.add_argument("--from-iter", type=int, default=None)
    parser.add_argument("-n", "--num-agents", type=int, default=None)
    parser.add_argument("--obs", type=int, default=None)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--full-observation", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--stochastic", action="store_true", default=False)
    parser.add_argument("--log", action="store_true", default=False)
    parser.add_argument("--output-csv", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("-z", type=str, default=None)
    parser.add_argument("--area-size", type=parse_jax_array, default=None)
    parser.add_argument("--scene-mode", type=str, default=None,
                        choices=["random", "handmade", "uftstc_left", "uftstc_straight"])
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--visible-devices", type=str, default=None)

    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
