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
    comm_radius = args.comm_radius
    if comm_radius is None and hasattr(config, "comm_radius"):
        comm_radius = config.comm_radius
    env = make_env(
        env_id=config.env if args.env is None else args.env,
        num_agents=num_agents,
        num_obs=config.obs if args.obs is None else args.obs,
        max_step=args.max_step,
        full_observation=args.full_observation,
        area_size=config.area_size if args.area_size is None else args.area_size,
        reward_min=config.reward_min if args.reward_min is None else args.reward_min,
        reward_max=config.reward_max if args.reward_max is None else args.reward_max,
        comm_radius=comm_radius
    )
    if args.scene_mode is not None:
        env.params["scene_mode"] = args.scene_mode
    if args.deterministic_scene and not hasattr(env, "reset_deterministic"):
        raise ValueError(
            f"Environment {type(env).__name__} does not provide the four "
            "deterministic demonstration scenes."
        )

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

    num_episodes = 4 if args.deterministic_scene else args.epi
    test_key = jr.PRNGKey(args.seed)
    test_keys = jr.split(test_key, num_episodes)
    if not args.deterministic_scene:
        test_keys = test_keys[args.offset:]
    else:
        print("> Deterministic scenes enabled: running scene 0 through 3")

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

    for i_epi in range(num_episodes):
        key_x0, _ = jr.split(test_keys[i_epi], 2)
        if args.deterministic_scene:
            rollout: Rollout = rollout_fn(
                key_x0,
                deterministic_scene_index=jnp.asarray(
                    i_epi, dtype=jnp.int32
                ),
            )
        else:
            rollout: Rollout = rollout_fn(key_x0)

        if args.output_csv:
            T_graph = rollout.graph
            T = T_graph.states.shape[0]
            # Preview-goal environments expose two GOAL nodes per ego.  Keep
            # the historical one-goal behavior for every existing environment.
            num_graph_goals = getattr(
                env, "num_graph_goals", env.num_agents
            )
            agent_states = []
            goal_states = []
            obst_states = []

            for t in range(T):
                graph_t = jax.tree_util.tree_map(lambda x: x[t], T_graph)
                agent_states.append(graph_t.type_states(type_idx=MVE.AGENT, n_type=env.num_agents))
                goal_states.append(
                    graph_t.type_states(
                        type_idx=MVE.GOAL, n_type=num_graph_goals
                    )
                )
                obst_states.append(graph_t.type_states(type_idx=MVE.OBST, n_type=env.num_obsts))

            agent_states = jnp.stack(agent_states)
            goal_states = jnp.stack(goal_states)
            obst_states = jnp.stack(obst_states)

            state_dir = os.path.join(args.path, "state_csv")
            os.makedirs(state_dir, exist_ok=True)

            def save_entity_csv(states, entity_name, entity_id):
                csv_path = os.path.join(state_dir, f"{stamp_str}_epi{i_epi:02d}_{entity_name}{entity_id:02d}_states.csv")
                state_dim = states.shape[-1]
                # Low-speed environments expose semantic state labels shared
                # by ego, goals, and obstacles.  Keep a generic fallback for
                # environments with a different or unspecified state layout.
                semantic_labels = tuple(getattr(env, "state_labels", ()))
                state_cols = (
                    list(semantic_labels)
                    if len(semantic_labels) == state_dim
                    else [f"s{i}" for i in range(state_dim)]
                )
                with open(csv_path, "w") as f:
                    f.write(",".join(["time_step"] + state_cols) + "\n")
                    for t in range(states.shape[0]):
                        # float32 needs roughly nine significant decimal digits
                        # for a round trip.  Six decimal places can change which
                        # polygon edge is active in safety diagnostics.
                        values = ",".join(
                            f"{float(v):.9e}" for v in states[t, entity_id]
                        )
                        f.write(f"{t},{values}\n")
                return csv_path

            def save_safety_debug_csv(
                agent_id,
                raw_actions,
                transformed_actions,
                costs,
                costs_real,
                diagnostics,
            ):
                """Save action-dependent safety values without recomputation later."""
                csv_path = os.path.join(
                    state_dir,
                    f"{stamp_str}_epi{i_epi:02d}_agent{agent_id:02d}_safety_debug.csv",
                )
                columns = ["time_step"]
                columns += [f"raw_action{idx}" for idx in range(raw_actions.shape[-1])]
                columns += [
                    f"transformed_action{idx}"
                    for idx in range(transformed_actions.shape[-1])
                ]
                columns += ["applied_steering"]
                cost_names = [name.replace(" ", "_") for name in env.cost_components]
                columns += [f"cost_{name}" for name in cost_names]
                columns += [f"cost_real_{name}" for name in cost_names]

                def boundary_columns(prefix):
                    return [
                        f"{prefix}_alpha",
                        f"{prefix}_alpha_grad_x",
                        f"{prefix}_alpha_grad_y",
                        f"{prefix}_alpha_grad_heading_x",
                        f"{prefix}_alpha_grad_heading_y",
                        f"{prefix}_h_dot",
                        f"{prefix}_g_dot",
                    ]

                has_split_road_boundaries = hasattr(
                    diagnostics, "lower_boundary_alpha"
                )
                if has_split_road_boundaries:
                    columns += boundary_columns("lower_boundary")
                    columns += boundary_columns("upper_boundary")
                else:
                    # Intersection environments expose one combined polygon
                    # boundary constraint and retain the existing column names.
                    columns += boundary_columns("boundary")
                for obstacle_id in range(diagnostics.obstacle_alpha.shape[2]):
                    prefix = f"obstacle{obstacle_id:02d}"
                    columns += [
                        f"{prefix}_alpha",
                        f"{prefix}_alpha_grad_x",
                        f"{prefix}_alpha_grad_y",
                        f"{prefix}_alpha_grad_heading_x",
                        f"{prefix}_alpha_grad_heading_y",
                        f"{prefix}_h_dot",
                        f"{prefix}_g_dot",
                    ]

                with open(csv_path, "w") as f:
                    f.write(",".join(columns) + "\n")
                    for t in range(raw_actions.shape[0]):
                        values = []
                        values.extend(raw_actions[t, agent_id])
                        values.extend(transformed_actions[t, agent_id])
                        values.append(diagnostics.applied_steering[t, agent_id])
                        values.extend(costs[t, agent_id])
                        values.extend(costs_real[t, agent_id])
                        if has_split_road_boundaries:
                            for prefix in ("lower_boundary", "upper_boundary"):
                                values.append(
                                    getattr(diagnostics, f"{prefix}_alpha")[
                                        t, agent_id
                                    ]
                                )
                                values.extend(
                                    getattr(diagnostics, f"{prefix}_alpha_grad")[
                                        t, agent_id
                                    ]
                                )
                                values.append(
                                    getattr(diagnostics, f"{prefix}_h_dot")[
                                        t, agent_id
                                    ]
                                )
                                values.append(
                                    getattr(diagnostics, f"{prefix}_g_dot")[
                                        t, agent_id
                                    ]
                                )
                        else:
                            values.append(diagnostics.boundary_alpha[t, agent_id])
                            values.extend(diagnostics.boundary_alpha_grad[t, agent_id])
                            values.append(diagnostics.boundary_h_dot[t, agent_id])
                            values.append(diagnostics.boundary_g_dot[t, agent_id])
                        for obstacle_id in range(diagnostics.obstacle_alpha.shape[2]):
                            values.append(
                                diagnostics.obstacle_alpha[t, agent_id, obstacle_id]
                            )
                            values.extend(
                                diagnostics.obstacle_alpha_grad[
                                    t, agent_id, obstacle_id
                                ]
                            )
                            values.append(
                                diagnostics.obstacle_h_dot[t, agent_id, obstacle_id]
                            )
                            values.append(
                                diagnostics.obstacle_g_dot[t, agent_id, obstacle_id]
                            )
                        formatted = ",".join(f"{float(v):.9e}" for v in values)
                        f.write(f"{t},{formatted}\n")
                return csv_path

            csv_paths = []
            for agent_id in range(agent_states.shape[1]):
                csv_paths.append(save_entity_csv(agent_states, "agent", agent_id))
            for goal_id in range(goal_states.shape[1]):
                csv_paths.append(save_entity_csv(goal_states, "goal", goal_id))
            for obst_id in range(obst_states.shape[1]):
                csv_paths.append(save_entity_csv(obst_states, "obst", obst_id))

            # The actor output stored in Rollout is normalized.  Environment
            # costs use the transformed action and the rate-filtered steering,
            # so save all three to make every frame reproducible.
            raw_actions = jnp.asarray(rollout.actions)
            transformed_actions = jax.vmap(env.transform_action)(raw_actions)
            if hasattr(env, "get_safety_diagnostics"):
                diagnostics = jax.jit(
                    jax.vmap(env.get_safety_diagnostics)
                )(T_graph, transformed_actions)
                for agent_id in range(agent_states.shape[1]):
                    csv_paths.append(
                        save_safety_debug_csv(
                            agent_id,
                            raw_actions,
                            transformed_actions,
                            jnp.asarray(rollout.costs),
                            jnp.asarray(rollout.costs_real),
                            diagnostics,
                        )
                    )
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
            f.write(f"{env.num_agents},{num_episodes},{env.max_episode_steps},"
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
    parser.add_argument(
        "--deterministic-scene",
        action="store_true",
        default=False,
        help="Ignore --epi/--offset and run the four fixed demonstration scenes.",
    )
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
    parser.add_argument("--comm-radius", type=float, default=None)
    parser.add_argument("--scene-mode", type=str, default=None,
                        choices=["random", "handmade", "uftstc_left", "uftstc_straight"])
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--visible-devices", type=str, default=None)

    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
