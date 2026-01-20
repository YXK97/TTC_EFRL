import jax
import jax.numpy as jnp

from UFTSTC.longitudinal_controller import PIDController
from UFTSTC.latral_controller import UFTSTCController
from defmarl.env import MultiAgentEnv
from defmarl.utils.typing import PRNGKey
from defmarl.trainer.data import Rollout


def parse_uftstc_controller_args(args, env: MultiAgentEnv) -> dict:
    """UFTSTC controller args: num_agents:int, num_obsts:int, dt:float, Af: float, r:float, mu:float, c:float, k1:float,
                 k2:float, k3:float, k4:float, v:float, Delta1:float, Delta2:float, p_num:int, p_den:int, alpha:float,
                 Cf:float, Cr:float, Lf:float, Lr:float, m:float, Iz:float,
                 max_deltaf:float, # deg
                 min_deltaf:float  # deg"""
    action_lower_limit, action_upper_limit = env.action_lim()
    uftstc_control_args = {"num_agents": args.num_agents, "num_obsts": args.obs, "dt": env.dt, "Af": args.Af,
                           "r": args.r, "mu": args.mu, "c": args.c, "k1": args.k1, "k2": args.k2, "k3": args.k3,
                    "k4": args.k4, "v": args.v, "Delta1": args.Delta1, "Delta2": args.Delta2, "p_num": args.p_num,
                    "p_den": args.p_den, "alpha": args.alpha,
                    "Cf": env.params["ego_Cf"], "Cr": env.params["ego_Cr"], "Lf": env.params["ego_lf"],
                    "Lr": env.params["ego_lr"], "m": env.params["ego_m"], "Iz": env.params["ego_Iz"],
                    "max_deltaf": action_upper_limit[:,1], "min_deltaf": action_lower_limit[:,1]}
    return uftstc_control_args


def parse_pid_controller_args(args, env: MultiAgentEnv) -> dict:
    """pid controller args: num_agents:int, kp:float, ki:float, kd:float, dt:float, max_integral:float, min_integral:float,
                 max_ax:float, # m/s^2
                 min_ax:float  # m/s^2"""
    action_lower_limit, action_upper_limit = env.action_lim()
    pid_control_args = {"num_agents": args.num_agents, "kp": args.kp, "ki": args.ki, "kd": args.kd,
                        "dt": env.dt, "max_integral": args.max_integral, "min_integral": args.min_integral,
                        "max_ax": action_upper_limit[:,0], "min_ax": action_lower_limit[:,0]}
    return pid_control_args


def eval_rollout_uftstc(
        env: MultiAgentEnv,
        lateral_controller: UFTSTCController,
        longitudinal_controller: PIDController,
        key: PRNGKey,
):
    init_graph, init_dsYddt = env.reset(key)

    def body(data, xs):
        graph, dsYddt = data
        a_deltaf = lateral_controller.calc_deltaf(graph, dsYddt)
        a_ax = longitudinal_controller.calc_ax(graph)

        # debug
        zeros = jnp.zeros_like(a_ax)
        #action = jnp.stack([a_ax, zeros], axis=1)

        action = jnp.stack([a_ax, a_deltaf], axis=1)
        next_graph, next_dsYddt, reward, cost, cost_real, done, info = env.step(graph, action)
        return ((next_graph, next_dsYddt),
                (graph, action, None, reward, cost, cost_real, done, None, next_graph, dsYddt))

    _, (graphs, actions, rnn_states, rewards, costs, costs_real, dones, log_pis, next_graphs, dsYddts) = (
        jax.lax.scan(body,
                     (init_graph, init_dsYddt),
                     None,
                     length=env.max_episode_steps))
    rollout_data = Rollout(graphs, actions, rnn_states, rewards, costs, costs_real, dones, log_pis, next_graphs, dsYddts)
    return rollout_data