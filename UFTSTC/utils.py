import jax
import jax.numpy as jnp

from UFTSTC.longitudinal_controller import PIDController
#from UFTSTC.latral_controller import UFTSTCController
from UFTSTC.latral_controller_pid import UFTSTCController_pid
from defmarl.env import MultiAgentEnv
from defmarl.utils.typing import PRNGKey
from defmarl.trainer.data import Rollout, Record




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
                    "max_deltaf": action_upper_limit[:,1], "min_deltaf": action_lower_limit[:,1],  "y_min": args.y_min,         # 传递左边界 y
                           "y_max": args.y_max,         # 传递右边界 y
                           "Af_lane": args.Af_lane,     # Lane parameters
                           "r_lane": args.r_lane,
                           "mu_lane": args.mu_lane,
                           "c_lane": args.c_lane,
                           "leak_near_lane": args.leak_near_lane,
                           "leak_far_lane": args.leak_far_lane,
                           "kp_d": args.kp_d,
                           "ki_d": args.ki_d,
                           "kd_d": args.kd_d,
                           "max_integral_d": args.max_integral_d,
                           "min_integral_d": args.min_integral_d}
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
        #lateral_controller: UFTSTCController,
        lateral_controller_pid: UFTSTCController_pid,
        longitudinal_controller: PIDController,
        key: PRNGKey,
):
    init_graph, init_dsYddt = env.reset(key)

    def body(data, xs):
        graph, dsYddt = data
        a_deltaf, a_Psid_metric, ao_BD ,BD_lane, a_Ye ,aS_agent_states , oS_obst_states , a_Yd , deltaf_clip_deg , T_goal_states= lateral_controller_pid.calc_deltaf(graph, dsYddt)
        a_ax , a_ax_clip = longitudinal_controller.calc_ax(graph)



        # debug
        zeros = jnp.zeros_like(a_ax)
        #action = jnp.stack([a_ax, zeros], axis=1)

        action = jnp.stack([a_ax, a_deltaf], axis=1)
        action_sum = jnp.stack([a_ax_clip, deltaf_clip_deg], axis=1)
        next_graph, next_dsYddt, reward, cost, cost_real, done, info = env.step(graph, action)
        return ((next_graph, next_dsYddt),
                (graph, action, action_sum, None, reward, cost, cost_real, done, None, next_graph, dsYddt, \
                 a_ax, a_deltaf, a_Psid_metric, ao_BD, BD_lane, a_Ye ,aS_agent_states , oS_obst_states , a_Yd , T_goal_states))

    _, (graphs, actions , action_sums , rnn_states, rewards, costs, costs_real, dones, log_pis, next_graphs, dsYddts, \
        a_axs, a_deltafs, a_Psid_metrics, ao_BDs, BD_lanes, a_Yes, aS_agent_statess , oS_obst_statess , a_Yds , T_goal_statess) = (
        jax.lax.scan(body,
                     (init_graph, init_dsYddt),
                     None,
                     length=env.max_episode_steps))
    rollout_data = Rollout(graphs, actions, rnn_states, rewards, costs, costs_real, dones, log_pis, next_graphs, dsYddts)
    record_data = Record(a_axs, a_deltafs, a_Psid_metrics,ao_BDs,BD_lanes, a_Yes,aS_agent_statess , oS_obst_statess , a_Yds , action_sums , T_goal_statess)
    return rollout_data, record_data