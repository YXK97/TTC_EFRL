from typing import Tuple

import jax
import jax.numpy as jnp
from typing_extensions import override

from .mve_lowspeed_CBF import MVELaneChangeAndOverTake_LowSpeed_CBF
from defmarl.utils.graph import GraphsTuple
from defmarl.utils.scaling_lowspeed import scaling_calc
from defmarl.utils.typing import Action, Cost, Reward
from defmarl.utils.utils import gen_i_j_pairs


class MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF(MVELaneChangeAndOverTake_LowSpeed_CBF):
    """Low-speed environment with an adaptive robust ISSf-CBF cost.

    Longitudinal acceleration is intentionally excluded from the CBF condition.
    """

    PARAMS = MVELaneChangeAndOverTake_LowSpeed_CBF.PARAMS.copy()
    PARAMS.update({
        "gamma": 10.0,
        "issf_epsilon_0": 1.0,
        "issf_epsilon_rate": 1.0,
        "issf_epsilon_min": 100.0,
    })

    @override
    def get_cost(self, graph: GraphsTuple, action: Action) -> Tuple[Cost, Cost]:
        thresh = self.params["alpha_thresh"]
        gamma = self.params["gamma"]
        num_agents = graph.env_states.agent.shape[0]
        num_obsts = graph.env_states.obstacle.shape[0]
        delta = self._filter_delta(graph.env_states.agent[:, 5], action[:, 1])

        def epsilon(h):
            epsilon_0 = self.params["issf_epsilon_0"]
            epsilon_rate = self.params["issf_epsilon_rate"]
            epsilon_min = self.params["issf_epsilon_min"]
            return epsilon_min + epsilon_0 * jax.nn.softplus(epsilon_rate * h)

        def issf_cbf_between(s1, s2, delta_rad, bb_size, lr):
            def alpha_fn(z):
                full = jnp.array([z[0], z[1], z[2], z[3], s1[4], s1[5]])
                return scaling_calc(
                    full,
                    s2,
                    self.params["ego_bb_size"],
                    self.params["ego_lr"],
                    bb_size,
                    lr,
                )

            z = s1[:4]
            alpha, grad_z = jax.value_and_grad(alpha_fn)(z)
            alpha = jnp.nan_to_num(alpha, nan=0.0, posinf=1e6, neginf=0.0)
            grad_z = jnp.nan_to_num(grad_z, nan=0.0, posinf=0.0, neginf=0.0)
            hvec = z[2:4] / jnp.maximum(jnp.linalg.norm(z[2:4]), 1e-6)
            omega = s1[4] / self.params["ego_L"] * jnp.tan(delta_rad)
            z_dot = jnp.array(
                [s1[4] * hvec[0], s1[4] * hvec[1], -hvec[1] * omega, hvec[0] * omega]
            )
            h = alpha - thresh
            h_dot = jnp.dot(grad_z, z_dot)
            g_dot = jnp.dot(
                grad_z,
                jnp.array([
                    0.0,
                    0.0,
                    -hvec[1] * s1[4] / self.params["ego_L"],
                    hvec[0] * s1[4] / self.params["ego_L"],
                ]),
            )
            young_penalty = jnp.square(g_dot) / epsilon(h)
            residual = h_dot / gamma + h - young_penalty / gamma
            cost = jnp.nan_to_num(-residual, nan=10.0, posinf=10.0, neginf=-3.0)
            return cost, 1 - alpha

        a_agent_cost = -jnp.ones((num_agents,), dtype=jnp.float32) * 3.0
        a_agent_cost_real = -jnp.ones((num_agents,), dtype=jnp.float32) * 3.0

        if num_obsts == 0:
            a_obst_cost = -jnp.ones((num_agents,), dtype=jnp.float32) * 3.0
            a_obst_cost_real = -jnp.ones((num_agents,), dtype=jnp.float32) * 3.0
        else:
            i_pairs, j_pairs = gen_i_j_pairs(num_agents, num_obsts)
            costs, reals = jax.vmap(issf_cbf_between, in_axes=(0, 0, 0, None, None))(
                graph.env_states.agent[i_pairs],
                graph.env_states.obstacle[j_pairs],
                delta[i_pairs],
                self.params["obst_bb_size"],
                self.params["obst_lr"],
            )
            a_obst_cost = jnp.max(costs.reshape((num_agents, num_obsts)), axis=1)
            a_obst_cost_real = jnp.max(reals.reshape((num_agents, num_obsts)), axis=1)

        bounds = self.generate_bound(graph.env_states.agent, self.params["bound_bb_size"])
        a_low_cost, a_low_real = jax.vmap(issf_cbf_between, in_axes=(0, 0, 0, None, None))(
            graph.env_states.agent,
            bounds[::2],
            delta,
            self.params["bound_bb_size"],
            0.0,
        )
        a_high_cost, a_high_real = jax.vmap(issf_cbf_between, in_axes=(0, 0, 0, None, None))(
            graph.env_states.agent,
            bounds[1::2],
            delta,
            self.params["bound_bb_size"],
            0.0,
        )

        cost = jnp.stack([a_agent_cost, a_obst_cost, a_low_cost, a_high_cost], axis=1)
        cost_real = jnp.stack([a_agent_cost_real, a_obst_cost_real, a_low_real, a_high_real], axis=1)
        cost = jnp.where(cost <= 0.0, cost, cost + 1.0)
        return jnp.clip(cost, a_min=-10.0, a_max=10.0), cost_real

    @override
    def get_reward(self, graph: GraphsTuple, action: Action) -> Reward:
        agent = self._observable(graph.env_states.agent)
        goal = self._observable(graph.env_states.goal)
        e = agent - goal
        W = jnp.diag(jnp.array([1e-3, 1e-3, 0, 0, 1e-3, 0]))
        reward = -jnp.sqrt(jnp.einsum("ai,ij,ja->a", e, W, e.transpose())).mean()
        reward -= (action[:, 0] ** 2).mean() * 0.0001
        reward -= (action[:, 1] ** 2).mean() * 0.0001
        return reward
