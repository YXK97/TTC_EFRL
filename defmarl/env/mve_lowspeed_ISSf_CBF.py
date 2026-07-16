from typing import Tuple

import jax
import jax.numpy as jnp
from typing_extensions import override

from .mve_lowspeed_CBF import MVELaneChangeAndOverTake_LowSpeed_CBF
from defmarl.utils.graph import GraphsTuple
from defmarl.utils.scaling_lowspeed import scaling_calc
from defmarl.utils.typing import Action, Cost
from defmarl.utils.utils import gen_i_j_pairs


class MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF(MVELaneChangeAndOverTake_LowSpeed_CBF):
    """Low-speed environment with an adaptive robust ISSf-CBF cost.

    The steering dynamics use the small-angle affine model ``tan(delta) ~= delta``.
    Longitudinal acceleration is intentionally excluded from the CBF condition.
    """

    PARAMS = MVELaneChangeAndOverTake_LowSpeed_CBF.PARAMS.copy()
    PARAMS.update({
        "gamma": 6.0,
        "issf_epsilon_0": 1.0,
        "issf_epsilon_rate": 0.5,
        "issf_epsilon_min": 1e-2,
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
            alpha, grad_h = jax.value_and_grad(alpha_fn)(z)
            alpha = jnp.nan_to_num(alpha, nan=0.0, posinf=1e6, neginf=0.0)
            grad_h = jnp.nan_to_num(grad_h, nan=0.0, posinf=0.0, neginf=0.0)

            heading = z[2:4] / jnp.maximum(jnp.linalg.norm(z[2:4]), 1e-6)
            speed = s1[4]
            f = jnp.array([speed * heading[0], speed * heading[1], 0.0, 0.0])
            g = jnp.array([
                0.0,
                0.0,
                -heading[1] * speed / self.params["ego_L"],
                heading[0] * speed / self.params["ego_L"],
            ])

            h = alpha - thresh
            lf_h = jnp.dot(grad_h, f)
            lg_h = jnp.dot(grad_h, g)
            young_penalty = jnp.square(lg_h) / epsilon(h)
            residual = lf_h + lg_h * delta_rad + gamma * h - young_penalty
            cost = jnp.nan_to_num(-residual / gamma, nan=10.0, posinf=10.0, neginf=-3.0)
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
