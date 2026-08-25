from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from typing_extensions import override

from .mve_lowspeed_CBF import MVELaneChangeAndOverTake_LowSpeed_CBF
from defmarl.utils.graph import GraphsTuple
from defmarl.utils.scaling_lowspeed import (
    scaling_calc,
    scaling_calc_parameterized,
    scaling_calc_unbounded_bound,
)
from defmarl.utils.typing import Action, Array, Cost, Reward, State
from defmarl.utils.utils import gen_i_j_pairs


class LowSpeedSafetyDiagnostics(NamedTuple):
    """Per-constraint values needed to reproduce straight-road ISSf costs."""

    applied_steering: Array
    obstacle_alpha: Array
    obstacle_alpha_grad: Array
    obstacle_h_dot: Array
    obstacle_g_dot: Array
    lower_boundary_alpha: Array
    lower_boundary_alpha_grad: Array
    lower_boundary_h_dot: Array
    lower_boundary_g_dot: Array
    upper_boundary_alpha: Array
    upper_boundary_alpha_grad: Array
    upper_boundary_h_dot: Array
    upper_boundary_g_dot: Array


class MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF(MVELaneChangeAndOverTake_LowSpeed_CBF):
    """Low-speed environment with an adaptive robust ISSf-CBF cost.

    Longitudinal acceleration is intentionally excluded from the CBF condition.
    """

    USE_UNBOUNDED_ISSF_ROAD_BOUNDS = False
    USE_PARAMETERIZED_ISSF_OBSTACLE_SCALING = False

    PARAMS = MVELaneChangeAndOverTake_LowSpeed_CBF.PARAMS.copy()
    PARAMS.update({
        "gamma": 10.0,
        "issf_epsilon_0": 1.0,
        "issf_epsilon_rate": 1.0,
        "issf_epsilon_min": 100.0,
    })

    def _safety_diagnostic_terms(
        self,
        alpha_fn,
        state: State,
        steering: Array,
    ) -> Tuple[Array, Array, Array, Array]:
        """Return alpha, its ego-pose gradient, h_dot, and g_dot.

        ``alpha_fn`` accepts a complete six-dimensional ego state.  Only the
        ego pose is differentiated, exactly as in ``get_cost``; speed and the
        previous steering state remain closed-over constants.
        """

        def alpha_from_pose(pose):
            full_state = jnp.array(
                [pose[0], pose[1], pose[2], pose[3], state[4], state[5]]
            )
            return alpha_fn(full_state)

        alpha, alpha_grad = jax.value_and_grad(alpha_from_pose)(state[:4])
        alpha = jnp.nan_to_num(alpha, nan=0.0, posinf=1e6, neginf=0.0)
        alpha_grad = jnp.nan_to_num(
            alpha_grad, nan=0.0, posinf=0.0, neginf=0.0
        )
        heading = state[2:4] / jnp.maximum(
            jnp.linalg.norm(state[2:4]), 1e-6
        )
        angular_speed = state[4] / self.params["ego_L"] * jnp.tan(steering)
        pose_dot = jnp.array(
            [
                state[4] * heading[0],
                state[4] * heading[1],
                -heading[1] * angular_speed,
                heading[0] * angular_speed,
            ]
        )
        steering_channel = jnp.array(
            [
                0.0,
                0.0,
                -heading[1] * state[4] / self.params["ego_L"],
                heading[0] * state[4] / self.params["ego_L"],
            ]
        )
        h_dot = jnp.dot(alpha_grad, pose_dot)
        g_dot = jnp.dot(alpha_grad, steering_channel)
        return alpha, alpha_grad, h_dot, g_dot

    def get_safety_diagnostics(
        self,
        graph: GraphsTuple,
        transformed_action: Action,
    ) -> LowSpeedSafetyDiagnostics:
        """Compute exact lower/high and per-obstacle straight-road diagnostics.

        The caller must pass actions already transformed into physical units.
        Scaling-function selection and steering filtering intentionally mirror
        ``get_cost`` so exported values reproduce the rollout cost path.
        """
        agents = graph.env_states.agent
        obstacles = graph.env_states.obstacle
        num_agents = agents.shape[0]
        num_obstacles = obstacles.shape[0]
        steering = self._filter_delta(agents[:, 5], transformed_action[:, 1])
        obstacle_scaling_fn = (
            scaling_calc_parameterized
            if self.USE_PARAMETERIZED_ISSF_OBSTACLE_SCALING
            else scaling_calc
        )

        if num_obstacles == 0:
            obstacle_alpha = jnp.empty((num_agents, 0), dtype=jnp.float32)
            obstacle_grad = jnp.empty((num_agents, 0, 4), dtype=jnp.float32)
            obstacle_h_dot = jnp.empty((num_agents, 0), dtype=jnp.float32)
            obstacle_g_dot = jnp.empty((num_agents, 0), dtype=jnp.float32)
        else:
            i_pairs, j_pairs = gen_i_j_pairs(num_agents, num_obstacles)

            def obstacle_terms(state, obstacle, steering_value):
                def obstacle_alpha_fn(ego_state):
                    return obstacle_scaling_fn(
                        ego_state,
                        obstacle,
                        self.params["ego_bb_size"],
                        self.params["ego_lr"],
                        self.params["obst_bb_size"],
                        self.params["obst_lr"],
                    )

                return self._safety_diagnostic_terms(
                    obstacle_alpha_fn, state, steering_value
                )

            obstacle_alpha, obstacle_grad, obstacle_h_dot, obstacle_g_dot = (
                jax.vmap(obstacle_terms)(
                    agents[i_pairs],
                    obstacles[j_pairs],
                    steering[i_pairs],
                )
            )
            obstacle_alpha = obstacle_alpha.reshape(
                (num_agents, num_obstacles)
            )
            obstacle_grad = obstacle_grad.reshape(
                (num_agents, num_obstacles, 4)
            )
            obstacle_h_dot = obstacle_h_dot.reshape(
                (num_agents, num_obstacles)
            )
            obstacle_g_dot = obstacle_g_dot.reshape(
                (num_agents, num_obstacles)
            )

        if self.USE_UNBOUNDED_ISSF_ROAD_BOUNDS:
            y_low = self.params["default_state_range"][2]
            y_high = self.params["default_state_range"][3]
            lower_A = jnp.array([[0.0, 1.0]])
            lower_b = jnp.array([y_low])
            upper_A = jnp.array([[0.0, -1.0]])
            upper_b = jnp.array([-y_high])

            def unbounded_terms(state, steering_value, A, b):
                def boundary_alpha_fn(ego_state):
                    return scaling_calc_unbounded_bound(
                        ego_state,
                        self.params["ego_bb_size"],
                        self.params["ego_lr"],
                        A,
                        b,
                    )

                return self._safety_diagnostic_terms(
                    boundary_alpha_fn, state, steering_value
                )

            lower_terms = jax.vmap(
                unbounded_terms, in_axes=(0, 0, None, None)
            )(agents, steering, lower_A, lower_b)
            upper_terms = jax.vmap(
                unbounded_terms, in_axes=(0, 0, None, None)
            )(agents, steering, upper_A, upper_b)
        else:
            bounds = self.generate_bound(
                agents, self.params["bound_bb_size"]
            )

            def virtual_boundary_terms(state, boundary, steering_value):
                def boundary_alpha_fn(ego_state):
                    return obstacle_scaling_fn(
                        ego_state,
                        boundary,
                        self.params["ego_bb_size"],
                        self.params["ego_lr"],
                        self.params["bound_bb_size"],
                        0.0,
                    )

                return self._safety_diagnostic_terms(
                    boundary_alpha_fn, state, steering_value
                )

            lower_terms = jax.vmap(virtual_boundary_terms)(
                agents, bounds[::2], steering
            )
            upper_terms = jax.vmap(virtual_boundary_terms)(
                agents, bounds[1::2], steering
            )

        return LowSpeedSafetyDiagnostics(
            steering,
            obstacle_alpha,
            obstacle_grad,
            obstacle_h_dot,
            obstacle_g_dot,
            lower_terms[0],
            lower_terms[1],
            lower_terms[2],
            lower_terms[3],
            upper_terms[0],
            upper_terms[1],
            upper_terms[2],
            upper_terms[3],
        )

    @override
    def get_cost(self, graph: GraphsTuple, action: Action) -> Tuple[Cost, Cost]:
        thresh = self.params["alpha_thresh"]
        gamma = self.params["gamma"]
        num_agents = graph.env_states.agent.shape[0]
        num_obsts = graph.env_states.obstacle.shape[0]
        delta = self._filter_delta(graph.env_states.agent[:, 5], action[:, 1])
        obstacle_scaling_fn = (
            scaling_calc_parameterized
            if self.USE_PARAMETERIZED_ISSF_OBSTACLE_SCALING
            else scaling_calc
        )

        def epsilon(h):
            epsilon_0 = self.params["issf_epsilon_0"]
            epsilon_rate = self.params["issf_epsilon_rate"]
            epsilon_min = self.params["issf_epsilon_min"]
            return epsilon_min + epsilon_0 * jax.nn.softplus(epsilon_rate * h)

        def issf_cbf_constraint(alpha_fn, s1, delta_rad):
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

        def issf_cbf_between(s1, s2, delta_rad, bb_size, lr):
            def alpha_fn(z):
                full = jnp.array([z[0], z[1], z[2], z[3], s1[4], s1[5]])
                return obstacle_scaling_fn(
                    full,
                    s2,
                    self.params["ego_bb_size"],
                    self.params["ego_lr"],
                    bb_size,
                    lr,
                )

            return issf_cbf_constraint(alpha_fn, s1, delta_rad)

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

        if self.USE_UNBOUNDED_ISSF_ROAD_BOUNDS:
            def issf_cbf_bound(s1, delta_rad, A, b):
                def alpha_fn(z):
                    full = jnp.array([z[0], z[1], z[2], z[3], s1[4], s1[5]])
                    return scaling_calc_unbounded_bound(
                        full,
                        self.params["ego_bb_size"],
                        self.params["ego_lr"],
                        A,
                        b,
                    )

                return issf_cbf_constraint(alpha_fn, s1, delta_rad)

            y_low = self.params["default_state_range"][2]
            y_high = self.params["default_state_range"][3]
            lower_A = jnp.array([[0.0, 1.0]])
            lower_b = jnp.array([y_low])
            upper_A = jnp.array([[0.0, -1.0]])
            upper_b = jnp.array([-y_high])
            a_low_cost, a_low_real = jax.vmap(
                issf_cbf_bound, in_axes=(0, 0, None, None)
            )(graph.env_states.agent, delta, lower_A, lower_b)
            a_high_cost, a_high_real = jax.vmap(
                issf_cbf_bound, in_axes=(0, 0, None, None)
            )(graph.env_states.agent, delta, upper_A, upper_b)
        else:
            bounds = self.generate_bound(
                graph.env_states.agent, self.params["bound_bb_size"]
            )
            a_low_cost, a_low_real = jax.vmap(
                issf_cbf_between, in_axes=(0, 0, 0, None, None)
            )(
                graph.env_states.agent,
                bounds[::2],
                delta,
                self.params["bound_bb_size"],
                0.0,
            )
            a_high_cost, a_high_real = jax.vmap(
                issf_cbf_between, in_axes=(0, 0, 0, None, None)
            )(
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
