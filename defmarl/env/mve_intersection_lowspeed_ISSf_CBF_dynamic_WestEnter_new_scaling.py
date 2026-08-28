"""Fixed-west-entry intersection environment with parameterized ray scaling."""

from typing import Tuple

import jax
import jax.numpy as jnp
from typing_extensions import override

from defmarl.utils.graph import GraphsTuple
from defmarl.utils.issf_barrier import (
    compress_safe_barrier,
    safe_barrier_derivative,
)
from defmarl.utils.scaling_lowspeed import (
    heading_rot_matrix,
    rear_to_center,
    scaling_calc_parameterized,
)
from defmarl.utils.typing import Action, Array, Cost, ObstState, Reward, State
from defmarl.utils.utils import find_closest_goal_indices, gen_i_j_pairs

from .designed_scene_gen_intersection_deterministic_WestEnter import (
    DYNAMIC_ACCEL as DETERMINISTIC_DYNAMIC_ACCEL,
    DYNAMIC_TARGET_SPEED as DETERMINISTIC_DYNAMIC_TARGET_SPEED,
    gen_deterministic_scene_WestEnter_with_id,
)
from .designed_scene_gen_intersection_split_dynamic_WestEnter import (
    gen_scene_randomly_split_dynamic_WestEnter_with_id,
)
from .mve_intersection_lowspeed_ISSf_CBF_dynamic_new_scaling import (
    MVEIntersection_LowSpeed_ISSf_CBF_Dynamic_NewScaling,
    ray_convex_entry_scaling,
    ray_rectangle_extreme_scaling,
)
from .mve_intersection_lowspeed_ISSf_CBF_dynamic import (
    IntersectionSafetyDiagnostics,
    MVEIntersectionLowSpeedDynamicState,
    intersection_corner_extreme_points,
    intersection_corner_halfspaces,
)


class MVEIntersection_LowSpeed_ISSf_CBF_Dynamic_WestEnter_NewScaling(
    MVEIntersection_LowSpeed_ISSf_CBF_Dynamic_NewScaling
):
    """West-entry task whose road-boundary cost uses parameterized rays.

    Dynamics, graph construction, and rendering come from the new-scaling
    parent.  This leaf class owns the WestEnter scene source and reward, and
    overrides safety evaluation to use gate-free parameterized rays plus the
    safe-side-compressed ISSf barrier.
    """

    SAFETY_SIGNALS_ENABLED = True

    PARAMS = MVEIntersection_LowSpeed_ISSf_CBF_Dynamic_NewScaling.PARAMS.copy()
    PARAMS.update(
        {
            "gamma": 10.0,
            "issf_epsilon_0": 2.0,
            "issf_epsilon_rate": 2.0,
            "issf_epsilon_min": 10.0,

            # Match the straight-road environment: apply a small constant
            # penalty until ego has longitudinally passed the static vehicle.
            "pre_static_penalty": 0.00,
            # Adjacent Bezier lanes have slightly different curvature.  A
            # small margin prevents equal curve progress in the other lane
            # from being classified as already past because of projection
            # error (measured worst case below 0.15 m for this geometry).
            "static_pass_margin": 0.25,
            "v_min": 1.0 / 3.6,
            "v_max": 30.0 / 3.6,
            # Keep alpha and its hard minimum unchanged.  Only the safe side
            # of h=alpha-alpha_thresh is compressed before ISSf evaluation.
            "issf_safe_barrier_kappa": 1.0,
            # Total probability of drawing one of the four fixed WestEnter
            # scenes during training.  Each fixed scene therefore has
            # probability 0.02 / 4 = 0.5%; all other resets retain the split
            # scene distribution.
            "deterministic_scene_train_probability": 0.02,
        }
    )

    @override
    def _intersection_alpha(self, state: State) -> Array:
        """Parameterized two-ray-group scaling without a sigmoid gate.

        A ray entry is already zero when the ego scaling origin lies inside a
        forbidden corner.  The historical sigmoid(1e6 * gamma_0) multiplier is
        therefore redundant and would introduce an artificial gradient spike.
        """
        all_A, all_b = intersection_corner_halfspaces(
            self.params["main_road_half_width"],
            self.params["auxiliary_road_half_width"],
            self.params["intersection_radius"],
        )
        all_extreme_points = intersection_corner_extreme_points(
            self.params["main_road_half_width"],
            self.params["auxiliary_road_half_width"],
            self.params["intersection_radius"],
        )
        corner_alphas = jax.vmap(
            _parameterized_corner_alpha_without_gate,
            in_axes=(None, 0, 0, 0, None, None),
        )(
            state,
            all_A,
            all_b,
            all_extreme_points,
            self.params["ego_bb_size"],
            self.params["ego_lr"],
        )
        return jnp.min(corner_alphas)

    def reset_deterministic(
        self, scene_index: Array
    ) -> Tuple[GraphsTuple, Array]:
        """Reset to one of the four fixed WestEnter demonstration scenes."""
        scene = gen_deterministic_scene_WestEnter_with_id(
            scene_index,
            self.num_agents,
            self.num_goals,
            self.params["default_state_range"][:2],
            self.params["default_state_range"][2:4],
            self.params["lane_width"],
            self.params["lane_centers"],
        )
        return _build_westenter_reset(self, scene)

    @override
    def reset(self, key: Array) -> Tuple[GraphsTuple, Array]:
        """Mix the four fixed WestEnter scenes into training at low rate."""
        select_key, fixed_index_key, split_key = jax.random.split(key, 3)
        fixed_probability = jnp.clip(
            jnp.asarray(
                self.params.get(
                    "deterministic_scene_train_probability", 0.02
                ),
                dtype=jnp.float32,
            ),
            0.0,
            1.0,
        )
        use_fixed_scene = jax.random.bernoulli(
            select_key, p=fixed_probability
        )
        fixed_scene_index = jax.random.randint(
            fixed_index_key, (), minval=0, maxval=4, dtype=jnp.int32
        )

        def make_fixed_scene(_):
            return gen_deterministic_scene_WestEnter_with_id(
                fixed_scene_index,
                self.num_agents,
                self.num_goals,
                self.params["default_state_range"][:2],
                self.params["default_state_range"][2:4],
                self.params["lane_width"],
                self.params["lane_centers"],
            )

        def make_split_scene(_):
            return gen_scene_randomly_split_dynamic_WestEnter_with_id(
                split_key,
                self.num_agents,
                self.num_goals,
                self.params["default_state_range"][:2],
                self.params["default_state_range"][2:4],
                self.params["lane_width"],
                self.params["lane_centers"],
            )

        # Select only the generated arrays inside lax.cond.  Assigning traced
        # arrays to self from either branch would leak tracers under rollout
        # JIT, so graph construction and cached references happen afterwards.
        scene = jax.lax.cond(
            use_fixed_scene, make_fixed_scene, make_split_scene, operand=None
        )
        return _build_westenter_reset(self, scene)

    @override
    def obst_step_euler(
        self,
        obstacle_states: ObstState,
        dynamic_accel: Array,
        dynamic_target_speed: Array,
    ) -> ObstState:
        """Use unclipped speed control only for the fixed demonstration data."""
        parent_next = super().obst_step_euler(
            obstacle_states, dynamic_accel, dynamic_target_speed
        )
        is_deterministic = jnp.logical_and(
            jnp.isclose(dynamic_accel, DETERMINISTIC_DYNAMIC_ACCEL),
            jnp.isclose(
                dynamic_target_speed, DETERMINISTIC_DYNAMIC_TARGET_SPEED
            ),
        )

        def make_deterministic_next(_):
            static_obstacle = obstacle_states[0]
            dynamic_obstacle = obstacle_states[1]
            heading = dynamic_obstacle[2:4] / jnp.maximum(
                jnp.linalg.norm(dynamic_obstacle[2:4]), 1e-6
            )
            speed = dynamic_obstacle[4]
            speed_error = dynamic_target_speed - speed
            speed_step = jnp.sign(speed_error) * dynamic_accel * self.dt
            speed_next = jnp.where(
                jnp.abs(speed_error) <= jnp.abs(speed_step),
                dynamic_target_speed,
                speed + speed_step,
            )
            speed_mid = 0.5 * (speed + speed_next)
            position_next = (
                dynamic_obstacle[:2] + speed_mid * heading * self.dt
            )
            dynamic_obstacle_next = (
                dynamic_obstacle.at[:2]
                .set(position_next)
                .at[2:4]
                .set(heading)
                .at[4]
                .set(speed_next)
            )
            return jnp.stack(
                [static_obstacle, dynamic_obstacle_next], axis=0
            )

        return jax.lax.cond(
            is_deterministic,
            make_deterministic_next,
            lambda _: parent_next,
            operand=None,
        )

    @override
    def _issf_constraint(
        self, alpha_fn, state: State, steering: Array
    ) -> Tuple[Array, Array]:
        alpha, _, barrier_dot, steering_dot = self._safety_diagnostic_terms(
            alpha_fn, state, steering
        )
        barrier = compress_safe_barrier(
            alpha,
            self.params["alpha_thresh"],
            self.params["issf_safe_barrier_kappa"],
        )
        epsilon = self.params["issf_epsilon_min"] + self.params[
            "issf_epsilon_0"
        ] * jax.nn.softplus(
            self.params["issf_epsilon_rate"] * barrier
        )
        young_penalty = jnp.square(steering_dot) / epsilon
        residual = (
            barrier_dot / self.params["gamma"]
            + barrier
            - young_penalty / self.params["gamma"]
        )
        cost = jnp.nan_to_num(
            -residual, nan=10.0, posinf=10.0, neginf=-3.0
        )
        # Real collision/violation reporting remains based on raw geometry.
        return cost, 1.0 - alpha

    @override
    def _safety_diagnostic_terms(
        self, alpha_fn, state: State, steering: Array
    ) -> Tuple[Array, Array, Array, Array]:
        """Return raw alpha/gradient and the barrier's actual time terms."""

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
        barrier_grad = safe_barrier_derivative(
            alpha,
            self.params["alpha_thresh"],
            self.params["issf_safe_barrier_kappa"],
        ) * alpha_grad
        heading = state[2:4] / jnp.maximum(
            jnp.linalg.norm(state[2:4]), 1e-6
        )
        angular_speed = (
            state[4] / self.params["ego_L"] * jnp.tan(steering)
        )
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
        return (
            alpha,
            alpha_grad,
            jnp.dot(barrier_grad, pose_dot),
            jnp.dot(barrier_grad, steering_channel),
        )

    @override
    def get_safety_diagnostics(
        self, graph: GraphsTuple, transformed_action: Action
    ) -> IntersectionSafetyDiagnostics:
        agents = graph.env_states.agent
        obstacles = graph.env_states.obstacle
        num_agents = agents.shape[0]
        num_obstacles = obstacles.shape[0]
        steering = self._filter_delta(agents[:, 5], transformed_action[:, 1])

        if num_obstacles == 0:
            obstacle_alpha = jnp.empty((num_agents, 0), dtype=jnp.float32)
            obstacle_grad = jnp.empty((num_agents, 0, 4), dtype=jnp.float32)
            obstacle_h_dot = jnp.empty((num_agents, 0), dtype=jnp.float32)
            obstacle_g_dot = jnp.empty((num_agents, 0), dtype=jnp.float32)
        else:
            i_pairs, j_pairs = gen_i_j_pairs(num_agents, num_obstacles)

            def obstacle_terms(state, obstacle, steering_value):
                def alpha_fn(ego_state):
                    return scaling_calc_parameterized(
                        ego_state,
                        obstacle,
                        self.params["ego_bb_size"],
                        self.params["ego_lr"],
                        self.params["obst_bb_size"],
                        self.params["obst_lr"],
                    )

                return self._safety_diagnostic_terms(
                    alpha_fn, state, steering_value
                )

            terms = jax.vmap(obstacle_terms)(
                agents[i_pairs], obstacles[j_pairs], steering[i_pairs]
            )
            obstacle_alpha = terms[0].reshape((num_agents, num_obstacles))
            obstacle_grad = terms[1].reshape((num_agents, num_obstacles, 4))
            obstacle_h_dot = terms[2].reshape((num_agents, num_obstacles))
            obstacle_g_dot = terms[3].reshape((num_agents, num_obstacles))

        boundary_terms = jax.vmap(
            lambda state, steering_value: self._safety_diagnostic_terms(
                self._intersection_alpha, state, steering_value
            )
        )(agents, steering)
        return IntersectionSafetyDiagnostics(
            steering,
            obstacle_alpha,
            obstacle_grad,
            obstacle_h_dot,
            obstacle_g_dot,
            boundary_terms[0],
            boundary_terms[1],
            boundary_terms[2],
            boundary_terms[3],
        )

    @override
    def get_cost(self, graph: GraphsTuple, action: Action) -> Tuple[Cost, Cost]:
        """Use parameterized scaling for obstacles and road corners."""
        num_agents = graph.env_states.agent.shape[0]
        if not self.SAFETY_SIGNALS_ENABLED:
            fixed_cost = -jnp.ones(
                (num_agents, self.n_cost), dtype=jnp.float32
            )
            return fixed_cost, fixed_cost

        num_obstacles = graph.env_states.obstacle.shape[0]
        steering = self._filter_delta(graph.env_states.agent[:, 5], action[:, 1])
        agent_cost = -jnp.ones((num_agents,), dtype=jnp.float32) * 3.0
        agent_cost_real = -jnp.ones((num_agents,), dtype=jnp.float32) * 3.0

        if num_obstacles == 0:
            obstacle_cost = -jnp.ones((num_agents,), dtype=jnp.float32) * 3.0
            obstacle_cost_real = obstacle_cost
        else:
            i_pairs, j_pairs = gen_i_j_pairs(num_agents, num_obstacles)

            def between(state, obstacle, steering_value):
                def alpha_fn(ego_state):
                    return scaling_calc_parameterized(
                        ego_state,
                        obstacle,
                        self.params["ego_bb_size"],
                        self.params["ego_lr"],
                        self.params["obst_bb_size"],
                        self.params["obst_lr"],
                    )

                return self._issf_constraint(
                    alpha_fn, state, steering_value
                )

            pair_cost, pair_real = jax.vmap(between)(
                graph.env_states.agent[i_pairs],
                graph.env_states.obstacle[j_pairs],
                steering[i_pairs],
            )
            obstacle_cost = jnp.max(
                pair_cost.reshape((num_agents, num_obstacles)), axis=1
            )
            obstacle_cost_real = jnp.max(
                pair_real.reshape((num_agents, num_obstacles)), axis=1
            )

        boundary_cost, boundary_real = jax.vmap(
            lambda state, steering_value: self._issf_constraint(
                self._intersection_alpha, state, steering_value
            )
        )(graph.env_states.agent, steering)
        cost = jnp.stack(
            [agent_cost, obstacle_cost, boundary_cost], axis=1
        )
        cost_real = jnp.stack(
            [agent_cost_real, obstacle_cost_real, boundary_real], axis=1
        )
        cost = jnp.where(cost <= 0.0, cost, cost + 1.0)
        return jnp.clip(cost, a_min=-10.0, a_max=10.0), cost_real

    @override
    def _generate_scene(self, key: Array):
        return gen_scene_randomly_split_dynamic_WestEnter_with_id(
            key,
            self.num_agents,
            self.num_goals,
            self.params["default_state_range"][:2],
            self.params["default_state_range"][2:4],
            self.params["lane_width"],
            self.params["lane_centers"],
        )

    @override
    def get_reward(self, graph: GraphsTuple, action: Action) -> Reward:
        agent = self._observable(graph.env_states.agent)
        goal = self._observable(graph.env_states.goal)
        error = agent - goal
        weight = jnp.diag(jnp.array([1e-5, 1e-5, 0, 0, 1e-5, 0]))
        reward = -jnp.sqrt(
            jnp.einsum("ai,ij,ja->a", error, weight, error.transpose())
        ).mean()

        static_obstacle = graph.env_states.obstacle[0]
        static_heading = static_obstacle[2:4] / jnp.maximum(
            jnp.linalg.norm(static_obstacle[2:4]), 1e-6
        )
        # States store rear-axle positions.  Compare geometric centers so the
        # pass event is not biased when ego and the obstacle have different
        # headings in the turn or in the adjacent bypass lane.
        agent_heading = agent[:, 2:4] / jnp.maximum(
            jnp.linalg.norm(agent[:, 2:4], axis=1, keepdims=True), 1e-6
        )
        agent_center = agent[:, :2] + self.params["ego_lr"] * agent_heading
        static_center = (
            static_obstacle[:2]
            + self.params["obst_lr"] * static_heading
        )

        # Map the world-position difference into the static obstacle's local
        # Frenet longitudinal coordinate.  Its sign remains a valid before/after
        # ordering on both the entrance and the first part of a 90-degree turn,
        # unlike a comparison of world x alone.
        relative_longitudinal = jnp.einsum(
            "ai,i->a", agent_center - static_center, static_heading
        )
        ego_not_past_static = (
            relative_longitudinal <= self.params["static_pass_margin"]
        ).astype(jnp.float32)
        reward -= (
            self.params["pre_static_penalty"] * ego_not_past_static.mean()
        )
        return reward


MVEIntersectionLowSpeedISSfCBFDynamicWestEnterNewScaling = (
    MVEIntersection_LowSpeed_ISSf_CBF_Dynamic_WestEnter_NewScaling
)


def _build_westenter_reset(env, scene):
    """Convert either WestEnter scene source into the common graph state."""
    (
        agents,
        obstacles,
        all_goals,
        all_derivatives,
        dynamic_accel,
        dynamic_target_speed,
        scene_id,
    ) = scene
    env.all_goals = all_goals
    env.all_dsYddts = all_derivatives
    goal_indices = find_closest_goal_indices(
        env._observable(agents), env._observable(all_goals)
    )
    agent_indices = jnp.arange(agents.shape[0])
    goals = all_goals[agent_indices, goal_indices, :]
    derivatives = all_derivatives[agent_indices, goal_indices, :]
    env.num_obsts = obstacles.shape[0]
    env_state = MVEIntersectionLowSpeedDynamicState(
        agents,
        goals,
        obstacles,
        dynamic_accel,
        dynamic_target_speed,
        scene_id,
    )
    return env.get_graph(env_state), derivatives


@jax.jit
def _parameterized_corner_alpha_without_gate(
    state: State,
    A: Array,
    b: Array,
    extreme_points: Array,
    bb_size: Array,
    rear_to_center_offset: Array,
) -> Array:
    """Evaluate both ray groups for one forbidden road-corner polygon."""
    center = rear_to_center(state, rear_to_center_offset)
    rotation = heading_rot_matrix(state)
    half_length, half_width = bb_size[0] / 2.0, bb_size[1] / 2.0
    local_vertices = jnp.array(
        [
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, half_width],
            [-half_length, -half_width],
        ]
    )
    vertices = center + local_vertices @ rotation.T
    vertex_scaling = jnp.min(
        jax.vmap(
            ray_convex_entry_scaling, in_axes=(None, 0, None, None)
        )(center, vertices, A, b)
    )

    rectangle_template = jnp.array(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    )
    ego_A = rectangle_template @ rotation.T
    ego_b = (
        jnp.array([half_length, half_length, half_width, half_width])
        + ego_A @ center
    )
    extreme_scaling = jnp.min(
        jax.vmap(
            ray_rectangle_extreme_scaling,
            in_axes=(None, 0, None, None),
        )(center, extreme_points, ego_A, ego_b)
    )
    return jnp.nan_to_num(
        jnp.minimum(vertex_scaling, extreme_scaling),
        nan=0.0,
        posinf=1e6,
        neginf=0.0,
    )
