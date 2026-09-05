from typing import Tuple
import numpy as np
import jax
import jax.numpy as jnp

from typing_extensions import override

from .designed_scene_gen_two_lane_deterministic import (
    gen_deterministic_scene_two_lane_with_id,
)
from .designed_scene_gen_two_lane_split_dynamic import (
    gen_scene_randomly_split_dynamic_with_id,
)
from .mve_lowspeed_CBF_dynamic import (
    MVEDynamicEnvState,
    MVELaneChangeAndOverTake_LowSpeed_CBF_Dynamic,
)
from .mve_lowspeed_ISSf_CBF import (
    LowSpeedSafetyDiagnostics,
    MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF,
)
from defmarl.utils.graph import GraphsTuple
from defmarl.utils.issf_barrier import (
    compress_safe_barrier,
    safe_barrier_derivative,
)
from defmarl.utils.scaling_lowspeed import (
    scaling_calc_parameterized,
    scaling_calc_unbounded_bound,
)
from defmarl.utils.typing import Action, Array, Cost, Reward
from defmarl.utils.utils import find_closest_goal_indices, gen_i_j_pairs


class MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF_Dynamic(MVELaneChangeAndOverTake_LowSpeed_CBF_Dynamic):
    """Dynamic-obstacle low-speed environment with ego-only ISSf-CBF costs."""

    # Model the off-road regions directly as two unbounded half-planes.  This
    # avoids introducing finite extreme points through virtual boundary cars.
    USE_UNBOUNDED_ISSF_ROAD_BOUNDS = True
    USE_PARAMETERIZED_ISSF_OBSTACLE_SCALING = True

    PARAMS = MVELaneChangeAndOverTake_LowSpeed_CBF_Dynamic.PARAMS.copy()
    PARAMS.update({
        "obst_bb_size": jnp.array([4, 2]),
        "v_min": 5.0 / 3.6,
        "v_max": 30.0 / 3.6,
        "gamma": 10.0,
        "issf_epsilon_0": 2.0,
        "issf_epsilon_rate": 2.0,
        "issf_epsilon_min": 10.0,
        # Compress only alpha > alpha_thresh before applying ISSf-CBF.  The
        # unsafe side and cost_real remain exactly geometric.
        "issf_safe_barrier_kappa": 1.0,
        # Applied until ego's center passes the static obstacle. Keep this
        # ISSf-specific so the ordinary dynamic CBF environment is unchanged.
        "pre_static_penalty": 0.00,
        # Total probability of drawing one of the four fixed demonstration
        # scenes during training. Each fixed scene therefore has probability
        # 0.05 / 4 = 1.25%; all other resets retain the split distribution.
        "deterministic_scene_train_probability": 0.05,
    })

    _SCENE_PHASE_NAMES = (
        "START",
        "APPROACH",
        "SIDE",
        "PASSED",
        "DONE",
        "YIELD_RESUME",
        "EGO_FIRST",
    )

    def get_render_scene_label(self, graph: GraphsTuple) -> str:
        scene_id = int(np.asarray(graph.env_states.scene_id))
        reference_type = "LANE_CHANGE" if scene_id < len(self._SCENE_PHASE_NAMES) else "OVERTAKE"
        phase = self._SCENE_PHASE_NAMES[scene_id % len(self._SCENE_PHASE_NAMES)]
        return f"Scene: {reference_type} / {phase}"

    def reset_deterministic(
        self, scene_index: Array
    ) -> Tuple[GraphsTuple, Array]:
        """Reset to one of the four fixed two-lane demonstration scenes."""
        return _reset_deterministic_two_lane(self, scene_index)

    @override
    def reset(self, key: Array) -> Tuple[GraphsTuple, Array]:
        """Mix the four fixed scenes into training with a small probability."""
        return _reset_training_scene_mixture(self, key)

    @override
    def get_cost(self, graph: GraphsTuple, action: Action) -> Tuple[Cost, Cost]:
        return _get_safe_compressed_cost(self, graph, action)

    # This dynamic environment delegates ISSf cost evaluation instead of
    # inheriting from the ISSf class.  Delegate both diagnostic methods as well
    # so their alpha/gradient path remains identical to ``get_cost``.
    def _safety_diagnostic_terms(self, alpha_fn, state, steering):
        return _safe_compressed_diagnostic_terms(
            self, alpha_fn, state, steering
        )

    def get_safety_diagnostics(
        self, graph: GraphsTuple, transformed_action: Action
    ) -> LowSpeedSafetyDiagnostics:
        return MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF.get_safety_diagnostics(
            self, graph, transformed_action
        )

    @override
    def get_reward(self, graph: GraphsTuple, action: Action) -> Reward:
        agent = self._observable(graph.env_states.agent)
        goal = self._observable(graph.env_states.goal)
        e = agent - goal
        W = jnp.diag(jnp.array([3e-5, 3e-5, 0, 0, 3e-5, 0]))
        reward = -jnp.sqrt(jnp.einsum("ai,ij,ja->a", e, W, e.transpose())).mean()
        # reward -= (action[:, 0] ** 2).mean() * 0.0001
        # reward -= (action[:, 1] ** 2).mean() * 0.0001
        static_x = graph.env_states.obstacle[0, 0]
        ego_not_past_static = (agent[:, 0] <= static_x).astype(jnp.float32)
        reward -= self.params["pre_static_penalty"] * ego_not_past_static.mean()
        return reward


def _safe_compressed_diagnostic_terms(env, alpha_fn, state, steering):
    """Return raw geometry plus derivatives of the barrier used by ISSf.

    ``alpha_grad`` remains the raw scaling-factor gradient for geometric
    inspection.  ``h_dot`` and ``g_dot`` use the compressed barrier gradient,
    so they reproduce the actual cost computation.
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
    barrier_grad = safe_barrier_derivative(
        alpha,
        env.params["alpha_thresh"],
        env.params["issf_safe_barrier_kappa"],
    ) * alpha_grad
    heading = state[2:4] / jnp.maximum(
        jnp.linalg.norm(state[2:4]), 1e-6
    )
    angular_speed = state[4] / env.params["ego_L"] * jnp.tan(steering)
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
            -heading[1] * state[4] / env.params["ego_L"],
            heading[0] * state[4] / env.params["ego_L"],
        ]
    )
    return (
        alpha,
        alpha_grad,
        jnp.dot(barrier_grad, pose_dot),
        jnp.dot(barrier_grad, steering_channel),
    )


def _reset_deterministic_two_lane(env, scene_index):
    """Shared deterministic reset used by both straight ISSf variants."""
    scene = gen_deterministic_scene_two_lane_with_id(
        scene_index,
        env.num_agents,
        env.num_goals,
        env.params["default_state_range"][:2],
        env.params["default_state_range"][2:4],
        env.params["lane_width"],
        env.params["lane_centers"],
    )
    return _build_two_lane_reset(env, scene)


def _reset_training_scene_mixture(env, key):
    """Select split or fixed scene data, then construct one common reset."""
    select_key, fixed_index_key, split_key = jax.random.split(key, 3)
    fixed_probability = jnp.clip(
        jnp.asarray(
            env.params.get("deterministic_scene_train_probability", 0.05),
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
        return gen_deterministic_scene_two_lane_with_id(
            fixed_scene_index,
            env.num_agents,
            env.num_goals,
            env.params["default_state_range"][:2],
            env.params["default_state_range"][2:4],
            env.params["lane_width"],
            env.params["lane_centers"],
        )

    def make_split_scene(_):
        return gen_scene_randomly_split_dynamic_with_id(
            split_key,
            env.num_agents,
            env.num_goals,
            env.params["default_state_range"][:2],
            env.params["default_state_range"][2:4],
            env.params["lane_width"],
            env.params["lane_centers"],
        )

    scene = jax.lax.cond(
        use_fixed_scene, make_fixed_scene, make_split_scene, operand=None
    )
    return _build_two_lane_reset(env, scene)


def _build_two_lane_reset(env, scene):
    """Convert generated scene arrays into the shared dynamic graph state."""
    (
        agents,
        obstacles,
        all_goals,
        all_derivatives,
        dynamic_accel,
        dynamic_max_speed,
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
    env_state = MVEDynamicEnvState(
        agents,
        goals,
        obstacles,
        dynamic_accel,
        dynamic_max_speed,
        scene_id,
    )
    return env.get_graph(env_state), derivatives


def _get_safe_compressed_cost(env, graph, action):
    """Straight-road ISSf cost with safe-side barrier compression."""
    threshold = env.params["alpha_thresh"]
    gamma = env.params["gamma"]
    num_agents = graph.env_states.agent.shape[0]
    num_obstacles = graph.env_states.obstacle.shape[0]
    steering = env._filter_delta(graph.env_states.agent[:, 5], action[:, 1])

    def epsilon(barrier):
        return env.params["issf_epsilon_min"] + env.params[
            "issf_epsilon_0"
        ] * jax.nn.softplus(env.params["issf_epsilon_rate"] * barrier)

    def constraint(alpha_fn, state, steering_value):
        alpha, _, barrier_dot, steering_dot = (
            _safe_compressed_diagnostic_terms(
                env, alpha_fn, state, steering_value
            )
        )
        barrier = compress_safe_barrier(
            alpha, threshold, env.params["issf_safe_barrier_kappa"]
        )
        young_penalty = jnp.square(steering_dot) / epsilon(barrier)
        residual = barrier_dot / gamma + barrier - young_penalty / gamma
        cost = jnp.nan_to_num(
            -residual, nan=3.0, posinf=3.0, neginf=-3.0
        )
        return cost, 1.0 - alpha

    def obstacle_constraint(state, obstacle, steering_value):
        def alpha_fn(ego_state):
            return scaling_calc_parameterized(
                ego_state,
                obstacle,
                env.params["ego_bb_size"],
                env.params["ego_lr"],
                env.params["obst_bb_size"],
                env.params["obst_lr"],
            )

        return constraint(alpha_fn, state, steering_value)

    agent_cost = -jnp.ones((num_agents,), dtype=jnp.float32) * 3.0
    agent_cost_real = -jnp.ones((num_agents,), dtype=jnp.float32) * 3.0
    if num_obstacles == 0:
        obstacle_cost = -jnp.ones((num_agents,), dtype=jnp.float32) * 3.0
        obstacle_cost_real = obstacle_cost
    else:
        i_pairs, j_pairs = gen_i_j_pairs(num_agents, num_obstacles)
        pair_cost, pair_real = jax.vmap(obstacle_constraint)(
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

    def boundary_constraint(state, steering_value, A, b):
        def alpha_fn(ego_state):
            return scaling_calc_unbounded_bound(
                ego_state,
                env.params["ego_bb_size"],
                env.params["ego_lr"],
                A,
                b,
            )

        return constraint(alpha_fn, state, steering_value)

    y_low = env.params["default_state_range"][2]
    y_high = env.params["default_state_range"][3]
    lower_A, lower_b = jnp.array([[0.0, 1.0]]), jnp.array([y_low])
    upper_A, upper_b = jnp.array([[0.0, -1.0]]), jnp.array([-y_high])
    lower_cost, lower_real = jax.vmap(
        boundary_constraint, in_axes=(0, 0, None, None)
    )(graph.env_states.agent, steering, lower_A, lower_b)
    upper_cost, upper_real = jax.vmap(
        boundary_constraint, in_axes=(0, 0, None, None)
    )(graph.env_states.agent, steering, upper_A, upper_b)

    cost = jnp.stack(
        [agent_cost, obstacle_cost, lower_cost, upper_cost], axis=1
    )
    cost_real = jnp.stack(
        [agent_cost_real, obstacle_cost_real, lower_real, upper_real], axis=1
    )
    cost = jnp.where(cost <= 0.0, cost, cost + 1.0)
    return jnp.clip(cost, a_min=-3.0, a_max=3.0), cost_real
