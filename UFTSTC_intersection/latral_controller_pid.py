import functools as ft
import math
from typing import Tuple

import jax
import jax.numpy as jnp

from UFTSTC.latral_controller_pid import UFTSTCController_pid
from defmarl.env.mve import MVE, MVEEnvState
from defmarl.utils.graph import GraphsTuple


class UFTSTCIntersectionControllerPid(UFTSTCController_pid):
    """UFTSTC-PID lateral controller with intersection-local path frames.

    For the west-to-north left-turn handmade scene, the controller uses the
    west/east frame before the north transition and switches to a northbound
    frame after the agent reaches the north road transition.
    """

    def __init__(self, *args, scene_mode: str = "uftstc_left", switch_y: float = 17.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.scene_mode = scene_mode
        self.switch_y = switch_y
        turn_half = 17.5
        lane_offset = 3.0
        turn_radius = turn_half + lane_offset
        self.left_turn_mid_x = -turn_half + turn_radius / math.sqrt(2.0)
        self.left_turn_mid_y = -lane_offset + turn_radius * (1.0 - 1.0 / math.sqrt(2.0))

    @ft.partial(jax.jit, static_argnums=(0,))
    def _wrap_deg(self, angle_deg: jnp.ndarray) -> jnp.ndarray:
        return (angle_deg + 180.0) % 360.0 - 180.0

    @ft.partial(jax.jit, static_argnums=(0,))
    def _transform_state(self, states: jnp.ndarray, direction: jnp.ndarray, normal: jnp.ndarray) -> jnp.ndarray:
        pos = states[:, :2]
        vel = states[:, 2:4]
        local_x = pos @ direction
        local_y = pos @ normal
        local_vx = vel @ direction
        local_vy = vel @ normal
        theta_rad = states[:, 4] * jnp.pi / 180.0
        heading = jnp.stack([jnp.cos(theta_rad), jnp.sin(theta_rad)], axis=1)
        local_theta = jnp.arctan2(heading @ normal, heading @ direction) * 180.0 / jnp.pi
        return states.at[:, 0].set(local_x).at[:, 1].set(local_y) \
            .at[:, 2].set(local_vx).at[:, 3].set(local_vy).at[:, 4].set(local_theta)

    @ft.partial(jax.jit, static_argnums=(0,))
    def _to_local_graph(self, graph: GraphsTuple, a4_dsYddt: jnp.ndarray) -> Tuple[GraphsTuple, jnp.ndarray]:
        goal_states = graph.type_states(MVE.GOAL, n_type=self.num_agents)
        # Use reference-path progress, not actual vehicle heading, because avoidance can rotate the ego vehicle.
        # The fixed left-turn scene switches after the current reference goal has passed the arc midpoint.
        use_north = (
            (self.scene_mode == "uftstc_left")
            & (goal_states[0, 0] >= self.left_turn_mid_x)
            & (goal_states[0, 1] >= self.left_turn_mid_y)
        )

        west_dir = jnp.array([1.0, 0.0], dtype=jnp.float32)
        west_normal = jnp.array([0.0, 1.0], dtype=jnp.float32)
        north_dir = jnp.array([0.0, 1.0], dtype=jnp.float32)
        north_normal = jnp.array([-1.0, 0.0], dtype=jnp.float32)
        direction = jnp.where(use_north, north_dir, west_dir)
        normal = jnp.where(use_north, north_normal, west_normal)

        local_agent = self._transform_state(graph.env_states.agent, direction, normal)
        local_goal = self._transform_state(graph.env_states.goal, direction, normal)
        local_obst = self._transform_state(graph.env_states.obstacle, direction, normal)
        local_env_state = MVEEnvState(local_agent, local_goal, local_obst)

        local_states = self._transform_state(graph.states, direction, normal)
        local_nodes = graph.nodes.at[:, :local_states.shape[1]].set(local_states)
        local_graph = graph._replace(nodes=local_nodes, states=local_states, env_states=local_env_state)

        local_goal_states = local_graph.type_states(MVE.GOAL, n_type=self.num_agents)
        local_theta_rad = local_goal_states[:, 4] * jnp.pi / 180.0
        cos_theta = jnp.cos(local_theta_rad)
        safe_cos_theta = jnp.where(
            jnp.abs(cos_theta) < 1e-3,
            jnp.sign(cos_theta + 1e-6) * 1e-3,
            cos_theta,
        )
        dys = jnp.tan(local_theta_rad)
        speed_mps = jnp.linalg.norm(local_goal_states[:, 2:4], axis=1) / 3.6
        theta_dot_radps = local_goal_states[:, 5] * jnp.pi / 180.0
        curvature = theta_dot_radps / jnp.maximum(speed_mps, 1e-3)
        vxs_mps = local_goal_states[:, 2] / 3.6
        ddys = curvature / safe_cos_theta ** 3
        dddys = 3.0 * curvature ** 2 * jnp.sin(local_theta_rad) / safe_cos_theta ** 5
        north_dsYddt = jnp.stack([
            local_goal_states[:, 1],
            vxs_mps * dys,
            vxs_mps ** 2 * ddys,
            vxs_mps ** 3 * dddys,
        ], axis=1)
        local_dsYddt = jnp.where(use_north, north_dsYddt, a4_dsYddt)
        return local_graph, local_dsYddt

    @ft.partial(jax.jit, static_argnums=(0,))
    def calc_deltaf(self, graph: GraphsTuple, a4_dsYddt: jnp.ndarray):
        local_graph, local_dsYddt = self._to_local_graph(graph, a4_dsYddt)
        return super().calc_deltaf(local_graph, local_dsYddt)
