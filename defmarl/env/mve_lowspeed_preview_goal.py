"""Shared two-goal observation support for low-speed graph environments.

Each ego observes two points from its own stored reference trajectory:

1. the reference point closest to the current rear-axle position;
2. a speed-dependent preview point farther along the same trajectory.

The two nodes deliberately use the same GOAL type and the same feature layout.
Their different relative positions, headings, and reference steering values let
the graph transformer learn how much attention to assign to each point without
an explicit role flag.
"""

from typing import List, Optional, Tuple

import jax.numpy as jnp

from .mve import MVE
from defmarl.trainer.data import Rollout
from defmarl.utils.graph import EdgeBlock, GetGraph, GraphsTuple
from defmarl.utils.typing import AgentState, Array, State
from defmarl.utils.utils import find_closest_goal_indices, gen_i_j_pairs


class LowSpeedPreviewGoalMixin:
    """Add a nearest goal and a route-progress preview goal for every ego."""

    @property
    def goals_per_agent(self) -> int:
        return 2

    @property
    def num_graph_goals(self) -> int:
        """Number of active GOAL nodes in one environment graph."""
        return self.goals_per_agent * self.num_agents

    def _select_preview_goals(
        self,
        agents: AgentState,
        all_goals: State,
        all_derivatives: Array,
    ) -> Tuple[State, Array]:
        """Return flattened [tracking goals, preview goals] and tracking data.

        Preview selection advances the closest reference index instead of
        searching around a straight-line projection of the ego pose.  It
        therefore remains on the correct route through turns and lane changes.
        Reference derivatives keep the historical one-per-ego shape because
        they describe the current tracking point, not the observation preview.
        """
        observable_agents = self._observable(agents)
        observable_references = self._observable(all_goals)
        tracking_indices = find_closest_goal_indices(
            observable_agents, observable_references
        )

        base_distance = jnp.asarray(
            self.params["preview_base_distance"], dtype=agents.dtype
        )
        preview_time = jnp.asarray(
            self.params["preview_time"], dtype=agents.dtype
        )
        maximum_distance = jnp.asarray(
            self.params["preview_max_distance"], dtype=agents.dtype
        )
        point_interval = jnp.maximum(
            jnp.asarray(
                self.params["preview_reference_point_interval"],
                dtype=agents.dtype,
            ),
            1e-6,
        )
        preview_distance = jnp.clip(
            base_distance + preview_time * agents[:, 4],
            base_distance,
            maximum_distance,
        )
        # Round to the nearest reference sample while guaranteeing forward
        # progress and keeping the index inside the fixed reference array.
        preview_steps = jnp.floor(
            preview_distance / point_interval + 0.5
        ).astype(jnp.int32)
        preview_indices = jnp.minimum(
            tracking_indices + preview_steps,
            all_goals.shape[1] - 1,
        )

        agent_indices = jnp.arange(agents.shape[0])
        tracking_goals = all_goals[
            agent_indices, tracking_indices, :
        ]
        preview_goals = all_goals[
            agent_indices, preview_indices, :
        ]
        tracking_derivatives = all_derivatives[
            agent_indices, tracking_indices, :
        ]
        goals = jnp.concatenate(
            [tracking_goals, preview_goals], axis=0
        )
        return goals, tracking_derivatives

    def goal_dsYddt_step(
        self, agent_states_new: AgentState
    ) -> Tuple[State, Array]:
        """Update both goal nodes while retaining current-point derivatives."""
        return self._select_preview_goals(
            agent_states_new, self.all_goals, self.all_dsYddts
        )

    def _tracking_goals(self, graph: GraphsTuple) -> State:
        """Extract the first, reward-bearing goal associated with each ego."""
        return graph.env_states.goal[: self.num_agents]

    def _graph_with_tracking_goals(self, graph: GraphsTuple) -> GraphsTuple:
        """Present only current goals to an inherited reward implementation."""
        tracking_env_state = graph.env_states._replace(
            goal=self._tracking_goals(graph)
        )
        return graph._replace(env_states=tracking_env_state)

    def edge_blocks(self, state) -> List[EdgeBlock]:
        """Connect both route goals, plus the existing obstacles, to each ego."""
        num_agents = state.agent.shape[0]
        num_goals = state.goal.shape[0]
        num_obstacles = state.obstacle.shape[0]
        assert num_goals == self.goals_per_agent * num_agents

        agent_observations = self._observable(state.agent)
        goal_observations = self._observable(state.goal)
        agent_ids = jnp.arange(num_agents)
        edges = []

        # Goal layout is [tracking goal for every ego, preview goal for every
        # ego].  Keeping both edges in one block also guarantees a stable edge
        # order under JIT compilation.
        for agent_index in range(num_agents):
            goal_local_indices = jnp.array(
                [agent_index, num_agents + agent_index], dtype=jnp.int32
            )
            relative_states = (
                agent_observations[agent_index][None, :]
                - goal_observations[goal_local_indices]
            )
            edges.append(
                EdgeBlock(
                    relative_states[None, :, :],
                    jnp.ones((1, self.goals_per_agent), dtype=jnp.bool_),
                    jnp.array([agent_index], dtype=jnp.int32),
                    num_agents + goal_local_indices,
                )
            )

        boundary_edges, obstacle_offset = self._boundary_edge_blocks(
            state, num_agents + num_goals
        )
        edges.extend(boundary_edges)

        if num_obstacles > 0:
            obstacle_observations = self._observable(state.obstacle)
            distances = jnp.linalg.norm(
                agent_observations[:, None, :2]
                - obstacle_observations[None, :, :2],
                axis=-1,
            )
            mask = distances < self.params["comm_radius"]
            agent_pairs, obstacle_pairs = gen_i_j_pairs(
                num_agents, num_obstacles
            )
            relative_pairs = (
                agent_observations[agent_pairs]
                - obstacle_observations[obstacle_pairs]
            )
            edges.append(
                EdgeBlock(
                    relative_pairs.reshape(
                        (num_agents, num_obstacles, self.state_dim)
                    ),
                    mask,
                    agent_ids,
                    jnp.arange(num_obstacles) + obstacle_offset,
                )
            )
        return edges

    def get_graph(
        self, env_state, obst_as_agent: bool = False
    ) -> GraphsTuple:
        """Build a graph containing two GOAL nodes for every ego."""
        del obst_as_agent
        num_agents = env_state.agent.shape[0]
        num_goals = env_state.goal.shape[0]
        num_obstacles = env_state.obstacle.shape[0]
        assert num_goals == self.goals_per_agent * num_agents

        total = (
            num_agents
            + num_goals
            + self._num_boundary_nodes(env_state)
            + num_obstacles
        )
        node_features = jnp.zeros((total, self.node_dim))
        node_type = -jnp.ones((total,), dtype=jnp.int32)

        node_features = node_features.at[
            :num_agents, : self.state_dim
        ].set(self._observable(env_state.agent))
        node_features = node_features.at[
            num_agents : num_agents + num_goals, : self.state_dim
        ].set(self._observable(env_state.goal))
        node_type = node_type.at[:num_agents].set(MVE.AGENT)
        node_type = node_type.at[
            num_agents : num_agents + num_goals
        ].set(MVE.GOAL)
        node_features = node_features.at[:num_agents, -1].set(1.0)
        node_features = node_features.at[
            num_agents : num_agents + num_goals, -2
        ].set(1.0)

        cursor = num_agents + num_goals
        node_features, node_type, cursor = self._add_boundary_nodes(
            node_features, node_type, env_state, cursor
        )
        if num_obstacles > 0:
            node_features = node_features.at[
                cursor:, : self.state_dim
            ].set(self._observable(env_state.obstacle))
            node_features = node_features.at[cursor:, -3].set(1.0)
            node_type = node_type.at[cursor:].set(MVE.OBST)

        states = node_features[:, : self.state_dim]
        return GetGraph(
            node_features,
            node_type,
            self.edge_blocks(env_state),
            env_state,
            states,
        ).to_padded()

    def render_video(
        self,
        rollout: Rollout,
        video_path,
        Ta_is_unsafe=None,
        n_goals: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Render with the correct two-goal count and both live goal edges."""
        del n_goals
        return super().render_video(
            rollout,
            video_path,
            Ta_is_unsafe=Ta_is_unsafe,
            n_goals=self.num_graph_goals,
            **kwargs,
        )
