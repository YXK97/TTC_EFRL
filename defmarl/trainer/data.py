import jax.numpy as jnp

from typing import NamedTuple, Optional

from ..utils.typing import Array
from ..utils.typing import Action, Reward, Cost, Done
from ..utils.graph import GraphsTuple


class Rollout(NamedTuple):
    graph: GraphsTuple
    actions: Action
    rnn_states: Array
    rewards: Reward
    costs: Cost
    costs_real: Cost
    dones: Done
    log_pis: Optional[Array]
    next_graph: GraphsTuple
    dYddts: Optional[Array]
    zs: Optional[Array] = None
    z_global: Optional[Array] = None

    @property
    def length(self) -> int:
        return self.rewards.shape[0]

    @property
    def time_horizon(self) -> int:
        return self.rewards.shape[1]

    @property
    def num_agents(self) -> int:
        return self.rewards.shape[2]

    @property
    def n_data(self) -> int:
        return self.length * self.time_horizon

class Record(NamedTuple):
    ax: jnp.ndarray
    deltaf: jnp.ndarray
    Psid: jnp.ndarray
    ao_BD: jnp.ndarray
    BD_lane:jnp.ndarray
    a_Ye:jnp.ndarray
    aS_agent_states :jnp.ndarray
    oS_obst_states :jnp.ndarray
    a_Yd:jnp.ndarray
    action_sum :jnp.ndarray
    T_goal_states:jnp.ndarray