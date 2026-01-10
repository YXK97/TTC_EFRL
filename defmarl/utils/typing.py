from flax.core import FrozenDict
from jaxtyping import Array, Bool, Float, Int, Shaped
from typing import Dict, Any, Tuple, Union
from numpy import ndarray


# jax types
PRNGKey = Float[Array, '2']

BoolScalar = Bool[Array, ""]
ABool = Bool[Array, "num_agents"]
Shape = Tuple[int, ...]

BFloat = Float[Array, "b"]
BInt = Int[Array, "b"]
FloatScalar = Union[float, Float[Array, ""]]
IntScalar = Union[int, Int[Array, ""]]
TFloat = Float[Array, "T"]

# environment types
Action = Float[Array, 'num_agents action_dim']
Reward = Float[Array, '']
Cost = Float[Array, 'nh']
Done = BoolScalar
Info = Dict[str, Shaped[Array, '']]
EdgeIndex = Float[Array, '2 n_edge']
AgentState = Float[Array, 'num_agents agent_state_dim']
ObstState = Float[Array, 'num_obsts obst_state_dim']
State = Union[Float[Array, 'num_states state_dim'], type]
Node = Float[Array, 'num_nodes node_dim']
SingleNode = Float[Array, 'node_dim']
EdgeAttr = Float[Array, 'num_edges edge_dim']
Pos2d = Union[Float[Array, '2'], Float[ndarray, '2']]
Pos3d = Union[Float[Array, '3'], Float[ndarray, '3']]
Pos = Union[Pos2d, Pos3d]
Radius = Union[Float[Array, ''], float]
PathEff = Float[Array, 'curve_order_plus_1']
PathRefs = Float[Array, 'sample_points_num state_dim']


# neural network types
Params = Union[Dict[str, Any], FrozenDict[str, Any]]

# obstacles
ObsType = Int[Array, '']
ObsWidth = Float[Array, '']
ObsHeight = Float[Array, '']
ObsLength = Float[Array, '']
ObsTheta = Float[Array, '']
ObsQuaternion = Float[Array, '4']
