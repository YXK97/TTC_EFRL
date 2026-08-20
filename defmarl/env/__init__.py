from typing import Optional

from defmarl.env.base import MultiAgentEnv
from defmarl.env.mve_lanechangeANDovertake import MVELaneChangeAndOverTake
from defmarl.env.mve_normedGraph import MVENormedGraph
from defmarl.env.mve_normedGraph_CBF import MVENormedGraph_CBF
from defmarl.env.mve_lowspeed_normal import MVELaneChangeAndOverTake_LowSpeed
from defmarl.env.mve_lowspeed_dynamic import MVELaneChangeAndOverTake_LowSpeed_Dynamic
from defmarl.env.mve_lowspeed_CBF import MVELaneChangeAndOverTake_LowSpeed_CBF
from defmarl.env.mve_lowspeed_CBF_dynamic import MVELaneChangeAndOverTake_LowSpeed_CBF_Dynamic
from defmarl.env.mve_lowspeed_CBF_dynamic2 import MVELaneChangeAndOverTake_LowSpeed_CBF_Dynamic2
from defmarl.env.mve_lowspeed_ISSf_CBF import MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF
from defmarl.env.mve_lowspeed_ISSf_CBF_dynamic import MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF_Dynamic
from defmarl.env.mve_lowspeed_ISSf_CBF_dynamic2 import MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF_Dynamic2
from defmarl.env.mve_lowspeed_ISSf_CBF2 import MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF2
from defmarl.env.mve_lanechange_rsh import MVELaneChangeAndOverTake_RSH
from defmarl.env.mve_lowspeed_bound import MVELaneChangeAndOverTake_LowSpeed_Bound
from defmarl.env.mve_intersection import MVEIntersection
from defmarl.env.mve_intersection_lowspeed_ISSf_CBF_dynamic import (
    MVEIntersection_LowSpeed_ISSf_CBF_Dynamic,
)
from defmarl.env.mve_intersection_lowspeed_ISSf_CBF_dynamic_WestEnter import (
    MVEIntersection_LowSpeed_ISSf_CBF_Dynamic_WestEnter,
)

ENV = {
    'MVELaneChange': MVELaneChangeAndOverTake,
    'MVENormedGraph': MVENormedGraph,
    'MVENormedGraph_CBF': MVENormedGraph_CBF,
    'MVELaneChangeAndOverTake_LowSpeed': MVELaneChangeAndOverTake_LowSpeed,
    'MVELaneChangeAndOverTake_LowSpeed_Dynamic': MVELaneChangeAndOverTake_LowSpeed_Dynamic,
    'MVELaneChangeAndOverTake_LowSpeed_CBF': MVELaneChangeAndOverTake_LowSpeed_CBF,
    'MVELaneChangeAndOverTake_LowSpeed_CBF_Dynamic': MVELaneChangeAndOverTake_LowSpeed_CBF_Dynamic,
    'MVELaneChangeAndOverTake_LowSpeed_CBF_Dynamic2': MVELaneChangeAndOverTake_LowSpeed_CBF_Dynamic2,
    'MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF': MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF,
    'MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF_Dynamic': MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF_Dynamic,
    'MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF_Dynamic2': MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF_Dynamic2,
    'MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF2': MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF2,
    'MVELaneChangeAndOverTake_RSH': MVELaneChangeAndOverTake_RSH,
    'MVELaneChangeAndOverTake_LowSpeed_Bound': MVELaneChangeAndOverTake_LowSpeed_Bound,
    'MVEIntersection': MVEIntersection,
    'MVEIntersection_LowSpeed_ISSf_CBF_Dynamic': MVEIntersection_LowSpeed_ISSf_CBF_Dynamic,
    'MVEIntersection_LowSpeed_ISSf_CBF_Dynamic_WestEnter': MVEIntersection_LowSpeed_ISSf_CBF_Dynamic_WestEnter,
}


DEFAULT_MAX_STEP = 256


def make_env(
        env_id: str,
        num_agents: int,
        area_size: float = None,
        max_step: int = None,
        full_observation: bool = False,
        max_travel: Optional[float] = None,
        num_obs: Optional[int] = None,
        reward_min: float = -20.0,
        reward_max: float = 0.5,
        comm_radius: Optional[float] = None
) -> MultiAgentEnv:
    assert env_id in ENV.keys(), f'Environment {env_id} not implemented.'
    params = ENV[env_id].PARAMS.copy()
    if num_obs is not None:
        params['n_obsts'] = num_obs
    if full_observation:
        area_size = params['default_state_range'][:4] if area_size is None else area_size
        params['comm_radius'] = max(area_size) * 10
    if comm_radius is not None:
        params['comm_radius'] = comm_radius
    if max_step is None:
        return ENV[env_id](
            num_agents=num_agents,
            area_size=area_size,
            max_travel=max_travel,
            reward_min=reward_min,
            reward_max=reward_max,
            params=params
        )
    else:
        return ENV[env_id](
            num_agents=num_agents,
            area_size=area_size,
            max_step=max_step,
            max_travel=max_travel,
            reward_min=reward_min,
            reward_max=reward_max,
            params=params
        )
