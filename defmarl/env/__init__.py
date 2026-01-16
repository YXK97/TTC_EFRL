from typing import Optional

from .base import MultiAgentEnv
from .mve_lanechangeANDovertake import MVELaneChangeAndOverTake

ENV = {
    'MVELaneChange': MVELaneChangeAndOverTake,
}


DEFAULT_MAX_STEP = 128


def make_env(
        env_id: str,
        num_agents: int,
        area_size: float = None,
        max_step: int = None,
        full_observation: bool = False,
        max_travel: Optional[float] = None,
        num_obs: Optional[int] = None,
        reward_min: float = -20.0,
        reward_max: float = 0.5
) -> MultiAgentEnv:
    assert env_id in ENV.keys(), f'Environment {env_id} not implemented.'
    params = ENV[env_id].PARAMS
    if num_obs is not None:
        params['n_obsts'] = num_obs
    if full_observation:
        area_size = params['default_state_range'][:4] if area_size is None else area_size
        params['comm_radius'] = max(area_size) * 10
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
