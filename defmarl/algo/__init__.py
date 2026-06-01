from .base import Algorithm
from .informarl import InforMARL
from .defmarl import DefMARL
from .informarl_lagr import InforMARLLagr
from .defmarl_normedGraph import DefMARL_normedGraph
from .defmarl_CBFs import DefMARL_CBFs
from .ddpg import DDPG


def make_algo(algo: str, **kwargs) -> Algorithm:
    if algo == 'informarl':
        return InforMARL(**kwargs)
    elif algo == 'defmarl':
        return DefMARL(**kwargs)
    elif algo == 'informarl_lagr':
        return InforMARLLagr(**kwargs)
    elif algo == 'defmarl_normedGraph':
        return DefMARL_normedGraph(**kwargs)
    elif algo == 'defmarl_CBFs':
        return DefMARL_CBFs(**kwargs)
    elif algo == 'ddpg':
        return DDPG(**kwargs)
    elif algo == 'ddpg_efrl':
        return DDPG(safety_mode='efrl', **kwargs)
    elif algo == 'ddpg_lagr':
        return DDPG(safety_mode='lagr', **kwargs)
    else:
        raise ValueError(f'Unknown algorithm: {algo}')
