"""Straight-road preview ablation with scaling-only safety costs.

This environment keeps the scene generation, dynamics, observations, reward,
and rendering of the ISSf-CBF preview environment.  Its raw safety margin is
``alpha_thresh - alpha`` with the existing extra unit penalty on positive
values.  Real collision reporting remains ``1 - alpha``.
"""

from typing import Tuple

import jax.numpy as jnp
from typing_extensions import override

from defmarl.utils.graph import GraphsTuple
from defmarl.utils.typing import Action, Cost

from .mve_lowspeed_ISSf_CBF_dynamic_preview import (
    MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF_Dynamic_Preview,
)


class MVELaneChangeAndOverTake_LowSpeed_Normal_Dynamic_Preview(
    MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF_Dynamic_Preview
):
    """Preview environment whose cost is purely geometric scaling margin."""

    PARAMS = (
        MVELaneChangeAndOverTake_LowSpeed_ISSf_CBF_Dynamic_Preview.PARAMS.copy()
    )

    @override
    def get_cost(
        self, graph: GraphsTuple, action: Action
    ) -> Tuple[Cost, Cost]:
        """Return margin-adjusted scaling cost and unchanged real violations.

        ``_scaling_cost`` uses the same parameterized vehicle scaling and
        unbounded road-boundary scaling as the source ISSf environment.  Its
        first channel is a sentinel because inter-ego collision constraints
        are disabled; preserve the source environment's ``-3`` sentinel there.
        """
        del action
        _, cost_real = self._scaling_cost(graph)
        cost = cost_real + (self.params["alpha_thresh"] - 1.0)

        # No alpha is computed for the disabled inter-ego channel.  Keeping its
        # fixed value also makes cost_real identical to the source environment.
        cost = cost.at[:, 0].set(-3.0)
        cost_real = cost_real.at[:, 0].set(-3.0)

        # Retain the training margin used by the source environment: unsafe
        # scaling costs receive an additional unit penalty.  cost_real remains
        # the unclipped geometric collision indicator ``1 - alpha``.
        cost = jnp.where(cost <= 0.0, cost, cost + 1.0)
        return jnp.clip(cost, a_min=-3.0, a_max=3.0), cost_real


MVELaneChangeAndOverTakeLowSpeedNormalDynamicPreview = (
    MVELaneChangeAndOverTake_LowSpeed_Normal_Dynamic_Preview
)
