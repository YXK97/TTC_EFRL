"""WestEnter intersection preview ablation with scaling-only safety costs.

Only the cost definition differs from the source ISSf-CBF preview environment:
raw obstacle and road-boundary margins are ``alpha_thresh - alpha`` with the
existing extra unit penalty on positive values.  They contain no CBF/ISSf
derivatives or action term, while ``cost_real`` stays ``1 - alpha``.
"""

from typing import Tuple

import jax.numpy as jnp
from typing_extensions import override

from defmarl.utils.graph import GraphsTuple
from defmarl.utils.typing import Action, Cost

from .mve_intersection_lowspeed_ISSf_CBF_dynamic_WestEnter_new_scaling_preview import (
    MVEIntersection_LowSpeed_ISSf_CBF_Dynamic_WestEnter_NewScaling_Preview,
)


class MVEIntersection_LowSpeed_Normal_Dynamic_WestEnter_NewScaling_Preview(
    MVEIntersection_LowSpeed_ISSf_CBF_Dynamic_WestEnter_NewScaling_Preview
):
    """WestEnter preview environment with action-independent scaling costs."""

    PARAMS = (
        MVEIntersection_LowSpeed_ISSf_CBF_Dynamic_WestEnter_NewScaling_Preview.PARAMS.copy()
    )

    @override
    def get_cost(
        self, graph: GraphsTuple, action: Action
    ) -> Tuple[Cost, Cost]:
        """Return geometric threshold margin and unchanged real violations."""
        del action
        cost, cost_real = self._scaling_cost_intersection(graph)

        # Inter-ego scaling is disabled in the source environment, so this
        # channel has no alpha.  Match its existing fixed-safe sentinel exactly.
        cost = cost.at[:, 0].set(-3.0)
        cost_real = cost_real.at[:, 0].set(-3.0)

        # Preserve the positive-side unit margin from the source environment,
        # then bound only the training cost.  cost_real remains raw geometry.
        cost = jnp.where(cost <= 0.0, cost, cost + 1.0)
        return jnp.clip(cost, a_min=-3.0, a_max=3.0), cost_real


MVEIntersectionLowSpeedNormalDynamicWestEnterNewScalingPreview = (
    MVEIntersection_LowSpeed_Normal_Dynamic_WestEnter_NewScaling_Preview
)
