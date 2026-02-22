"""CR@K — Context Recall (fraction of user context features matched by any top-k item)."""
from torch import Tensor
from warprec.utils.registry import metric_registry
from .base_cars_metric import BaseCARSMetric


@metric_registry.register("CR")
class CR(BaseCARSMetric):
    """CR@K — Context Recall."""

    def _metric_name(self) -> str:
        return f"CR@{self.k}"

    def _compute_cars(self, match, user_ctx, item_ctx, valid) -> dict:
        # match: [N, k, F]
        any_match = match.any(dim=1).float()   # [N, F]
        per_user = any_match.mean(dim=-1)      # [N]
        return {f"CR@{self.k}": self._weighted_mean(per_user, valid)}