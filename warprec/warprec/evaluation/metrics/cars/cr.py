"""CR@K — Context Recall (fraction of user context features matched by top-k items)."""
from torch import Tensor
from warprec.utils.registry import metric_registry
from .base_cars_metric import BaseCARSMetric


@metric_registry.register("CR")
class CR(BaseCARSMetric):
    """CR@K — Context Recall."""

    def _metric_name(self) -> str:
        return f"CR@{self.k}"

    def _compute_cars(self, match, user_ctx, item_ctx, valid) -> dict:
        # -1 is the sentinel for "no context data"
        # Only features with a real value (>= 0) are counted as active
        query_active = (user_ctx != -1).float()                        # [N, F]
        n_query = query_active.sum(dim=-1).clamp(min=1e-10)            # [N]

        intersection = (match * query_active.unsqueeze(1)).sum(dim=-1) # [N, k]
        per_item = intersection / n_query.unsqueeze(1)                 # [N, k]
        per_user = per_item.mean(dim=1)                                # [N]

        return {f"CR@{self.k}": self._weighted_mean(per_user, valid)}