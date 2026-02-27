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
        # match: [N, k, F]  —  1.0 where item feature == user feature
        #
        # Theoretical CR: per-item recall, then mean over K items
        # cr_i = |c_q ∩ c_i| / |c_q|  -->  fraction of query features matched by item i
        #
        # match already encodes "feature matches user context" per position.
        # mean over F gives the fraction of features matched by each item.
        per_item = match.mean(dim=-1)      # [N, k]  — per-item recall
        per_user = per_item.mean(dim=1)    # [N]     — average over K items
        return {f"CR@{self.k}": self._weighted_mean(per_user, valid)}
