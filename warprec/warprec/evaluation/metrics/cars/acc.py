"""ACC@K — Average Context Consistency (exact match on all features)."""
from torch import Tensor
from warprec.utils.registry import metric_registry
from .base_cars_metric import BaseCARSMetric


@metric_registry.register("ACC")
class ACC(BaseCARSMetric):
    """ACC@K — Exact context match averaged over top-k items and users."""

    def _metric_name(self) -> str:
        return f"ACC@{self.k}"

    def _compute_cars(self, match, user_ctx, item_ctx, valid) -> dict:
        # match: [N, k, F]
        exact = match.all(dim=-1).float()   # [N, k]
        per_user = exact.mean(dim=1)        # [N]
        return {f"ACC@{self.k}": self._weighted_mean(per_user, valid)}