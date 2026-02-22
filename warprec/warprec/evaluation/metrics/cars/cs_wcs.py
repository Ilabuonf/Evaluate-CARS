"""
CS@K  — Context Satisfaction (Jaccard-like with penalty).
WCS@K — Weighted Context Satisfaction (IDF-weighted CS).
"""
import torch
from torch import Tensor
from warprec.utils.registry import metric_registry
from .base_cars_metric import BaseCARSMetric


class _CSBase(BaseCARSMetric):
    def __init__(self, alpha: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def _cs_score(self, match: Tensor, weights: Tensor) -> Tensor:
        w = weights.to(match.device)                              # [F]
        inter = (match * w).sum(dim=-1)                           # [N, k]
        union = w.sum().expand(match.shape[0], match.shape[1])    # [N, k]
        mismatch = ((1 - match) * w).sum(dim=-1)                  # [N, k]
        penalty = self.alpha * mismatch / w.sum().clamp(min=1e-10)
        score = inter / (union + penalty).clamp(min=1e-10)        # [N, k]
        return score.mean(dim=1)                                   # [N]


@metric_registry.register("CS")
class CS(_CSBase):
    """CS@K — Context Satisfaction (unweighted)."""

    def _metric_name(self) -> str:
        return f"CS@{self.k}"

    def _compute_cars(self, match, user_ctx, item_ctx, valid) -> dict:
        ones = torch.ones(self.num_features, device=match.device)
        per_user = self._cs_score(match, ones)
        return {f"CS@{self.k}": self._weighted_mean(per_user, valid)}


@metric_registry.register("WCS")
class WCS(_CSBase):
    """WCS@K — Weighted Context Satisfaction (IDF-weighted)."""

    def _metric_name(self) -> str:
        return f"WCS@{self.k}"

    def _compute_cars(self, match, user_ctx, item_ctx, valid) -> dict:
        per_user = self._cs_score(match, self.feature_weights)
        return {f"WCS@{self.k}": self._weighted_mean(per_user, valid)}