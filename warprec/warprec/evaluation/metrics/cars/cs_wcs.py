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

    def _cs_score(self, match: Tensor, weights: Tensor,
                  user_ctx: Tensor, item_ctx: Tensor) -> Tensor:
        w = weights.to(match.device)                              # [F]

        # -1 is the sentinel for "no context data"
        # For binary features (BGG, Frappe): real values are {0, 1}, so != -1 is always True
        # For categorical features (Yelp): real values are >= 0, so != -1 is always True
        # Only truly missing entries (filled with -1) are masked out
        uq = (user_ctx != -1).float()                            # [N, F]
        iq = (item_ctx != -1).float()                            # [N, k, F]

        inter   = (match * w).sum(dim=-1)                        # [N, k]
        union_m = ((uq.unsqueeze(1) + iq).clamp(max=1) * w).sum(dim=-1)  # [N, k]
        diff    = (uq.unsqueeze(1) * (1 - iq) * w).sum(dim=-1)  # [N, k]
        denom_q = (uq * w).sum(dim=-1).clamp(min=1e-10)         # [N]

        penalty = self.alpha * diff / denom_q.unsqueeze(1)
        score   = inter / (union_m + penalty).clamp(min=1e-10)   # [N, k]
        return score.mean(dim=1)                                  # [N]


@metric_registry.register("CS")
class CS(_CSBase):
    """CS@K — Context Satisfaction (unweighted)."""

    def _metric_name(self) -> str:
        return f"CS@{self.k}"

    def _compute_cars(self, match, user_ctx, item_ctx, valid) -> dict:
        ones = torch.ones(self.num_features, device=match.device)
        per_user = self._cs_score(match, ones, user_ctx, item_ctx)
        return {f"CS@{self.k}": self._weighted_mean(per_user, valid)}


@metric_registry.register("WCS")
class WCS(_CSBase):
    """WCS@K — Weighted Context Satisfaction (IDF-weighted)."""

    def _metric_name(self) -> str:
        return f"WCS@{self.k}"

    def _compute_cars(self, match, user_ctx, item_ctx, valid) -> dict:
        per_user = self._cs_score(match, self.feature_weights, user_ctx, item_ctx)
        return {f"WCS@{self.k}": self._weighted_mean(per_user, valid)}