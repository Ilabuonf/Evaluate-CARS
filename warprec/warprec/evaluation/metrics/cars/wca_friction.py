"""
WCA@K    — Weighted Context Accuracy (cosine similarity).
Friction@K — Proportion of matching features (1 - normalized Hamming).
"""
import torch
from torch import Tensor
from warprec.utils.registry import metric_registry
from .base_cars_metric import BaseCARSMetric


@metric_registry.register("Friction")
class Friction(BaseCARSMetric):
    """Friction@K — proportion of matching features."""

    def _metric_name(self) -> str:
        return f"Friction@{self.k}"

    def _compute_cars(self, match, user_ctx, item_ctx, valid) -> dict:
        per_item = match.mean(dim=-1)    # [N, k]
        per_user = per_item.mean(dim=1)  # [N]
        return {f"Friction@{self.k}": self._weighted_mean(per_user, valid)}


@metric_registry.register("WCA")
class WCA(BaseCARSMetric):
    """WCA@K — Weighted Context Accuracy (IDF-weighted cosine similarity)."""
    def _metric_name(self) -> str:
        return f"WCA@{self.k}"

    def _compute_cars(self, match, user_ctx, item_ctx, valid) -> dict:
        w = self.feature_weights.to(match.device)                      # [F]

        # Numerator: sum_f w_f * c_q_f * c_i_f
        dot = (user_ctx.unsqueeze(1) * item_ctx * w).sum(dim=-1)       # [N, k]

        # Denominator: sqrt(sum_f w_f * c_q_f^2) * sqrt(sum_f w_f * c_i_f^2)
        u_norm = ((user_ctx ** 2) * w).sum(dim=-1).sqrt().clamp(min=1e-10)          # [N]
        i_norm = ((item_ctx ** 2) * w.unsqueeze(0).unsqueeze(0)).sum(dim=-1).sqrt().clamp(min=1e-10)  # [N, k]

        cos = dot / (u_norm.unsqueeze(1) * i_norm)                     # [N, k]
        per_user = cos.mean(dim=1)                                      # [N]
        return {f"WCA@{self.k}": self._weighted_mean(per_user, valid)}