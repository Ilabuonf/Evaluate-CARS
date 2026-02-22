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
        u = user_ctx * w                                               # [N, F]
        i = item_ctx * w.unsqueeze(0).unsqueeze(0)                     # [N, k, F]
        u_norm = u.norm(dim=-1, keepdim=True).clamp(min=1e-10)         # [N, 1]
        i_norm = i.norm(dim=-1).clamp(min=1e-10)                       # [N, k]
        dot = (i * u.unsqueeze(1)).sum(dim=-1)                         # [N, k]
        cos = dot / (i_norm * u_norm)                                  # [N, k]
        per_user = cos.mean(dim=1)                                     # [N]
        return {f"WCA@{self.k}": self._weighted_mean(per_user, valid)}