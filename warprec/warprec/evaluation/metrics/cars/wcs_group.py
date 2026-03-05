"""
WCS_Group@K — Weighted Context Satisfaction broken down by feature group.
"""
import torch
from torch import Tensor
from warprec.utils.registry import metric_registry
from .base_cars_metric import BaseCARSMetric


@metric_registry.register("WCS_Group")
class WCSGroup(BaseCARSMetric):
    """WCS_Group@K — Per-group Weighted Context Satisfaction."""

    def __init__(self, alpha: float = 0.5, group_definitions: dict = None, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self._group_definitions = group_definitions or {}

    def _metric_name(self) -> str:
        return f"WCS_Group@{self.k}"

    def _compute_cars(self, match, user_ctx, item_ctx, valid) -> dict:
        if not self._group_definitions:
            return {f"WCS_Group@{self.k}": 0.0}

        device = match.device
        results = {}

        for group_name, feat_indices in self._group_definitions.items():
            if not feat_indices:
                results[f"WCS_{group_name}@{self.k}"] = 0.0
                continue

            idx = torch.tensor(feat_indices, dtype=torch.long, device=device)

            group_match   = match[:, :, idx]           # [N, k, |group|]
            group_weights = self.feature_weights[idx]  # [|group|]
            w = group_weights

            # -1 is the sentinel for "no context data"
            uq = (user_ctx[:, idx] != -1).float()      # [N, |group|]
            iq = (item_ctx[:, :, idx] != -1).float()   # [N, k, |group|]

            inter   = (group_match * w).sum(dim=-1)                          # [N, k]
            union_m = ((uq.unsqueeze(1) + iq).clamp(max=1) * w).sum(dim=-1) # [N, k]
            diff    = (uq.unsqueeze(1) * (1 - iq) * w).sum(dim=-1)          # [N, k]
            denom_q = (uq * w).sum(dim=-1).clamp(min=1e-10)                 # [N]

            penalty  = self.alpha * diff / denom_q.unsqueeze(1)
            score    = inter / (union_m + penalty).clamp(min=1e-10)          # [N, k]
            per_user = score.mean(dim=1)                                      # [N]

            results[f"WCS_{group_name}@{self.k}"] = self._weighted_mean(per_user, valid)

        return results
