"""
WCS_Group@K — Weighted Context Satisfaction broken down by feature group.
"""

import torch
from torch import Tensor
from warprec.utils.registry import metric_registry
from .base_cars_metric import BaseCARSMetric


@metric_registry.register("WCS_Group")
class WCSGroup(BaseCARSMetric):
    """WCS_Group@K — Per-group Weighted Context Satisfaction.

    Returns one key per group in the results dict, e.g.:
        {"WCS_Temporal@5": 0.42, "WCS_Activity@5": 0.87, "WCS_Environment@5": 0.38}

    Args:
        group_definitions (dict): Maps group name -> list of feature indices.
        alpha (float): Penalty weight, same as CS/WCS. Default 0.5.
    """

    def __init__(self, alpha: float = 0.5, group_definitions: dict = None, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        # Store under a private attribute to avoid any name clash
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

            # Slice to group features only
            group_match   = match[:, :, idx]           # [N, k, |group|]
            group_weights = self.feature_weights[idx]  # [|group|]

            w        = group_weights
            inter    = (group_match * w).sum(dim=-1)                      # [N, k]
            union    = w.sum().expand(group_match.shape[0],
                                      group_match.shape[1])               # [N, k]
            mismatch = ((1 - group_match) * w).sum(dim=-1)                # [N, k]
            penalty  = self.alpha * mismatch / w.sum().clamp(min=1e-10)   # [N, k]
            score    = inter / (union + penalty).clamp(min=1e-10)         # [N, k]
            per_user = score.mean(dim=1)                                   # [N]

            results[f"WCS_{group_name}@{self.k}"] = self._weighted_mean(per_user, valid)

        return results