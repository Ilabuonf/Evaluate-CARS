"""CGB@K — Context Group Balance across feature groups."""
import torch
from torch import Tensor
from warprec.utils.registry import metric_registry
from .base_cars_metric import BaseCARSMetric


@metric_registry.register("CGB")
class CGB(BaseCARSMetric):
    """CGB@K — Context Group Balance."""

    def _metric_name(self) -> str:
        return f"CGB@{self.k}"

    def _compute_cars(self, match, user_ctx, item_ctx, valid) -> dict:
        if not self.feature_groups:
            return {f"CGB@{self.k}": 0.0}

        group_recalls = []
        for group_name, feat_indices in self.feature_groups.items():
            if not feat_indices:
                continue
            idx = torch.tensor(feat_indices, device=match.device)
            group_match = match[:, :, idx]               # [N, k, |group|]
            any_match = group_match.any(dim=1).float()   # [N, |group|]
            recall = any_match.mean(dim=-1)              # [N]
            group_recalls.append(self._weighted_mean(recall, valid))

        if len(group_recalls) < 2:
            return {f"CGB@{self.k}": 1.0}

        recalls_tensor = torch.tensor(group_recalls)
        std = recalls_tensor.std().item()
        cgb = 1.0 - min(std / 0.5, 1.0)
        return {f"CGB@{self.k}": cgb}