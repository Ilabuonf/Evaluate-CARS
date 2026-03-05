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
        n_groups = sum(1 for fi in self.feature_groups.values() if fi)
        for group_name, feat_indices in self.feature_groups.items():
            if not feat_indices:
                continue
            idx = torch.tensor(feat_indices, device=match.device)
            group_match = match[:, :, idx]              # [N, k, |group|]
            # per-item recall within group, then mean over k
            per_item = group_match.mean(dim=-1)         # [N, k]
            per_user = per_item.mean(dim=1)             # [N]
            group_recalls.append(self._weighted_mean(per_user, valid))
        if len(group_recalls) < 2:
            return {f"CGB@{self.k}": 1.0}
        recalls_tensor = torch.tensor(group_recalls)
        mu = recalls_tensor.mean()
        variance = ((recalls_tensor - mu) ** 2).mean()
        sigma = variance.sqrt().item()
        # sigma_max formula 
        g = len(group_recalls)
        sigma_max = ((g - 1) / (g ** 2)) ** 0.5
        cgb = 1.0 - min(sigma / sigma_max, 1.0)
        return {f"CGB@{self.k}": cgb}