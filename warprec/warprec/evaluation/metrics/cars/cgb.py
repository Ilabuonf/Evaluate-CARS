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
            group_match = match[:, :, idx]                             # [N, k, |group|]

            # Number of active query features in this group
            uq_group = (user_ctx[:, idx] != -1).float()               # [N, |group|]
            n_query_g = uq_group.sum(dim=-1).clamp(min=1e-10)         # [N]

            # Per-item recall within group: |c_q^g ∩ c_i^g| / |c_q^g|
            intersection_g = (group_match * uq_group.unsqueeze(1)).sum(dim=-1)  # [N, k]
            per_item_recall = intersection_g / n_query_g.unsqueeze(1)           # [N, k]

            # Mean over top-K, then weighted mean over users
            per_user = per_item_recall.mean(dim=1)                     # [N]
            group_recalls.append(self._weighted_mean(per_user, valid))

        if len(group_recalls) < 2:
            return {f"CGB@{self.k}": 1.0}

        recalls_tensor = torch.tensor(group_recalls)
        mu = recalls_tensor.mean()
        sigma = ((recalls_tensor - mu) ** 2).mean().sqrt().item()

        g = len(group_recalls)
        sigma_max = ((g - 1) / (g ** 2)) ** 0.5
        cgb = 1.0 - min(sigma / sigma_max, 1.0)
        return {f"CGB@{self.k}": cgb}