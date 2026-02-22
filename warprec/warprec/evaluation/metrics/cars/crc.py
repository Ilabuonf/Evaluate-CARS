"""CRC@K — Context Ranking Correlation (Spearman approximation via Pearson on ranks)."""
import torch
from torch import Tensor
from warprec.utils.registry import metric_registry
from .base_cars_metric import BaseCARSMetric


@metric_registry.register("CRC")
class CRC(BaseCARSMetric):
    """CRC@K — Context Ranking Correlation."""

    _REQUIRED_COMPONENTS = BaseCARSMetric._REQUIRED_COMPONENTS | set()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Tensor accumulator for top-k scores
        self.add_state(
            "all_top_k_scores",
            default=torch.zeros(0, self.k, dtype=torch.float),
            dist_reduce_fx="cat",
        )

    def update(self, preds: Tensor, **kwargs) -> None:
        super().update(preds, **kwargs)
        top_k_scores = kwargs.get(f"top_{self.k}_values")  # [B, k]
        if top_k_scores is None:
            return
        device = self.item_context_lookup.device
        top_k_scores = top_k_scores.float().to(device)
        self.all_top_k_scores = torch.cat(
            [self.all_top_k_scores, top_k_scores], dim=0
        )

    def _metric_name(self) -> str:
        return f"CRC@{self.k}"

    def _compute_cars(self, match, user_ctx, item_ctx, valid) -> dict:
        if self.all_top_k_scores.shape[0] == 0:
            return {f"CRC@{self.k}": 0.0}

        device = match.device
        top_k_scores = self.all_top_k_scores.to(device)   # [N, k]
        ctx_scores = match.mean(dim=-1)                    # [N, k]

        def rank_tensor(t: Tensor) -> Tensor:
            order = t.argsort(dim=1, descending=True)
            ranks = torch.zeros_like(order, dtype=torch.float, device=device)
            ranks.scatter_(
                1, order,
                torch.arange(1, t.shape[1] + 1, dtype=torch.float, device=device)
                    .unsqueeze(0).expand_as(order),
            )
            return ranks

        def pearson(a: Tensor, b: Tensor) -> Tensor:
            a_mean = a.mean(dim=1, keepdim=True)
            b_mean = b.mean(dim=1, keepdim=True)
            num = ((a - a_mean) * (b - b_mean)).sum(dim=1)
            den = (
                ((a - a_mean) ** 2).sum(dim=1) *
                ((b - b_mean) ** 2).sum(dim=1)
            ).sqrt()
            return (num / den.clamp(min=1e-10)).nan_to_num(0.0)

        pred_ranks = rank_tensor(top_k_scores)
        ctx_ranks = rank_tensor(ctx_scores)
        corr = pearson(pred_ranks, ctx_ranks)   # [N]
        return {f"CRC@{self.k}": self._weighted_mean(corr, valid)}