import torch
from torch import Tensor
from warprec.utils.registry import metric_registry
from .base_cars_metric import BaseCARSMetric

class _CWBase(BaseCARSMetric):
    """Shared context-weighted ranking computation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("all_binary_rel", default=[], dist_reduce_fx=None)

    def update(self, preds: Tensor, **kwargs) -> None:
        super().update(preds, **kwargs)
        binary_rel = kwargs.get(f"top_{self.k}_binary_relevance")
        if binary_rel is not None:
            self.all_binary_rel.append(binary_rel)

    def _context_weights(self, match: Tensor) -> Tensor:
        """Context similarity per item in top-k. Returns [N, k]."""
        return match.mean(dim=-1)  # fraction of features matching

    def _cw_relevance(self, binary_rel: Tensor, ctx_weights: Tensor) -> Tensor:
        """Context-weighted relevance: rel * ctx_weight. Returns [N, k]."""
        return binary_rel.to(ctx_weights.device) * ctx_weights


@metric_registry.register("CW-nDCG")
class CWnDCG(_CWBase):
    """CW-nDCG@K — Context-Weighted Normalized Discounted Cumulative Gain."""

    def _metric_name(self) -> str:
        return f"CW-nDCG@{self.k}"

    def _compute_cars(self, match, user_ctx, item_ctx, valid) -> dict:
        if not self.all_binary_rel:
            return {f"CW-nDCG@{self.k}": 0.0}
    
        device    = match.device
        bin_rel   = torch.cat(self.all_binary_rel, dim=0).to(device)  # [N, k]
        ctx_w     = self._context_weights(match)                       # [N, k]
        cw_rel    = self._cw_relevance(bin_rel, ctx_w)                 # [N, k]  in [0, 1]
    
        positions = torch.arange(1, self.k + 1, dtype=torch.float, device=device)
        discount  = 1.0 / torch.log2(positions + 1)                   # [k]
    
        # exponential gain (2^rel_cw − 1) 
        gain      = (2.0 ** cw_rel) - 1.0                             # [N, k]
        dcg       = (gain * discount).sum(dim=1)                      # [N]
    
        # IDCG: sort cw_rel in descending order
        ideal_cw_rel, _ = cw_rel.sort(dim=1, descending=True)   # [N, k]
        ideal_gain = (2.0 ** ideal_cw_rel) - 1.0                 # [N, k]
        idcg = (ideal_gain * discount).sum(dim=1).clamp(min=1e-10)
    
        ndcg = (dcg / idcg).nan_to_num(0.0)
        return {f"CW-nDCG@{self.k}": self._weighted_mean(ndcg, valid)}



@metric_registry.register("CW-MAP")
class CWMAP(_CWBase):
    """CW-MAP@K — Context-Weighted Mean Average Precision."""

    def _metric_name(self) -> str:
        return f"CW-MAP@{self.k}"

    def _compute_cars(self, match, user_ctx, item_ctx, valid) -> dict:
        if not self.all_binary_rel:
            return {f"CW-MAP@{self.k}": 0.0}

        device = match.device
        binary_rel = torch.cat(self.all_binary_rel, dim=0).to(device)  # [N, k]
        ctx_w = self._context_weights(match)                         # [N, k]
        cw_rel = self._cw_relevance(binary_rel, ctx_w)               # [N, k]

        # Precision@i for each position
        cumsum = cw_rel.cumsum(dim=1)                                # [N, k]
        positions = torch.arange(1, self.k + 1, dtype=torch.float,
                                  device=device)
        precision_at_i = cumsum / positions                          # [N, k]

        # AP = sum(precision_at_i * rel_i) / num_relevant
        ap = (precision_at_i * cw_rel).sum(dim=1)
        num_relevant = cw_rel.sum(dim=1).clamp(min=1e-10)
        ap = (ap / num_relevant).nan_to_num(0.0)

        return {f"CW-MAP@{self.k}": self._weighted_mean(ap, valid)}