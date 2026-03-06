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

    def _context_weights(self, match: Tensor, user_ctx: Tensor, item_ctx: Tensor) -> Tensor:
        """CS score with alpha=0.5 as defined in the thesis. Returns [N, k]."""
        alpha = 0.5
        ones = torch.ones(match.shape[-1], device=match.device)
        uq = (user_ctx != -1).float()                                  # [N, F]
        iq = (item_ctx != -1).float()                                  # [N, k, F]
        inter   = match.sum(dim=-1)                                    # [N, k]
        union_m = (uq.unsqueeze(1) + iq).clamp(max=1).sum(dim=-1)     # [N, k]
        diff    = (uq.unsqueeze(1) * (1 - iq)).sum(dim=-1)            # [N, k]
        denom_q = uq.sum(dim=-1).clamp(min=1e-10)                     # [N]
        penalty = alpha * diff / denom_q.unsqueeze(1)
        return inter / (union_m + penalty).clamp(min=1e-10)            # [N, k]

    def _compute_cars(self, match, user_ctx, item_ctx, valid) -> dict:
        if not self.all_binary_rel:
            return {f"CW-nDCG@{self.k}": 0.0}

        device  = match.device
        bin_rel = torch.cat(self.all_binary_rel, dim=0).to(device)    # [N, k]
        ctx_w   = self._context_weights(match, user_ctx, item_ctx)    # [N, k]
        cw_rel  = bin_rel.to(ctx_w.device) * ctx_w                    # [N, k]

        positions = torch.arange(1, self.k + 1, dtype=torch.float, device=device)
        discount  = 1.0 / torch.log2(positions + 1)                   # [k]

        gain = (2.0 ** cw_rel) - 1.0                                  # [N, k]
        dcg  = (gain * discount).sum(dim=1)                           # [N]

        ideal_cw_rel, _ = cw_rel.sort(dim=1, descending=True)
        ideal_gain = (2.0 ** ideal_cw_rel) - 1.0
        idcg = (ideal_gain * discount).sum(dim=1).clamp(min=1e-10)

        ndcg = (dcg / idcg).nan_to_num(0.0)
        return {f"CW-nDCG@{self.k}": self._weighted_mean(ndcg, valid)}



@metric_registry.register("CW-MAP")
class CWMAP(_CWBase):
    """CW-MAP@K — Context-Weighted Mean Average Precision."""
    def _metric_name(self) -> str:
        return f"CW-MAP@{self.k}"

    def _context_weights(self, match: Tensor, user_ctx: Tensor, item_ctx: Tensor) -> Tensor:
        """CS score with alpha=0.5 as defined in the thesis. Returns [N, k]."""
        alpha = 0.5
        uq = (user_ctx != -1).float()                                  # [N, F]
        iq = (item_ctx != -1).float()                                  # [N, k, F]
        inter   = match.sum(dim=-1)                                    # [N, k]
        union_m = (uq.unsqueeze(1) + iq).clamp(max=1).sum(dim=-1)     # [N, k]
        diff    = (uq.unsqueeze(1) * (1 - iq)).sum(dim=-1)            # [N, k]
        denom_q = uq.sum(dim=-1).clamp(min=1e-10)                     # [N]
        penalty = alpha * diff / denom_q.unsqueeze(1)
        return inter / (union_m + penalty).clamp(min=1e-10)            # [N, k]

    def _compute_cars(self, match, user_ctx, item_ctx, valid) -> dict:
        if not self.all_binary_rel:
            return {f"CW-MAP@{self.k}": 0.0}

        device     = match.device
        binary_rel = torch.cat(self.all_binary_rel, dim=0).to(device)  # [N, k]
        ctx_w      = self._context_weights(match, user_ctx, item_ctx)  # [N, k]
        cw_rel     = binary_rel.to(ctx_w.device) * ctx_w               # [N, k]

        # Precision_cw@i = sum_{j=1}^{i} rel_cw(j) / i
        cumsum       = cw_rel.cumsum(dim=1)                            # [N, k]
        positions    = torch.arange(1, self.k + 1, dtype=torch.float, device=device)
        precision_cw = cumsum / positions                              # [N, k]

        # AP_cw = (1 / |Relevant_q|) * sum_k precision_cw@k * rel_cw(k)
        # |Relevant_q| = number of binary relevant items (not context-weighted)
        num_relevant = binary_rel.sum(dim=1).clamp(min=1e-10)          # [N]
        ap = (precision_cw * cw_rel).sum(dim=1) / num_relevant         # [N]
        ap = ap.nan_to_num(0.0)

        return {f"CW-MAP@{self.k}": self._weighted_mean(ap, valid)}