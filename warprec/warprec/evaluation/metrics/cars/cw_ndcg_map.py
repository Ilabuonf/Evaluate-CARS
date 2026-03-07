import torch
from torch import Tensor
from warprec.utils.registry import metric_registry
from .base_cars_metric import BaseCARSMetric

class _CWBase(BaseCARSMetric):
    """
    Shared context-weighted ranking computation base class.
    Handles the accumulation of binary relevance labels and provides
    the core contextual weight computation logic.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize state for TorchMetrics using a tensor buffer.
        # This stores ground truth binary relevance for the top-k items across all batches.
        k_val = kwargs.get('k', 10)
        self.add_state("all_binary_rel", default=torch.zeros(0, k_val, dtype=torch.float), dist_reduce_fx="cat")

    def update(self, preds: Tensor, **kwargs) -> None:
        """
        Extract and concatenate the binary relevance signal from the current batch.
        """
        super().update(preds, **kwargs)
        binary_rel = kwargs.get(f"top_{self.k}_binary_relevance")
        if binary_rel is not None:
            # Cast to float and move to the current device before concatenation
            self.all_binary_rel = torch.cat([self.all_binary_rel, binary_rel.float().to(self.all_binary_rel.device)], dim=0)

    def _context_weights(self, match: Tensor, user_ctx: Tensor, item_ctx: Tensor) -> Tensor:
        """
        Implementation of the Context Satisfaction (CS) formula with alpha penalty.
        Matches the definition in Thesis Section 4.4.2:
        CS = |inter| / (|union| + alpha * |missing_query_features| / |total_query_features|)
        """
        alpha = 0.5
        # Identify active features (ignore padding/missing values denoted by -1)
        uq = (user_ctx != -1).float()  # Query context mask [N, F]
        iq = (item_ctx != -1).float()  # Item context mask [N, k, F]
        
        # Intersection: Active features present in both query and item
        inter = match.sum(dim=-1)      # [N, k]
        
        # Union: Active features present in either query or item
        union_m = (uq.unsqueeze(1) + iq).clamp(max=1).sum(dim=-1)
        
        # Difference: Features requested in query but missing from item
        diff = (uq.unsqueeze(1) * (1 - iq)).sum(dim=-1)
        
        # Denominator: Total number of features in the query context
        denom_q = uq.sum(dim=-1).clamp(min=1e-10)
        
        # Penalty term for situational misalignment
        penalty = alpha * diff / denom_q.unsqueeze(1)
        
        return inter / (union_m + penalty).clamp(min=1e-10)

@metric_registry.register("CW-nDCG")
class CWnDCG(_CWBase):
    """
    CW-nDCG@K — Context-Weighted Normalized Discounted Cumulative Gain.

    Adapts nDCG for context-aware evaluation by replacing standard binary
    relevance with context-weighted relevance:

        rel_cw(k) = rel(k) * C(c_q, c_ik, alpha)

    where C is the Context Satisfaction score (CS, alpha=0.5).

    The ideal ranking IDCG_cw assumes that all relevant items appear at
    the top positions with perfect contextual alignment (C = 1), i.e.:

        IDCG_cw@N = sum_{k=1}^{min(R_q, N)} 1 / log2(k+1)

    This defines the theoretical maximum as a recommender that retrieves
    all relevant items AND aligns each perfectly with the query context.
    As a consequence, CW-nDCG <= nDCG always holds, and the gap between
    the two metrics quantifies how much relevant items are recommended in
    contextually appropriate situations.
    """
    def _metric_name(self) -> str:
        return f"CW-nDCG@{self.k}"

    def _compute_cars(self, match, user_ctx, item_ctx, valid) -> dict:
        if self.all_binary_rel.shape[0] == 0:
            return {f"CW-nDCG@{self.k}": 0.0}

        device  = match.device
        bin_rel = self.all_binary_rel.to(device)                       # [N, k] binary relevance

        # Compute CS score for each recommended item: C(c_q, c_ik, alpha)
        # Returns values in [0, 1] where 1 = perfect contextual alignment
        ctx_w = self._context_weights(match, user_ctx, item_ctx)       # [N, k]

        # Context-weighted relevance: rel_cw(k) = rel(k) * C(c_q, c_ik, alpha)
        # Non-relevant items: rel_cw = 0 regardless of context score
        # Relevant items: rel_cw = ctx_w in [0, 1]
        cw_rel = bin_rel * ctx_w                                       # [N, k]

        positions = torch.arange(1, self.k + 1, dtype=torch.float, device=device)
        discount  = 1.0 / torch.log2(positions + 1)                   # [k]

        # CW-DCG: exponential gain scaled by contextual relevance
        # gain = 2^rel_cw - 1: relevant item with ctx_w=1 contributes 1.0,
        # relevant item with ctx_w=0.5 contributes ~0.41, non-relevant contributes 0.0
        gain   = (2.0 ** cw_rel) - 1.0                                # [N, k]
        dcg_cw = (gain * discount).sum(dim=1)                         # [N]

        # IDCG_cw: theoretical maximum assuming all relevant items are at the
        # top positions with perfect contextual alignment (C = 1, so rel_cw* = 1).
        # Numerically equivalent to standard IDCG, but interpreted here as the
        # context-aware upper bound: a perfect system that ranks all relevant
        # items first AND achieves C = 1 for each of them.
        # Practical note: iterates over users to handle variable numbers of
        # relevant items per query (n_cap = min(R_q, K)).
        num_rel = bin_rel.sum(dim=1).long()
        idcg_cw = torch.zeros_like(dcg_cw)
        for i, n in enumerate(num_rel):
            if n > 0:
                n_cap = min(n.item(), self.k)
                # gain = 2^1 - 1 = 1.0 per relevant item with perfect context
                idcg_cw[i] = discount[:n_cap].sum()

        ndcg_cw = (dcg_cw / idcg_cw.clamp(min=1e-10)).nan_to_num(0.0)
        return {f"CW-nDCG@{self.k}": self._weighted_mean(ndcg_cw, valid)}


@metric_registry.register("CW-MAP")
class CWMAP(_CWBase):
    """
    CW-MAP@K — Context-Weighted Mean Average Precision.

    Extends MAP by replacing binary relevance with context-weighted
    relevance in both the precision calculation and the AP accumulation
    (Thesis Section 3.2):

        Precision_cw@k = (sum_{j=1}^{k} rel_cw(j)) / k
        AP_cw = (1 / |Relevant_q|) * sum_k Precision_cw@k * rel_cw(k)

    Normalisation uses the total number of binary relevant items |Relevant_q|,
    consistent with the standard MAP definition. A substantial decrease in
    CW-MAP relative to MAP indicates that the model retrieves relevant items
    but recommends them in contextually inappropriate situations.
    """
    def _metric_name(self) -> str:
        return f"CW-MAP@{self.k}"

    def _compute_cars(self, match, user_ctx, item_ctx, valid) -> dict:
        if self.all_binary_rel.shape[0] == 0:
            return {f"CW-MAP@{self.k}": 0.0}

        device  = match.device
        bin_rel = self.all_binary_rel.to(device)                       # [N, k]

        # Compute CS score for each recommended item: C(c_q, c_ik, alpha)
        ctx_w = self._context_weights(match, user_ctx, item_ctx)       # [N, k]

        # Context-weighted relevance: rel_cw(k) = rel(k) * C(c_q, c_ik, alpha)
        cw_rel = bin_rel * ctx_w                                       # [N, k]

        # Context-Weighted Precision at position k:
        # Precision_cw@k = (sum_{j=1}^{k} rel_cw(j)) / k
        cumsum       = cw_rel.cumsum(dim=1)                            # [N, k]
        positions    = torch.arange(1, self.k + 1, dtype=torch.float, device=device)
        precision_cw = cumsum / positions                              # [N, k]

        # AP_cw = (1 / |Relevant_q|) * sum_k Precision_cw@k * rel_cw(k)
        # Normalisation by binary |Relevant_q| ensures AP_cw <= AP always holds,
        # with the gap reflecting contextual misalignment of relevant items.
        num_relevant = bin_rel.sum(dim=1).clamp(min=1e-10)            # [N]
        ap = (precision_cw * cw_rel).sum(dim=1) / num_relevant        # [N]

        return {f"CW-MAP@{self.k}": self._weighted_mean(ap.nan_to_num(0.0), valid)}