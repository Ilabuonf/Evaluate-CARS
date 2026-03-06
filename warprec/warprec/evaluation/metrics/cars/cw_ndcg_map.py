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
    Evaluates ranking quality by scaling the relevance gain by contextual similarity.
    Calculates IDCG assuming a perfect situational match (ctx_w=1.0).
    """
    def _metric_name(self) -> str:
        return f"CW-nDCG@{self.k}"

    def _compute_cars(self, match, user_ctx, item_ctx, valid) -> dict:
        if self.all_binary_rel.shape[0] == 0:
            return {f"CW-nDCG@{self.k}": 0.0}
    
        device = match.device
        bin_rel = self.all_binary_rel.to(device)
        ctx_w = self._context_weights(match, user_ctx, item_ctx)
        
        # Calculate context-weighted relevance: rel_cw(k) = rel(k) * ctx_w(k)
        cw_rel = bin_rel * ctx_w 
    
        # Logarithmic discount positions
        positions = torch.arange(1, self.k + 1, dtype=torch.float, device=device)
        discount = 1.0 / torch.log2(positions + 1)
        
        # DCG_cw: sum ( (2^rel_cw - 1) / log2(pos+1) )
        gain = (2.0 ** cw_rel) - 1.0
        dcg_cw = (gain * discount).sum(dim=1)
    
        # IDCG_cw: The ideal ranking assumes that all relevant items retrieved
        # have a perfect contextual similarity score of 1.0.
        # This creates a baseline for a "perfect" Situational Recommender.
        num_rel = bin_rel.sum(dim=1).long()
        idcg_cw = torch.zeros_like(dcg_cw)
        for i, n in enumerate(num_rel):
            if n > 0:
                n_cap = min(n.item(), self.k)
                # Ideal Gain is (2^1 - 1) = 1.0 per relevant item
                idcg_cw[i] = discount[:n_cap].sum()

        # Compute final normalized score, handling potential division by zero
        ndcg_cw = (dcg_cw / idcg_cw.clamp(min=1e-10)).nan_to_num(0.0)
        return {f"CW-nDCG@{self.k}": self._weighted_mean(ndcg_cw, valid)}


@metric_registry.register("CW-MAP")
class CWMAP(_CWBase):
    """
    CW-MAP@K — Context-Weighted Mean Average Precision.
    Extends MAP by using context-weighted relevance in the precision calculation.
    """
    def _metric_name(self) -> str:
        return f"CW-MAP@{self.k}"

    def _compute_cars(self, match, user_ctx, item_ctx, valid) -> dict:
        if self.all_binary_rel.shape[0] == 0:
            return {f"CW-MAP@{self.k}": 0.0}

        device = match.device
        bin_rel = self.all_binary_rel.to(device)
        ctx_w = self._context_weights(match, user_ctx, item_ctx)
        
        # Integrated relevance: rel_cw(k) = rel(k) * ctx_w(k)
        cw_rel = bin_rel * ctx_w

        # Context-Weighted Precision at i: (sum_{j=1}^{i} rel_cw(j)) / i
        cumsum = cw_rel.cumsum(dim=1)
        positions = torch.arange(1, self.k + 1, dtype=torch.float, device=device)
        precision_cw = cumsum / positions

        # AP_cw = (1 / |Relevant_q|) * sum_k (precision_cw@k * rel_cw(k))
        # Normalization is based on the total number of binary relevant items
        num_relevant = bin_rel.sum(dim=1).clamp(min=1e-10)
        ap = (precision_cw * cw_rel).sum(dim=1) / num_relevant
        
        return {f"CW-MAP@{self.k}": self._weighted_mean(ap.nan_to_num(0.0), valid)}