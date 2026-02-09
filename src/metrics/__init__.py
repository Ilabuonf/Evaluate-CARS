"""
Context-Aware Metrics Package
==============================

Implementations of all context-aware evaluation metrics.

Metrics:
    - ACC@K: Average Context Consistency
    - CS@K, WCS@K: Context Satisfaction (weighted)
    - WCA, Friction: Alternative similarity metrics
    - CR, CRC, CGB: Advanced context metrics
    - CW-nDCG, CW-MAP: Context-weighted ranking
"""

from .context_consistency import compute_acc, ACCEvaluator
from .context_satisfaction import (
    compute_cs_wcs, 
    compute_idf_weights,
    context_satisfaction_score,
    weighted_context_satisfaction_score,
    CSSatisfactionEvaluator
)
from .similarity_metrics import (
    compute_similarity_metrics,
    compute_wca_score,
    compute_friction_score,
    SimilarityEvaluator
)
from .advanced_metrics import (
    compute_context_recall,
    compute_context_ranking_correlation,
    compute_context_group_balance,
    AdvancedMetricsEvaluator
)
from .weighted_ranking import (
    compute_context_weighted_ndcg,
    compute_context_weighted_map,
    WeightedRankingEvaluator
)

__all__ = [
    # Context Consistency
    'compute_acc',
    'ACCEvaluator',
    
    # Context Satisfaction
    'compute_cs_wcs',
    'compute_idf_weights',
    'context_satisfaction_score',
    'weighted_context_satisfaction_score',
    'CSSatisfactionEvaluator',
    
    # Similarity
    'compute_similarity_metrics',
    'compute_wca_score',
    'compute_friction_score',
    'SimilarityEvaluator',
    
    # Advanced
    'compute_context_recall',
    'compute_context_ranking_correlation',
    'compute_context_group_balance',
    'AdvancedMetricsEvaluator',
    
    # Weighted Ranking
    'compute_context_weighted_ndcg',
    'compute_context_weighted_map',
    'WeightedRankingEvaluator',
]