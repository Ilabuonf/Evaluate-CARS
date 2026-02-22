"""
Similarity-Based Context Metrics (WCA, Friction)
=================================================

Alternative context matching metrics based on similarity measures.

Metrics:
    - WCA (Weighted Context Alignment): IDF-weighted cosine similarity
    - Friction: Inverted Hamming distance

WCA Formula (corrected cosine similarity):
    WCA = Σ_f (w_f * c_q,f) * (w_f * c_i,f)
          ─────────────────────────────────────────────────────
          sqrt(Σ_{f∈q} w_f²) * sqrt(Σ_{f∈i} w_f²)

    where:
        c_q,f = 1 if feature f is present and valid in query context
        c_i,f = 1 if feature f is present and valid in item context
        w_f   = IDF weight of feature f

    Since c_q,f and c_i,f are binary:
        numerator   = Σ_{f: q_val==i_val, both valid} w_f²
        denom_query = sqrt( Σ_{f∈q, valid} w_f² )
        denom_item  = sqrt( Σ_{f∈i, valid} w_f² )

Friction Formula:
    Friction = 1 - (Hamming distance / # features)

Interpretation:
    WCA:
        - Range: [0, 1]
        - 1.0: Perfect cosine alignment (all features match)
        - 0.0: No shared features
        - Penalises rare-feature mismatches more than common ones (via w_f²)

    Friction:
        - Range: [0, 1]
        - 1.0: No mismatches (zero friction)
        - 0.0: All features differ (maximum friction)

Key difference from WCS:
    WCS uses a Jaccard-like formula with an asymmetric penalty for missing
    query features. WCA uses symmetric cosine similarity — a mismatch on a
    rare feature (high IDF, high w_f²) contributes quadratically more to the
    distance than a mismatch on a common feature.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from pathlib import Path


def compute_wca_score(query_context: pd.Series,
                      item_context: pd.Series,
                      idf_weights: Dict[str, float],
                      context_features: List[str]) -> float:
    """
    Weighted Context Alignment: IDF-weighted cosine similarity.

    Implements the formula from the thesis:
        WCA = Σ(w_f·c_q,f)(w_f·c_i,f) / [sqrt(Σ(w_f·c_q,f)²) · sqrt(Σ(w_f·c_i,f)²)]

    For binary context vectors this simplifies to:
        numerator   = Σ_{f: match} w_f²
        denom_query = sqrt( Σ_{f∈q} w_f² )
        denom_item  = sqrt( Σ_{f∈i} w_f² )

    Args:
        query_context: Series mapping feature names to query values.
        item_context:  Series mapping feature names to item values.
        idf_weights:   Dict mapping feature names to IDF weights.
        context_features: Ordered list of feature names to consider.

    Returns:
        WCA score in [0, 1].
    """
    numerator    = 0.0
    sum_sq_query = 0.0
    sum_sq_item  = 0.0

    for feat in context_features:
        w   = idf_weights.get(feat, 1.0)
        w2  = w * w

        q_val = str(query_context.get(feat, '')).strip()
        i_val = str(item_context.get(feat, '')).strip()

        q_valid = q_val and q_val.lower() != 'nan'
        i_valid = i_val and i_val.lower() != 'nan'

        if q_valid:
            sum_sq_query += w2          # c_q,f = 1  →  (w_f · 1)² = w_f²
        if i_valid:
            sum_sq_item  += w2          # c_i,f = 1  →  (w_f · 1)² = w_f²
        if q_valid and i_valid and q_val == i_val:
            numerator    += w2          # both active and equal → w_f² · 1 · 1

    denom_query = np.sqrt(sum_sq_query)
    denom_item  = np.sqrt(sum_sq_item)
    denominator = denom_query * denom_item

    if denominator == 0.0:
        return 0.0

    return float(np.clip(numerator / denominator, 0.0, 1.0))


def compute_friction_score(query_context: pd.Series,
                           item_context: pd.Series,
                           context_features: List[str]) -> float:
    """
    Context Friction: inverted normalised Hamming distance.

    Friction = 1 - (number of mismatched features / total features)

    Args:
        query_context: Series mapping feature names to query values.
        item_context:  Series mapping feature names to item values.
        context_features: Ordered list of feature names to consider.

    Returns:
        Friction score in [0, 1]. Higher means fewer mismatches.
    """
    if not context_features:
        return 1.0

    distance = 0
    for feat in context_features:
        q_val = str(query_context.get(feat, '')) \
                if pd.notna(query_context.get(feat)) else ''
        i_val = str(item_context.get(feat, '')) \
                if pd.notna(item_context.get(feat)) else ''
        if q_val != i_val:
            distance += 1

    return 1.0 - (distance / len(context_features))


def compute_similarity_metrics(predictions_df: pd.DataFrame,
                                context_info: pd.DataFrame,
                                context_features: List[str],
                                k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """
    Compute WCA (cosine similarity) and Friction metrics at multiple cutoffs.

    Fix applied (v2):
        - Query context features are read from actual feature columns when
          available (set by the evaluator's lookup-table step), rather than
          always re-parsing q_context_id via string split.  This mirrors the
          branch logic already present in compute_cs_wcs() and prevents WCA
          from silently using incorrect feature values.

    Args:
        predictions_df: Predictions DataFrame.  Must contain 'user_id:token',
                        'item_id:token', 'q_context_id', 'prediction', and
                        optionally pre-filled feature columns.
        context_info:   Item-context lookup (one row per item, mode context).
        context_features: Ordered list of context feature names.
        k_values:       Cutoff values for top-K evaluation.

    Returns:
        Dict with WCA@K, Friction@K for each K, plus WCA@all, Friction@all.
    """
    results = {}

    pred_df = predictions_df.copy()
    ctx_df  = context_info.copy()

    # ------------------------------------------------------------------
    # Step 1: Extract query context features
    # Use actual feature columns if already populated (lookup-table path);
    # fall back to parsing q_context_id only when columns are absent.
    # ------------------------------------------------------------------
    query_context_available = all(feat in pred_df.columns
                                  for feat in context_features)

    # NOTE: feature columns in the predictions file (daytime, weekday, ...)
    # represent the ITEM context, not the query context.
    # Query context must always be extracted from q_context_id.
    if not query_context_available:
        # Standard path: parse q_context_id string
        context_splits = pred_df['q_context_id'].astype(str).str.split(
            '_', expand=True)
        for i, feat in enumerate(context_features):
            if i < context_splits.shape[1]:
                pred_df[f'{feat}_query'] = (context_splits[i]
                                            .astype(str).str.strip())
            else:
                pred_df[f'{feat}_query'] = ''
    else:
        # Feature columns present but they are ITEM context values.
        # Still extract query from q_context_id.
        context_splits = pred_df['q_context_id'].astype(str).str.split(
            '_', expand=True)
        for i, feat in enumerate(context_features):
            if i < context_splits.shape[1]:
                pred_df[f'{feat}_query'] = (context_splits[i]
                                            .astype(str).str.strip())
            else:
                pred_df[f'{feat}_query'] = ''

    # ------------------------------------------------------------------
    # Step 2: Prepare item context columns
    # ------------------------------------------------------------------
    ctx_df['item_id:token'] = ctx_df['item_id:token'].astype(str).str.strip()
    pred_df['item_id:token'] = pred_df['item_id:token'].astype(str).str.strip()

    for feat in context_features:
        if feat in ctx_df.columns:
            ctx_df[feat] = ctx_df[feat].astype(str).str.strip()

    rename_dict = {feat: f'{feat}_item'
                   for feat in context_features if feat in ctx_df.columns}
    ctx_df = ctx_df.rename(columns=rename_dict)

    merged = pred_df.merge(ctx_df, on='item_id:token', how='left')

    # ------------------------------------------------------------------
    # Step 3: Compute rank if missing
    # ------------------------------------------------------------------
    if 'rank' not in merged.columns:
        merged = merged.sort_values(
            ['user_id:token', 'q_context_id', 'prediction'],
            ascending=[True, True, False]
        )
        merged['rank'] = (merged
                          .groupby(['user_id:token', 'q_context_id'])
                          .cumcount() + 1)

    # ------------------------------------------------------------------
    # Step 4: Compute IDF weights from query-side features
    # ------------------------------------------------------------------
    from src.metrics.context_satisfaction import compute_idf_weights
    idf_weights = compute_idf_weights(pred_df, context_features)

    # ------------------------------------------------------------------
    # Step 5: Score every prediction row
    # ------------------------------------------------------------------
    wca_scores      = []
    friction_scores = []

    for _, row in merged.iterrows():
        q_ctx = pd.Series({f: row.get(f'{f}_query') for f in context_features})
        i_ctx = pd.Series({f: row.get(f'{f}_item')  for f in context_features})

        wca_scores.append(
            compute_wca_score(q_ctx, i_ctx, idf_weights, context_features))
        friction_scores.append(
            compute_friction_score(q_ctx, i_ctx, context_features))

    merged['wca_score'] = wca_scores
    merged['friction']  = friction_scores

    # ------------------------------------------------------------------
    # Step 6: Aggregate by cutoff K
    # ------------------------------------------------------------------
    for k in k_values:
        top_k = (merged
                 .sort_values(['user_id:token', 'q_context_id', 'rank'])
                 .groupby(['user_id:token', 'q_context_id'])
                 .head(k))

        wca_k      = (top_k
                      .groupby(['user_id:token', 'q_context_id'])['wca_score']
                      .mean().mean())
        friction_k = (top_k
                      .groupby(['user_id:token', 'q_context_id'])['friction']
                      .mean().mean())

        results[f'WCA@{k}']      = float(wca_k)
        results[f'Friction@{k}'] = float(friction_k)

    # Overall (all ranks)
    results['WCA@all']      = float(merged['wca_score'].mean())
    results['Friction@all'] = float(merged['friction'].mean())

    return results


class SimilarityEvaluator:
    """Standalone evaluator for WCA and Friction metrics."""

    def __init__(self,
                 context_features: List[str],
                 k_values: List[int] = [5, 10, 20]):
        self.context_features = context_features
        self.k_values         = k_values
        self.results          = {}

    def evaluate_model(self,
                       model_name: str,
                       predictions_path: Path,
                       context_info_path: Path) -> Dict[str, float]:
        """Evaluate WCA/Friction for a single model."""
        pred_df = pd.read_csv(predictions_path, sep='\t')
        ctx_df  = pd.read_csv(context_info_path, sep='\t')

        res = compute_similarity_metrics(pred_df, ctx_df,
                                         self.context_features,
                                         self.k_values)
        self.results[model_name] = res
        return res

    def evaluate_all(self,
                     results_dir: Path,
                     context_info_path: Path) -> pd.DataFrame:
        """Evaluate all models in a results directory."""
        exclude_dirs = {'context_metrics', 'evaluation', '__pycache__'}

        model_dirs = [d for d in results_dir.iterdir()
                      if d.is_dir() and d.name not in exclude_dirs]

        for model_dir in sorted(model_dirs):
            model_name = model_dir.name.capitalize()
            pred_files = list((model_dir / 'result').glob('*predictions.tsv'))
            if not pred_files:
                continue
            print(f"Evaluating {model_name}...")
            self.evaluate_model(model_name, pred_files[0], context_info_path)

        return pd.DataFrame(self.results).T.round(4)