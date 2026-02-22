"""
Context Satisfaction Metrics (CS@K, WCS@K)
===========================================
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Set
from pathlib import Path


def compute_idf_weights(df: pd.DataFrame, context_features: List[str]) -> Dict[str, float]:
    """Calculate IDF weights for context features."""
    # Count unique context rows to determine N
    n_contexts = df.drop_duplicates().shape[0] if not df.empty else 0
    idf_weights = {}
    
    for feat in context_features:
        if feat in df.columns:
            # Count unique values for this specific feature
            df_f = df[feat].astype(str).nunique()
            # Standard IDF formula with smoothing
            idf_weights[feat] = np.log((n_contexts + 1) / (df_f + 1)) + 1
        else:
            idf_weights[feat] = 1.0
    return idf_weights


def context_satisfaction_score(query_context: pd.Series,
                               item_context: pd.Series,
                               alpha: float = 0.5) -> float:
    """Calculate the CS score for a single query-item pair."""
    # Filter valid values and convert to sets
    q_set = set(query_context.dropna().astype(str).values)
    i_set = set(item_context.dropna().astype(str).values)
    
    if not q_set: return 0.0
    
    intersection = len(q_set & i_set)
    union = len(q_set | i_set)
    difference = len(q_set - i_set)
    
    # Calculate penalty based on missing query features in the item context
    penalty = alpha * (difference / len(q_set))
    return intersection / (union + penalty) if (union + penalty) > 0 else 0.0


def weighted_context_satisfaction_score(query_context: pd.Series,
                                        item_context: pd.Series,
                                        idf_weights: Dict[str, float],
                                        context_features: List[str],
                                        alpha: float = 0.5) -> float:
    """Calculate the WCS score using IDF weights."""
    query_features = []
    item_features = []
    matched_features = []
    
    for feat in context_features:
        q_val = str(query_context.get(feat, '')).strip()
        i_val = str(item_context.get(feat, '')).strip()
        
        # Check if values are valid (not empty or NaN)
        is_q_valid = q_val and q_val.lower() != 'nan'
        is_i_valid = i_val and i_val.lower() != 'nan'
        
        if is_q_valid: query_features.append(feat)
        if is_i_valid: item_features.append(feat)
        if is_q_valid and is_i_valid and q_val == i_val:
            matched_features.append(feat)
            
    if not query_features: return 0.0
    
    # Use sets to handle union and difference logic correctly
    q_feat_set = set(query_features)
    i_feat_set = set(item_features)
    union_feat_set = q_feat_set | i_feat_set
    missing_feat_set = q_feat_set - i_feat_set
    
    # Weighted calculations using IDF weights
    w_intersection = sum(idf_weights.get(f, 1.0) for f in matched_features)
    w_union = sum(idf_weights.get(f, 1.0) for f in union_feat_set)
    w_missing = sum(idf_weights.get(f, 1.0) for f in missing_feat_set)
    w_query = sum(idf_weights.get(f, 1.0) for f in q_feat_set)
    
    # Penalty adjusted by the importance (IDF) of the missing features
    penalty = alpha * (w_missing / w_query)
    return w_intersection / (w_union + penalty) if (w_union + penalty) > 0 else 0.0


def compute_dimensional_wcs(predictions_df: pd.DataFrame,
                           context_info: pd.DataFrame,
                           feature_groups: Dict[str, List[str]],
                           alpha: float = 0.5,
                           k: int = 5) -> Dict[str, float]:
    """Compute WCS separately for each feature group."""
    # Initialize the dictionary immediately to prevent "name 'results' is not defined"
    results = {}
    all_features = list(set([f for group in feature_groups.values() for f in group]))
    
    # If features are missing from columns (e.g., baseline models), return 0.0
    if not all(feat in predictions_df.columns for feat in all_features):
        return {f'WCS_{group_name}@{k}': 0.0 for group_name in feature_groups}

    pred_df = predictions_df.copy()
    ctx_df = context_info.copy()
    
    # Force conversion to string for safe merging
    pred_df['item_id:token'] = pred_df['item_id:token'].astype(str).str.strip()
    ctx_df['item_id:token'] = ctx_df['item_id:token'].astype(str).str.strip()
    
    # Map item contexts by adding suffix
    rename_dict = {f: f'{f}_item' for f in all_features if f in ctx_df.columns}
    ctx_df = ctx_df.rename(columns=rename_dict)
    
    # Merge predictions with item context metadata
    merged = pred_df.merge(ctx_df, on='item_id:token', how='left')
    
    if 'rank' not in merged.columns:
        # Calculate rank based on prediction scores if column is missing
        merged['rank'] = merged.groupby(['user_id:token', 'q_context_id'])['prediction'].rank(
            ascending=False, method='first'
        )
    
    top_k = merged[merged['rank'] <= k].copy()

    for group_name, group_features in feature_groups.items():
        scores = []
        for _, row in top_k.iterrows():
            # Perform robust comparison between query context and item context
            matches = sum(1 for f in group_features 
                         if str(row.get(f)).strip() == str(row.get(f'{f}_item')).strip())
            scores.append(matches / len(group_features) if group_features else 0.0)
            
        results[f'WCS_{group_name}@{k}'] = float(np.mean(scores)) if scores else 0.0
        
    return results


def compute_cs_wcs(predictions_df: pd.DataFrame,
                   context_info: pd.DataFrame,
                   context_features: List[str],
                   alpha: float = 0.5,
                   k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """Compute CS@K and WCS@K metrics """
    pred_df = predictions_df.copy()
    ctx_df = context_info.copy()
    
    # String types for correct merging
    pred_df['item_id:token'] = pred_df['item_id:token'].astype(str).str.strip()
    ctx_df['item_id:token'] = ctx_df['item_id:token'].astype(str).str.strip()

    # Parse query context features from the q_context_id string
    splits = pred_df['q_context_id'].astype(str).str.split('_', expand=True)
    for i, feat in enumerate(context_features):
        if i < splits.shape[1]:
            pred_df[f'{feat}_query'] = splits[i].str.strip()

    # Prepare item context columns
    rename_dict = {f: f'{f}_item' for f in context_features if f in ctx_df.columns}
    ctx_df = ctx_df.rename(columns=rename_dict)
    
    merged = pred_df.merge(ctx_df, on='item_id:token', how='left')
    
    if 'rank' not in merged.columns:
        merged['rank'] = merged.groupby(['user_id:token', 'q_context_id']).cumcount() + 1
        
    idf_weights = compute_idf_weights(pred_df, context_features)
    
    def apply_metrics(row):
        # Create helper Series for current query and item contexts
        q_ctx = pd.Series({f: row.get(f'{f}_query') for f in context_features})
        i_ctx = pd.Series({f: row.get(f'{f}_item') for f in context_features})
        return pd.Series({
            'cs': context_satisfaction_score(q_ctx, i_ctx, alpha),
            'wcs': weighted_context_satisfaction_score(q_ctx, i_ctx, idf_weights, context_features, alpha)
        })

    merged[['cs_score', 'wcs_score']] = merged.apply(apply_metrics, axis=1)

    results = {}
    for k in k_values:
        top_k = merged[merged['rank'] <= k]
        if not top_k.empty:
            # Average scores per query first, then mean across all queries
            q_avg = top_k.groupby(['user_id:token', 'q_context_id'])[['cs_score', 'wcs_score']].mean()
            results[f'CS@{k}'] = float(q_avg['cs_score'].mean())
            results[f'WCS@{k}'] = float(q_avg['wcs_score'].mean())
        else:
            results[f'CS@{k}'] = 0.0
            results[f'WCS@{k}'] = 0.0
        
    results['CS@all'] = float(merged['cs_score'].mean()) if not merged.empty else 0.0
    results['WCS@all'] = float(merged['wcs_score'].mean()) if not merged.empty else 0.0
    
    return results


class CSSatisfactionEvaluator:
    """Standalone evaluator for Context Satisfaction metrics"""
    
    def __init__(self, context_features: List[str], alpha: float = 0.5, k_values: List[int] = [5, 10, 20]):
        self.context_features = context_features
        self.alpha = alpha
        self.k_values = k_values
        self.results = {}
    
    def evaluate_model(self, model_name: str, predictions_path: Path, context_info_path: Path) -> Dict[str, float]:
        """Evaluate a specific model's output file."""
        pred_df = pd.read_csv(predictions_path, sep='\t')
        ctx_df = pd.read_csv(context_info_path, sep='\t')
        res = compute_cs_wcs(pred_df, ctx_df, self.context_features, self.alpha, self.k_values)
        self.results[model_name] = res
        return res
    
    def evaluate_all(self, results_dir: Path, context_info_path: Path) -> pd.DataFrame:
        """Evaluate all models within a results directory."""
        exclude = {'context_metrics', 'evaluation', '__pycache__'}
        for model_dir in results_dir.iterdir():
            if model_dir.is_dir() and model_dir.name not in exclude:
                # Search for the tsv prediction file inside the result subfolder
                preds = list((model_dir / 'result').glob('*predictions.tsv'))
                if preds:
                    print(f"Evaluation in progress: {model_dir.name}...")
                    self.evaluate_model(model_dir.name.capitalize(), preds[0], context_info_path)
        
        return pd.DataFrame(self.results).T.round(4)