"""
Context Satisfaction Metrics (CS@K, WCS@K)
===========================================

Modified Jaccard-based metrics for partial context matching.

Metrics:
    - CS@K: Context Satisfaction (unweighted)
    - WCS@K: Weighted Context Satisfaction (IDF-weighted)

CS Formula:
    CS = |C_q ∩ C_i| / (|C_q ∪ C_i| + α·|C_q \ C_i|/|C_q|)
    
    where:
        C_q = query context set
        C_i = item context set
        α = penalty parameter for missing features

WCS Formula:
    WCS = Σ(matched × IDF) / (Σ(union × IDF) + α·Σ(missing × IDF)/Σ(query × IDF))

Interpretation:
    - Range: [0, 1]
    - 1.0: Perfect context match
    - 0.5: Moderate overlap
    - 0.0: No overlap
    - α controls penalty for missing query features

Use case:
    Flexible context matching with partial overlap
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Set
from pathlib import Path


def compute_idf_weights(df: pd.DataFrame, context_features: List[str]) -> Dict[str, float]:
    """
    Compute IDF weights for context features.
    
    Formula: IDF = log((N+1)/(df+1)) + 1
    
    Args:
        df: DataFrame with context features
        context_features: List of feature names
    
    Returns:
        Dict mapping feature names to IDF weights
    """
    n_contexts = df.drop_duplicates().shape[0]
    idf_weights = {}
    
    for feat in context_features:
        if feat in df.columns:
            df_f = df[feat].astype(str).nunique()
            idf_weights[feat] = np.log((n_contexts + 1) / (df_f + 1)) + 1
    
    return idf_weights


def context_satisfaction_score(query_context: pd.Series,
                               item_context: pd.Series,
                               alpha: float = 0.5) -> float:
    """
    Compute Context Satisfaction score for single query-item pair.
    
    Args:
        query_context: Query context values
        item_context: Item context values
        alpha: Penalty parameter for missing features
    
    Returns:
        CS score in [0, 1]
    """
    q_set = set(query_context[query_context.notna()].astype(str).values)
    i_set = set(item_context[item_context.notna()].astype(str).values)
    
    intersection = len(q_set & i_set)
    union = len(q_set | i_set)
    difference = len(q_set - i_set)
    q_size = len(q_set)
    
    if q_size == 0 or union == 0:
        return 0.0
    
    penalty = alpha * (difference / q_size)
    cs = intersection / (union + penalty)
    
    return cs


def weighted_context_satisfaction_score(query_context: pd.Series,
                                        item_context: pd.Series,
                                        idf_weights: Dict[str, float],
                                        context_features: List[str],
                                        alpha: float = 0.5) -> float:
    """
    Compute Weighted Context Satisfaction using IDF weights.
    
    Args:
        query_context: Query context values
        item_context: Item context values
        idf_weights: IDF weight for each feature
        context_features: List of all context features
        alpha: Penalty parameter
    
    Returns:
        WCS score in [0, 1]
    """
    query_features: Set[str] = set()
    item_features: Set[str] = set()
    matched_features: Set[str] = set()
    
    for feat in context_features:
        q_val = str(query_context.get(feat, '')).strip()
        i_val = str(item_context.get(feat, '')).strip()
        
        if q_val and q_val != 'nan':
            query_features.add(feat)
        
        if i_val and i_val != 'nan':
            item_features.add(feat)
        
        if q_val and i_val and q_val == i_val:
            matched_features.add(feat)
    
    union_features = query_features | item_features
    missing_features = query_features - item_features
    
    # Weighted sums
    intersection_weight = sum(idf_weights.get(f, 1.0) for f in matched_features)
    union_weight = sum(idf_weights.get(f, 1.0) for f in union_features)
    missing_weight = sum(idf_weights.get(f, 1.0) for f in missing_features)
    requested_weight = sum(idf_weights.get(f, 1.0) for f in query_features)
    
    if union_weight == 0 or requested_weight == 0:
        return 0.0
    
    penalty = alpha * (missing_weight / requested_weight)
    wcs = intersection_weight / (union_weight + penalty)
    
    return wcs


def compute_cs_wcs(predictions_df: pd.DataFrame,
                   context_info: pd.DataFrame,
                   context_features: List[str],
                   alpha: float = 0.5,
                   k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """
    Compute CS@K and WCS@K metrics.
    
    Args:
        predictions_df: Predictions with q_context_id
        context_info: Item contexts
        context_features: List of context feature names
        alpha: Penalty parameter
        k_values: List of K values
    
    Returns:
        Dict with CS@K and WCS@K scores
    """
    results = {}
    
    # Prepare data
    pred_df = predictions_df.copy()
    ctx_df = context_info.copy()
    
    # Extract query context from q_context_id
    context_splits = pred_df['q_context_id'].str.split('_', expand=True)
    
    for i, feat in enumerate(context_features):
        if i < context_splits.shape[1]:
            pred_df[f'{feat}_query'] = context_splits[i].astype(str).str.strip()
    
    # Prepare item contexts
    ctx_df['item_id:token'] = ctx_df['item_id:token'].astype(str).str.strip()
    pred_df['item_id:token'] = pred_df['item_id:token'].astype(str).str.strip()
    
    for feat in context_features:
        if feat in ctx_df.columns:
            ctx_df[feat] = ctx_df[feat].astype(str).str.strip()
    
    rename_dict = {feat: f'{feat}_item' for feat in context_features 
                  if feat in ctx_df.columns}
    ctx_df = ctx_df.rename(columns=rename_dict)
    
    # Merge
    merged = pred_df.merge(ctx_df, on='item_id:token', how='left')
    
    # Add rank if missing
    if 'rank' not in merged.columns:
        merged = merged.sort_values(
            ['user_id:token', 'q_context_id', 'prediction'],
            ascending=[True, True, False]
        )
        merged['rank'] = (merged.groupby(['user_id:token', 'q_context_id'])
                                .cumcount() + 1)
    
    # Compute IDF weights
    idf_weights = compute_idf_weights(pred_df, context_features)
    
    # Compute CS and WCS for each prediction
    cs_scores = []
    wcs_scores = []
    
    query_cols = [f'{f}_query' for f in context_features]
    item_cols = [f'{f}_item' for f in context_features]
    
    for idx, row in merged.iterrows():
        q_ctx = row[query_cols]
        i_ctx = row[item_cols]
        
        cs = context_satisfaction_score(q_ctx, i_ctx, alpha)
        cs_scores.append(cs)
        
        q_dict = pd.Series({f: row.get(f'{f}_query') for f in context_features})
        i_dict = pd.Series({f: row.get(f'{f}_item') for f in context_features})
        wcs = weighted_context_satisfaction_score(q_dict, i_dict, idf_weights, 
                                                  context_features, alpha)
        wcs_scores.append(wcs)
    
    merged['cs_score'] = cs_scores
    merged['wcs_score'] = wcs_scores
    
    # Aggregate by K
    for k in k_values:
        top_k = (merged
                .sort_values(['user_id:token', 'q_context_id', 'rank'])
                .groupby(['user_id:token', 'q_context_id'])
                .head(k))
        
        cs_k = top_k.groupby(['user_id:token', 'q_context_id'])['cs_score'].mean().mean()
        wcs_k = top_k.groupby(['user_id:token', 'q_context_id'])['wcs_score'].mean().mean()
        
        results[f'CS@{k}'] = float(cs_k)
        results[f'WCS@{k}'] = float(wcs_k)
    
    # Overall scores
    results['CS@all'] = float(merged['cs_score'].mean())
    results['WCS@all'] = float(merged['wcs_score'].mean())
    
    return results


class CSSatisfactionEvaluator:
    """Standalone evaluator for Context Satisfaction metrics"""
    
    def __init__(self, context_features: List[str], 
                 alpha: float = 0.5,
                 k_values: List[int] = [5, 10, 20]):
        self.context_features = context_features
        self.alpha = alpha
        self.k_values = k_values
        self.results = {}
    
    def evaluate_model(self, model_name: str, 
                      predictions_path: Path,
                      context_info_path: Path) -> Dict[str, float]:
        """Evaluate CS/WCS for a single model"""
        pred_df = pd.read_csv(predictions_path, sep='\t')
        ctx_df = pd.read_csv(context_info_path, sep='\t')
        
        results = compute_cs_wcs(pred_df, ctx_df, self.context_features,
                                self.alpha, self.k_values)
        
        self.results[model_name] = results
        return results
    
    def evaluate_all(self, results_dir: Path, 
                    context_info_path: Path) -> pd.DataFrame:
        """Evaluate all models in directory"""
        exclude_dirs = {'context_metrics', 'evaluation', '__pycache__'}
        
        model_dirs = [d for d in results_dir.iterdir() 
                     if d.is_dir() and d.name not in exclude_dirs]
        
        for model_dir in model_dirs:
            model_name = model_dir.name.capitalize()
            
            pred_files = list((model_dir / 'result').glob('*predictions.tsv'))
            if not pred_files:
                continue
            
            pred_file = pred_files[0]
            
            print(f"Evaluating {model_name}...")
            self.evaluate_model(model_name, pred_file, context_info_path)
        
        results_df = pd.DataFrame(self.results).T
        return results_df.round(4)