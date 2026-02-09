"""
Context-Weighted Ranking Metrics (CW-nDCG, CW-MAP)
===================================================

Traditional ranking metrics weighted by context satisfaction.

Metrics:
    - CW-nDCG: Context-Weighted Normalized Discounted Cumulative Gain
    - CW-MAP: Context-Weighted Mean Average Precision

CW-nDCG Formula:
    CW-nDCG@K = DCG@K / IDCG@K
    where DCG@K = Σ_{i=1}^K (CS_i × rel_i) / log2(i+1)
    
CW-MAP Formula:
    CW-MAP@K = (1/K) Σ_{i=1}^K (CS_i × rel_i × P@i)

where CS_i is context satisfaction score for item at rank i.

Interpretation:
    - Combines relevance with context matching
    - Higher scores mean relevant items also match context well
    - Lower scores if context-mismatched items rank high

Use case:
    Joint optimization of relevance and context matching
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from pathlib import Path


def compute_context_weighted_ndcg(predictions_df: pd.DataFrame,
                                  context_info: pd.DataFrame,
                                  context_features: List[str],
                                  k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """
    Compute Context-Weighted nDCG (CW-nDCG@K).
    
    Traditional nDCG but gains are weighted by context satisfaction.
    
    Args:
        predictions_df: Predictions with label (relevance)
        context_info: Item contexts
        context_features: List of context feature names
        k_values: List of K values
    
    Returns:
        Dict with CW-nDCG@K scores
    """
    from src.metrics.context_satisfaction import context_satisfaction_score
    
    results = {}
    
    # Prepare data
    pred_df = predictions_df.copy()
    ctx_df = context_info.copy()
    
    # Extract query context
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
    
    # Add rank
    if 'rank' not in merged.columns:
        merged = merged.sort_values(
            ['user_id:token', 'q_context_id', 'prediction'],
            ascending=[True, True, False]
        )
        merged['rank'] = (merged.groupby(['user_id:token', 'q_context_id'])
                                .cumcount() + 1)
    
    # Compute context satisfaction scores
    cs_scores = []
    query_cols = [f'{f}_query' for f in context_features]
    item_cols = [f'{f}_item' for f in context_features]
    
    for idx, row in merged.iterrows():
        q_ctx = row[query_cols]
        i_ctx = row[item_cols]
        cs = context_satisfaction_score(q_ctx, i_ctx, alpha=0.5)
        cs_scores.append(cs)
    
    merged['cs_score'] = cs_scores
    
    # Get relevance labels (binary)
    if 'label' in merged.columns:
        merged['relevance'] = merged['label']
    elif 'label:float' in merged.columns:
        merged['relevance'] = merged['label:float']
    else:
        # Assume all items are relevant if no label
        merged['relevance'] = 1.0
    
    # Compute CW-nDCG for each K
    for k in k_values:
        ndcg_scores = []
        
        for (user, ctx), group in merged.groupby(['user_id:token', 'q_context_id']):
            group = group.sort_values('rank').head(k)
            
            if len(group) == 0:
                continue
            
            # DCG: sum of (CS × relevance × discount)
            dcg = 0.0
            for i, row in enumerate(group.itertuples(), 1):
                cs = row.cs_score
                rel = row.relevance
                discount = np.log2(i + 1)
                dcg += (cs * rel) / discount
            
            # IDCG: best possible ordering (sorted by CS × relevance)
            ideal_group = group.copy()
            ideal_group['ideal_score'] = ideal_group['cs_score'] * ideal_group['relevance']
            ideal_group = ideal_group.sort_values('ideal_score', ascending=False)
            
            idcg = 0.0
            for i, row in enumerate(ideal_group.itertuples(), 1):
                cs = row.cs_score
                rel = row.relevance
                discount = np.log2(i + 1)
                idcg += (cs * rel) / discount
            
            # nDCG
            if idcg > 0:
                ndcg = dcg / idcg
            else:
                ndcg = 0.0
            
            ndcg_scores.append(ndcg)
        
        results[f'CW-nDCG@{k}'] = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
    
    return results


def compute_context_weighted_map(predictions_df: pd.DataFrame,
                                 context_info: pd.DataFrame,
                                 context_features: List[str],
                                 k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """
    Compute Context-Weighted MAP (CW-MAP@K).
    
    Traditional MAP but precision is weighted by context satisfaction.
    
    Args:
        predictions_df: Predictions with label (relevance)
        context_info: Item contexts
        context_features: List of context feature names
        k_values: List of K values
    
    Returns:
        Dict with CW-MAP@K scores
    """
    from src.metrics.context_satisfaction import context_satisfaction_score
    
    results = {}
    
    # Prepare data (same as CW-nDCG)
    pred_df = predictions_df.copy()
    ctx_df = context_info.copy()
    
    context_splits = pred_df['q_context_id'].str.split('_', expand=True)
    
    for i, feat in enumerate(context_features):
        if i < context_splits.shape[1]:
            pred_df[f'{feat}_query'] = context_splits[i].astype(str).str.strip()
    
    ctx_df['item_id:token'] = ctx_df['item_id:token'].astype(str).str.strip()
    pred_df['item_id:token'] = pred_df['item_id:token'].astype(str).str.strip()
    
    for feat in context_features:
        if feat in ctx_df.columns:
            ctx_df[feat] = ctx_df[feat].astype(str).str.strip()
    
    rename_dict = {feat: f'{feat}_item' for feat in context_features 
                  if feat in ctx_df.columns}
    ctx_df = ctx_df.rename(columns=rename_dict)
    
    merged = pred_df.merge(ctx_df, on='item_id:token', how='left')
    
    if 'rank' not in merged.columns:
        merged = merged.sort_values(
            ['user_id:token', 'q_context_id', 'prediction'],
            ascending=[True, True, False]
        )
        merged['rank'] = (merged.groupby(['user_id:token', 'q_context_id'])
                                .cumcount() + 1)
    
    # Compute CS scores
    cs_scores = []
    query_cols = [f'{f}_query' for f in context_features]
    item_cols = [f'{f}_item' for f in context_features]
    
    for idx, row in merged.iterrows():
        q_ctx = row[query_cols]
        i_ctx = row[item_cols]
        cs = context_satisfaction_score(q_ctx, i_ctx, alpha=0.5)
        cs_scores.append(cs)
    
    merged['cs_score'] = cs_scores
    
    # Get relevance
    if 'label' in merged.columns:
        merged['relevance'] = merged['label']
    elif 'label:float' in merged.columns:
        merged['relevance'] = merged['label:float']
    else:
        merged['relevance'] = 1.0
    
    # Compute CW-MAP for each K
    for k in k_values:
        ap_scores = []
        
        for (user, ctx), group in merged.groupby(['user_id:token', 'q_context_id']):
            group = group.sort_values('rank').head(k)
            
            if len(group) == 0:
                continue
            
            # Compute weighted precision at each position
            cumulative_relevant = 0
            cumulative_cs_weighted = 0.0
            precisions = []
            
            for i, row in enumerate(group.itertuples(), 1):
                cs = row.cs_score
                rel = row.relevance
                
                # Count as relevant if label = 1
                if rel > 0:
                    cumulative_relevant += 1
                    cumulative_cs_weighted += cs
                    
                    # Weighted precision
                    weighted_precision = cumulative_cs_weighted / i
                    precisions.append(weighted_precision)
            
            # Average precision
            if len(precisions) > 0:
                ap = np.mean(precisions)
            else:
                ap = 0.0
            
            ap_scores.append(ap)
        
        results[f'CW-MAP@{k}'] = float(np.mean(ap_scores)) if ap_scores else 0.0
    
    return results


class WeightedRankingEvaluator:
    """Standalone evaluator for context-weighted ranking metrics"""
    
    def __init__(self, context_features: List[str],
                 k_values: List[int] = [5, 10, 20]):
        self.context_features = context_features
        self.k_values = k_values
        self.results = {}
    
    def evaluate_model(self, model_name: str,
                      predictions_path: Path,
                      context_info_path: Path) -> Dict[str, float]:
        """Evaluate CW-nDCG/CW-MAP for a single model"""
        pred_df = pd.read_csv(predictions_path, sep='\t')
        ctx_df = pd.read_csv(context_info_path, sep='\t')
        
        results = {}
        
        # CW-nDCG
        ndcg = compute_context_weighted_ndcg(pred_df, ctx_df,
                                            self.context_features,
                                            self.k_values)
        results.update(ndcg)
        
        # CW-MAP
        cwmap = compute_context_weighted_map(pred_df, ctx_df,
                                            self.context_features,
                                            self.k_values)
        results.update(cwmap)
        
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