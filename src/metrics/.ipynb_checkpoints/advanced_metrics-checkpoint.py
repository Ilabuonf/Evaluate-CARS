"""
Advanced Context Metrics (CR, CRC, CGB)
========================================

Sophisticated context-aware metrics for comprehensive evaluation.

Metrics:
    - CR (Context Recall): Feature coverage
    - CRC (Context Ranking Correlation): Ranking coherence
    - CGB (Context Group Balance): Dimensional fairness

CR Formula:
    CR@K = |C_q ∩ C_i| / |C_q|
    
CRC Formula:
    CRC@K = (ρ + 1) / 2
    where ρ = Spearman correlation between rank and context score

CGB Formula:
    CGB@K = 1 - min(σ / 0.5, 1.0)
    where σ = std dev of group-wise recall scores

Use cases:
    - CR: Check if query features are covered
    - CRC: Check if better-matching items rank higher
    - CGB: Check if all feature groups are balanced
"""

import pandas as pd
import numpy as np
import warnings
from typing import List, Dict, Set
from pathlib import Path
from scipy.stats import spearmanr, ConstantInputWarning


def compute_context_recall(predictions_df: pd.DataFrame,
                           context_info: pd.DataFrame,
                           context_features: List[str],
                           k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """
    Compute Context Recall (CR@K).
    
    Measures what proportion of query features are present in recommended items.
    
    Args:
        predictions_df: Predictions with q_context_id
        context_info: Item contexts
        context_features: List of context feature names
        k_values: List of K values
    
    Returns:
        Dict with CR@K scores
    """
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
    
    # Compute recall for each prediction
    recall_scores = []
    
    for idx, row in merged.iterrows():
        q_features: Set[str] = set()
        i_features: Set[str] = set()
        
        for feat in context_features:
            q_val = row.get(f'{feat}_query')
            i_val = row.get(f'{feat}_item')
            
            if pd.notna(q_val):
                q_features.add(f'{feat}={q_val}')
            if pd.notna(i_val):
                i_features.add(f'{feat}={i_val}')
        
        if len(q_features) > 0:
            recall = len(q_features & i_features) / len(q_features)
        else:
            recall = 0.0
        
        recall_scores.append(recall)
    
    merged['recall'] = recall_scores
    
    # Aggregate by K
    for k in k_values:
        top_k = (merged
                .sort_values(['user_id:token', 'q_context_id', 'rank'])
                .groupby(['user_id:token', 'q_context_id'])
                .head(k))
        
        cr_k = top_k.groupby(['user_id:token', 'q_context_id'])['recall'].mean().mean()
        results[f'CR@{k}'] = float(cr_k)
    
    results['CR@all'] = float(merged['recall'].mean())
    return results


def compute_context_ranking_correlation(predictions_df: pd.DataFrame,
                                        context_info: pd.DataFrame,
                                        context_features: List[str],
                                        k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """
    Compute Context Ranking Correlation (CRC@K).
    
    Measures if items with better context match are ranked higher.
    Uses Spearman correlation between rank and context satisfaction.
    
    Args:
        predictions_df: Predictions with q_context_id
        context_info: Item contexts
        context_features: List of context feature names
        k_values: List of K values
    
    Returns:
        Dict with CRC@K scores
    """
    from src.metrics.context_satisfaction import context_satisfaction_score
    
    results = {}
    
    # Prepare data (same as other metrics)
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
    
    # Compute correlation for each K
    for k in k_values:
        correlations = []
        
        for (user, ctx), group in merged.groupby(['user_id:token', 'q_context_id']):
            group = group.sort_values('rank').head(k)
            
            if len(group) < 2:
                continue
            
            ranks = group['rank'].values
            scores = group['cs_score'].values
            
            if np.std(ranks) > 0 and np.std(scores) > 0:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=ConstantInputWarning)
                    try:
                        rho, _ = spearmanr(ranks, scores)
                        if not np.isnan(rho):
                            # Transform to [0, 1] range
                            crc = (rho + 1) / 2
                            correlations.append(crc)
                    except:
                        pass
        
        # Average correlation (0.5 = no correlation baseline)
        results[f'CRC@{k}'] = float(np.mean(correlations)) if correlations else 0.5
    
    results['CRC@all'] = results.get(f'CRC@{max(k_values)}', 0.5)
    return results


def compute_context_group_balance(predictions_df: pd.DataFrame,
                                  context_info: pd.DataFrame,
                                  context_features: List[str],
                                  feature_groups: Dict[str, List[str]],
                                  k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """
    Compute Context Group Balance (CGB@K).
    
    Measures fairness across feature groups.
    High CGB means all groups are equally well-covered.
    
    Args:
        predictions_df: Predictions with q_context_id
        context_info: Item contexts
        context_features: List of context feature names
        feature_groups: Dict mapping group names to feature lists
        k_values: List of K values
    
    Returns:
        Dict with CGB@K scores
    """
    results = {}
    
    # Prepare data
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
    
    # Compute balance for each K
    for k in k_values:
        top_k = (merged
                .sort_values(['user_id:token', 'q_context_id', 'rank'])
                .groupby(['user_id:token', 'q_context_id'])
                .head(k))
        
        group_recalls = []
        
        for group_name, group_features in feature_groups.items():
            recalls = []
            
            for idx, row in top_k.iterrows():
                q_feats: Set[str] = set()
                i_feats: Set[str] = set()
                
                for feat in group_features:
                    q_val = row.get(f'{feat}_query')
                    i_val = row.get(f'{feat}_item')
                    
                    if pd.notna(q_val):
                        q_feats.add(f'{feat}={q_val}')
                    if pd.notna(i_val):
                        i_feats.add(f'{feat}={i_val}')
                
                if len(q_feats) > 0:
                    recall = len(q_feats & i_feats) / len(q_feats)
                else:
                    recall = 0.0
                
                recalls.append(recall)
            
            group_recall = np.mean(recalls) if recalls else 0.0
            group_recalls.append(group_recall)
        
        # Compute balance: 1 - normalized std dev
        if len(group_recalls) > 1:
            std_dev = np.std(group_recalls)
            cgb = 1 - min(std_dev / 0.5, 1.0)  # 0.5 is max expected std
        else:
            cgb = 1.0  # Perfect balance if only one group
        
        results[f'CGB@{k}'] = float(cgb)
    
    results['CGB@all'] = results.get(f'CGB@{max(k_values)}', 1.0)
    return results


class AdvancedMetricsEvaluator:
    """Standalone evaluator for advanced metrics"""
    
    def __init__(self, context_features: List[str],
                 feature_groups: Dict[str, List[str]],
                 k_values: List[int] = [5, 10, 20]):
        self.context_features = context_features
        self.feature_groups = feature_groups
        self.k_values = k_values
        self.results = {}
    
    def evaluate_model(self, model_name: str,
                      predictions_path: Path,
                      context_info_path: Path) -> Dict[str, float]:
        """Evaluate CR/CRC/CGB for a single model"""
        pred_df = pd.read_csv(predictions_path, sep='\t')
        ctx_df = pd.read_csv(context_info_path, sep='\t')
        
        results = {}
        
        # CR
        cr = compute_context_recall(pred_df, ctx_df, self.context_features, self.k_values)
        results.update(cr)
        
        # CRC
        crc = compute_context_ranking_correlation(pred_df, ctx_df, 
                                                  self.context_features, 
                                                  self.k_values)
        results.update(crc)
        
        # CGB
        cgb = compute_context_group_balance(pred_df, ctx_df,
                                           self.context_features,
                                           self.feature_groups,
                                           self.k_values)
        results.update(cgb)
        
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