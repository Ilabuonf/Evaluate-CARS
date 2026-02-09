"""
Similarity-Based Context Metrics (WCA, Friction)
=================================================

Alternative context matching metrics based on similarity measures.

Metrics:
    - WCA (Weighted Context Alignment): Normalized weighted match
    - Friction: Inverted Hamming distance

WCA Formula:
    WCA = Σ(matched features × IDF) / Σ(query features × IDF)
    
Friction Formula:
    Friction = 1 - (Hamming distance / # features)

Interpretation:
    WCA:
        - Range: [0, 1]
        - 1.0: All query features matched
        - 0.0: No query features matched
        
    Friction:
        - Range: [0, 1]
        - 1.0: No mismatches (low friction)
        - 0.0: All features differ (high friction)

Use case:
    - WCA: Focus on query coverage
    - Friction: Penalize any mismatch equally
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
    Weighted Context Alignment: IDF-weighted match score.
    
    Returns proportion of query features that matched (weighted).
    """
    matched_weight = 0.0
    total_query_weight = 0.0
    
    for feat in context_features:
        q_val = str(query_context.get(feat, '')).strip()
        i_val = str(item_context.get(feat, '')).strip()
        
        w = idf_weights.get(feat, 1.0)
        
        if q_val and q_val != 'nan':
            total_query_weight += w
            
            if i_val and i_val != 'nan' and q_val == i_val:
                matched_weight += w
    
    if total_query_weight == 0:
        return 0.0
    
    wca = matched_weight / total_query_weight
    return np.clip(wca, 0.0, 1.0)


def compute_friction_score(query_context: pd.Series,
                          item_context: pd.Series,
                          context_features: List[str]) -> float:
    """
    Context Friction: Inverted Hamming distance.
    
    Returns 1 - (proportion of features that differ).
    """
    distance = 0
    max_distance = len(context_features)
    
    for feat in context_features:
        q_val = str(query_context.get(feat, '')) if pd.notna(query_context.get(feat)) else ''
        i_val = str(item_context.get(feat, '')) if pd.notna(item_context.get(feat)) else ''
        
        if q_val != i_val:
            distance += 1
    
    if max_distance == 0:
        return 1.0
    
    return 1.0 - (distance / max_distance)


def compute_similarity_metrics(predictions_df: pd.DataFrame,
                               context_info: pd.DataFrame,
                               context_features: List[str],
                               k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """
    Compute WCA and Friction metrics.
    
    Args:
        predictions_df: Predictions with q_context_id
        context_info: Item contexts
        context_features: List of context feature names
        k_values: List of K values
    
    Returns:
        Dict with WCA@K and Friction@K scores
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
    
    # Add rank if missing
    if 'rank' not in merged.columns:
        merged = merged.sort_values(
            ['user_id:token', 'q_context_id', 'prediction'],
            ascending=[True, True, False]
        )
        merged['rank'] = (merged.groupby(['user_id:token', 'q_context_id'])
                                .cumcount() + 1)
    
    # Compute IDF weights
    from src.metrics.context_satisfaction import compute_idf_weights
    idf_weights = compute_idf_weights(pred_df, context_features)
    
    # Compute WCA and Friction for each prediction
    wca_scores = []
    friction_scores = []
    
    for idx, row in merged.iterrows():
        q_ctx = pd.Series({f: row.get(f'{f}_query') for f in context_features})
        i_ctx = pd.Series({f: row.get(f'{f}_item') for f in context_features})
        
        wca = compute_wca_score(q_ctx, i_ctx, idf_weights, context_features)
        friction = compute_friction_score(q_ctx, i_ctx, context_features)
        
        wca_scores.append(wca)
        friction_scores.append(friction)
    
    merged['wca_score'] = wca_scores
    merged['friction'] = friction_scores
    
    # Aggregate by K
    for k in k_values:
        top_k = (merged
                .sort_values(['user_id:token', 'q_context_id', 'rank'])
                .groupby(['user_id:token', 'q_context_id'])
                .head(k))
        
        wca_k = top_k.groupby(['user_id:token', 'q_context_id'])['wca_score'].mean().mean()
        friction_k = top_k.groupby(['user_id:token', 'q_context_id'])['friction'].mean().mean()
        
        results[f'WCA@{k}'] = float(wca_k)
        results[f'Friction@{k}'] = float(friction_k)
    
    # Overall scores
    results['WCA@all'] = float(merged['wca_score'].mean())
    results['Friction@all'] = float(merged['friction'].mean())
    
    return results


class SimilarityEvaluator:
    """Standalone evaluator for similarity metrics"""
    
    def __init__(self, context_features: List[str],
                 k_values: List[int] = [5, 10, 20]):
        self.context_features = context_features
        self.k_values = k_values
        self.results = {}
    
    def evaluate_model(self, model_name: str,
                      predictions_path: Path,
                      context_info_path: Path) -> Dict[str, float]:
        """Evaluate WCA/Friction for a single model"""
        pred_df = pd.read_csv(predictions_path, sep='\t')
        ctx_df = pd.read_csv(context_info_path, sep='\t')
        
        results = compute_similarity_metrics(pred_df, ctx_df, 
                                            self.context_features,
                                            self.k_values)
        
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