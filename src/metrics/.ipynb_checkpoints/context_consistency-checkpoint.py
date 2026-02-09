"""
Context Consistency Metric (ACC@K)
===================================

Measures percentage of top-K items with EXACT context match to query.

Formula:
    ACC@K = (1/|Q|) Σ_q (1/K) Σ_{i=1}^K I(C_q = C_i)

where:
    C_q = query context vector
    C_i = context of item i
    I(·) = indicator function (1 if exact match, 0 otherwise)

Interpretation:
    - Range: [0, 1]
    - 1.0: All top-K items have exact context match
    - 0.0: No items match the query context
    - Expected random: Very low (< 0.01) for high-dimensional contexts

Use case:
    Strict context matching evaluation
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from pathlib import Path


def compute_acc(predictions_df: pd.DataFrame,
                context_info: pd.DataFrame,
                context_features: List[str],
                k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """
    Compute Average Context Consistency (ACC@K).
    
    Args:
        predictions_df: Predictions with columns:
            - user_id:token
            - item_id:token
            - q_context_id (format: feat1_feat2_feat3)
            - prediction (score)
            - rank (optional)
        context_info: Item contexts with columns:
            - item_id:token
            - ...context_features
        context_features: List of context feature names
        k_values: List of K values to evaluate
    
    Returns:
        Dict with keys like 'ACC@5', 'ACC@10'
    
    Example:
        >>> predictions = pd.read_csv('predictions.tsv', sep='\\t')
        >>> contexts = pd.read_csv('context_info.tsv', sep='\\t')
        >>> features = ['playing_time', 'gaming_mood', 'social_companion']
        >>> results = compute_acc(predictions, contexts, features, [5, 10])
        >>> print(results)
        {'ACC@5': 0.0823, 'ACC@10': 0.0756}
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
    
    # Rename to _item suffix
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
    
    # Compute ACC for each K
    for k in k_values:
        top_k = (merged
                .sort_values(['user_id:token', 'q_context_id', 'rank'])
                .groupby(['user_id:token', 'q_context_id'])
                .head(k))
        
        query_cols = [f'{f}_query' for f in context_features]
        item_cols = [f'{f}_item' for f in context_features]
        
        if not all(col in top_k.columns for col in query_cols + item_cols):
            results[f'ACC@{k}'] = 0.0
            continue
        
        # Check if ALL features match
        feature_matches = []
        for feat in context_features:
            q_col = f'{feat}_query'
            i_col = f'{feat}_item'
            match = (top_k[q_col].astype(str) == top_k[i_col].astype(str))
            feature_matches.append(match)
        
        # All features must match for context_match = 1
        top_k['context_match'] = pd.concat(feature_matches, axis=1).all(axis=1).astype(int)
        
        # Average per query, then across all queries
        acc = top_k.groupby(['user_id:token', 'q_context_id'])['context_match'].mean().mean()
        results[f'ACC@{k}'] = float(acc)
    
    return results


class ACCEvaluator:
    """
    Standalone evaluator for Average Context Consistency.
    
    Can be used for batch evaluation of multiple models.
    """
    
    def __init__(self, context_features: List[str], k_values: List[int] = [5, 10, 20]):
        self.context_features = context_features
        self.k_values = k_values
        self.results = {}
    
    def evaluate_model(self, model_name: str, predictions_path: Path, 
                      context_info_path: Path) -> Dict[str, float]:
        """
        Evaluate ACC for a single model.
        
        Args:
            model_name: Name of the model
            predictions_path: Path to predictions TSV file
            context_info_path: Path to context info TSV file
        
        Returns:
            Dict with ACC@K scores
        """
        # Load data
        pred_df = pd.read_csv(predictions_path, sep='\t')
        ctx_df = pd.read_csv(context_info_path, sep='\t')
        
        # Compute ACC
        results = compute_acc(pred_df, ctx_df, self.context_features, self.k_values)
        
        # Store
        self.results[model_name] = results
        
        return results
    
    def evaluate_all(self, results_dir: Path, context_info_path: Path) -> pd.DataFrame:
        """
        Evaluate ACC for all models in a results directory.
        
        Args:
            results_dir: Directory containing model subdirectories
            context_info_path: Path to context info file
        
        Returns:
            DataFrame with all results
        """
        exclude_dirs = {'context_metrics', 'evaluation', '__pycache__'}
        
        model_dirs = [d for d in results_dir.iterdir() 
                     if d.is_dir() and d.name not in exclude_dirs]
        
        for model_dir in model_dirs:
            model_name = model_dir.name.capitalize()
            
            # Find predictions file
            pred_files = list((model_dir / 'result').glob('*predictions.tsv'))
            if not pred_files:
                continue
            
            pred_file = pred_files[0]
            
            print(f"Evaluating {model_name}...")
            self.evaluate_model(model_name, pred_file, context_info_path)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(self.results).T
        return results_df.round(4)