"""
Evaluation Utility Functions
=============================

Helper functions for running evaluations and generating reports.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime


def collect_all_predictions(results_dir: Path,
                           exclude_dirs: Optional[List[str]] = None) -> Dict[str, Path]:
    """
    Collect prediction files from all model directories.
    
    Args:
        results_dir: Base results directory
        exclude_dirs: Directory names to exclude
    
    Returns:
        Dict mapping model names to prediction file paths
    """
    if exclude_dirs is None:
        exclude_dirs = ['context_metrics', 'evaluation', '__pycache__']
    
    predictions = {}
    
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name in exclude_dirs:
            continue
        
        # Look for predictions in result/ subdirectory
        result_dir = model_dir / 'result'
        if not result_dir.exists():
            continue
        
        pred_files = list(result_dir.glob('*predictions.tsv'))
        if pred_files:
            model_name = model_dir.name.capitalize()
            predictions[model_name] = pred_files[0]
    
    return predictions


def compute_all_metrics(predictions_path: Path,
                       context_info_path: Path,
                       context_features: List[str],
                       feature_groups: Dict[str, List[str]],
                       k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """
    Compute all context-aware metrics for a single model.
    
    Args:
        predictions_path: Path to predictions file
        context_info_path: Path to context info file
        context_features: List of context feature names
        feature_groups: Dict mapping group names to features
        k_values: List of K values
    
    Returns:
        Dict with all metric scores
    """
    from src.metrics import (
        compute_acc,
        compute_cs_wcs,
        compute_similarity_metrics,
        compute_context_recall,
        compute_context_ranking_correlation,
        compute_context_group_balance,
        compute_context_weighted_ndcg,
        compute_context_weighted_map
    )
    
    # Load data
    pred_df = pd.read_csv(predictions_path, sep='\t')
    ctx_df = pd.read_csv(context_info_path, sep='\t')
    
    results = {}
    
    # Context Consistency
    acc_results = compute_acc(pred_df, ctx_df, context_features, k_values)
    results.update(acc_results)
    
    # Context Satisfaction
    cs_results = compute_cs_wcs(pred_df, ctx_df, context_features, 
                               alpha=0.5, k_values=k_values)
    results.update(cs_results)
    
    # Similarity Metrics
    sim_results = compute_similarity_metrics(pred_df, ctx_df, 
                                            context_features, k_values)
    results.update(sim_results)
    
    # Advanced Metrics
    cr_results = compute_context_recall(pred_df, ctx_df, context_features, k_values)
    results.update(cr_results)
    
    crc_results = compute_context_ranking_correlation(pred_df, ctx_df, 
                                                      context_features, k_values)
    results.update(crc_results)
    
    cgb_results = compute_context_group_balance(pred_df, ctx_df,
                                               context_features,
                                               feature_groups, k_values)
    results.update(cgb_results)
    
    # Weighted Ranking Metrics
    ndcg_results = compute_context_weighted_ndcg(pred_df, ctx_df,
                                                context_features, k_values)
    results.update(ndcg_results)
    
    map_results = compute_context_weighted_map(pred_df, ctx_df,
                                              context_features, k_values)
    results.update(map_results)
    
    return results


def create_results_table(all_results: Dict[str, Dict[str, float]],
                        metric_groups: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
    """
    Create formatted results table from metric dict.
    
    Args:
        all_results: Dict mapping model names to metric dicts
        metric_groups: Optional grouping of metrics
    
    Returns:
        DataFrame with models as rows, metrics as columns
    """
    df = pd.DataFrame(all_results).T
    
    # Round to 4 decimals
    df = df.round(4)
    
    # Sort columns by metric groups if provided
    if metric_groups:
        sorted_cols = []
        for group_name, metrics in metric_groups.items():
            for metric in metrics:
                if metric in df.columns:
                    sorted_cols.append(metric)
        
        # Add any remaining columns
        remaining = [c for c in df.columns if c not in sorted_cols]
        df = df[sorted_cols + remaining]
    
    return df


def generate_evaluation_report(all_results: Dict[str, Dict[str, float]],
                              output_path: Path,
                              dataset_name: str,
                              context_features: List[str],
                              k_values: List[int]) -> None:
    """
    Generate comprehensive evaluation report.
    
    Args:
        all_results: Dict mapping model names to metric dicts
        output_path: Path to save report
        dataset_name: Name of dataset
        context_features: List of context features
        k_values: List of K values evaluated
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"CONTEXT-AWARE EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Context Features: {', '.join(context_features)}\n")
        f.write(f"K Values: {k_values}\n\n")
        
        f.write("="*80 + "\n")
        f.write("OVERALL RESULTS\n")
        f.write("="*80 + "\n\n")
        
        # Create results table
        df = pd.DataFrame(all_results).T.round(4)
        f.write(df.to_string())
        f.write("\n\n")
        
        # Best models per metric
        f.write("="*80 + "\n")
        f.write("BEST MODELS PER METRIC\n")
        f.write("="*80 + "\n\n")
        
        for metric in df.columns:
            best_model = df[metric].idxmax()
            best_score = df[metric].max()
            f.write(f"{metric:20s}: {best_model:15s} ({best_score:.4f})\n")
        
        f.write("\n")
        
        # Summary statistics
        f.write("="*80 + "\n")
        f.write("METRIC STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        stats_df = df.describe().loc[['mean', 'std', 'min', 'max']].T
        f.write(stats_df.to_string())
        f.write("\n")
    
    print(f"✓ Report saved to {output_path}")


def compare_models(results_df: pd.DataFrame,
                  baseline_model: str = 'Random',
                  metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compare all models to a baseline.
    
    Args:
        results_df: DataFrame with model results
        baseline_model: Name of baseline model
        metrics: List of metrics to compare (None = all)
    
    Returns:
        DataFrame with improvement percentages
    """
    if baseline_model not in results_df.index:
        raise ValueError(f"Baseline model '{baseline_model}' not found")
    
    if metrics is None:
        metrics = results_df.columns.tolist()
    
    baseline_scores = results_df.loc[baseline_model, metrics]
    
    improvements = pd.DataFrame(index=results_df.index, columns=metrics)
    
    for model in results_df.index:
        for metric in metrics:
            model_score = results_df.loc[model, metric]
            baseline_score = baseline_scores[metric]
            
            if baseline_score > 0:
                improvement = ((model_score - baseline_score) / baseline_score) * 100
            else:
                improvement = 0.0
            
            improvements.loc[model, metric] = improvement
    
    return improvements.astype(float).round(2)


def rank_models(results_df: pd.DataFrame,
               metrics: Optional[List[str]] = None,
               weights: Optional[Dict[str, float]] = None) -> pd.Series:
    """
    Rank models by weighted average of metrics.
    
    Args:
        results_df: DataFrame with model results
        metrics: List of metrics to use (None = all)
        weights: Dict mapping metrics to weights (None = equal)
    
    Returns:
        Series with weighted scores, sorted descending
    """
    if metrics is None:
        metrics = results_df.columns.tolist()
    
    if weights is None:
        weights = {m: 1.0 for m in metrics}
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    # Compute weighted average
    scores = pd.Series(0.0, index=results_df.index)
    
    for metric in metrics:
        weight = weights.get(metric, 0.0)
        scores += results_df[metric] * weight
    
    return scores.sort_values(ascending=False)


def save_results_json(all_results: Dict[str, Dict[str, float]],
                     output_path: Path,
                     metadata: Optional[Dict] = None) -> None:
    """
    Save results as JSON with metadata.
    
    Args:
        all_results: Dict mapping model names to metric dicts
        output_path: Path to save JSON
        metadata: Optional metadata dict
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output = {
        'timestamp': datetime.now().isoformat(),
        'results': all_results
    }
    
    if metadata:
        output['metadata'] = metadata
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✓ Results saved to {output_path}")


def load_results_json(input_path: Path) -> Dict:
    """
    Load results from JSON file.
    
    Args:
        input_path: Path to JSON file
    
    Returns:
        Dict with results and metadata
    """
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    return data