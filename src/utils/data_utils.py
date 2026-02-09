"""
Data Utility Functions
======================

Helper functions for data loading, preprocessing, and context handling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def create_context_id(df: pd.DataFrame, context_features: List[str]) -> pd.Series:
    """
    Create context_id string from context features.
    
    Args:
        df: DataFrame with context features
        context_features: List of feature names
    
    Returns:
        Series with context_id strings (format: feat1_feat2_feat3)
    
    Example:
        >>> df = pd.DataFrame({'time': ['morning', 'evening'], 
        ...                   'weather': ['sunny', 'rainy']})
        >>> create_context_id(df, ['time', 'weather'])
        0    morning_sunny
        1    evening_rainy
        dtype: object
    """
    context_values = df[context_features].astype(str)
    return context_values.apply(lambda row: '_'.join(row), axis=1)


def parse_context_id(context_id: str, context_features: List[str]) -> Dict[str, str]:
    """
    Parse context_id string back to feature dict.
    
    Args:
        context_id: Context ID string (format: feat1_feat2_feat3)
        context_features: List of feature names in order
    
    Returns:
        Dict mapping feature names to values
    
    Example:
        >>> parse_context_id('morning_sunny', ['time', 'weather'])
        {'time': 'morning', 'weather': 'sunny'}
    """
    values = context_id.split('_')
    return {feat: val for feat, val in zip(context_features, values)}


def load_recbole_predictions(pred_path: Path) -> pd.DataFrame:
    """
    Load RecBole prediction file with proper types.
    
    Args:
        pred_path: Path to predictions TSV file
    
    Returns:
        DataFrame with standardized columns
    """
    df = pd.read_csv(pred_path, sep='\t')
    
    # Standardize column names
    rename_map = {}
    if 'user' in df.columns:
        rename_map['user'] = 'user_id:token'
    if 'item' in df.columns:
        rename_map['item'] = 'item_id:token'
    
    if rename_map:
        df = df.rename(columns=rename_map)
    
    # Ensure string types for IDs
    if 'user_id:token' in df.columns:
        df['user_id:token'] = df['user_id:token'].astype(str).str.strip()
    if 'item_id:token' in df.columns:
        df['item_id:token'] = df['item_id:token'].astype(str).str.strip()
    
    return df


def create_context_info(df: pd.DataFrame,
                       item_column: str,
                       context_features: List[str],
                       aggregation: str = 'mode') -> pd.DataFrame:
    """
    Create item context info from interaction data.
    
    Uses mode (most common) or mean aggregation per item.
    
    Args:
        df: DataFrame with interactions
        item_column: Name of item ID column
        context_features: List of context feature names
        aggregation: 'mode' or 'mean'
    
    Returns:
        DataFrame with columns [item_id:token, ...context_features]
    """
    if aggregation == 'mode':
        agg_func = lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
    else:
        agg_func = 'mean'
    
    context_df = (
        df.groupby(item_column)[context_features]
        .agg(agg_func)
        .reset_index()
        .rename(columns={item_column: 'item_id:token'})
    )
    
    return context_df


def split_by_user_timestamp(df: pd.DataFrame,
                           user_col: str = 'user_id',
                           timestamp_col: str = 'timestamp',
                           train_ratio: float = 0.7,
                           valid_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by user-wise chronological order.
    
    For each user, earliest interactions go to train,
    middle to validation, latest to test.
    
    Args:
        df: DataFrame with user interactions
        user_col: User ID column name
        timestamp_col: Timestamp column name
        train_ratio: Proportion for training
        valid_ratio: Proportion for validation
    
    Returns:
        Tuple of (train_df, valid_df, test_df)
    """
    train_list = []
    valid_list = []
    test_list = []
    
    for user, group in df.groupby(user_col):
        # Sort by timestamp
        group = group.sort_values(timestamp_col)
        
        n = len(group)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)
        
        train_list.append(group.iloc[:n_train])
        valid_list.append(group.iloc[n_train:n_train+n_valid])
        test_list.append(group.iloc[n_train+n_valid:])
    
    train_df = pd.concat(train_list, ignore_index=True)
    valid_df = pd.concat(valid_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)
    
    return train_df, valid_df, test_df


def get_user_context_queries(df: pd.DataFrame,
                            user_col: str,
                            context_features: List[str]) -> pd.DataFrame:
    """
    Get unique user-context queries from dataset.
    
    Args:
        df: DataFrame with user and context columns
        user_col: User ID column name
        context_features: List of context feature names
    
    Returns:
        DataFrame with unique user-context combinations
    """
    cols = [user_col] + context_features
    queries = df[cols].drop_duplicates()
    queries['q_context_id'] = create_context_id(queries, context_features)
    
    return queries


def merge_predictions_with_context(predictions_df: pd.DataFrame,
                                  context_info_df: pd.DataFrame,
                                  context_features: List[str]) -> pd.DataFrame:
    """
    Merge predictions with item context information.
    
    Adds both query context (from q_context_id) and item context.
    
    Args:
        predictions_df: Predictions with q_context_id
        context_info_df: Item contexts
        context_features: List of context feature names
    
    Returns:
        DataFrame with query and item context columns
    """
    pred_df = predictions_df.copy()
    ctx_df = context_info_df.copy()
    
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
    
    return merged


def compute_dataset_statistics(df: pd.DataFrame,
                              user_col: str,
                              item_col: str,
                              context_features: List[str]) -> Dict:
    """
    Compute comprehensive dataset statistics.
    
    Args:
        df: DataFrame with interactions
        user_col: User ID column name
        item_col: Item ID column name
        context_features: List of context feature names
    
    Returns:
        Dict with statistics
    """
    stats = {
        'n_interactions': len(df),
        'n_users': df[user_col].nunique(),
        'n_items': df[item_col].nunique(),
        'sparsity': 1 - (len(df) / (df[user_col].nunique() * df[item_col].nunique())),
        'avg_interactions_per_user': len(df) / df[user_col].nunique(),
        'avg_interactions_per_item': len(df) / df[item_col].nunique(),
        'context_features': {}
    }
    
    # Context feature statistics
    for feat in context_features:
        if feat in df.columns:
            stats['context_features'][feat] = {
                'n_unique': df[feat].nunique(),
                'mode': df[feat].mode()[0] if len(df[feat].mode()) > 0 else None,
                'missing_rate': df[feat].isna().mean()
            }
    
    # Context combinations
    df['context_id'] = create_context_id(df, context_features)
    stats['n_unique_contexts'] = df['context_id'].nunique()
    
    return stats


def validate_predictions(pred_df: pd.DataFrame,
                        required_columns: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
    """
    Validate prediction DataFrame format.
    
    Args:
        pred_df: Predictions DataFrame
        required_columns: List of required column names
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    if required_columns is None:
        required_columns = ['user_id:token', 'item_id:token', 'q_context_id', 'prediction']
    
    # Check required columns
    missing_cols = set(required_columns) - set(pred_df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    
    # Check for NaN values
    if pred_df.isna().any().any():
        nan_cols = pred_df.columns[pred_df.isna().any()].tolist()
        errors.append(f"NaN values found in columns: {nan_cols}")
    
    # Check prediction scores
    if 'prediction' in pred_df.columns:
        if not pd.api.types.is_numeric_dtype(pred_df['prediction']):
            errors.append("Prediction scores must be numeric")
        
        if (pred_df['prediction'] < 0).any():
            errors.append("Negative prediction scores found")
    
    # Check ranks if present
    if 'rank' in pred_df.columns:
        if not pd.api.types.is_numeric_dtype(pred_df['rank']):
            errors.append("Ranks must be numeric")
        
        if (pred_df['rank'] < 1).any():
            errors.append("Ranks must be >= 1")
    
    is_valid = len(errors) == 0
    return is_valid, errors