"""
Utility Functions Package
==========================

Helper functions for data processing and evaluation.
"""

from .data_utils import (
    create_context_id,
    parse_context_id,
    load_recbole_predictions,
    create_context_info,
    split_by_user_timestamp,
    get_user_context_queries,
    merge_predictions_with_context,
    compute_dataset_statistics,
    validate_predictions
)

from .eval_utils import (
    collect_all_predictions,
    compute_all_metrics,
    create_results_table,
    generate_evaluation_report,
    compare_models,
    rank_models,
    save_results_json,
    load_results_json
)

__all__ = [
    # Data utils
    'create_context_id',
    'parse_context_id',
    'load_recbole_predictions',
    'create_context_info',
    'split_by_user_timestamp',
    'get_user_context_queries',
    'merge_predictions_with_context',
    'compute_dataset_statistics',
    'validate_predictions',
    
    # Eval utils
    'collect_all_predictions',
    'compute_all_metrics',
    'create_results_table',
    'generate_evaluation_report',
    'compare_models',
    'rank_models',
    'save_results_json',
    'load_results_json',
]