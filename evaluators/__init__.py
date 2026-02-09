"""
Evaluators Package
==================

Consolidated evaluators for context-aware recommendation systems.
Each evaluator handles a complete dataset evaluation pipeline.
"""

from .evaluate_bgg import BGGEvaluator
from .evaluate_frappe import FrappeEvaluator
from .evaluate_yelp import YelpEvaluator

__all__ = [
    'BGGEvaluator',
    'FrappeEvaluator', 
    'YelpEvaluator'
]

__version__ = '1.0.0'