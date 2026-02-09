"""
Dataset Pipelines Package
==========================

Training pipelines for different datasets.
"""

from .pipeline_template import BasePipeline
from .bgg_pipeline import BGGPipeline
from .frappe_pipeline import FrappePipeline
from .yelp_pipeline import YelpPipeline

__all__ = [
    'BasePipeline',
    'BGGPipeline',
    'FrappePipeline',
    'YelpPipeline'
]