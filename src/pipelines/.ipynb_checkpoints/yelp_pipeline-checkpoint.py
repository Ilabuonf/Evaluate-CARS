"""
Yelp Context-Aware Recommendation Pipeline
===========================================

Dataset-specific implementation for Yelp restaurant recommendations.

Context Features:
    - Temporal: time_of_day, day_of_week, is_weekend
    - Social: party_size, occasion
    - Environmental: weather, season

Usage:
    python src/pipelines/yelp_pipeline.py --config configs/yelp_config.yaml
"""

import sys
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipelines.pipeline_template import BasePipeline


class YelpPipeline(BasePipeline):
    """Yelp-specific pipeline implementation"""
    
    # Yelp context features
    CONTEXT_FEATURES = [
        'time_of_day', 'day_of_week', 'is_weekend',
        'party_size', 'occasion',
        'weather', 'season'
    ]
    
    # Feature groups
    FEATURE_GROUPS = {
        'temporal': ['time_of_day', 'day_of_week', 'is_weekend'],
        'social': ['party_size', 'occasion'],
        'environmental': ['weather', 'season']
    }
    
    def _get_column_names(self) -> Dict[str, str]:
        """Yelp column names"""
        return {
            'user': 'user_id',
            'item': 'business_id',
            'label': 'stars'  # Rating as label
        }
    
    def _load_dataset_splits(self):
        """Load Yelp CSV splits"""
        data_path = Path(self.config['paths']['data'])
        
        # Load splits
        self.train_df = pd.read_csv(data_path / 'yelp_train.csv')
        self.valid_df = pd.read_csv(data_path / 'yelp_valid.csv')
        self.test_df = pd.read_csv(data_path / 'yelp_test.csv')
        
        # Convert ratings to binary labels (>= 4 stars = positive)
        threshold = self.config.get('rating_threshold', 4.0)
        for df in [self.train_df, self.valid_df, self.test_df]:
            df['label'] = (df['stars'] >= threshold).astype(int)
        
        # Create context info from training data
        item_cols = ['business_id'] + self.CONTEXT_FEATURES
        self.context_info = (
            self.train_df[item_cols]
            .groupby('business_id')
            .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
            .reset_index()
            .rename(columns={'business_id': 'item_id:token'})
        )


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Yelp Pipeline')
    parser.add_argument('--config', type=str, default='configs/yelp_config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['dataset'] = {'name': 'Yelp'}
    
    # Check GPU
    try:
        import torch
        config['use_gpu'] = torch.cuda.is_available()
    except ImportError:
        config['use_gpu'] = False
    
    # Run pipeline
    pipeline = YelpPipeline(config)
    success = pipeline.run()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()