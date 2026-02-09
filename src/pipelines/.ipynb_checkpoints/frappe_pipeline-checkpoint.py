"""
Frappe Context-Aware Recommendation Pipeline
=============================================

Dataset-specific implementation for Frappe mobile app recommendations.

Context Features:
    - daytime, weekday, isweekend (temporal)
    - homework, cost (activity)
    - weather, country, city (environment)

Usage:
    python src/pipelines/frappe_pipeline.py --config configs/frappe_config.yaml
"""

import sys
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipelines.pipeline_template import BasePipeline


class FrappePipeline(BasePipeline):
    """Frappe-specific pipeline implementation"""
    
    # Frappe context features
    CONTEXT_FEATURES = [
        'daytime', 'weekday', 'isweekend',
        'homework', 'cost',
        'weather', 'country', 'city'
    ]
    
    # Feature groups
    FEATURE_GROUPS = {
        'temporal': ['daytime', 'weekday', 'isweekend'],
        'activity': ['homework', 'cost'],
        'environment': ['weather', 'country', 'city']
    }
    
    def _get_column_names(self) -> Dict[str, str]:
        """Frappe column names"""
        return {
            'user': 'user',
            'item': 'item',
            'label': 'label'
        }
    
    def _load_dataset_splits(self):
        """Load Frappe CSV splits"""
        data_path = Path(self.config['paths']['data'])
        
        # Load splits (already have headers)
        self.train_df = pd.read_csv(data_path / 'frappe_train.csv')
        self.valid_df = pd.read_csv(data_path / 'frappe_valid.csv')
        self.test_df = pd.read_csv(data_path / 'frappe_test.csv')
        
        # Create context info from training data
        item_cols = ['item'] + self.CONTEXT_FEATURES
        self.context_info = (
            self.train_df[item_cols]
            .groupby('item')
            .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
            .reset_index()
            .rename(columns={'item': 'item_id:token'})
        )


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Frappe Pipeline')
    parser.add_argument('--config', type=str, default='configs/frappe_config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['dataset'] = {'name': 'Frappe'}
    
    # Check GPU
    try:
        import torch
        config['use_gpu'] = torch.cuda.is_available()
    except ImportError:
        config['use_gpu'] = False
    
    # Run pipeline
    pipeline = FrappePipeline(config)
    success = pipeline.run()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()