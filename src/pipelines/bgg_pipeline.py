"""
BoardGameGeek (BGG) Context-Aware Recommendation Pipeline
==========================================================

Dataset-specific implementation for BGG.

Context Features:
    - playing_time: Duration categories
    - gaming_mood: Player experience preference
    - social_companion: Social context

Usage:
    python src/pipelines/bgg_pipeline.py --config configs/bgg_config.yaml
"""

import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipelines.pipeline_template import BasePipeline


class BGGPipeline(BasePipeline):
    """BoardGameGeek-specific pipeline implementation"""
    
    # Context features specific to BGG
    CONTEXT_FEATURES = [
        'playing_time',
        'gaming_mood',
        'social_companion'
    ]
    
    # Feature groups for dimensional analysis
    FEATURE_GROUPS = {
        'temporal': ['playing_time'],
        'experiential': ['gaming_mood'],
        'social': ['social_companion']
    }
    
    def _get_column_names(self) -> Dict[str, str]:
        """BGG uses these column names"""
        return {
            'user': 'user_id:token',
            'item': 'game_id:token',  # Original column name
            'label': 'label',
            'rating': 'rating:float'
        }
    
    def _load_dataset_splits(self):
        """Load pre-split BGG data and reconstruct categorical context"""
        data_path = Path(self.config['paths']['data'])
        
        dtypes = {
            'user_id:token': 'category',
            'game_id:token': 'category',
            'context_id': 'int32',
            'rating:float': 'float32'
        }
        
        print("  Loading base data files...")
        
        # 1. Load base data
        self.train_df = pd.read_csv(data_path / 'train_df.tsv', sep='\t', dtype=dtypes)
        self.valid_df = pd.read_csv(data_path / 'valid_df.tsv', sep='\t', dtype=dtypes)
        self.test_df = pd.read_csv(data_path / 'test_df.tsv', sep='\t', dtype=dtypes)
        
        print("  Loading context definitions...")
        
        # 2. Load and reconstruct context features
        context_info_path = data_path / 'context_info.tsv'
        
        if context_info_path.exists():
            context_info = pd.read_csv(context_info_path, sep='\t')
            
            # Check if context features need reconstruction from One-Hot
            if 'playing_time' not in context_info.columns:
                print("  Reconstructing context features from One-Hot encoding...")
                context_info = self._reconstruct_categorical_features(context_info)
            
            # Merge context features into data splits
            print("  Merging context features into splits...")
            for df in [self.train_df, self.valid_df, self.test_df]:
                # Merge context info
                for feat in self.CONTEXT_FEATURES:
                    if feat in context_info.columns:
                        # Create temporary merge
                        temp = context_info[['context_id', feat]].copy()
                        df[feat] = df['context_id'].map(
                            temp.set_index('context_id')[feat]
                        )
            
            self.context_info = context_info
            print("  ✓ Context features added to data splits")
        else:
            print(" Warning: context_info.tsv not found")
            # Create dummy context features
            for df in [self.train_df, self.valid_df, self.test_df]:
                for feat in self.CONTEXT_FEATURES:
                    df[feat] = 'unknown'
        
        # 3. Create binary labels
        rating_threshold = self.config.get('rating_threshold', 7.0)
        print(f"  Creating binary labels (threshold: {rating_threshold})...")
        
        for df in [self.train_df, self.valid_df, self.test_df]:
            df['label'] = (df['rating:float'] >= rating_threshold).astype(int)
        
        print("  ✓ Data loading completed")
    
    def _reconstruct_categorical_features(self, context_df: pd.DataFrame) -> pd.DataFrame:
        """
        Reconstruct categorical features from One-Hot encoding.
        
        Example:
            Input:  playing_time_short:float=1.0, playing_time_medium:float=0.0, ...
            Output: playing_time='short'
        """
        def collapse_onehot(df, prefix, new_col_name):
            """Find which One-Hot column has value 1.0 and extract category name"""
            cols = [c for c in df.columns if c.startswith(prefix)]
            if not cols:
                return df
            
            # Get category name from column with max value (should be 1.0)
            df[new_col_name] = (
                df[cols]
                .idxmax(axis=1)
                .str.replace(prefix, '')
                .str.replace(':float', '')
            )
            
            return df
        
        # Reconstruct each feature group
        context_df = collapse_onehot(context_df, 'playing_time_', 'playing_time')
        context_df = collapse_onehot(context_df, 'gaming_mood_', 'gaming_mood')
        context_df = collapse_onehot(context_df, 'social_companion_', 'social_companion')
        
        return context_df
    
    def _get_item_context_info(self) -> pd.DataFrame:
        """
        Get item context information from training data.
        
        For BGG, we aggregate context features per item using the mode (most common value).
        This is used for adding context info to predictions.
        """
        # Aggregate context features per item
        item_context = (
            self.train_df.groupby('game_id:token')[self.CONTEXT_FEATURES]
            .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
            .reset_index()
            .rename(columns={'game_id:token': 'item_id:token'})
        )
        
        # Convert to string for consistency
        item_context['item_id:token'] = item_context['item_id:token'].astype(str).str.strip()
        
        for feat in self.CONTEXT_FEATURES:
            if feat in item_context.columns:
                item_context[feat] = item_context[feat].astype(str).str.strip()
        
        return item_context


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='BGG Recommendation Pipeline')
    parser.add_argument('--config', type=str, default='configs/bgg_config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add dataset name
    config['dataset'] = {'name': 'BoardGameGeek'}
    
    # Check GPU
    try:
        import torch
        config['use_gpu'] = torch.cuda.is_available()
        if config['use_gpu']:
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    except ImportError:
        config['use_gpu'] = False
    
    # Run pipeline
    pipeline = BGGPipeline(config)
    success = pipeline.run()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()