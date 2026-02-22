"""
Frappe Context-Aware Recommendation Pipeline 
=====================================================

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
    """Frappe-specific pipeline implementation - FIXED"""
    
    # Frappe context features
    COLUMN_NAMES = [
        'label', 'user', 'item', 'daytime', 'weekday', 'isweekend',
        'homework', 'cost', 'weather', 'country', 'city'
    ]
    
    CONTEXT_FEATURES = COLUMN_NAMES[3:]
    
    # Context Groups
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
        """Load Frappe CSV splits and add context_id"""
        data_path = Path(self.config['paths']['data'])
        
        print("  Loading CSV files...")
        
        # Load splits (already have headers)
        self.train_df = pd.read_csv(data_path / 'frappe_train.csv')
        self.valid_df = pd.read_csv(data_path / 'frappe_valid.csv')
        self.test_df = pd.read_csv(data_path / 'frappe_test.csv')
        
        print(f"  Loaded: train={len(self.train_df)}, valid={len(self.valid_df)}, test={len(self.test_df)}")
        
        # Identify unique contexts and assign context_id
        print("  Identifying unique contexts...")
        
        all_data = pd.concat([self.train_df, self.valid_df, self.test_df], ignore_index=True)
        
        # Create unique context combinations
        unique_contexts = (
            all_data[self.CONTEXT_FEATURES]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        unique_contexts['context_id'] = range(len(unique_contexts))
        
        print(f"  ✓ Identified {len(unique_contexts)} unique contexts")
        
        # Merge context_id back into splits
        for split_name, df in [('train', self.train_df), 
                               ('valid', self.valid_df), 
                               ('test', self.test_df)]:
            df_with_ctx = df.merge(
                unique_contexts,
                on=self.CONTEXT_FEATURES,
                how='left'
            )
            
            if split_name == 'train':
                self.train_df = df_with_ctx
            elif split_name == 'valid':
                self.valid_df = df_with_ctx
            else:
                self.test_df = df_with_ctx
        
        print(f"  ✓ Added context_id to all splits")
        
        # Save context_info with context_id mapping
        context_info_path = data_path / 'context_info_with_id.tsv'
        unique_contexts.to_csv(context_info_path, sep='\t', index=False)
        print(f"  ✓ Saved context lookup: {context_info_path}")
        
        # Create item context info (for merging into predictions)
        item_cols = ['item'] + self.CONTEXT_FEATURES
        self.context_info = (
            self.train_df[item_cols]
            .groupby('item')
            .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
            .reset_index()
            .rename(columns={'item': 'item_id:token'})
        )
        
        print(f"  ✓ Created item context info: {len(self.context_info)} items")
    
    def _get_item_context_info(self) -> pd.DataFrame:
        """
        Get item context information from training data.
        
        OVERRIDE: Frappe doesn't need context_id in item context,
        just the feature values.
        """
        item_context = (
            self.train_df.groupby('item')[self.CONTEXT_FEATURES]
            .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
            .reset_index()
            .rename(columns={'item': 'item_id:token'})
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
    
    parser = argparse.ArgumentParser(description='Frappe Pipeline - Fixed')
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
        if config['use_gpu']:
            print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    except ImportError:
        config['use_gpu'] = False
    
    # Run pipeline
    pipeline = FrappePipeline(config)
    success = pipeline.run()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()