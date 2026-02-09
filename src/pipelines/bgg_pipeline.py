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
            'item': 'game_id:token',
            'label': 'label',  # Binary label from rating >= threshold
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
        
        # 1. Caricamento base
        self.train_df = pd.read_csv(data_path / 'train_df.tsv', sep='\t', dtype=dtypes)
        self.valid_df = pd.read_csv(data_path / 'valid_df.tsv', sep='\t', dtype=dtypes)
        self.test_df = pd.read_csv(data_path / 'test_df.tsv', sep='\t', dtype=dtypes)

        # 2. Gestione Contesto
        if (data_path / 'context_info.tsv').exists():
            context_info = pd.read_csv(data_path / 'context_info.tsv', sep='\t')
            
            # Funzione per collassare le colonne One-Hot in una categorica
            def collapse_onehot(df, prefix, new_name):
                # Prende le colonne che iniziano con il prefisso (es. 'playing_time_')
                cols = [c for c in df.columns if c.startswith(prefix)]
                if not cols: return df
                # Trova quale colonna ha valore 1.0 per ogni riga e ne estrae il nome
                df[new_name] = df[cols].idxmax(axis=1).str.replace(prefix, "").str.replace(":float", "")
                return df

            # Trasformiamo le colonne One-Hot nelle 3 colonne richieste dalla pipeline
            context_info = collapse_onehot(context_info, "playing_time_", "playing_time")
            context_info = collapse_onehot(context_info, "gaming_mood_", "gaming_mood")
            context_info = collapse_onehot(context_info, "social_companion_", "social_companion")

            # Merge dei dati puliti
            self.train_df = self.train_df.merge(context_info[['context_id', 'playing_time', 'gaming_mood', 'social_companion']], on='context_id', how='left')
            self.valid_df = self.valid_df.merge(context_info[['context_id', 'playing_time', 'gaming_mood', 'social_companion']], on='context_id', how='left')
            self.test_df = self.test_df.merge(context_info[['context_id', 'playing_time', 'gaming_mood', 'social_companion']], on='context_id', how='left')
            print("  ✓ Context columns reconstructed from One-Hot format")

        # 3. Target binario
        rating_threshold = self.config.get('rating_threshold', 7.0)
        for df in [self.train_df, self.valid_df, self.test_df]:
            df['label'] = (df['rating:float'] >= rating_threshold).astype(int)
        
        self.context_info = context_info


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
    except ImportError:
        config['use_gpu'] = False
    
    # Run pipeline
    pipeline = BGGPipeline(config)
    success = pipeline.run()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()