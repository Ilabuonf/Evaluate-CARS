"""
Context-Aware Recommendation Pipeline Template
===============================================

Reusable base class for dataset-specific pipelines.

Dataset-specific pipelines (BGG, Frappe, Yelp) inherit from this
and customize data loading and feature definitions.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.baselines import RandomModel, PopularityModel

# RecBole imports
try:
    from recbole.quick_start import run_recbole, load_data_and_model
    from recbole.data.interaction import Interaction  
    import torch
    RECBOLE_AVAILABLE = True
except ImportError:
    RECBOLE_AVAILABLE = False
    print(" RecBole not available")

# Ranx for evaluation
try:
    from ranx import Qrels, Run, evaluate
    RANX_AVAILABLE = True
except ImportError:
    RANX_AVAILABLE = False

GPU_AVAILABLE = torch.cuda.is_available() if RECBOLE_AVAILABLE else False


class BasePipeline(ABC):
    """
    Base pipeline for context-aware recommendation.
    
    Subclasses must implement:
        - CONTEXT_FEATURES: List[str]
        - FEATURE_GROUPS: Dict[str, List[str]]
        - _get_column_names() -> Dict[str, str]
        - _load_dataset_splits()
    """
    
    CONTEXT_FEATURES: List[str] = []
    FEATURE_GROUPS: Dict[str, List[str]] = {}
    
    def __init__(self, config: Dict):
        self.config = config
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.use_gpu = config.get('use_gpu', GPU_AVAILABLE)
        self.base_output = Path(config['paths']['output_base'])
        self.base_output.mkdir(parents=True, exist_ok=True)
        
        # Will be set by subclass
        self.train_df = None
        self.valid_df = None
        self.test_df = None
        self.context_info = None
    
    @abstractmethod
    def _get_column_names(self) -> Dict[str, str]:
        """
        Return dict mapping generic names to dataset-specific column names.
        
        Example:
            {'user': 'user_id', 'item': 'business_id', 'label': 'stars'}
        """
        pass
    
    @abstractmethod
    def _load_dataset_splits(self):
        """Load train, valid, test splits into self.train_df, etc."""
        pass
    
    def step1_load_data(self):
        """Load data splits"""
        print("\n" + "="*70)
        print("STEP 1: LOADING DATA")
        print("="*70)
        
        self._load_dataset_splits()
        
        print(f"\n  Dataset Statistics:")
        print(f"    Train: {len(self.train_df):,} interactions")
        print(f"    Valid: {len(self.valid_df):,} interactions")
        print(f"    Test:  {len(self.test_df):,} interactions")
        
        col_names = self._get_column_names()
        print(f"\n  Unique Entities:")
        print(f"    Users: {self.train_df[col_names['user']].nunique():,}")
        print(f"    Items: {self.train_df[col_names['item']].nunique():,}")
        
        print(f"\n  Context Features ({len(self.CONTEXT_FEATURES)}):")
        for feat in self.CONTEXT_FEATURES:
            n_values = self.train_df[feat].nunique()
            print(f"      {feat}: {n_values} unique values")
    
    def step2_prepare_recbole_data(self):
        """Convert to RecBole format"""
        print("\n" + "="*70)
        print("STEP 2: PREPARING RECBOLE FORMAT")
        print("="*70)
        
        dataset_name = self.config['dataset']['name'].lower()
        recbole_dir = Path(self.config['paths']['data']) / 'recbole' / dataset_name
        recbole_dir.mkdir(parents=True, exist_ok=True)
        
        col_names = self._get_column_names()
        
        def prepare_split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
            """Convert to RecBole format"""
            recbole_df = pd.DataFrame({
                'user_id:token': df[col_names['user']].astype(str),
                'item_id:token': df[col_names['item']].astype(str),
                'label:float': df[col_names['label']].astype(float)
            })
            
            for col in self.CONTEXT_FEATURES:
                if col in df.columns:
                    recbole_df[f'{col}:token'] = df[col].astype(str)
            
            return recbole_df
        
        # Prepare each split
        train_rb = prepare_split(self.train_df, 'train')
        valid_rb = prepare_split(self.valid_df, 'valid')
        test_rb = prepare_split(self.test_df, 'test')
        
        # Save
        train_rb.to_csv(recbole_dir / f'{dataset_name}.train.inter', sep='\t', index=False)
        valid_rb.to_csv(recbole_dir / f'{dataset_name}.valid.inter', sep='\t', index=False)
        test_rb.to_csv(recbole_dir / f'{dataset_name}.test.inter', sep='\t', index=False)
        
        print(f"\n  ✓ RecBole files created in {recbole_dir}")
        
        # Dataset config
        inter_cols = ['user_id', 'item_id', 'label'] + self.CONTEXT_FEATURES
        
        dataset_config = {
            'field_separator': '\t',
            'USER_ID_FIELD': 'user_id',
            'ITEM_ID_FIELD': 'item_id',
            'LABEL_FIELD': 'label',
            'load_col': {'inter': inter_cols},
            'user_inter_num_interval': '[0,inf)',
            'item_inter_num_interval': '[0,inf)',
            'additional_feat_suffix': ['token']
        }
        
        with open(recbole_dir / f'{dataset_name}.yaml', 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"  ✓ Dataset config saved")
    
    def step3_train_models(self):
        """Train all models (baselines + CTR)"""
        print("\n" + "="*70)
        print("STEP 3: TRAINING ALL MODELS")
        print("="*70)
        
        # Train baselines first
        self._train_baseline_models()
        
        # Train CTR models
        for model_name, model_config in self.config['models'].items():
            if model_name in ['Random', 'Pop']:
                continue
            self._train_ctr_model(model_name, model_config)
    
    def _train_baseline_models(self):
        """Train Random and Popularity baselines"""
        print(f"\n{'='*70}")
        print("TRAINING BASELINE MODELS")
        print('='*70)
        
        baseline_classes = {
            'Random': RandomModel,
            'Pop': PopularityModel
        }
        
        col_names = self._get_column_names()
        
        for model_name, ModelClass in baseline_classes.items():
            print(f"\n  → Training {model_name}...")
            
            output_dir = self.base_output / model_name.lower() / 'result'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            baseline = ModelClass(
                name=model_name,
                train_df=self.train_df,
                test_df=self.test_df,
                config=self.config,
                context_features=self.CONTEXT_FEATURES,
                column_names=col_names  # Pass column mapping
            )
            
            baseline.fit()
            top_k = self.config['evaluation']['top_k']
            baseline.predict(top_k=top_k)
            baseline.save_predictions(output_dir)
            
            self.results[model_name] = {
                'train_auc': np.nan,
                'train_logloss': np.nan,
                'model_type': 'baseline'
            }
            
            print(f"    ✓ {model_name} completed\n")
    
    def _train_ctr_model(self, model_name: str, model_config: Dict):
        """Train a single CTR model with RecBole"""
        if not RECBOLE_AVAILABLE:
            print(f"  ⚠ Skipping {model_name}: RecBole not available")
            return
        
        print(f"\n{'='*70}")
        print(f"TRAINING CTR MODEL: {model_name}")
        print('='*70)
        
        dataset_name = self.config['dataset']['name'].lower()
        params = model_config.get('params', {})
        
        config_dict = {
            'model': model_name,
            'dataset': dataset_name,
            'data_path': str(Path(self.config['paths']['data']) / 'recbole'),
            'USER_ID_FIELD': 'user_id',
            'ITEM_ID_FIELD': 'item_id',
            'LABEL_FIELD': 'label',
            'load_col': {
                'inter': ['user_id', 'item_id', 'label'] + self.CONTEXT_FEATURES
            },
            'epochs': params.get('epochs', 50),
            'train_batch_size': params.get('batch_size', 2048),
            'eval_batch_size': 4096,
            'learning_rate': params.get('lr', 0.001),
            'learner': params.get('learner', 'adam'),
            'stopping_step': params.get('stopping_step', 10),
            'benchmark_filename': ['train', 'valid', 'test'],
            'eval_args': {
                'split': {'RS': [1.0, 0.0, 0.0]},
                'group_by': 'user',
                'order': 'RO',
                'mode': 'labeled'
            },
            'metrics': ['AUC', 'LogLoss'],
            'valid_metric': 'AUC',
            'numerical_features': [],
            'embedding_size': params.get('embedding_size', 64),
            'reg_weight': params.get('reg_weight', 0.0001),
            'device': 'cuda' if self.use_gpu else 'cpu',
            'gpu_id': str(self.config['training']['gpu']),
            'checkpoint_dir': str(self.base_output / model_name.lower()),
            'state': 'INFO',
            'show_progress': True,
            'seed': self.config['training']['seed']
        }
        
        # Model-specific parameters
        if model_name in ['DeepFM', 'xDeepFM', 'NFM']:
            config_dict.update({
                'mlp_hidden_size': params.get('mlp_hidden_size', [128, 64]),
                'dropout_prob': params.get('dropout', 0.2)
            })
        
        if model_name == 'AFM':
            config_dict.update({
                'attention_size': params.get('attention_size', 64),
                'dropout_prob': params.get('dropout', 0.1)
            })
        
        try:
            result = run_recbole(
                model=model_name,
                dataset=dataset_name,
                config_dict=config_dict
            )
            
            valid_result = result.get('best_valid_result', {})
            
            self.results[model_name] = {
                'train_auc': valid_result.get('auc', np.nan),
                'train_logloss': valid_result.get('logloss', np.nan),
                'model_type': 'ctr'
            }
            
            print(f"\n  ✓ Training completed")
            print(f"    Validation AUC: {self.results[model_name]['train_auc']:.4f}")
            
        except Exception as e:
            print(f"\n  Training failed: {e}")
            self.results[model_name] = {
                'train_auc': np.nan,
                'train_logloss': np.nan,
                'model_type': 'ctr',
                'error': str(e)
            }
    
    # =========================================================================
    # STEP 4: GENERATE PREDICTIONS
    # =========================================================================
    
    def step4_generate_predictions(self):
        """Generate predictions for CTR models"""
        print("\n" + "="*70)
        print("STEP 4: GENERATING PREDICTIONS")
        print("="*70)
        
        print("\n  Note: Baseline predictions already generated in Step 3")
        
        # Generate predictions for each CTR model
        for model_name in self.config['models'].keys():
            if model_name in ['Random', 'Pop']:
                continue  # Skip baselines (already done)
            
            self._generate_ctr_predictions(model_name)
    
    def _generate_ctr_predictions(self, model_name: str):
        """Generate predictions for a trained CTR model"""
        if not RECBOLE_AVAILABLE:
            print(f"  Skipping {model_name}: RecBole not available")
            return
        
        print(f"\n  Generating predictions for {model_name}...")
        
        checkpoint_dir = self.base_output / model_name.lower()
        checkpoint_files = list(checkpoint_dir.glob('*.pth'))
        
        if not checkpoint_files:
            print(f"    No checkpoint found in {checkpoint_dir}")
            return
        
        # Get most recent checkpoint
        checkpoint_file = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        print(f"    Using checkpoint: {checkpoint_file.name}")
        
        try:
            # Load trained model
            config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
                model_file=str(checkpoint_file)
            )
            
            model.eval()
            device = torch.device('cuda' if self.use_gpu else 'cpu')
            model = model.to(device)
            
            # Get test user-context pairs
            col_names = self._get_column_names()
            test_df = self.test_df.copy()
            user_contexts = test_df[[col_names['user']] + self.CONTEXT_FEATURES].drop_duplicates()
            
            # Get all items
            all_items = list(dataset.field2id_token['item_id'])
            num_items = len(all_items)
            top_k = self.config['evaluation']['top_k']
            
            print(f"    Scoring {num_items} items for {len(user_contexts):,} queries...")
            
            predictions = []
            skipped = 0
            
            for idx, row in tqdm(user_contexts.iterrows(), total=len(user_contexts),
                                desc="    Predicting", leave=False):
                user_orig = str(row[col_names['user']])
                
                # Check if user is in dataset
                if user_orig not in dataset.field2token_id['user_id']:
                    skipped += 1
                    continue
                
                user_id = dataset.field2token_id['user_id'][user_orig]
                
                # Map context features
                context_ids = {}
                skip_query = False
                
                for feat in self.CONTEXT_FEATURES:
                    feat_value = str(row[feat])
                    if feat_value not in dataset.field2token_id.get(feat, {}):
                        skip_query = True
                        break
                    context_ids[feat] = dataset.field2token_id[feat][feat_value]
                
                if skip_query:
                    skipped += 1
                    continue
                
                # Create interaction batch for all items
                interaction_dict = {
                    'user_id': torch.full((num_items,), user_id, dtype=torch.long),
                    'item_id': torch.arange(num_items, dtype=torch.long)
                }
                
                # Add context features
                for feat in self.CONTEXT_FEATURES:
                    interaction_dict[feat] = torch.full(
                        (num_items,), context_ids[feat], dtype=torch.long
                    )
                
                interaction = Interaction(interaction_dict).to(device)
                
                # Get predictions
                with torch.no_grad():
                    scores = model.predict(interaction)
                    
                    if isinstance(scores, torch.Tensor):
                        scores = scores.cpu().numpy()
                    
                    # Ensure 1D array
                    if len(scores.shape) > 1:
                        scores = scores.squeeze()
                    
                    # Safety check
                    if len(scores) != num_items:
                        if len(scores) > num_items:
                            scores = scores[:num_items]
                        else:
                            scores = np.pad(scores, (0, num_items - len(scores)),
                                          constant_values=-np.inf)
                
                # Get top-K items
                top_indices = np.argsort(-scores)[:top_k]
                
                # Create q_context_id
                q_context_id = '_'.join([str(row[f]) for f in self.CONTEXT_FEATURES])
                
                # Store predictions
                for rank, item_idx in enumerate(top_indices):
                    if item_idx >= num_items:
                        continue
                    
                    predictions.append({
                        'user_id:token': user_orig,
                        'item_id:token': str(all_items[item_idx]),
                        'q_context_id': q_context_id,
                        'prediction': float(scores[item_idx]),
                        'rank': rank + 1
                    })
            
            if skipped > 0:
                print(f"   Skipped {skipped} queries (not in vocab)")
            
            # Convert to DataFrame
            pred_df = pd.DataFrame(predictions)
            
            # Add item context information
            item_context = self._get_item_context_info()
            
            pred_df['item_id:token'] = pred_df['item_id:token'].astype(str).str.strip()
            item_context['item_id:token'] = item_context['item_id:token'].astype(str).str.strip()
            
            final_df = pred_df.merge(item_context, on='item_id:token', how='left')
            
            # Reorder columns
            cols = ['user_id:token', 'item_id:token', 'q_context_id',
                   'prediction', 'rank'] + self.CONTEXT_FEATURES
            final_df = final_df[[c for c in cols if c in final_df.columns]]
            
            # Save predictions
            output_dir = checkpoint_dir / 'result'
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f'{model_name}_final_predictions.tsv'
            final_df.to_csv(output_path, sep='\t', index=False)
            
            print(f"    ✓ {len(final_df):,} predictions saved to:")
            print(f"      {output_path}")
            
        except Exception as e:
            print(f"    Failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_item_context_info(self) -> pd.DataFrame:
        """
        Get item context information from training data.
        Uses mode (most common) value for each item-context pair.
        """
        col_names = self._get_column_names()
        
        item_context = (
            self.train_df.groupby(col_names['item'])[self.CONTEXT_FEATURES]
            .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
            .reset_index()
            .rename(columns={col_names['item']: 'item_id:token'})
        )
        
        return item_context
    
    # =========================================================================
    # MAIN EXECUTION
    # =========================================================================
    
    def run(self) -> bool:
        """Execute complete pipeline"""
        print("\n" + "="*70)
        print(f"{self.config['dataset']['name'].upper()} PIPELINE")
        print("="*70)
        print(f"  Timestamp: {self.timestamp}")
        print(f"  GPU: {'Available' if GPU_AVAILABLE else 'Not available'}")
        
        try:
            self.step1_load_data()
            self.step2_prepare_recbole_data()
            self.step3_train_models()
            self.step4_generate_predictions()
            
            print("\n" + "="*70)
            print("✓ PIPELINE COMPLETED!")
            print("="*70)
            print(f"\nOutputs saved to: {self.base_output}")
            print("\nNext step: Run evaluation")
            print(f"  python -m evaluators.evaluate_{self.config['dataset']['name'].lower()}")
            
            return True
            
        except Exception as e:
            print(f"\n{'='*70}")
            print(" PIPELINE FAILED")
            print('='*70)
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False