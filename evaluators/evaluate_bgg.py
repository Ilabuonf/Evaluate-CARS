"""
Complete BGG Evaluator - ALL Metrics
=====================================================

Computes ALL metrics:
1. Traditional: AUC, LogLoss, nDCG@5, nDCG@10, MAP@10, MRR@10, P@5, R@10
2. Context Similarity: ACC@5, CS@5, WCS@5, WCA@5, Friction@5
3. Advanced: CR@5, CRC@5, CGB@5
4. Dimensional: WCS by feature groups
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

# Import optimized metric modules
from src.metrics import (
    compute_acc,
    compute_cs_wcs,
    compute_similarity_metrics,
    compute_context_recall,
    compute_context_ranking_correlation,
    compute_context_group_balance
)

try:
    from ranx import Qrels, Run, evaluate
    RANX_AVAILABLE = True
except ImportError:
    RANX_AVAILABLE = False

try:
    from sklearn.metrics import roc_auc_score, log_loss
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class CompleteBGGEvaluator:
    """Complete BGG evaluator with proper prediction handling"""
    
    CONTEXT_FEATURES = ['playing_time', 'gaming_mood', 'social_companion']
    
    FEATURE_GROUPS = {
        'temporal': ['playing_time'],
        'experiential': ['gaming_mood'],
        'social': ['social_companion']
    }
    
    def __init__(self, config: Dict):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
    def load_test_set(self):
        """Load BGG test set"""
        print("="*70)
        print("LOADING BGG TEST SET")
        print("="*70)
        
        test_path = Path(self.config['test_path'])
        self.test_df = pd.read_csv(test_path, sep='\t')
        
        # Normalize data types
        self.test_df['user_id:token'] = self.test_df['user_id:token'].astype(str).str.strip()
        self.test_df['game_id:token'] = self.test_df['game_id:token'].astype(str).str.strip()
        self.test_df['context_id'] = self.test_df['context_id'].astype(str).str.strip()
        
        # Create query_id (user + context_id for matching)
        self.test_df['query_id'] = (
            self.test_df['user_id:token'] + '_' + self.test_df['context_id']
        )
        
        # Build ground truth for ranking metrics
        self.ground_truth = {}
        for _, row in self.test_df.iterrows():
            qid = row['query_id']
            item = row['game_id:token']
            rating = row['rating:float']
            
            if qid not in self.ground_truth:
                self.ground_truth[qid] = {}
            self.ground_truth[qid][item] = rating
        
        print(f"✓ Loaded: {self.test_df.shape}")
        print(f"  Queries: {len(self.ground_truth):,}")
        print(f"  Items: {self.test_df['game_id:token'].nunique():,}")
        print()
    
    def load_context_info(self):
        """Load or construct item-context mapping"""
        print("Loading context information...")
        
        # Try to load from context_info.tsv first
        ctx_path = Path(self.config['context_info_path'])
        
        if ctx_path.exists():
            context_info = pd.read_csv(ctx_path, sep='\t')
            
            # Check if needs reconstruction
            if 'playing_time' not in context_info.columns:
                print("  Reconstructing from One-Hot...")
                context_info = self._reconstruct_categorical(context_info)
        else:
            context_info = None
        
        # Load training data to create item-context mapping
        train_path = Path(self.config['test_path']).parent / 'train_df.tsv'
        
        if train_path.exists():
            print("  Loading training data for item-context mapping...")
            train_df = pd.read_csv(train_path, sep='\t')
            
            # Load context_info if we have it, otherwise use train directly
            if context_info is not None and 'context_id' in context_info.columns:
                # Merge train with context definitions
                train_with_ctx = train_df.merge(
                    context_info[['context_id'] + self.CONTEXT_FEATURES],
                    on='context_id',
                    how='left'
                )
            else:
                # Try to extract from train if features are there
                if all(f in train_df.columns for f in self.CONTEXT_FEATURES):
                    train_with_ctx = train_df
                else:
                    print("  ⚠ Warning: Cannot find context features")
                    train_with_ctx = train_df
                    for feat in self.CONTEXT_FEATURES:
                        if feat not in train_with_ctx.columns:
                            train_with_ctx[feat] = 'unknown'
            
            # Aggregate context features per item (mode)
            self.context_info = (
                train_with_ctx.groupby('game_id:token')[self.CONTEXT_FEATURES]
                .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
                .reset_index()
                .rename(columns={'game_id:token': 'item_id:token'})
            )
        else:
            print("  Warning: No training data found")
            # Create dummy context info
            unique_items = self.test_df['game_id:token'].unique()
            self.context_info = pd.DataFrame({
                'item_id:token': unique_items
            })
            for feat in self.CONTEXT_FEATURES:
                self.context_info[feat] = 'unknown'
        
        # Ensure string types
        self.context_info['item_id:token'] = self.context_info['item_id:token'].astype(str).str.strip()
        for feat in self.CONTEXT_FEATURES:
            if feat in self.context_info.columns:
                self.context_info[feat] = self.context_info[feat].astype(str).str.strip()
        
        print(f"✓ Context info: {len(self.context_info)} items")
        print()
    
    def _reconstruct_categorical(self, df):
        """Reconstruct categorical from One-Hot encoding"""
        def collapse(dataframe, prefix, new_name):
            cols = [c for c in dataframe.columns if c.startswith(prefix)]
            if not cols:
                return dataframe
            dataframe[new_name] = (
                dataframe[cols].idxmax(axis=1)
                .str.replace(prefix, "").str.replace(":float", "")
            )
            return dataframe
        
        df = collapse(df, "playing_time_", "playing_time")
        df = collapse(df, "gaming_mood_", "gaming_mood")
        df = collapse(df, "social_companion_", "social_companion")
        return df
    
    def evaluate_model(self, model_name: str, pred_path: Path) -> Dict:
        """Evaluate single model"""
        print(f"  → {model_name}")
        
        if not pred_path.exists():
            print(f"      ✗ File not found: {pred_path}")
            return {}
        
        # Load predictions
        try:
            pred_df = pd.read_csv(pred_path, sep='\t')
        except Exception as e:
            print(f"      Failed to load: {e}")
            return {}
        
        # Check if empty
        if len(pred_df) == 0:
            print(f"      File is empty (only header)")
            return {}
        
        print(f"      Loaded {len(pred_df):,} predictions")
        print(f"      Columns: {list(pred_df.columns)[:5]}...")
        
        results = {}
        
        # Run evaluations
        print(f"      Traditional metrics...")
        results.update(self._eval_traditional(pred_df))
        
        print(f"      Context metrics...")
        results.update(self._eval_context(pred_df))
        
        print(f"      Advanced metrics...")
        results.update(self._eval_advanced(pred_df))
        
        print(f"      Dimensional...")
        results.update(self._eval_dimensional(pred_df))
        
        print(f"      ✓ Completed")
        return results
    
    def _eval_traditional(self, pred_df):
        """Traditional metrics (AUC, LogLoss, nDCG, etc.)"""
        results = {}
        
        # Normalize columns
        pred_df = pred_df.copy()
        pred_df['user_id:token'] = pred_df['user_id:token'].astype(str).str.strip()
        pred_df['item_id:token'] = pred_df['item_id:token'].astype(str).str.strip()
        pred_df['q_context_id'] = pred_df['q_context_id'].astype(str).str.strip()
        
        # Create query_id to match test set
        # Assume q_context_id is either context_id or context features
        pred_df['query_id'] = pred_df['user_id:token'] + '_' + pred_df['q_context_id']
        
        # AUC & LogLoss
        if SKLEARN_AVAILABLE:
            test_subset = self.test_df[['query_id', 'game_id:token', 'rating:float']].copy()
            threshold = self.config.get('rating_threshold', 7.0)
            test_subset['label'] = (test_subset['rating:float'] >= threshold).astype(int)
            
            merged = pred_df.merge(
                test_subset[['query_id', 'game_id:token', 'label']],
                left_on=['query_id', 'item_id:token'],
                right_on=['query_id', 'game_id:token'],
                how='inner'
            )
            
            if len(merged) > 0 and len(np.unique(merged['label'])) > 1:
                try:
                    results['AUC'] = roc_auc_score(merged['label'], merged['prediction'])
                except:
                    results['AUC'] = np.nan
                
                y_norm = np.clip(
                    (merged['prediction'] - merged['prediction'].min()) / 
                    (merged['prediction'].max() - merged['prediction'].min() + 1e-10),
                    1e-10, 1 - 1e-10
                )
                try:
                    results['LogLoss'] = log_loss(merged['label'], y_norm)
                except:
                    results['LogLoss'] = np.nan
        
        # Ranking metrics
        if RANX_AVAILABLE:
            qrels = Qrels({qid: {i: int(r) for i, r in items.items()} 
                          for qid, items in self.ground_truth.items()})
            
            run_dict = (pred_df.groupby('query_id')
                       .apply(lambda g: dict(zip(g['item_id:token'], g['prediction'])))
                       .to_dict())
            
            run = Run(run_dict)
            
            try:
                ranx_res = evaluate(qrels, run,
                                   ['ndcg@5', 'ndcg@10', 'map@10', 'mrr@10', 
                                    'precision@5', 'recall@10'],
                                   make_comparable=True)
                
                results['nDCG@5'] = ranx_res.get('ndcg@5', np.nan)
                results['nDCG@10'] = ranx_res.get('ndcg@10', np.nan)
                results['MAP@10'] = ranx_res.get('map@10', np.nan)
                results['MRR@10'] = ranx_res.get('mrr@10', np.nan)
                results['P@5'] = ranx_res.get('precision@5', np.nan)
                results['R@10'] = ranx_res.get('recall@10', np.nan)
            except Exception as e:
                print(f"        Ranx error: {e}")
                for m in ['nDCG@5', 'nDCG@10', 'MAP@10', 'MRR@10', 'P@5', 'R@10']:
                    results[m] = np.nan
        
        return results
    
    def _eval_context(self, pred_df):
        """Context similarity metrics"""
        # Check if has context features
        has_context = any(f in pred_df.columns for f in self.CONTEXT_FEATURES)
        
        if not has_context:
            print(f"        Baseline - skipping")
            return {}
        
        try:
            results = {}
            k = [5]
            
            results.update(compute_acc(pred_df, self.context_info, self.CONTEXT_FEATURES, k))
            results.update(compute_cs_wcs(pred_df, self.context_info, self.CONTEXT_FEATURES, 0.5, k))
            results.update(compute_similarity_metrics(pred_df, self.context_info, self.CONTEXT_FEATURES, k))
            
            return results
        except Exception as e:
            print(f"        Error: {e}")
            return {}
    
    def _eval_advanced(self, pred_df):
        """Advanced context metrics"""
        has_context = any(f in pred_df.columns for f in self.CONTEXT_FEATURES)
        if not has_context:
            return {}
        
        try:
            results = {}
            k = [5]
            
            results.update(compute_context_recall(pred_df, self.context_info, self.CONTEXT_FEATURES, k))
            results.update(compute_context_ranking_correlation(pred_df, self.context_info, self.CONTEXT_FEATURES, k))
            results.update(compute_context_group_balance(pred_df, self.context_info, 
                                                        self.CONTEXT_FEATURES, self.FEATURE_GROUPS, k))
            return results
        except Exception as e:
            print(f"        Error: {e}")
            return {}
    
    def _eval_dimensional(self, pred_df):
        """Dimensional WCS metrics"""
        has_context = any(f in pred_df.columns for f in self.CONTEXT_FEATURES)
        if not has_context:
            return {}
        
        try:
            results = {}
            k = [5]
            
            for group_name, group_feats in self.FEATURE_GROUPS.items():
                if not group_feats:
                    continue
                
                wcs_res = compute_cs_wcs(pred_df, self.context_info, group_feats, 0.5, k)
                if 'WCS@5' in wcs_res:
                    results[f'WCS_{group_name}@5'] = wcs_res['WCS@5']
            
            return results
        except Exception as e:
            print(f"        Error: {e}")
            return {}
    
    def evaluate_all_models(self):
        """Evaluate all models"""
        print("\n" + "="*70)
        print("EVALUATING ALL MODELS")
        print("="*70)
        print()
        
        results_dir = Path(self.config['results_dir'])
        exclude = {'__pycache__', 'context_metrics', 'evaluation', 'complete_metrics'}
        model_dirs = [d for d in results_dir.iterdir() 
                     if d.is_dir() and d.name not in exclude and not d.name.startswith('.')]
        
        print(f"Found {len(model_dirs)} model directories\n")
        
        for model_dir in sorted(model_dirs):
            model_name = model_dir.name.capitalize()
            
            pred_file = model_dir / 'result' / f'{model_name}_final_predictions.tsv'
            if not pred_file.exists():
                pred_files = list((model_dir / 'result').glob('*predictions.tsv'))
                if pred_files:
                    pred_file = pred_files[0]
                else:
                    print(f"  ⚠ {model_name}: No predictions\n")
                    continue
            
            results = self.evaluate_model(model_name, pred_file)
            if results:
                self.results[model_name] = results
            print()
    
    def save_results(self):
        """Save results"""
        print("="*70)
        print("SAVING RESULTS")
        print("="*70)
        
        if not self.results:
            print("✗ No results")
            return None
        
        df = pd.DataFrame(self.results).T.round(4)
        
        # Order columns by table
        cols_order = (
            [c for c in ['AUC', 'LogLoss', 'nDCG@5', 'nDCG@10', 'MAP@10', 'MRR@10', 'P@5', 'R@10'] if c in df.columns] +
            [c for c in ['ACC@5', 'CS@5', 'WCS@5', 'WCA@5', 'Friction@5'] if c in df.columns] +
            [c for c in ['CR@5', 'CRC@5', 'CGB@5'] if c in df.columns] +
            [c for c in df.columns if 'WCS_' in c]
        )
        
        df = df[[c for c in cols_order if c in df.columns]]
        
        csv_path = self.output_dir / f"bgg_all_metrics_{self.timestamp}.csv"
        df.to_csv(csv_path)
        
        print(f"✓ Saved: {csv_path}\n")
        print(df.to_string())
        print()
        
        return df
    
    def run(self):
        """Execute evaluation"""
        print("\n" + "="*70)
        print("BGG COMPLETE EVALUATOR")
        print("="*70)
        print()
        
        try:
            self.load_test_set()
            self.load_context_info()
            self.evaluate_all_models()
            self.save_results()
            
            print("\n" + "="*70)
            print("✓ COMPLETED")
            print("="*70)
            return True
        except Exception as e:
            print(f"\n✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == '__main__':
    config = {
        'test_path': './datasets/bgg/test_df.tsv',
        'context_info_path': './datasets/bgg/context_info.tsv',
        'results_dir': './outputs/bgg',
        'output_dir': './results/bgg/complete_metrics',
        'rating_threshold': 7.0
    }
    
    evaluator = CompleteBGGEvaluator(config)
    evaluator.run()