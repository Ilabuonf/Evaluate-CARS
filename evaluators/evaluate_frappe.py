"""
Complete Frappe Evaluator - ALL Metrics 
=========================================================

Computes ALL metrics:
- Traditional: AUC, LogLoss, nDCG, MAP, MRR, P, R
- Context Similarity: ACC, CS, WCS (+ @all), WCA, Friction (+ @all)
- Advanced: CR (+ @all), CRC (+ @all), CGB (+ @all)
- Context-Weighted: CW-nDCG, CW-MAP
- Dimensional: WCS by feature groups 
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from src.metrics import (
    compute_acc,
    compute_cs_wcs,
    compute_similarity_metrics,
    compute_context_recall,
    compute_context_ranking_correlation,
    compute_context_group_balance
)

from src.metrics.weighted_ranking import (
    compute_context_weighted_ndcg,
    compute_context_weighted_map
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


class CompleteFrappeEvaluator:
    """Complete Frappe evaluator with ALL metrics - FIXED context_id handling"""
    
    CONTEXT_FEATURES = [
        'daytime', 'weekday', 'isweekend',
        'homework', 'cost',
        'weather', 'country', 'city'
    ]
    
    FEATURE_GROUPS = {
        'temporal': ['daytime', 'weekday', 'isweekend'],
        'activity': ['homework', 'cost'],
        'environment': ['weather', 'country', 'city']
    }
    
    def __init__(self, config: Dict):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.context_lookup = None  # Will store context_id → features mapping
    
    def _smart_read_csv(self, filepath):
        """Read CSV with auto separator detection"""
        for sep in ['\t', ',']:
            try:
                df = pd.read_csv(filepath, sep=sep)
                if len(df.columns) > 1:
                    return df
            except:
                continue
        return pd.read_csv(filepath)
    
    def load_test_set(self):
        """Load Frappe test set and create context lookup"""
        print("="*70)
        print("LOADING FRAPPE TEST SET")
        print("="*70)
        
        test_path = Path(self.config['test_path'])
        self.test_df = self._smart_read_csv(test_path)
        
        # Detect column names
        if 'user_id:token' in self.test_df.columns:
            user_col, item_col = 'user_id:token', 'item_id:token'
        elif 'user' in self.test_df.columns:
            user_col, item_col = 'user', 'item'
        else:
            raise ValueError(f"Unknown columns: {list(self.test_df.columns)}")
        
        self.test_df[user_col] = self.test_df[user_col].astype(str).str.strip()
        self.test_df[item_col] = self.test_df[item_col].astype(str).str.strip()
        self.user_col, self.item_col = user_col, item_col
        
        # Convert context features to string
        for feat in self.CONTEXT_FEATURES:
            if feat in self.test_df.columns:
                self.test_df[feat] = self.test_df[feat].astype(str).str.strip()
        
        # Create context_id → features lookup table
        # Try to load from context_info_with_id.tsv first (from pipeline)
        context_lookup_path = test_path.parent / 'context_info_with_id.tsv'
        
        if context_lookup_path.exists():
            print(f"  ✓ Loading context lookup from: {context_lookup_path.name}")
            context_lookup_df = pd.read_csv(context_lookup_path, sep='\t')
            
            # Convert all columns to string for consistent merge
            for feat in self.CONTEXT_FEATURES:
                if feat in context_lookup_df.columns:
                    context_lookup_df[feat] = context_lookup_df[feat].astype(str).str.strip()
            
            self.context_lookup = context_lookup_df.set_index('context_id')
            print(f"  ✓ Context lookup table loaded: {len(self.context_lookup)} unique contexts")
            
            # If test_df doesn't have context_id, add it
            if 'context_id' not in self.test_df.columns:
                print(f"  Adding context_id to test set...")
                self.test_df = self.test_df.merge(
                    context_lookup_df,
                    on=self.CONTEXT_FEATURES,
                    how='left'
                )
        elif 'context_id' in self.test_df.columns:
            self.context_lookup = (
                self.test_df[['context_id'] + self.CONTEXT_FEATURES]
                .drop_duplicates()
                .copy()
            )
            self.context_lookup['context_id'] = self.context_lookup['context_id'].astype(str)
            self.context_lookup = self.context_lookup.set_index('context_id')
            
            print(f"  ✓ Context lookup table created from test set: {len(self.context_lookup)} unique contexts")
        else:
            # Create context_id from feature concatenation
            print(f"  No context_id found - creating from feature values")
            
            # Create synthetic context_id by concatenating features
            self.test_df['context_id'] = (
                self.test_df[self.CONTEXT_FEATURES]
                .astype(str)
                .apply('_'.join, axis=1)
            )
            
            self.context_lookup = (
                self.test_df[['context_id'] + self.CONTEXT_FEATURES]
                .drop_duplicates()
                .copy()
            )
            self.context_lookup = self.context_lookup.set_index('context_id')
            
            print(f"  ✓ Context lookup table created: {len(self.context_lookup)} unique contexts (from features)")
        
        # Create query_id for evaluation
        context_str = self.test_df[self.CONTEXT_FEATURES].astype(str).apply('_'.join, axis=1)
        self.test_df['query_id'] = self.test_df[user_col] + '_' + context_str
        
        # VECTORIZED ground truth
        self.ground_truth = (
            self.test_df.groupby('query_id')
            .apply(lambda g: dict(zip(g[item_col], g['label'].astype(int))))
            .to_dict()
        )
        
        print(f"✓ Loaded: {self.test_df.shape}, Queries: {len(self.ground_truth):,}\n")
    
    def load_context_info(self):
        """Load context info from training data"""
        print("Loading context information...")
        
        train_files = [
            Path(self.config.get('train_path', './datasets/frappe/frappe_train.csv')),
            Path(self.config['test_path']).parent / 'frappe_train.csv'
        ]
        
        train_df = None
        for train_file in train_files:
            if train_file.exists():
                try:
                    train_df = self._smart_read_csv(train_file)
                    print(f"  Loaded: {train_file.name}")
                    break
                except:
                    pass
        
        if train_df is None:
            raise FileNotFoundError("Could not find Frappe training data")
        
        item_col = 'item' if 'item' in train_df.columns else self.item_col
        
        # Convert context features to string
        for feat in self.CONTEXT_FEATURES:
            if feat in train_df.columns:
                train_df[feat] = train_df[feat].astype(str).str.strip()
        
        # Aggregate by item (mode)
        self.context_info = (
            train_df.groupby(item_col)[self.CONTEXT_FEATURES]
            .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
            .reset_index()
            .rename(columns={item_col: 'item_id:token'})
        )
        self.context_info['item_id:token'] = self.context_info['item_id:token'].astype(str).str.strip()
        
        print(f"✓ Context info: {len(self.context_info)} items\n")
    
    def _map_context_id_to_features(self, q_context_id: str) -> Dict[str, str]:
        """
        Map q_context_id to feature values using lookup table.
        
        Args:
            q_context_id: Can be:
                - Single context_id: "123"
                - Composite context_ids: "5041_5046_5053_..."
                - Feature values: "morning_monday_0_0_0_sunny_US_NewYork"
        
        Returns:
            Dict mapping feature names to values
        """
        if self.context_lookup is None:
            return {f: '' for f in self.CONTEXT_FEATURES}
        
        # Strategy 1: Try exact match (works for feature concatenation)
        try:
            if q_context_id in self.context_lookup.index:
                return self.context_lookup.loc[q_context_id].to_dict()
        except:
            pass
        
        # Strategy 2: Try first part of composite ID
        try:
            first_id = str(q_context_id).split('_')[0]
            if first_id in self.context_lookup.index:
                return self.context_lookup.loc[first_id].to_dict()
        except:
            pass
        
        # Strategy 3: Assume q_context_id IS the feature concatenation
        # Split and map to feature names
        try:
            parts = str(q_context_id).split('_')
            if len(parts) == len(self.CONTEXT_FEATURES):
                return dict(zip(self.CONTEXT_FEATURES, parts))
        except:
            pass
        
        # Fallback: empty values
        return {f: '' for f in self.CONTEXT_FEATURES}
    
    def _add_query_features_to_predictions(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add query context features to predictions using context_id lookup.
        
        This replaces the broken parsing approach.
        """
        pred_with_features = pred_df.copy()
        
        # Map each q_context_id to its feature values
        for feat in self.CONTEXT_FEATURES:
            pred_with_features[feat] = pred_with_features['q_context_id'].apply(
                lambda x: self._map_context_id_to_features(str(x)).get(feat, '')
            )
        
        return pred_with_features
    
    def evaluate_model(self, model_name: str, pred_path: Path) -> Dict:
        """Evaluate single model with ALL metrics"""
        print(f"  → {model_name}")
        
        pred_df = pd.read_csv(pred_path, sep='\t')
        
        if len(pred_df) == 0:
            print(f"   Empty predictions file")
            return {}
        
        print(f"      Loaded {len(pred_df):,} predictions")
        
        results = {}
        
        # Traditional metrics
        print(f"      Traditional...")
        results.update(self._evaluate_traditional(pred_df))
        
        # Context similarity metrics
        print(f"      Context metrics...")
        k_val = [5]
        try:
            # Add query features to predictions first
            pred_with_features = self._add_query_features_to_predictions(pred_df)
            
            # ACC
            results.update(compute_acc(pred_with_features, self.context_info, 
                                      self.CONTEXT_FEATURES, k_val))
            
            # CS, WCS (both @K and @all)
            cs_wcs_res = compute_cs_wcs(pred_with_features, self.context_info, 
                                        self.CONTEXT_FEATURES, 
                                        alpha=0.5, k_values=k_val)
            results.update(cs_wcs_res)
            
            # WCA, Friction (both @K and @all)
            sim_res = compute_similarity_metrics(pred_with_features, self.context_info, 
                                                 self.CONTEXT_FEATURES, k_val)
            results.update(sim_res)
            
            # CR (+ @all)
            cr_res = compute_context_recall(pred_with_features, self.context_info, 
                                           self.CONTEXT_FEATURES, k_val)
            results.update(cr_res)
            
            # CRC (+ @all)
            crc_res = compute_context_ranking_correlation(pred_with_features, self.context_info, 
                                                          self.CONTEXT_FEATURES, k_val)
            results.update(crc_res)
            
            # CGB (+ @all)
            cgb_res = compute_context_group_balance(pred_with_features, self.context_info, 
                                                   self.CONTEXT_FEATURES, 
                                                   self.FEATURE_GROUPS, k_val)
            results.update(cgb_res)
            
        except Exception as e:
            print(f"        Context metrics error: {e}")
            import traceback
            traceback.print_exc()
        
        # CW-nDCG, CW-MAP
        print(f"      CW-nDCG, CW-MAP...")
        try:
            # Use pred_with_features if available, otherwise original
            cw_pred = pred_with_features if 'pred_with_features' in locals() else pred_df.copy()
            
            cw_pred['user_id:token'] = cw_pred['user_id:token'].astype(str).str.strip()
            cw_pred['item_id:token'] = cw_pred['item_id:token'].astype(str).str.strip()
            cw_pred['q_context_id'] = cw_pred['q_context_id'].astype(str).str.strip()
            cw_pred['query_id'] = cw_pred['user_id:token'] + '_' + cw_pred['q_context_id']
            
            # Add labels from ground truth
            gt_list = [
                {'query_id': qid, 'item_id:token': item, 'label': label}
                for qid, items in self.ground_truth.items()
                for item, label in items.items()
            ]
            
            if gt_list:
                gt_df = pd.DataFrame(gt_list)
                cw_pred = cw_pred.merge(gt_df, on=['query_id', 'item_id:token'], how='left')
                cw_pred['label'] = cw_pred['label'].fillna(0).astype(int)
            else:
                cw_pred['label'] = 0
            
            # Compute CW metrics
            results.update(compute_context_weighted_ndcg(cw_pred, self.context_info, 
                                                        self.CONTEXT_FEATURES, 
                                                        k_values=[5, 10, 20]))
            results.update(compute_context_weighted_map(cw_pred, self.context_info, 
                                                       self.CONTEXT_FEATURES, 
                                                       k_values=[5, 10, 20]))
        except Exception as e:
            print(f"        CW error: {e}")
        
        # Dimensional analysis
        print(f"      Dimensional...")
        results.update(self._evaluate_dimensional(pred_df))
        
        print(f"      ✓ Done")
        return results
    
    def _evaluate_traditional(self, pred_df):
        """Traditional metrics: AUC, LogLoss, nDCG, MAP, MRR, P, R"""
        res = {}
        
        pred_df = pred_df.copy()
        pred_df['user_id:token'] = pred_df['user_id:token'].astype(str).str.strip()
        pred_df['item_id:token'] = pred_df['item_id:token'].astype(str).str.strip()
        pred_df['q_context_id'] = pred_df['q_context_id'].astype(str).str.strip()
        pred_df['query_id'] = pred_df['user_id:token'] + '_' + pred_df['q_context_id']
        
        # AUC & LogLoss
        if SKLEARN_AVAILABLE:
            test_subset = self.test_df[['query_id', self.item_col, 'label']].copy()
            test_subset[self.item_col] = test_subset[self.item_col].astype(str).str.strip()
            
            merged = pred_df.merge(
                test_subset,
                left_on=['query_id', 'item_id:token'],
                right_on=['query_id', self.item_col],
                how='inner'
            )
            
            if len(merged) > 0 and len(np.unique(merged['label'])) > 1:
                y_true, y_scores = merged['label'].values, merged['prediction'].values
                
                try:
                    res['AUC'] = roc_auc_score(y_true, y_scores)
                except:
                    res['AUC'] = np.nan
                
                # Normalize scores for LogLoss
                y_norm = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-10)
                y_norm = np.clip(y_norm, 1e-10, 1 - 1e-10)
                try:
                    res['LogLoss'] = log_loss(y_true, y_norm)
                except:
                    res['LogLoss'] = np.nan
        
        # Ranking metrics
        if RANX_AVAILABLE:
            qrels = Qrels(self.ground_truth)
            
            # VECTORIZED run dict
            run_dict = pred_df.groupby('query_id').apply(
                lambda g: dict(zip(g['item_id:token'], g['prediction']))
            ).to_dict()
            run = Run(run_dict)
            
            try:
                ranx_res = evaluate(qrels, run, 
                                   ['ndcg@5', 'ndcg@10', 'map@10', 'mrr@10', 
                                    'precision@5', 'recall@10'],
                                   make_comparable=True)
                
                res['nDCG@5'] = ranx_res.get('ndcg@5', np.nan)
                res['nDCG@10'] = ranx_res.get('ndcg@10', np.nan)
                res['MAP@10'] = ranx_res.get('map@10', np.nan)
                res['MRR@10'] = ranx_res.get('mrr@10', np.nan)
                res['P@5'] = ranx_res.get('precision@5', np.nan)
                res['R@10'] = ranx_res.get('recall@10', np.nan)
            except Exception as e:
                print(f"      Ranx error: {e}")
                for m in ['nDCG@5', 'nDCG@10', 'MAP@10', 'MRR@10', 'P@5', 'R@10']:
                    res[m] = np.nan
        
        return res
    
    def _evaluate_dimensional(self, pred_df):
        """
        Dimensional WCS metrics using context_id lookup.
        """
        results = {}
        k = 5
        
        # Add query features using lookup table
        pred_with_features = self._add_query_features_to_predictions(pred_df)
        
        # Verify features were added
        missing_features = [f for f in self.CONTEXT_FEATURES if f not in pred_with_features.columns]
        if missing_features:
            print(f"        ⚠ Warning: Missing features after lookup: {missing_features}")
            for feat in missing_features:
                pred_with_features[feat] = ''
        
        # Now compute WCS for each group
        for g_name, g_feats in self.FEATURE_GROUPS.items():
            try:
                # Check if group features exist
                if not all(f in pred_with_features.columns for f in g_feats):
                    print(f"        {g_name}: Missing features, skipping")
                    results[f'WCS_{g_name}@{k}'] = 0.0
                    continue
                
                # Filter to top-K
                if 'rank' not in pred_with_features.columns:
                    pred_with_features['rank'] = (
                        pred_with_features.groupby(['user_id:token', 'q_context_id'])['prediction']
                        .rank(ascending=False, method='first')
                    )
                
                top_k = pred_with_features[pred_with_features['rank'] <= k].copy()
                
                if len(top_k) == 0:
                    results[f'WCS_{g_name}@{k}'] = 0.0
                    continue
                
                # Prepare item context for this group
                ctx_subset = self.context_info[['item_id:token'] + g_feats].copy()
                ctx_subset['item_id:token'] = ctx_subset['item_id:token'].astype(str).str.strip()
                top_k['item_id:token'] = top_k['item_id:token'].astype(str).str.strip()
                
                # Merge item context
                merged = top_k.merge(ctx_subset, on='item_id:token', how='left', 
                                    suffixes=('_query', '_item'))
                
                # Compute simple WCS: fraction of matching features
                scores = []
                for _, row in merged.iterrows():
                    matches = 0
                    for f in g_feats:
                        q_val = str(row.get(f'_query' if f'{f}_query' not in row else f'{f}_query', 
                                           row.get(f, ''))).strip()
                        i_val = str(row.get(f'{f}_item', '')).strip()
                        
                        # Handle case where features are already in row without suffix
                        if f'{f}_query' not in row and f'{f}_item' not in row:
                            q_val = str(row.get(f, '')).strip()
                            i_val = str(row.get(f'{f}_item', '')).strip()
                        
                        if q_val and i_val and q_val == i_val:
                            matches += 1
                    
                    scores.append(matches / len(g_feats) if g_feats else 0.0)
                
                wcs_score = float(np.mean(scores)) if scores else 0.0
                results[f'WCS_{g_name}@{k}'] = wcs_score
                print(f"        {g_name}: {wcs_score:.4f}")
                
            except Exception as e:
                print(f"        {g_name} error: {e}")
                import traceback
                traceback.print_exc()
                results[f'WCS_{g_name}@{k}'] = 0.0
        
        return results
    
    def evaluate_all_models(self):
        """Evaluate all models"""
        print("\n" + "="*70)
        print("EVALUATING ALL MODELS")
        print("="*70)
        print()
        
        r_dir = Path(self.config['results_dir'])
        exclude = {'.', '__pycache__', 'context_metrics', 'evaluation', 'complete_metrics'}
        model_dirs = [d for d in r_dir.iterdir() 
                     if d.is_dir() and d.name not in exclude and not d.name.startswith('.')]
        
        print(f"Found {len(model_dirs)} model directories\n")
        
        for md in sorted(model_dirs):
            model_name = md.name.capitalize()
            
            # Find predictions file
            p_file = md / 'result' / f'{model_name}_final_predictions.tsv'
            if not p_file.exists():
                # Try alternative names
                pred_files = list((md / 'result').glob('*predictions.tsv'))
                if pred_files:
                    p_file = pred_files[0]
                else:
                    print(f"  ⚠ {model_name}: No predictions found\n")
                    continue
            
            results = self.evaluate_model(model_name, p_file)
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
        
        # Order columns logically
        col_categories = {
            'Traditional': ['AUC', 'LogLoss', 'nDCG@5', 'nDCG@10', 'MAP@10', 'MRR@10', 'P@5', 'R@10'],
            'Context_Sim': ['ACC@5', 'CS@5', 'WCS@5', 'CS@all', 'WCS@all', 'WCA@5', 'Friction@5', 
                           'WCA@all', 'Friction@all'],
            'Advanced': ['CR@5', 'CR@all', 'CRC@5', 'CRC@all', 'CGB@5', 'CGB@all'],
            'CW_Ranking': ['CW-nDCG@5', 'CW-nDCG@10', 'CW-nDCG@20', 'CW-MAP@5', 'CW-MAP@10', 'CW-MAP@20'],
            'Dimensional': [f'WCS_{g}@5' for g in self.FEATURE_GROUPS.keys()]
        }
        
        # Build ordered column list
        ordered_cols = []
        for category, cols in col_categories.items():
            ordered_cols.extend([c for c in cols if c in df.columns])
        
        # Add any remaining columns
        ordered_cols.extend([c for c in df.columns if c not in ordered_cols])
        
        df = df[ordered_cols]
        
        # Save
        csv_path = self.output_dir / f"frappe_all_metrics_{self.timestamp}.csv"
        df.to_csv(csv_path)
        
        print(f"✓ Saved: {csv_path}\n")
        print(df.to_string())
        print()
        
        # Print dimensional results separately for visibility
        print("\n" + "="*70)
        print("DIMENSIONAL ANALYSIS RESULTS")
        print("="*70)
        dim_cols = [c for c in df.columns if 'WCS_' in c and '@5' in c]
        if dim_cols:
            print("\nWCS by Feature Group (@5):")
            print(df[dim_cols].to_string())
            print()
            
            # Highlight best model per group
            print("Best model per feature group:")
            for col in dim_cols:
                best_model = df[col].idxmax()
                best_val = df[col].max()
                print(f"  {col}: {best_model} ({best_val:.4f})")
        
        return df
    
    def run(self):
        """Execute complete evaluation"""
        print("\n" + "="*70)
        print("FRAPPE COMPLETE EVALUATOR")
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
        'test_path': './datasets/frappe/frappe_test.csv',
        'train_path': './datasets/frappe/frappe_train.csv',
        'results_dir': './outputs/frappe',
        'output_dir': './results/frappe/complete_metrics'
    }
    
    evaluator = CompleteFrappeEvaluator(config)
    evaluator.run()