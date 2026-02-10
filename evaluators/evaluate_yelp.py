"""Complete Yelp Evaluator with CW Metrics"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict
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

# CW Metrics
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

class CompleteYelpEvaluator:
    CONTEXT_FEATURES = ['hour_of_day', 'day_of_week', 'is_weekend', 'city', 'category', 'price_range']
    FEATURE_GROUPS = {
        'temporal': ['hour_of_day', 'day_of_week', 'is_weekend'],
        'business': ['city', 'category', 'price_range']
    }
    
    def __init__(self, config: Dict):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def _smart_read_csv(self, filepath):
        for sep in ['\t', ',']:
            try:
                df = pd.read_csv(filepath, sep=sep)
                if len(df.columns) > 1: return df
            except: continue
        return pd.read_csv(filepath)

    def load_test_set(self):
        print("="*70 + "\nLOADING YELP TEST SET\n" + "="*70)
        test_path = Path(self.config['test_path'])
        self.test_df = self._smart_read_csv(test_path)
        
        rename_map = {'user_id': 'user_id:token', 'business_id': 'item_id:token', 'stars': 'rating'}
        self.test_df = self.test_df.rename(columns=rename_map)
        
        for col in ['user_id:token', 'item_id:token']:
            self.test_df[col] = self.test_df[col].astype(str).str.strip()
            
        ctx_str = self.test_df[self.CONTEXT_FEATURES].astype(str).apply('_'.join, axis=1)
        self.test_df['query_id'] = self.test_df['user_id:token'] + '_' + ctx_str
        
        self.ground_truth = {}
        for _, row in self.test_df.iterrows():
            qid, item, rel = row['query_id'], row['item_id:token'], row['rating']
            if qid not in self.ground_truth: self.ground_truth[qid] = {}
            self.ground_truth[qid][item] = 1.0 if rel >= self.config.get('stars_threshold', 4.0) else 0.0
            
        print(f"✓ Loaded: {self.test_df.shape}\n  Queries: {len(self.ground_truth):,}\n")

    def load_context_info(self):
        print("Loading context information...")
        train_path = Path(self.config['train_path'])
        train_df = self._smart_read_csv(train_path)
        
        item_col = 'business_id' if 'business_id' in train_df.columns else 'item_id:token'
        train_df = train_df.rename(columns={item_col: 'item_id:token'})
        
        if 'price_range' in train_df.columns:
            train_df['price_range'] = train_df['price_range'].fillna('2')

        self.context_info = train_df.groupby('item_id:token')[self.CONTEXT_FEATURES].agg(
            lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]
        ).reset_index()
        self.context_info['item_id:token'] = self.context_info['item_id:token'].astype(str).str.strip()
        
        print(f"✓ Context info: {len(self.context_info)} items\n")

    def evaluate_model(self, model_name: str, pred_path: Path) -> Dict:
        print(f"  → {model_name}")
        pred_df = pd.read_csv(pred_path, sep='\t')
        pred_df = pred_df.rename(columns={'business_id': 'item_id:token', 'user_id': 'user_id:token'})
        for col in ['user_id:token', 'item_id:token', 'q_context_id']:
            pred_df[col] = pred_df[col].astype(str).str.strip()
            
        results = {}
        
        # Traditional
        print(f"      Traditional...")
        results.update(self._evaluate_traditional(pred_df))
        
        # Context metrics
        print(f"      Context metrics...")
        k_val = [5]
        try:
            results.update(compute_acc(pred_df, self.context_info, self.CONTEXT_FEATURES, k_val))
            results.update(compute_cs_wcs(pred_df, self.context_info, self.CONTEXT_FEATURES, alpha=0.5, k_values=k_val))
            results.update(compute_similarity_metrics(pred_df, self.context_info, self.CONTEXT_FEATURES, k_val))
            results.update(compute_context_recall(pred_df, self.context_info, self.CONTEXT_FEATURES, k_val))
            results.update(compute_context_ranking_correlation(pred_df, self.context_info, self.CONTEXT_FEATURES, k_val))
            results.update(compute_context_group_balance(pred_df, self.context_info, self.CONTEXT_FEATURES, self.FEATURE_GROUPS, k_val))
        except Exception as e:
            print(f"        Error: {e}")
        
        # CW Ranking Metrics
        print(f"      CW-nDCG, CW-MAP...")
        try:
            cw_pred = pred_df.copy()
            cw_pred['query_id'] = cw_pred['user_id:token'] + '_' + cw_pred['q_context_id']
            labels = []
            for _, row in cw_pred.iterrows():
                qid, item = row['query_id'], row['item_id:token']
                labels.append(self.ground_truth.get(qid, {}).get(item, 0.0))
            cw_pred['label'] = labels
            
            cw_ndcg = compute_context_weighted_ndcg(cw_pred, self.context_info, self.CONTEXT_FEATURES, k_values=[5, 10, 20])
            results.update(cw_ndcg)
            
            cw_map = compute_context_weighted_map(cw_pred, self.context_info, self.CONTEXT_FEATURES, k_values=[5, 10, 20])
            results.update(cw_map)
        except Exception as e:
            print(f"        CW metrics error: {e}")
        
        # Dimensional
        print(f"      Dimensional...")
        for g_name, g_feats in self.FEATURE_GROUPS.items():
            try:
                w_res = compute_cs_wcs(pred_df, self.context_info, g_feats, alpha=0.5, k_values=[5])
                if 'WCS@5' in w_res: results[f'WCS_{g_name}@5'] = w_res['WCS@5']
            except: pass

        print(f"      ✓ Completed")
        return results

    def _evaluate_traditional(self, pred_df):
        res = {}
        test_subset = self.test_df[['user_id:token', 'item_id:token', 'rating']].copy()
        merged = pred_df.merge(test_subset, on=['user_id:token', 'item_id:token'], how='inner')
        
        if len(merged) > 0:
            y_true = (merged['rating'] >= 4.0).astype(int)
            if len(np.unique(y_true)) > 1:
                try:
                    res['AUC'] = roc_auc_score(y_true, merged['prediction'])
                except:
                    res['AUC'] = np.nan
                try:
                    res['LogLoss'] = log_loss(y_true, merged['prediction'], labels=[0,1])
                except:
                    res['LogLoss'] = np.nan

        if RANX_AVAILABLE:
            pred_df['query_id'] = pred_df['user_id:token'] + '_' + pred_df['q_context_id']
            qrels = Qrels(self.ground_truth)
            run_dict = {qid: dict(zip(g['item_id:token'], g['prediction'])) for qid, g in pred_df.groupby('query_id')}
            run = Run(run_dict)
            try:
                ranx_res = evaluate(qrels, run, ['ndcg@5', 'ndcg@10', 'map@10', 'mrr@10', 'precision@5', 'recall@10'], make_comparable=True)
                res['nDCG@5'], res['nDCG@10'], res['MAP@10'] = ranx_res.get('ndcg@5', np.nan), ranx_res.get('ndcg@10', np.nan), ranx_res.get('map@10', np.nan)
                res['MRR@10'], res['P@5'], res['R@10'] = ranx_res.get('mrr@10', np.nan), ranx_res.get('precision@5', np.nan), ranx_res.get('recall@10', np.nan)
            except Exception as e:
                print(f"      Ranx error: {e}")
                for m in ['nDCG@5', 'nDCG@10', 'MAP@10', 'MRR@10', 'P@5', 'R@10']: res[m] = np.nan
        return res

    def run(self):
        try:
            self.load_test_set()
            self.load_context_info()
            
            r_dir = Path(self.config['results_dir'])
            model_dirs = [d for d in r_dir.iterdir() if d.is_dir() and not d.name.startswith(('.', 'context', 'evaluation'))]
            for md in sorted(model_dirs):
                p_file = next(md.rglob('*predictions.tsv'), None)
                if p_file: self.results[md.name.capitalize()] = self.evaluate_model(md.name, p_file)
            
            if self.results:
                df = pd.DataFrame(self.results).T.round(4)
                df.to_csv(self.output_dir / f"yelp_all_metrics_{self.timestamp}.csv")
                print(f"\n✓ Saved\n{df.to_string()}")
            print("\n" + "="*70 + "\n✓ COMPLETED\n" + "="*70)
            return True
        except Exception as e:
            print(f"\n✗ Failed: {e}"); import traceback; traceback.print_exc(); return False

if __name__ == '__main__':
    config = {
        'test_path': './datasets/yelp/yelp_test.csv',
        'train_path': './datasets/yelp/yelp_train.csv',
        'results_dir': './outputs/yelp',
        'output_dir': './results/yelp/complete_metrics',
        'stars_threshold': 4.0
    }
    CompleteYelpEvaluator(config).run()