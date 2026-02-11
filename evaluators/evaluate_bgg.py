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
    compute_acc, compute_cs_wcs, compute_similarity_metrics,
    compute_context_recall, compute_context_ranking_correlation,
    compute_context_group_balance
)

from src.metrics.weighted_ranking import (
    compute_context_weighted_ndcg, compute_context_weighted_map
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
        print("="*70 + "\nLOADING BGG TEST SET\n" + "="*70)
        test_path = Path(self.config['test_path'])
        self.test_df = pd.read_csv(test_path, sep='\t')
        self.test_df = self.test_df.rename(columns={'game_id:token': 'item_id:token'})
        
        for col in ['user_id:token', 'item_id:token', 'context_id']:
            self.test_df[col] = self.test_df[col].astype(str).str.strip()
        
        self.test_df['query_id'] = self.test_df['user_id:token'] + '_' + self.test_df['context_id']
        
        # VECTORIZED ground truth creation
        self.ground_truth = (
            self.test_df.groupby('query_id')
            .apply(lambda g: dict(zip(g['item_id:token'], g['rating:float'])))
            .to_dict()
        )
        
        print(f"✓ Loaded: {self.test_df.shape}, Queries: {len(self.ground_truth):,}\n")

    def load_context_info(self):
        print("Loading context information...")
        ctx_path = Path(self.config['context_info_path'])
        ctx_df = pd.read_csv(ctx_path, sep='\t')
        if 'playing_time' not in ctx_df.columns:
            ctx_df = self._reconstruct_categorical(ctx_df)
        
        train_path = Path(self.config['test_path']).parent / 'train_df.tsv'
        train_df = pd.read_csv(train_path, sep='\t', usecols=['game_id:token', 'context_id'])
        train_df = train_df.rename(columns={'game_id:token': 'item_id:token'})
        
        merged_ctx = train_df.merge(ctx_df, on='context_id', how='left')
        merged_ctx['item_id:token'] = merged_ctx['item_id:token'].astype(str).str.strip()
        self.context_info = merged_ctx.drop_duplicates(subset=['item_id:token'])
        
        # Convert to string
        for feat in self.CONTEXT_FEATURES:
            if feat in self.context_info.columns:
                self.context_info[feat] = self.context_info[feat].astype(str).str.strip()
        
        print(f"✓ Context info: {len(self.context_info)} items\n")

    def _reconstruct_categorical(self, df):
        def collapse(dataframe, prefix, new_name):
            cols = [c for c in dataframe.columns if c.startswith(prefix)]
            if not cols: return dataframe
            dataframe[new_name] = dataframe[cols].idxmax(axis=1).str.replace(prefix, "").str.replace(":float", "")
            return dataframe
        df = collapse(df, "playing_time_", "playing_time")
        df = collapse(df, "gaming_mood_", "gaming_mood")
        df = collapse(df, "social_companion_", "social_companion")
        return df

    def evaluate_model(self, model_name: str, pred_path: Path) -> Dict:
        print(f"  → {model_name}")
        pred_df = pd.read_csv(pred_path, sep='\t')
        pred_df = pred_df.rename(columns={'game_id:token': 'item_id:token'})
        for col in ['user_id:token', 'item_id:token', 'q_context_id']:
            pred_df[col] = pred_df[col].astype(str).str.strip()
        
        results = {}
        
        print(f"      Traditional...")
        results.update(self._run_traditional(pred_df))
        
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
        
        print(f"      CW-nDCG, CW-MAP...")
        try:
            # VECTORIZED label assignment
            cw_pred = pred_df.copy()
            cw_pred['query_id'] = cw_pred['user_id:token'] + '_' + cw_pred['q_context_id']
            
            gt_list = [
                {'query_id': qid, 'item_id:token': item, 'label': 1.0 if rating >= 7.0 else 0.0}
                for qid, items in self.ground_truth.items()
                for item, rating in items.items()
            ]
            
            if gt_list:
                gt_df = pd.DataFrame(gt_list)
                cw_pred = cw_pred.merge(gt_df, on=['query_id', 'item_id:token'], how='left')
                cw_pred['label'] = cw_pred['label'].fillna(0).astype(float)
            else:
                cw_pred['label'] = 0.0
            
            results.update(compute_context_weighted_ndcg(cw_pred, self.context_info, self.CONTEXT_FEATURES, k_values=[5, 10, 20]))
            results.update(compute_context_weighted_map(cw_pred, self.context_info, self.CONTEXT_FEATURES, k_values=[5, 10, 20]))
        except Exception as e:
            print(f"        CW error: {e}")
        
        print(f"      Dimensional...")
        for g_name, g_feats in self.FEATURE_GROUPS.items():
            try:
                w_res = compute_cs_wcs(pred_df, self.context_info, g_feats, alpha=0.5, k_values=[5])
                if 'WCS@5' in w_res: results[f'WCS_{g_name}@5'] = w_res['WCS@5']
            except: pass
            
        print(f"      ✓ Done")
        return results

    def _run_traditional(self, pred_df):
        res = {}
        
        if SKLEARN_AVAILABLE:
            m = pred_df.merge(
                self.test_df[['user_id:token', 'item_id:token', 'context_id', 'rating:float']], 
                left_on=['user_id:token', 'item_id:token', 'q_context_id'],
                right_on=['user_id:token', 'item_id:token', 'context_id'], 
                how='inner'
            )
            
            if len(m) > 0:
                y_true = (m['rating:float'] >= 7.0).astype(int)
                y_scores = m['prediction'].values
                
                if len(np.unique(y_true)) > 1:
                    try:
                        res['AUC'] = roc_auc_score(y_true, y_scores)
                    except:
                        res['AUC'] = np.nan
                    
                    y_norm = np.clip((y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-10), 1e-10, 1 - 1e-10)
                    try:
                        res['LogLoss'] = log_loss(y_true, y_norm, labels=[0, 1])
                    except:
                        res['LogLoss'] = np.nan
        
        if RANX_AVAILABLE:
            try:
                pred_df['query_id'] = pred_df['user_id:token'] + '_' + pred_df['q_context_id']
                
                qrels_dict = {qid: {item: int(rating >= 7.0) for item, rating in items.items()} for qid, items in self.ground_truth.items()}
                qrels = Qrels(qrels_dict)
                
                # VECTORIZED run dict
                run_dict = pred_df.groupby('query_id').apply(
                    lambda g: dict(zip(g['item_id:token'], g['prediction']))
                ).to_dict()
                run = Run(run_dict)
                
                ranx_res = evaluate(qrels, run, ['ndcg@5', 'ndcg@10', 'map@10', 'mrr@10', 'precision@5', 'recall@10'], make_comparable=True)
                
                res['nDCG@5'] = ranx_res.get('ndcg@5', np.nan)
                res['nDCG@10'] = ranx_res.get('ndcg@10', np.nan)
                res['MAP@10'] = ranx_res.get('map@10', np.nan)
                res['MRR@10'] = ranx_res.get('mrr@10', np.nan)
                res['P@5'] = ranx_res.get('precision@5', np.nan)
                res['R@10'] = ranx_res.get('recall@10', np.nan)
            except Exception as e:
                print(f"        Ranx error: {e}")
                for m in ['nDCG@5', 'nDCG@10', 'MAP@10', 'MRR@10', 'P@5', 'R@10']:
                    res[m] = np.nan
        
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
                df.to_csv(self.output_dir / f"bgg_all_metrics_{self.timestamp}.csv")
                print(f"\n✓ Saved\n{df.to_string()}")
            print("\n" + "="*70 + "\n✓ COMPLETED\n" + "="*70)
            return True
        except Exception as e:
            print(f"\n✗ Failed: {e}"); import traceback; traceback.print_exc(); return False

if __name__ == '__main__':
    config = {
        'test_path': './datasets/bgg/test_df.tsv',
        'context_info_path': './datasets/bgg/context_info.tsv',
        'results_dir': './outputs/bgg',
        'output_dir': './results/bgg/complete_metrics'
    }
    CompleteBGGEvaluator(config).run()