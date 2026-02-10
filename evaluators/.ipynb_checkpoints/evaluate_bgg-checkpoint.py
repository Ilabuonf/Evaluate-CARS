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


class CompleteBGGEvaluator:
    """Complete BGG evaluator matching all paper tables"""
    
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
        """Load BGG test set and standardize column names"""
        print("="*70)
        print("LOADING BGG TEST SET")
        print("="*70)
        
        test_path = Path(self.config['test_path'])
        self.test_df = pd.read_csv(test_path, sep='\t')
        
        # STANDARDIZZAZIONE: game_id -> item_id
        self.test_df = self.test_df.rename(columns={'game_id:token': 'item_id:token'})
        
        for col in ['user_id:token', 'item_id:token', 'context_id']:
            self.test_df[col] = self.test_df[col].astype(str).str.strip()
        
        self.test_df['query_id'] = self.test_df['user_id:token'] + '_' + self.test_df['context_id']
        
        self.ground_truth = {}
        for _, row in self.test_df.iterrows():
            qid, item, rating = row['query_id'], row['item_id:token'], row['rating:float']
            if qid not in self.ground_truth: self.ground_truth[qid] = {}
            self.ground_truth[qid][item] = rating
            
        print(f"✓ Loaded: {self.test_df.shape}\n  Queries: {len(self.ground_truth):,}\n")

    def load_context_info(self):
        """Load context info and LINK it to items via train set"""
        print("Loading context information...")
        
        # 1. Carica definizioni contesti
        ctx_path = Path(self.config['context_info_path'])
        ctx_df = pd.read_csv(ctx_path, sep='\t')
        if 'playing_time' not in ctx_df.columns:
            ctx_df = self._reconstruct_categorical(ctx_df)
        
        # 2. Carica train_df per mappare item <-> context_id
        train_path = Path(self.config['test_path']).parent / 'train_df.tsv'
        train_df = pd.read_csv(train_path, sep='\t', usecols=['game_id:token', 'context_id'])
        train_df = train_df.rename(columns={'game_id:token': 'item_id:token'})
        
        # 3. Merge per avere un DataFrame: [item_id, playing_time, gaming_mood, social_companion]
        merged_ctx = train_df.merge(ctx_df, on='context_id', how='left')
        
        # Standardizza nomi e tipi
        merged_ctx['item_id:token'] = merged_ctx['item_id:token'].astype(str).str.strip()
        self.context_info = merged_ctx.drop_duplicates(subset=['item_id:token'])
        
        print(f"✓ Context info linked to {len(self.context_info)} items\n")

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
        
        # Standardizza colonne
        pred_df = pred_df.rename(columns={'game_id:token': 'item_id:token'})
        for col in ['user_id:token', 'item_id:token', 'q_context_id']:
            pred_df[col] = pred_df[col].astype(str).str.strip()
        
        results = {}
        
        # 1. Traditional
        print(f"      Traditional...")
        results.update(self._run_traditional(pred_df))
        
        # 2. Context Metrics
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
            print(f"        Context metrics error: {e}")
        
        # 3. CW Ranking Metrics (NEW!)
        print(f"      CW-nDCG, CW-MAP...")
        try:
            # Prepare for CW metrics - need ground truth dict
            cw_pred = pred_df.copy()
            cw_pred['query_id'] = cw_pred['user_id:token'] + '_' + cw_pred['q_context_id']
            
            # Add label column from ground truth
            labels = []
            for _, row in cw_pred.iterrows():
                qid = row['query_id']
                item = row['item_id:token']
                rating = self.ground_truth.get(qid, {}).get(item, 0.0)
                labels.append(1.0 if rating >= 7.0 else 0.0)
            cw_pred['label'] = labels
            
            # Compute CW metrics
            cw_ndcg = compute_context_weighted_ndcg(
                cw_pred, self.context_info, self.CONTEXT_FEATURES, k_values=[5, 10, 20]
            )
            results.update(cw_ndcg)
            
            cw_map = compute_context_weighted_map(
                cw_pred, self.context_info, self.CONTEXT_FEATURES, k_values=[5, 10, 20]
            )
            results.update(cw_map)
        except Exception as e:
            print(f"        CW metrics error: {e}")
        
        # 4. Dimensional
        print(f"      Dimensional...")
        for g_name, g_feats in self.FEATURE_GROUPS.items():
            try:
                w_res = compute_cs_wcs(pred_df, self.context_info, g_feats, alpha=0.5, k_values=[5])
                if 'WCS@5' in w_res: 
                    results[f'WCS_{g_name}@5'] = w_res['WCS@5']
            except:
                pass
            
        print(f"      ✓ Completed")
        return results

    def _run_traditional(self, pred_df):
        """Traditional metrics with FIXED Ranx grouping"""
        res = {}
        
        # AUC/LogLoss
        if SKLEARN_AVAILABLE:
            m = pred_df.merge(
                self.test_df[['user_id:token', 'item_id:token', 'context_id', 'rating:float']], 
                left_on=['user_id:token', 'item_id:token', 'q_context_id'],
                right_on=['user_id:token', 'item_id:token', 'context_id'], 
                how='inner'
            )
            
            if len(m) > 0 and len(np.unique(m['rating:float'])) > 1:
                y_true = (m['rating:float'] >= 7.0).astype(int)
                y_scores = m['prediction'].values
                
                try:
                    res['AUC'] = roc_auc_score(y_true, y_scores)
                except:
                    res['AUC'] = np.nan
                
                y_norm = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-10)
                y_norm = np.clip(y_norm, 1e-10, 1 - 1e-10)
                try:
                    res['LogLoss'] = log_loss(y_true, y_norm, labels=[0, 1])
                except:
                    res['LogLoss'] = np.nan
        
        # Ranx - FIX: group by query_id (user + context), not just user_id!
        if RANX_AVAILABLE:
            try:
                # Create query_id
                pred_df['query_id'] = pred_df['user_id:token'] + '_' + pred_df['q_context_id']
                
                # Qrels
                qrels_dict = {}
                for qid, items in self.ground_truth.items():
                    qrels_dict[qid] = {item: int(rating >= 7.0) for item, rating in items.items()}
                qrels = Qrels(qrels_dict)
                
                # Run - group by QUERY_ID (not user_id!)
                run_dict = {}
                for query_id, group in pred_df.groupby('query_id'):
                    run_dict[query_id] = {
                        row['item_id:token']: float(row['prediction'])
                        for _, row in group.iterrows()
                    }
                run = Run(run_dict)
                
                # Evaluate
                ranx_res = evaluate(
                    qrels, run,
                    ['ndcg@5', 'ndcg@10', 'map@10', 'mrr@10', 'precision@5', 'recall@10'],
                    make_comparable=True
                )
                
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

    def evaluate_all_models(self):
        print("\n" + "="*70 + "\nEVALUATING ALL MODELS\n" + "="*70)
        r_dir = Path(self.config['results_dir'])
        model_dirs = [d for d in r_dir.iterdir() if d.is_dir() and not d.name.startswith(('.', 'context', 'evaluation'))]
        
        for md in sorted(model_dirs):
            p_file = next(md.rglob('*predictions.tsv'), None)
            if p_file: 
                self.results[md.name.capitalize()] = self.evaluate_model(md.name, p_file)

    def save_results(self):
        if not self.results: 
            print("✗ No results")
            return
        
        df = pd.DataFrame(self.results).T.round(4)
        
        # Column order
        col_order = []
        
        # Traditional
        for m in ['AUC', 'LogLoss', 'nDCG@5', 'nDCG@10', 'MAP@10', 'MRR@10', 'P@5', 'R@10']:
            if m in df.columns:
                col_order.append(m)
        
        # Context similarity
        for m in ['ACC@5', 'CS@5', 'WCS@5', 'WCA@5', 'Friction@5']:
            if m in df.columns:
                col_order.append(m)
        
        # Advanced
        for m in ['CR@5', 'CRC@5', 'CGB@5']:
            if m in df.columns:
                col_order.append(m)
        
        # CW Ranking
        for k in [5, 10, 20]:
            for m in [f'CW-nDCG@{k}', f'CW-MAP@{k}']:
                if m in df.columns:
                    col_order.append(m)
        
        # Dimensional
        for group in ['temporal', 'experiential', 'social']:
            col = f'WCS_{group}@5'
            if col in df.columns:
                col_order.append(m)
        
        df = df[[c for c in col_order if c in df.columns]]
        
        path = self.output_dir / f"bgg_all_metrics_{self.timestamp}.csv"
        df.to_csv(path)
        print(f"\n✓ Saved: {path}\n{df.to_string()}")

    def run(self):
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
            import traceback; traceback.print_exc()
            return False

if __name__ == '__main__':
    config = {
        'test_path': './datasets/bgg/test_df.tsv',
        'context_info_path': './datasets/bgg/context_info.tsv',
        'results_dir': './outputs/bgg',
        'output_dir': './results/bgg/complete_metrics'
    }
    CompleteBGGEvaluator(config).run()