"""
Frappe Dataset Evaluator
=========================

Consolidated evaluator for Frappe dataset.

Context features:
- Temporal: daytime, weekday, isweekend
- Activity: homework, cost
- Environment: weather, country, city

Usage:
    from evaluators import FrappeEvaluator
    
    config = {
        'test_path': './data/frappe/frappe_test.csv',
        'train_path': './data/frappe/frappe_train.csv',
        'results_dir': './outputs/frappe',
        'output_dir': './results/frappe/context_metrics',
        'cutoffs': [5, 10, 20],
        'alpha': 0.5
    }
    
    evaluator = FrappeEvaluator(config)
    evaluator.run()
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class FrappeEvaluator:
    """Complete evaluator for Frappe dataset"""
    
    CONTEXT_FEATURES = [
        'daytime', 'weekday', 'isweekend',
        'homework', 'cost', 'weather', 'country', 'city'
    ]
    
    def __init__(self, config):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.context_features = self.CONTEXT_FEATURES
        
    def load_test_set(self):
        """Load Frappe test set"""
        print("="*70)
        print("LOADING FRAPPE TEST SET")
        print("="*70)
        
        test_path = Path(self.config['test_path'])
        self.test_df = pd.read_csv(test_path)
        
        # Ensure string types
        self.test_df['user'] = self.test_df['user'].astype(str).str.strip()
        self.test_df['item'] = self.test_df['item'].astype(str).str.strip()
        
        # Create q_context_id
        context_cols = [col for col in self.context_features if col in self.test_df.columns]
        self.test_df['q_context_id'] = (
            self.test_df[context_cols].astype(str).agg('_'.join, axis=1)
        )
        
        # Create query_id
        self.test_df['query_id'] = (
            self.test_df['user'] + '_' + self.test_df['q_context_id']
        )
        
        # Store ground truth (binary labels)
        self.ground_truth = {}
        for _, row in self.test_df.iterrows():
            qid = row['query_id']
            item = str(row['item']).strip()
            relevance = float(row['label'])
            
            if qid not in self.ground_truth:
                self.ground_truth[qid] = {}
            self.ground_truth[qid][item] = relevance
        
        self.unique_query_ids = self.test_df['query_id'].unique()
        
        print(f"✓ Test set loaded: {self.test_df.shape}")
        print(f"  Unique queries: {len(self.unique_query_ids):,}")
        print(f"  Unique items: {self.test_df['item'].nunique():,}")
        print(f"  Label distribution: {self.test_df['label'].value_counts().to_dict()}")
        print()
        
    def load_context_info(self):
        """Load item context information"""
        print("Loading context definitions...")
        
        train_path = Path(self.config['train_path'])
        train_df = pd.read_csv(train_path)
        
        # Ensure string types
        train_df['item'] = train_df['item'].astype(str).str.strip()
        
        # Get unique item-context combinations
        item_cols = ['item'] + self.context_features
        self.item_context = train_df[item_cols].groupby('item').agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
        ).reset_index()
        
        print(f"✓ Item contexts loaded: {len(self.item_context)}")
        
        # Compute IDF weights
        self._compute_idf_weights(train_df)
        print()
    
    def _compute_idf_weights(self, df):
        """Compute IDF-based feature weights"""
        self.idf_weights = {}
        
        for feat in self.context_features:
            if feat in df.columns:
                n_total = len(df)
                n_unique = df[feat].nunique()
                idf = np.log((n_total + 1) / (n_unique + 1)) + 1.0
                self.idf_weights[feat] = idf
        
        print(f"  IDF weights computed for {len(self.idf_weights)} features:")
        sorted_weights = sorted(self.idf_weights.items(), key=lambda x: x[1], reverse=True)
        for feat, weight in sorted_weights[:3]:
            print(f"    {feat}: {weight:.4f}")
    
    def compute_cs_score(self, q_context, i_context, alpha=0.5):
        """Context Satisfaction"""
        q_vals = set(str(q_context.get(f, '')) for f in self.context_features 
                     if pd.notna(q_context.get(f)) and str(q_context.get(f, '')))
        i_vals = set(str(i_context.get(f, '')) for f in self.context_features 
                     if pd.notna(i_context.get(f)) and str(i_context.get(f, '')))
        
        intersection = len(q_vals & i_vals)
        union = len(q_vals | i_vals)
        difference = len(q_vals - i_vals)
        
        if len(q_vals) == 0 or union == 0:
            return 0.0
        
        penalty = alpha * (difference / len(q_vals))
        cs = intersection / (union + penalty)
        return cs
    
    def compute_wca_score(self, q_context, i_context):
        """Weighted Context Alignment"""
        q_vec = []
        i_vec = []
        
        for feat in self.context_features:
            w = self.idf_weights.get(feat, 1.0)
            q_val = 1.0 if pd.notna(q_context.get(feat)) else 0.0
            i_val = 1.0 if pd.notna(i_context.get(feat)) else 0.0
            q_vec.append(w * q_val)
            i_vec.append(w * i_val)
        
        q_vec = np.array(q_vec)
        i_vec = np.array(i_vec)
        
        dot_product = np.dot(q_vec, i_vec)
        norm_q = np.linalg.norm(q_vec)
        norm_i = np.linalg.norm(i_vec)
        
        if norm_q == 0 or norm_i == 0:
            return 0.0
        
        return dot_product / (norm_q * norm_i + 1e-10)
    
    def compute_friction_score(self, q_context, i_context):
        """Context Friction"""
        distance = 0
        total_weight = 0
        
        for feat in self.context_features:
            w = self.idf_weights.get(feat, 1.0)
            q_val = str(q_context.get(feat, '')) if pd.notna(q_context.get(feat)) else ''
            i_val = str(i_context.get(feat, '')) if pd.notna(i_context.get(feat)) else ''
            
            if q_val != i_val:
                distance += w
            total_weight += w
        
        if total_weight == 0:
            return 1.0
        
        return 1.0 - (distance / total_weight)
    
    def evaluate_model(self, model_name, predictions_path, cutoffs, alpha):
        """Evaluate CW-nDCG and CW-MAP"""
        print(f"→ Evaluating {model_name}...")
        
        if not predictions_path.exists():
            print(f"  ✗ Predictions not found")
            return None
        
        # Load predictions
        pred_df = pd.read_csv(predictions_path, sep='\t')
        
        pred_df['user_id:token'] = pred_df['user_id:token'].astype(str).str.strip()
        pred_df['item_id:token'] = pred_df['item_id:token'].astype(str).str.strip()
        pred_df['q_context_id'] = pred_df['q_context_id'].astype(str).str.strip()
        
        pred_df['query_id'] = pred_df['user_id:token'] + '_' + pred_df['q_context_id']
        
        # Extract query context
        context_splits = pred_df['q_context_id'].str.split('_', expand=True)
        for i, feat in enumerate(self.context_features):
            if i < context_splits.shape[1]:
                pred_df[f'{feat}_query'] = context_splits[i].astype(str).str.strip()
        
        # Build item context dict
        item_ctx_dict = {}
        for _, row in self.item_context.iterrows():
            item_id = str(row['item']).strip()
            ctx = {feat: row[feat] for feat in self.context_features if feat in row.index}
            item_ctx_dict[item_id] = ctx
        
        # Sort by prediction
        pred_df = pred_df.sort_values(
            ['query_id', 'prediction'],
            ascending=[True, False]
        ).reset_index(drop=True)
        
        # Group by query
        grouped = pred_df.groupby('query_id', sort=False)
        
        # Check overlap
        pred_query_ids = set(pred_df['query_id'].unique())
        test_query_ids = set(self.unique_query_ids)
        overlap = pred_query_ids & test_query_ids
        
        print(f"  Query overlap: {len(overlap)}/{len(test_query_ids)} ({len(overlap)/len(test_query_ids)*100:.1f}%)")
        
        if len(overlap) == 0:
            print(f"  ✗ NO OVERLAP!")
            return None
        
        print(f"  Computing metrics...")
        
        # Results storage
        results = {}
        for metric_name in ['CS', 'WCA', 'Friction']:
            for k in cutoffs:
                results[f'{metric_name}_CW-nDCG@{k}'] = []
                results[f'{metric_name}_CW-MAP@{k}'] = []
        
        queries_evaluated = 0
        
        for query_id in tqdm(self.unique_query_ids, desc="  ", leave=False):
            if query_id not in grouped.groups:
                continue
            
            group = grouped.get_group(query_id)
            
            # Get query context
            q_context = {}
            for feat in self.context_features:
                col = f'{feat}_query'
                if col in group.columns:
                    q_context[feat] = group.iloc[0][col]
            
            # Compute similarity scores
            ranked_items = []
            cs_scores = []
            wca_scores = []
            friction_scores = []
            
            for _, row in group.iterrows():
                item_id = row['item_id:token']
                
                if item_id in item_ctx_dict:
                    i_context = item_ctx_dict[item_id]
                    
                    cs = self.compute_cs_score(q_context, i_context, alpha)
                    wca = self.compute_wca_score(q_context, i_context)
                    friction = self.compute_friction_score(q_context, i_context)
                    
                    ranked_items.append(item_id)
                    cs_scores.append(cs)
                    wca_scores.append(wca)
                    friction_scores.append(friction)
            
            if not ranked_items:
                continue
            
            # Evaluate at each cutoff
            for k in cutoffs:
                # CS-based
                cs_ndcg = self._compute_cw_ndcg(query_id, ranked_items, cs_scores, k)
                cs_map = self._compute_cw_map(query_id, ranked_items, cs_scores, k)
                if not np.isnan(cs_ndcg):
                    results[f'CS_CW-nDCG@{k}'].append(cs_ndcg)
                if not np.isnan(cs_map):
                    results[f'CS_CW-MAP@{k}'].append(cs_map)
                
                # WCA-based
                wca_ndcg = self._compute_cw_ndcg(query_id, ranked_items, wca_scores, k)
                wca_map = self._compute_cw_map(query_id, ranked_items, wca_scores, k)
                if not np.isnan(wca_ndcg):
                    results[f'WCA_CW-nDCG@{k}'].append(wca_ndcg)
                if not np.isnan(wca_map):
                    results[f'WCA_CW-MAP@{k}'].append(wca_map)
                
                # Friction-based
                friction_ndcg = self._compute_cw_ndcg(query_id, ranked_items, friction_scores, k)
                friction_map = self._compute_cw_map(query_id, ranked_items, friction_scores, k)
                if not np.isnan(friction_ndcg):
                    results[f'Friction_CW-nDCG@{k}'].append(friction_ndcg)
                if not np.isnan(friction_map):
                    results[f'Friction_CW-MAP@{k}'].append(friction_map)
            
            queries_evaluated += 1
        
        if queries_evaluated == 0:
            print(f"  ✗ No queries evaluated")
            return None
        
        print(f"  ✓ Evaluated {queries_evaluated}/{len(self.unique_query_ids)} queries")
        
        # Calculate averages
        final_results = {}
        for metric_name, values in results.items():
            if values:
                final_results[metric_name] = np.mean(values)
        
        return final_results
    
    def _compute_cw_ndcg(self, query_id, ranked_items, similarity_scores, k):
        """Context-Weighted nDCG"""
        if query_id not in self.ground_truth:
            return np.nan
        
        gt = self.ground_truth[query_id]
        relevances = np.array([gt.get(item, 0.0) for item in ranked_items[:k]])
        similarities = np.array(similarity_scores[:k])
        
        if len(relevances) == 0:
            return 0.0
        
        gains = 2**relevances - 1
        weighted_gains = gains * similarities
        
        positions = np.arange(1, len(relevances) + 1)
        discounts = np.log2(positions + 1)
        
        cw_dcg = np.sum(weighted_gains / discounts)
        
        all_ratings = sorted(gt.values(), reverse=True)[:k]
        if not all_ratings:
            return 0.0
        
        ideal_gains = 2**np.array(all_ratings) - 1
        ideal_positions = np.arange(1, len(all_ratings) + 1)
        ideal_discounts = np.log2(ideal_positions + 1)
        cw_idcg = np.sum(ideal_gains / ideal_discounts)
        
        return cw_dcg / cw_idcg if cw_idcg > 0 else 0.0
    
    def _compute_cw_map(self, query_id, ranked_items, similarity_scores, k):
        """Context-Weighted MAP"""
        if query_id not in self.ground_truth:
            return np.nan
        
        gt = self.ground_truth[query_id]
        num_relevant = sum(1 for rel in gt.values() if rel > 0)
        
        if num_relevant == 0:
            return 0.0
        
        relevances = np.array([gt.get(item, 0.0) for item in ranked_items[:k]])
        similarities = np.array(similarity_scores[:k])
        
        if len(relevances) == 0:
            return 0.0
        
        binary_rel = (relevances > 0).astype(float)
        
        ap_sum = 0.0
        num_relevant_seen = 0
        
        for i in range(len(ranked_items[:k])):
            if binary_rel[i] > 0:
                num_relevant_seen += 1
                precision = num_relevant_seen / (i + 1)
                weighted_precision = precision * similarities[i]
                ap_sum += weighted_precision
        
        if num_relevant_seen == 0:
            return 0.0
        
        return ap_sum / num_relevant
    
    def evaluate_all_models(self):
        """Evaluate all models"""
        print("\n" + "="*70)
        print("EVALUATING ALL MODELS")
        print("="*70)
        print()
        
        cutoffs = self.config['cutoffs']
        alpha = self.config['alpha']
        results_dir = Path(self.config['results_dir'])
        
        all_results = {}
        
        exclude_dirs = {'__pycache__', 'context_metrics', 'evaluation'}
        
        model_dirs = [
            d for d in results_dir.iterdir()
            if d.is_dir() and d.name not in exclude_dirs and not d.name.startswith('.')
        ]
        
        print(f"Found {len(model_dirs)} model directories\n")
        
        for model_dir in sorted(model_dirs):
            model_name = model_dir.name.capitalize()
            
            pred_file = model_dir / 'result' / f"{model_name}_final_predictions.tsv"
            if not pred_file.exists():
                pred_files = list((model_dir / 'result').glob('*predictions.tsv'))
                if pred_files:
                    pred_file = pred_files[0]
                else:
                    print(f"⚠ Skipping {model_name}: no predictions\n")
                    continue
            
            results = self.evaluate_model(model_name, pred_file, cutoffs, alpha)
            
            if results:
                all_results[model_name] = results
            
            print()
        
        self.results = all_results
    
    def save_results(self):
        """Save results"""
        print("="*70)
        print("SAVING RESULTS")
        print("="*70)
        
        if not self.results:
            print("✗ No results to save")
            return None
        
        results_df = pd.DataFrame(self.results).T.round(4)
        
        csv_path = self.output_dir / f"frappe_cw_metrics_{self.timestamp}.csv"
        results_df.to_csv(csv_path)
        
        print(f"✓ Results saved to: {csv_path}\n")
        print(results_df.to_string())
        
        return results_df
    
    def run(self):
        """Execute evaluation pipeline"""
        print("\n" + "="*70)
        print("FRAPPE CONTEXT-WEIGHTED EVALUATOR")
        print("="*70)
        print(f"Timestamp: {self.timestamp}")
        print(f"Cutoffs: {self.config['cutoffs']}")
        print(f"Alpha: {self.config['alpha']}")
        print()
        
        try:
            self.load_test_set()
            self.load_context_info()
            self.evaluate_all_models()
            self.save_results()
            
            print("\n" + "="*70)
            print("✓ EVALUATION COMPLETED")
            print("="*70)
            print(f"\nResults: {self.output_dir}")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == '__main__':
    config = {
        'test_path': './datasets/frappe/frappe_test.csv',
        'train_path': './datasets/frappe/frappe_train.csv',
        'results_dir': './outputs/frappe',
        'output_dir': './results/frappe/context_metrics',
        'cutoffs': [5, 10, 20],
        'alpha': 0.5,
    }
    
    evaluator = FrappeEvaluator(config)
    success = evaluator.run()
    
    import sys
    sys.exit(0 if success else 1)