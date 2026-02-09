"""
BGG Dataset Evaluator
=====================

Consolidated evaluator for BoardGameGeek dataset that computes:
- Context Similarity Metrics (CS, WCA, Friction)
- Context-Weighted Ranking Metrics (CW-nDCG, CW-MAP)
- All advanced context metrics (ACC, CR, CRC, CGB)

Usage:
    from evaluators import BGGEvaluator
    
    config = {
        'test_path': './data/test_df.tsv',
        'context_info_path': './data/context_info.tsv',
        'results_dir': './outputs',
        'output_dir': './results/bgg',
        'cutoffs': [5, 10, 20],
        'alpha': 0.5,
        'relevance_threshold': 7.0
    }
    
    evaluator = BGGEvaluator(config)
    evaluator.run()
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import spearmanr, ConstantInputWarning
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')


class BGGEvaluator:
    """Complete evaluator for BoardGameGeek dataset"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store results
        self.results = {}
        
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    
    def load_test_set(self):
        """Load test set with ratings"""
        print("="*70)
        print("LOADING TEST SET")
        print("="*70)
        
        test_path = Path(self.config['test_path'])
        
        dtypes = {
            'user_id:token': 'category',
            'game_id:token': 'category',
            'context_id': 'int32',
            'rating:float': 'float32'
        }
        
        self.test_df = pd.read_csv(test_path, sep='\t', dtype=dtypes, engine='c')
        
        # Create query_id
        self.test_df['query_id'] = (
            self.test_df['user_id:token'].astype(str) + '_' + 
            self.test_df['context_id'].astype(str)
        )
        
        # Store ground truth
        self.ground_truth = {}
        for _, row in self.test_df.iterrows():
            qid = row['query_id']
            item = str(row['game_id:token'])
            rating = row['rating:float']
            
            if qid not in self.ground_truth:
                self.ground_truth[qid] = {}
            self.ground_truth[qid][item] = rating
        
        self.unique_query_ids = self.test_df['query_id'].unique()
        
        print(f"✓ Test set loaded: {self.test_df.shape}")
        print(f"  Unique queries: {len(self.unique_query_ids):,}")
        print(f"  Unique items: {self.test_df['game_id:token'].nunique():,}")
        print(f"  Rating range: [{self.test_df['rating:float'].min():.1f}, "
              f"{self.test_df['rating:float'].max():.1f}]")
        print()
        
    def load_context_info(self):
        """Load context definitions and compute IDF weights"""
        print("Loading context definitions...")
        ctx_path = Path(self.config['context_info_path'])
        self.ctx_info = pd.read_csv(ctx_path, sep='\t', index_col='context_id')
        
        print(f"✓ Context definitions loaded: {len(self.ctx_info)}")
        print(f"  Features: {len(self.ctx_info.columns)}")
        
        # Convert to numpy for vectorization
        self.ctx_array = self.ctx_info.values.astype(float)
        self.ctx_index = self.ctx_info.index.values
        self.ctx_dict = {cid: idx for idx, cid in enumerate(self.ctx_index)}
        
        # Compute IDF weights
        self._compute_feature_importance()
        print()
    
    def _compute_feature_importance(self):
        """Compute IDF-based feature weights"""
        df = self.ctx_array.sum(axis=0)
        N = self.ctx_array.shape[0]
        raw_idf = np.log((N + 1) / (df + 1)) + 1.0
        self.feature_weights = np.nan_to_num(raw_idf, nan=0.0)
        
        print(f"  IDF weights computed:")
        print(f"    Range: [{self.feature_weights.min():.4f}, "
              f"{self.feature_weights.max():.4f}]")
        print(f"    Mean: {self.feature_weights.mean():.4f}")
    
    # =========================================================================
    # SIMILARITY METRICS (Vectorized)
    # =========================================================================
    
    def compute_cs_vectorized(self, c_q, item_contexts, alpha):
        """Context Satisfaction - Modified Jaccard"""
        rec = c_q.astype(bool)
        items = item_contexts.astype(bool)
        
        w_int = rec & items
        w_uni = rec | items
        w_diff = rec & ~items
        
        num = np.sum(w_int.astype(float) * self.feature_weights, axis=1)
        den = np.sum(w_uni.astype(float) * self.feature_weights, axis=1)
        mis = np.sum(w_diff.astype(float) * self.feature_weights, axis=1)
        req = np.sum(rec.astype(float) * self.feature_weights)
        
        if req <= 0:
            return np.zeros(len(item_contexts))
        
        cs_scores = num / (den + alpha * (mis / req))
        return np.nan_to_num(cs_scores, nan=0.0)
    
    def compute_wca_vectorized(self, c_q, item_contexts):
        """Weighted Context Alignment - Cosine Similarity"""
        weighted_q = c_q * self.feature_weights
        weighted_items = item_contexts * self.feature_weights
        
        dot_products = np.dot(weighted_items, weighted_q)
        norm_q = np.linalg.norm(weighted_q)
        norm_items = np.linalg.norm(weighted_items, axis=1)
        
        if norm_q == 0:
            return np.zeros(len(item_contexts))
        
        wca_scores = dot_products / (norm_q * norm_items + 1e-10)
        return np.clip(wca_scores, 0.0, 1.0)
    
    def compute_friction_vectorized(self, c_q, item_contexts):
        """Context Friction - Inverted Weighted Hamming"""
        differences = (c_q != item_contexts).astype(float)
        weighted_friction = np.sum(differences * self.feature_weights, axis=1)
        max_friction = np.sum(self.feature_weights)
        
        if max_friction == 0:
            return np.ones(len(item_contexts))
        
        normalized_friction = weighted_friction / max_friction
        return 1.0 - normalized_friction
    
    # =========================================================================
    # RANKING METRICS
    # =========================================================================
    
    def compute_dcg(self, relevances, k):
        """Standard DCG"""
        relevances = np.array(relevances[:k])
        if len(relevances) == 0:
            return 0.0
        
        actual_k = len(relevances)
        gains = 2**relevances - 1
        discounts = np.log2(np.arange(2, actual_k + 2))
        return np.sum(gains / discounts)
    
    def compute_idcg(self, ground_truth_ratings, k):
        """Ideal DCG"""
        if not ground_truth_ratings:
            return 0.0
        
        sorted_ratings = sorted(ground_truth_ratings, reverse=True)
        return self.compute_dcg(sorted_ratings, k)
    
    def compute_cw_ndcg(self, query_id, ranked_items, similarity_scores, k):
        """Context-Weighted nDCG"""
        if query_id not in self.ground_truth:
            return np.nan
        
        gt = self.ground_truth[query_id]
        relevances = np.array([gt.get(item, 0.0) for item in ranked_items[:k]])
        similarities = np.array(similarity_scores[:k])
        
        if len(relevances) == 0:
            return 0.0
        
        # Weighted gains
        gains = 2**relevances - 1
        weighted_gains = gains * similarities
        
        # Position discount
        positions = np.arange(1, len(relevances) + 1)
        position_discount = np.log2(positions + 1)
        
        cw_dcg = np.sum(weighted_gains / position_discount)
        
        # IDCG
        all_ratings = sorted(gt.values(), reverse=True)[:k]
        if not all_ratings:
            return 0.0
        
        ideal_gains = 2**np.array(all_ratings) - 1
        ideal_positions = np.arange(1, len(all_ratings) + 1)
        ideal_discount = np.log2(ideal_positions + 1)
        cw_idcg = np.sum(ideal_gains / ideal_discount)
        
        return cw_dcg / cw_idcg if cw_idcg > 0 else 0.0
    
    def compute_cw_map(self, query_id, ranked_items, similarity_scores, k):
        """Context-Weighted MAP"""
        if query_id not in self.ground_truth:
            return np.nan
        
        gt = self.ground_truth[query_id]
        threshold = self.config.get('relevance_threshold', 7.0)
        
        num_relevant = sum(1 for rating in gt.values() if rating >= threshold)
        
        if num_relevant == 0:
            return 0.0
        
        relevances = []
        for item in ranked_items[:k]:
            rel = gt.get(item, 0.0)
            relevances.append(rel)
        
        if not relevances:
            return 0.0
        
        relevances = np.array(relevances)
        similarities = np.array(similarity_scores[:k])
        
        binary_rel = (relevances >= threshold).astype(float)
        
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
    
    # =========================================================================
    # MODEL EVALUATION
    # =========================================================================
    
    def evaluate_model(self, model_name, predictions_path, cutoffs, alpha):
        """Evaluate all metrics for a single model"""
        print(f"→ Evaluating {model_name}...")
        
        if not predictions_path.exists():
            print(f"  Predictions not found")
            return None
        
        # Load predictions
        pred_df = pd.read_csv(predictions_path, sep='\t', engine='c')
        
        pred_df['user_id:token'] = pred_df['user_id:token'].astype(str).str.strip()
        pred_df['item_id:token'] = pred_df['item_id:token'].astype(str).str.strip()
        pred_df['q_context_id'] = pred_df['q_context_id'].astype(str).str.strip()
        
        pred_df['query_id'] = (
            pred_df['user_id:token'] + '_' + pred_df['q_context_id']
        )
        
        # Check for item_context_id
        if 'item_context_id' not in pred_df.columns:
            print(f"  Baseline model - cannot compute context metrics")
            return None
        
        pred_df['item_context_id'] = pred_df['item_context_id'].astype('Int64')
        
        # Check coverage
        coverage = pred_df['item_context_id'].notna().sum() / len(pred_df) * 100
        print(f"  Context coverage: {coverage:.1f}%")
        
        if coverage < 10:
            print(f"  Insufficient context coverage")
            return None
        
        # Sort by score
        pred_df = pred_df.sort_values(
            ['query_id', 'prediction'],
            ascending=[True, False]
        ).reset_index(drop=True)
        
        # Pre-compute item context vectors
        unique_items = pred_df['item_context_id'].dropna().unique()
        item_ctx_vectors = {}
        
        for iid in unique_items:
            iid_int = int(iid)
            if iid_int in self.ctx_dict:
                item_ctx_vectors[iid_int] = self.ctx_array[self.ctx_dict[iid_int]]
        
        # Group by query
        grouped = pred_df.groupby('query_id', sort=False)
        
        print(f"  Computing similarity + ranking metrics...")
        
        # Results storage
        results = {}
        
        # Similarity metrics at cutoffs
        for metric_name in ['CS', 'WCA', 'Friction']:
            results[f'{metric_name}@all'] = []
            for k in cutoffs:
                results[f'{metric_name}@{k}'] = []
        
        # Ranking metrics
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
            q_context_id = group.iloc[0]['q_context_id']
            try:
                q_context_id = int(q_context_id)
            except (ValueError, TypeError):
                continue
            
            if q_context_id not in self.ctx_dict:
                continue
            
            c_vec = self.ctx_array[self.ctx_dict[q_context_id]]
            
            # Get ranked items
            ranked_items = []
            item_contexts_list = []
            
            for _, row in group.iterrows():
                item_id = row['item_id:token']
                item_ctx_id = row['item_context_id']
                
                if pd.notna(item_ctx_id):
                    item_ctx_id = int(item_ctx_id)
                    if item_ctx_id in item_ctx_vectors:
                        ranked_items.append(item_id)
                        item_contexts_list.append(item_ctx_vectors[item_ctx_id])
            
            if not ranked_items:
                continue
            
            # Convert to matrix
            item_matrix = np.array(item_contexts_list)
            
            # Compute similarities
            cs_scores = self.compute_cs_vectorized(c_vec, item_matrix, alpha)
            wca_scores = self.compute_wca_vectorized(c_vec, item_matrix)
            friction_scores = self.compute_friction_vectorized(c_vec, item_matrix)
            
            # Store average similarities
            results['CS@all'].append(np.mean(cs_scores))
            results['WCA@all'].append(np.mean(wca_scores))
            results['Friction@all'].append(np.mean(friction_scores))
            
            # Evaluate at each cutoff
            for k in cutoffs:
                # Average similarity at cutoff
                results[f'CS@{k}'].append(np.mean(cs_scores[:k]))
                results[f'WCA@{k}'].append(np.mean(wca_scores[:k]))
                results[f'Friction@{k}'].append(np.mean(friction_scores[:k]))
                
                # CS-based ranking
                cs_ndcg = self.compute_cw_ndcg(query_id, ranked_items, cs_scores, k)
                cs_map = self.compute_cw_map(query_id, ranked_items, cs_scores, k)
                
                if not np.isnan(cs_ndcg):
                    results[f'CS_CW-nDCG@{k}'].append(cs_ndcg)
                if not np.isnan(cs_map):
                    results[f'CS_CW-MAP@{k}'].append(cs_map)
                
                # WCA-based ranking
                wca_ndcg = self.compute_cw_ndcg(query_id, ranked_items, wca_scores, k)
                wca_map = self.compute_cw_map(query_id, ranked_items, wca_scores, k)
                
                if not np.isnan(wca_ndcg):
                    results[f'WCA_CW-nDCG@{k}'].append(wca_ndcg)
                if not np.isnan(wca_map):
                    results[f'WCA_CW-MAP@{k}'].append(wca_map)
                
                # Friction-based ranking
                friction_ndcg = self.compute_cw_ndcg(query_id, ranked_items, friction_scores, k)
                friction_map = self.compute_cw_map(query_id, ranked_items, friction_scores, k)
                
                if not np.isnan(friction_ndcg):
                    results[f'Friction_CW-nDCG@{k}'].append(friction_ndcg)
                if not np.isnan(friction_map):
                    results[f'Friction_CW-MAP@{k}'].append(friction_map)
            
            queries_evaluated += 1
        
        if queries_evaluated == 0:
            print(f"  No queries evaluated")
            return None
        
        print(f"  Evaluated {queries_evaluated}/{len(self.unique_query_ids)} queries")
        
        # Calculate averages
        final_results = {}
        for metric_name, values in results.items():
            if values:
                final_results[metric_name] = np.mean(values)
        
        # Print results
        print(f"\n  Similarity Metrics:")
        for sim in ['CS', 'WCA', 'Friction']:
            print(f"    {sim}:")
            for k in ['all'] + cutoffs:
                key = f'{sim}@{k}'
                if key in final_results:
                    print(f"      @{k}: {final_results[key]:.4f}")
        
        print(f"\n  Ranking Metrics:")
        for sim in ['CS', 'WCA', 'Friction']:
            print(f"    {sim}-weighted:")
            for k in cutoffs:
                ndcg_key = f'{sim}_CW-nDCG@{k}'
                map_key = f'{sim}_CW-MAP@{k}'
                if ndcg_key in final_results:
                    print(f"      CW-nDCG@{k}: {final_results[ndcg_key]:.4f}")
                if map_key in final_results:
                    print(f"      CW-MAP@{k}: {final_results[map_key]:.4f}")
        
        print(f"  Completed")
        return final_results
    
    def evaluate_all_models(self):
        """Evaluate all models"""
        print("\n" + "="*70)
        print("EVALUATING ALL METRICS")
        print("="*70)
        print()
        
        cutoffs = self.config['cutoffs']
        alpha = self.config['alpha']
        results_dir = Path(self.config['results_dir'])
        
        all_results = {}
        
        # Auto-detect models
        exclude_dirs = {
            'context_consistency', 'context_metrics', 'context_satisfaction',
            'context_similarity', 'context_weighted_ranking', 
            'context_weighted_metrics', 'evaluation', '__pycache__'
        }
        
        model_dirs = [
            d for d in results_dir.iterdir()
            if d.is_dir() and d.name not in exclude_dirs and not d.name.startswith('.')
        ]
        
        print(f"Found {len(model_dirs)} model directories")
        print()
        
        for model_dir in sorted(model_dirs):
            model_name = self._extract_model_name(model_dir.name)
            pred_file = self._find_predictions_file(model_dir, model_name)
            
            if not pred_file:
                print(f"⚠ Skipping {model_name}: no predictions")
                print()
                continue
            
            print(f"✓ Found: {pred_file.relative_to(results_dir)}")
            
            results = self.evaluate_model(model_name, pred_file, cutoffs, alpha)
            
            if results:
                all_results[model_name] = results
            
            print()
        
        self.results = all_results
    
    def _find_predictions_file(self, model_dir, model_name):
        """Find predictions file"""
        candidates = [
            model_dir / 'result' / f"{model_name}_final_predictions.tsv",
            model_dir / 'result' / f"{model_dir.name}_final_predictions.tsv",
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return candidate
        
        if (model_dir / 'result').exists():
            pred_files = list((model_dir / 'result').glob('*predictions.tsv'))
            if pred_files:
                return pred_files[0]
        
        return None
    
    def _extract_model_name(self, dirname):
        """Extract model name"""
        name = dirname.split('_f0')[0].split('_seed')[0]
        
        if name.lower() in ['random', 'pop', 'popularity']:
            return name.capitalize()
        
        if name.upper() == name and len(name) <= 5:
            return name.upper()
        
        return name.capitalize() if name else dirname
    
    # =========================================================================
    # REPORTING
    # =========================================================================
    
    def save_results(self):
        """Save results with analysis"""
        print("="*70)
        print("SAVING RESULTS")
        print("="*70)
        
        if not self.results:
            print("✗ No results to save")
            return None
        
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.round(4)
        
        # Organize columns
        cutoffs = self.config['cutoffs']
        
        ordered_cols = []
        
        # Similarity metrics
        for sim in ['CS', 'WCA', 'Friction']:
            ordered_cols.append(f'{sim}@all')
            for k in cutoffs:
                col = f'{sim}@{k}'
                if col in results_df.columns:
                    ordered_cols.append(col)
        
        # Ranking metrics
        for sim in ['CS', 'WCA', 'Friction']:
            for k in cutoffs:
                ndcg_col = f'{sim}_CW-nDCG@{k}'
                map_col = f'{sim}_CW-MAP@{k}'
                if ndcg_col in results_df.columns:
                    ordered_cols.append(ndcg_col)
                if map_col in results_df.columns:
                    ordered_cols.append(map_col)
        
        results_df = results_df[ordered_cols]
        results_df = results_df.sort_index()
        
        # Save CSV
        csv_path = self.output_dir / f"bgg_context_metrics_{self.timestamp}.csv"
        results_df.to_csv(csv_path)
        
        print(f"Results saved to: {csv_path}")
        print()
        print(results_df.to_string())
        print()
        
        # Best models
        print("="*70)
        print("BEST MODELS BY METRIC")
        print("="*70)
        
        print("\nSimilarity Metrics:")
        for sim in ['CS', 'WCA', 'Friction']:
            print(f"\n  {sim}:")
            for k in ['all'] + cutoffs:
                col = f'{sim}@{k}'
                if col in results_df.columns:
                    best_model = results_df[col].idxmax()
                    best_value = results_df[col].max()
                    print(f"    @{k}: {best_model} ({best_value:.4f})")
        
        print("\n\nRanking Metrics:")
        for sim in ['CS', 'WCA', 'Friction']:
            print(f"\n  {sim}-weighted:")
            for k in cutoffs:
                ndcg_col = f'{sim}_CW-nDCG@{k}'
                map_col = f'{sim}_CW-MAP@{k}'
                if ndcg_col in results_df.columns:
                    best_model = results_df[ndcg_col].idxmax()
                    best_value = results_df[ndcg_col].max()
                    print(f"    CW-nDCG@{k}: {best_model} ({best_value:.4f})")
                if map_col in results_df.columns:
                    best_model = results_df[map_col].idxmax()
                    best_value = results_df[map_col].max()
                    print(f"    CW-MAP@{k}: {best_model} ({best_value:.4f})")
        
        return results_df
    
    def create_visualizations(self, results_df):
        """Create comparison plots"""
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        if results_df is None or results_df.empty:
            print("✗ No data to visualize")
            return
        
        cutoffs = self.config['cutoffs']
        
        # Figure 1: Similarity Metrics
        fig1, axes1 = plt.subplots(1, 3, figsize=(20, 6))
        fig1.suptitle('Context Similarity Metrics', fontsize=16, fontweight='bold')
        
        for idx, sim in enumerate(['CS', 'WCA', 'Friction']):
            ax = axes1[idx]
            cols = [f'{sim}@{k}' for k in cutoffs if f'{sim}@{k}' in results_df.columns]
            if cols:
                results_df[cols].plot(kind='bar', ax=ax, rot=45)
                ax.set_title(f'{sim}', fontweight='bold')
                ax.set_ylabel('Score')
                ax.legend(title='Cutoff')
                ax.grid(axis='y', alpha=0.3)
                ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        plot1_path = self.output_dir / f"similarity_metrics_{self.timestamp}.png"
        plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
        print(f"Similarity plots saved to: {plot1_path}")
        plt.close()
        
        # Figure 2: Ranking Metrics
        fig2, axes2 = plt.subplots(2, 3, figsize=(20, 12))
        fig2.suptitle('Context-Weighted Ranking Metrics', fontsize=16, fontweight='bold')
        
        # CW-nDCG row
        for idx, sim in enumerate(['CS', 'WCA', 'Friction']):
            ax = axes2[0, idx]
            cols = [f'{sim}_CW-nDCG@{k}' for k in cutoffs if f'{sim}_CW-nDCG@{k}' in results_df.columns]
            if cols:
                results_df[cols].plot(kind='bar', ax=ax, rot=45)
                ax.set_title(f'{sim}-weighted nDCG', fontweight='bold')
                ax.set_ylabel('Score')
                ax.legend(title='Cutoff')
                ax.grid(axis='y', alpha=0.3)
                ax.set_ylim([0, 1.05])
        
        # CW-MAP row
        for idx, sim in enumerate(['CS', 'WCA', 'Friction']):
            ax = axes2[1, idx]
            cols = [f'{sim}_CW-MAP@{k}' for k in cutoffs if f'{sim}_CW-MAP@{k}' in results_df.columns]
            if cols:
                results_df[cols].plot(kind='bar', ax=ax, rot=45)
                ax.set_title(f'{sim}-weighted MAP', fontweight='bold')
                ax.set_ylabel('Score')
                ax.legend(title='Cutoff')
                ax.grid(axis='y', alpha=0.3)
                ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        plot2_path = self.output_dir / f"ranking_metrics_{self.timestamp}.png"
        plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
        print(f"Ranking plots saved to: {plot2_path}")
        plt.close()
    
    # =========================================================================
    # MAIN EXECUTION
    # =========================================================================
    
    def run(self):
        """Execute evaluation pipeline"""
        print("\n" + "="*70)
        print("BGG CONTEXT-AWARE EVALUATOR")
        print("="*70)
        print(f"Timestamp: {self.timestamp}")
        print(f"Cutoffs: {self.config['cutoffs']}")
        print(f"Alpha (CS penalty): {self.config['alpha']}")
        print(f"Relevance threshold: {self.config.get('relevance_threshold', 7.0)}")
        print()
        
        try:
            self.load_test_set()
            self.load_context_info()
            self.evaluate_all_models()
            results_df = self.save_results()
            
            if results_df is not None:
                self.create_visualizations(results_df)
            
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


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == '__main__':
    config = {
        'test_path': './data/test_df.tsv',
        'context_info_path': './data/context_info.tsv',
        'results_dir': './outputs',
        'output_dir': './results/bgg/context_metrics',
        'cutoffs': [5, 10, 20],
        'alpha': 0.5,
        'relevance_threshold': 7.0,
    }
    
    evaluator = BGGEvaluator(config)
    success = evaluator.run()
    
    import sys
    sys.exit(0 if success else 1)