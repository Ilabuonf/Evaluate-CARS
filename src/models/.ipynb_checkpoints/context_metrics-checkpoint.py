"""
Shared Context Metrics Implementation
======================================

Universal implementation of context-aware metrics.
Used by multiple dataset pipelines (Frappe, Yelp).

Metrics:
    - Context Consistency (ACC@K)
    - Context Satisfaction (CS@K)
    - Weighted Context Satisfaction (WCS@K)
    - Dimensional Satisfaction (by feature groups)
    - Alternative Similarity Metrics (WCA, Friction)
    - Advanced Metrics (CR, CRC, CGB)
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from scipy.stats import spearmanr
import warnings
from scipy.stats import ConstantInputWarning


class ContextMetrics:
    """
    Universal context-aware metric implementations.
    
    CRITICAL: Single merge in _prepare_merge_dataframes, no double merging.
    """
    
    def __init__(self, context_features: List[str], feature_groups: Dict[str, List[str]]):
        """
        Args:
            context_features: List of context feature names
            feature_groups: Dict mapping group names to feature lists
                Example: {'temporal': ['daytime', 'weekday'], 
                         'spatial': ['city', 'country']}
        """
        self.context_features = context_features
        self.feature_groups = feature_groups
        self.idf_weights = {}
        
    def compute_idf_weights(self, context_df: pd.DataFrame):
        """
        Compute IDF (Inverse Document Frequency) weights for context features.
        
        Rare features get higher weights.
        Formula: IDF = log((N+1)/(df+1)) + 1
        
        Args:
            context_df: DataFrame with context features
        """
        n_contexts = context_df.drop_duplicates().shape[0]
        
        for feat in self.context_features:
            if feat in context_df.columns:
                df_f = context_df[feat].astype(str).nunique()
                self.idf_weights[feat] = np.log((n_contexts + 1) / (df_f + 1)) + 1
        
        print(f"\n  IDF Weights computed for {len(self.idf_weights)} features:")
        sorted_weights = sorted(self.idf_weights.items(), key=lambda x: x[1], reverse=True)
        print("    Most discriminative features:")
        for feat, weight in sorted_weights[:5]:
            print(f"      {feat}: {weight:.4f}")
        print("    Least discriminative features:")
        for feat, weight in sorted_weights[-5:]:
            print(f"      {feat}: {weight:.4f}")
    
    def _prepare_merge_dataframes(self, predictions_df: pd.DataFrame, 
                                  context_info: pd.DataFrame) -> pd.DataFrame:
        """
        CRITICAL: Does ALL the merging. Returns complete DataFrame.
        
        This method:
        1. Extracts query context from q_context_id
        2. Merges with item context from context_info
        3. Adds rank column if missing
        
        Returns single DataFrame with both query and item contexts.
        """
        pred_df = predictions_df.copy()
        ctx_df = context_info.copy()
        
        # Extract query context from q_context_id (format: feat1_feat2_feat3_...)
        context_splits = pred_df['q_context_id'].str.split('_', expand=True)
        
        if context_splits.shape[1] < len(self.context_features):
            print(f"    ⚠ WARNING: q_context_id has {context_splits.shape[1]} parts, "
                  f"expected {len(self.context_features)}")
        
        # Assign query context columns
        for i, feat in enumerate(self.context_features):
            if i < context_splits.shape[1]:
                pred_df[f'{feat}_query'] = context_splits[i].astype(str).str.strip()
            else:
                pred_df[f'{feat}_query'] = ''
        
        # Prepare context_info for merge
        ctx_df['item_id:token'] = ctx_df['item_id:token'].astype(str).str.strip()
        pred_df['item_id:token'] = pred_df['item_id:token'].astype(str).str.strip()
        
        for feat in self.context_features:
            if feat in ctx_df.columns:
                ctx_df[feat] = ctx_df[feat].astype(str).str.strip()
        
        # Rename to _item suffix
        rename_dict = {feat: f'{feat}_item' for feat in self.context_features 
                      if feat in ctx_df.columns}
        ctx_df = ctx_df.rename(columns=rename_dict)
        
        # SINGLE merge - this is the only merge!
        pred_df = pred_df.merge(ctx_df, on='item_id:token', how='left')
        
        # Ensure rank exists
        if 'rank' not in pred_df.columns:
            pred_df = pred_df.sort_values(
                ['user_id:token', 'q_context_id', 'prediction'], 
                ascending=[True, True, False]
            )
            pred_df['rank'] = (pred_df.groupby(['user_id:token', 'q_context_id'])
                                      .cumcount() + 1)
        
        return pred_df
    
    # =========================================================================
    # CONTEXT CONSISTENCY (ACC@K)
    # =========================================================================
    
    def context_consistency(self, predictions_df: pd.DataFrame, 
                          context_info: pd.DataFrame,
                          k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """
        Compute Average Context Consistency (ACC@K).
        
        Percentage of top-K items with EXACT context match to query.
        
        Args:
            predictions_df: Predictions with columns [user_id:token, item_id:token, 
                           q_context_id, prediction]
            context_info: Item contexts with columns [item_id:token, ...context_features]
            k_values: List of K values to evaluate
        
        Returns:
            Dict with keys like 'ACC@5', 'ACC@10'
        """
        results = {}
        
        # Get prepared data - NO additional merge!
        pred_with_context = self._prepare_merge_dataframes(predictions_df, context_info)
        
        for k in k_values:
            top_k = (pred_with_context
                    .sort_values(['user_id:token', 'q_context_id', 'rank'])
                    .groupby(['user_id:token', 'q_context_id'])
                    .head(k))
            
            query_context_cols = [f'{f}_query' for f in self.context_features]
            item_context_cols = [f'{f}_item' for f in self.context_features]
            
            if not all(col in top_k.columns for col in query_context_cols + item_context_cols):
                results[f'ACC@{k}'] = 0.0
                continue
            
            # Compare each feature
            feature_matches = []
            for feat in self.context_features:
                q_col = f'{feat}_query'
                i_col = f'{feat}_item'
                match = (top_k[q_col].astype(str) == top_k[i_col].astype(str))
                feature_matches.append(match)
            
            # All features must match
            top_k['context_match'] = pd.concat(feature_matches, axis=1).all(axis=1).astype(int)
            
            acc = top_k.groupby(['user_id:token', 'q_context_id'])['context_match'].mean().mean()
            results[f'ACC@{k}'] = acc
        
        return results
    
    # =========================================================================
    # CONTEXT SATISFACTION (CS@K)
    # =========================================================================
    
    def context_satisfaction_score(self, query_context: pd.Series, 
                                   item_context: pd.Series,
                                   alpha: float = 0.5) -> float:
        """
        Compute Context Satisfaction score for a single item.
        
        Modified Jaccard with penalty for missing features:
        CS = |C_q ∩ C_i| / (|C_q ∪ C_i| + α·|C_q \ C_i|/|C_q|)
        """
        q_set = set(query_context[query_context.notna()].astype(str).values)
        i_set = set(item_context[item_context.notna()].astype(str).values)
        
        intersection = len(q_set & i_set)
        union = len(q_set | i_set)
        difference = len(q_set - i_set)
        q_size = len(q_set)
        
        if q_size == 0 or union == 0:
            return 0.0
        
        penalty = alpha * (difference / q_size)
        cs = intersection / (union + penalty)
        return cs
    
    def context_satisfaction(self, predictions_df: pd.DataFrame,
                           context_info: pd.DataFrame,
                           alpha: float = 0.5,
                           k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """Compute Average Context Satisfaction (ACS@K)"""
        results = {}
        
        # Get prepared data
        pred_with_context = self._prepare_merge_dataframes(predictions_df, context_info)
        
        query_context_cols = [f'{f}_query' for f in self.context_features]
        item_context_cols = [f'{f}_item' for f in self.context_features]
        
        # Compute CS for each prediction
        cs_scores = []
        for idx, row in pred_with_context.iterrows():
            q_ctx = row[query_context_cols]
            i_ctx = row[item_context_cols]
            cs = self.context_satisfaction_score(q_ctx, i_ctx, alpha)
            cs_scores.append(cs)
        
        pred_with_context['cs_score'] = cs_scores
        
        # Aggregate by K
        for k in k_values:
            top_k = (pred_with_context
                    .sort_values(['user_id:token', 'q_context_id', 'rank'])
                    .groupby(['user_id:token', 'q_context_id'])
                    .head(k))
            
            acs = top_k.groupby(['user_id:token', 'q_context_id'])['cs_score'].mean().mean()
            results[f'CS@{k}'] = acs
        
        results['CS@all'] = pred_with_context['cs_score'].mean()
        return results
    
    # =========================================================================
    # WEIGHTED CONTEXT SATISFACTION (WCS@K)
    # =========================================================================
    
    def weighted_context_satisfaction_score(self, query_context, item_context, alpha=0.5):
        """
        Compute Weighted Context Satisfaction using IDF weights.
        
        WCS = Σ(matched features × IDF) / 
              (Σ(union features × IDF) + α × Σ(missing features × IDF) / Σ(requested features × IDF))
        """
        query_features = set()
        item_features = set()
        matched_features = set()
        
        for feat in self.context_features:
            q_val = str(query_context.get(feat, '')).strip()
            i_val = str(item_context.get(feat, '')).strip()
            
            # Feature present in query
            if q_val and q_val != 'nan':
                query_features.add(feat)
            
            # Feature present in item
            if i_val and i_val != 'nan':
                item_features.add(feat)
            
            # Feature matched (same value)
            if q_val and i_val and q_val == i_val:
                matched_features.add(feat)
        
        # Compute weighted sets
        union_features = query_features | item_features
        missing_features = query_features - item_features
        
        # IDF weights
        intersection_weight = sum(self.idf_weights.get(f, 1.0) for f in matched_features)
        union_weight = sum(self.idf_weights.get(f, 1.0) for f in union_features)
        missing_weight = sum(self.idf_weights.get(f, 1.0) for f in missing_features)
        requested_weight = sum(self.idf_weights.get(f, 1.0) for f in query_features)
        
        # Avoid division by zero
        if union_weight == 0 or requested_weight == 0:
            return 0.0
        
        # WCS formula
        penalty = alpha * (missing_weight / requested_weight)
        wcs = intersection_weight / (union_weight + penalty)
        
        return wcs
    
    def weighted_context_satisfaction(self, predictions_df: pd.DataFrame,
                                    context_info: pd.DataFrame,
                                    alpha: float = 0.5,
                                    k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """Compute Average Weighted Context Satisfaction (WCS@K)"""
        results = {}
        
        pred_with_context = self._prepare_merge_dataframes(predictions_df, context_info)
        
        # Compute WCS for each prediction
        wcs_scores = []
        for idx, row in pred_with_context.iterrows():
            q_ctx = pd.Series({f: row.get(f'{f}_query') for f in self.context_features})
            i_ctx = pd.Series({f: row.get(f'{f}_item') for f in self.context_features})
            wcs = self.weighted_context_satisfaction_score(q_ctx, i_ctx, alpha)
            wcs_scores.append(wcs)
        
        pred_with_context['wcs_score'] = wcs_scores
        
        # Aggregate by K
        for k in k_values:
            top_k = (pred_with_context
                    .sort_values(['user_id:token', 'q_context_id', 'rank'])
                    .groupby(['user_id:token', 'q_context_id'])
                    .head(k))
            
            awcs = top_k.groupby(['user_id:token', 'q_context_id'])['wcs_score'].mean().mean()
            results[f'WCS@{k}'] = awcs
        
        results['WCS@all'] = pred_with_context['wcs_score'].mean()
        return results
    
    # =========================================================================
    # DIMENSIONAL ANALYSIS
    # =========================================================================
    
    def dimensional_satisfaction(self, predictions_df: pd.DataFrame,
                                context_info: pd.DataFrame,
                                alpha: float = 0.5,
                                k: int = 5) -> Dict[str, float]:
        """Compute WCS separately for each feature group"""
        results = {}
        
        pred_with_context = self._prepare_merge_dataframes(predictions_df, context_info)
        
        top_k = (pred_with_context
                .sort_values(['user_id:token', 'q_context_id', 'rank'])
                .groupby(['user_id:token', 'q_context_id'])
                .head(k))
        
        for group_name, group_features in self.feature_groups.items():
            group_scores = []
            
            for idx, row in top_k.iterrows():
                q_ctx = pd.Series({f: row.get(f'{f}_query') for f in group_features})
                i_ctx = pd.Series({f: row.get(f'{f}_item') for f in group_features})
                wcs = self.weighted_context_satisfaction_score(q_ctx, i_ctx, alpha)
                group_scores.append(wcs)
            
            results[f'WCS_{group_name}@{k}'] = np.mean(group_scores) if group_scores else 0.0
        
        return results
    
    # =========================================================================
    # ALTERNATIVE SIMILARITY METRICS
    # =========================================================================
    
    def cosine_similarity_score(self, query_context: pd.Series,
                           item_context: pd.Series) -> float:
        """
        Weighted Context Alignment (WCA) - Normalized weighted match score.
        
        Returns WCA score in [0, 1] where 1.0 = all query features matched.
        """
        matched_weight = 0.0
        total_query_weight = 0.0
        
        for feat in self.context_features:
            q_val = str(query_context.get(feat, '')).strip()
            i_val = str(item_context.get(feat, '')).strip()
            
            w = self.idf_weights.get(feat, 1.0)
            
            # Count weight if query has this feature
            if q_val and q_val != 'nan':
                total_query_weight += w
                
                # Add to matched weight if item also has it AND values match
                if i_val and i_val != 'nan' and q_val == i_val:
                    matched_weight += w
        
        if total_query_weight == 0:
            return 0.0
        
        wca = matched_weight / total_query_weight
        return np.clip(wca, 0.0, 1.0)
    
    def hamming_distance(self, query_context: pd.Series,
                    item_context: pd.Series) -> float:
        """Context Friction (inverted Hamming distance)"""
        distance = 0
        max_distance = len(self.context_features)
        
        for feat in self.context_features:
            q_val = str(query_context.get(feat, '')) if pd.notna(query_context.get(feat)) else ''
            i_val = str(item_context.get(feat, '')) if pd.notna(item_context.get(feat)) else ''
            if q_val != i_val:
                distance += 1
        
        if max_distance == 0:
            return 1.0
        
        return 1.0 - (distance / max_distance)
    
    def alternative_metrics(self, predictions_df: pd.DataFrame,
                          context_info: pd.DataFrame,
                          k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """Compute WCA and Friction metrics"""
        results = {}
        
        pred_with_context = self._prepare_merge_dataframes(predictions_df, context_info)
        
        wca_scores = []
        friction_scores = []
        
        for idx, row in pred_with_context.iterrows():
            q_ctx = pd.Series({f: row.get(f'{f}_query') for f in self.context_features})
            i_ctx = pd.Series({f: row.get(f'{f}_item') for f in self.context_features})
            
            wca = self.cosine_similarity_score(q_ctx, i_ctx)
            friction = self.hamming_distance(q_ctx, i_ctx)
            
            wca_scores.append(wca)
            friction_scores.append(friction)
        
        pred_with_context['wca_score'] = wca_scores
        pred_with_context['friction'] = friction_scores
        
        for k in k_values:
            top_k = (pred_with_context
                    .sort_values(['user_id:token', 'q_context_id', 'rank'])
                    .groupby(['user_id:token', 'q_context_id'])
                    .head(k))
            
            results[f'WCA@{k}'] = top_k.groupby(['user_id:token', 'q_context_id'])['wca_score'].mean().mean()
            results[f'Friction@{k}'] = top_k.groupby(['user_id:token', 'q_context_id'])['friction'].mean().mean()
        
        results['WCA@all'] = pred_with_context['wca_score'].mean()
        results['Friction@all'] = pred_with_context['friction'].mean()
        return results
    
    # =========================================================================
    # ADVANCED METRICS (CR, CRC, CGB)
    # =========================================================================
    
    def context_recall(self, predictions_df: pd.DataFrame,
                      context_info: pd.DataFrame,
                      k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """Context Recall (CR@K): Feature coverage metric"""
        results = {}
        
        pred_with_context = self._prepare_merge_dataframes(predictions_df, context_info)
        
        recall_scores = []
        for idx, row in pred_with_context.iterrows():
            q_features = set()
            i_features = set()
            
            for feat in self.context_features:
                q_val = row.get(f'{feat}_query')
                i_val = row.get(f'{feat}_item')
                
                if pd.notna(q_val):
                    q_features.add(f'{feat}={q_val}')
                if pd.notna(i_val):
                    i_features.add(f'{feat}={i_val}')
            
            if len(q_features) > 0:
                recall = len(q_features & i_features) / len(q_features)
            else:
                recall = 0.0
            
            recall_scores.append(recall)
        
        pred_with_context['recall'] = recall_scores
        
        for k in k_values:
            top_k = (pred_with_context
                    .sort_values(['user_id:token', 'q_context_id', 'rank'])
                    .groupby(['user_id:token', 'q_context_id'])
                    .head(k))
            
            results[f'CR@{k}'] = top_k.groupby(['user_id:token', 'q_context_id'])['recall'].mean().mean()
        
        results['CR@all'] = pred_with_context['recall'].mean()
        return results
    
    def context_ranking_correlation(self, predictions_df: pd.DataFrame,
                                   context_info: pd.DataFrame,
                                   k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """Context Ranking Correlation (CRC@K): Ranking coherence metric"""
        results = {}
        
        pred_with_context = self._prepare_merge_dataframes(predictions_df, context_info)
        
        # Compute context scores
        cs_scores = []
        for idx, row in pred_with_context.iterrows():
            q_ctx = pd.Series({f: row.get(f'{f}_query') for f in self.context_features})
            i_ctx = pd.Series({f: row.get(f'{f}_item') for f in self.context_features})
            cs = self.context_satisfaction_score(q_ctx, i_ctx, alpha=0.5)
            cs_scores.append(cs)
        
        pred_with_context['cs_score'] = cs_scores
        
        for k in k_values:
            correlations = []
            
            for (user, ctx), group in pred_with_context.groupby(['user_id:token', 'q_context_id']):
                group = group.sort_values('rank').head(k)
                
                if len(group) < 2:
                    continue
                
                ranks = group['rank'].values
                scores = group['cs_score'].values

                if np.std(ranks) > 0 and np.std(scores) > 0:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=ConstantInputWarning)
                        try:
                            rho, _ = spearmanr(ranks, scores)
                            if not np.isnan(rho):
                                crc = (rho + 1) / 2
                                correlations.append(crc)
                        except:
                            pass
            
            results[f'CRC@{k}'] = np.mean(correlations) if correlations else 0.5
        
        results['CRC@all'] = results.get(f'CRC@{max(k_values)}', 0.5)
        return results
    
    def context_group_balance(self, predictions_df: pd.DataFrame,
                             context_info: pd.DataFrame,
                             k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """Context Group Balance (CGB@K): Dimensional fairness metric"""
        results = {}
        
        pred_with_context = self._prepare_merge_dataframes(predictions_df, context_info)
        
        for k in k_values:
            top_k = (pred_with_context
                    .sort_values(['user_id:token', 'q_context_id', 'rank'])
                    .groupby(['user_id:token', 'q_context_id'])
                    .head(k))
            
            group_recalls = []
            
            for group_name, group_features in self.feature_groups.items():
                recalls = []
                
                for idx, row in top_k.iterrows():
                    q_feats = set()
                    i_feats = set()
                    
                    for feat in group_features:
                        q_val = row.get(f'{feat}_query')
                        i_val = row.get(f'{feat}_item')
                        
                        if pd.notna(q_val):
                            q_feats.add(f'{feat}={q_val}')
                        if pd.notna(i_val):
                            i_feats.add(f'{feat}={i_val}')
                    
                    if len(q_feats) > 0:
                        recall = len(q_feats & i_feats) / len(q_feats)
                    else:
                        recall = 0.0
                    
                    recalls.append(recall)
                
                group_recall = np.mean(recalls) if recalls else 0.0
                group_recalls.append(group_recall)
            
            # Compute balance
            if len(group_recalls) > 1:
                std_dev = np.std(group_recalls)
                cgb = 1 - min(std_dev / 0.5, 1.0)
            else:
                cgb = 1.0
            
            results[f'CGB@{k}'] = cgb
        
        results['CGB@all'] = results.get(f'CGB@{max(k_values)}', 1.0)
        return results