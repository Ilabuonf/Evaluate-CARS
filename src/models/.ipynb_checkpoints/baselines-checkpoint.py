"""
Baseline Models for Context-Aware Recommendation
=================================================

Non-context-aware baseline models that serve as performance lower bounds.

Models:
    - Random: Uniform random recommendations
    - Popularity: Most popular items to everyone
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm


class BaselineModel:
    """Base class for baseline models"""
    
    def __init__(self, name: str, train_df: pd.DataFrame, test_df: pd.DataFrame,
                 config: Dict, context_features: List[str], column_names: Dict[str, str]):
        """
        Args:
            name: Model name
            train_df: Training data
            test_df: Test data
            config: Configuration dictionary
            context_features: List of context feature names
            column_names: Dict mapping generic names to dataset columns
                Example: {'user': 'user_id:token', 'item': 'game_id:token', 'label': 'label'}
        """
        self.name = name
        self.train_df = train_df
        self.test_df = test_df
        self.config = config
        self.context_features = context_features
        self.column_names = column_names
        self.all_items = None
        self.predictions = []
        
    def fit(self):
        """Train the baseline model"""
        raise NotImplementedError
        
    def predict(self, top_k: int = 50):
        """Generate top-K predictions for all user-context pairs in test set"""
        raise NotImplementedError
        
    def _create_context_id(self, row: pd.Series) -> str:
        """Create context_id string from context features"""
        return '_'.join([str(row[f]) for f in self.context_features])
    
    def save_predictions(self, output_dir: Path):
        """Save predictions in RecBole-compatible format"""
        pred_df = pd.DataFrame(self.predictions)
        
        # Add rank if not present
        if 'rank' not in pred_df.columns:
            pred_df = pred_df.sort_values(
                ['user_id:token', 'q_context_id', 'prediction'], 
                ascending=[True, True, False]
            )
            pred_df['rank'] = (pred_df.groupby(['user_id:token', 'q_context_id'])
                                      .cumcount() + 1)
        
        # Get item context from training data (mode for each item)
        item_cols = self._get_item_column_name()
        item_context = (
            self.train_df.groupby(item_cols)[self.context_features]
            .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
            .reset_index()
            .rename(columns={item_cols: 'item_id:token'})
        )
        
        # Merge with predictions
        pred_df['item_id:token'] = pred_df['item_id:token'].astype(str).str.strip()
        item_context['item_id:token'] = item_context['item_id:token'].astype(str).str.strip()
        
        final_df = pred_df.merge(item_context, on='item_id:token', how='left')
        
        # Select columns
        cols = ['user_id:token', 'item_id:token', 'q_context_id', 
                'prediction', 'rank'] + self.context_features
        final_df = final_df[[c for c in cols if c in final_df.columns]]
        
        # Save
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'{self.name}_final_predictions.tsv'
        final_df.to_csv(output_path, sep='\t', index=False)
        
        print(f"    ✓ {len(final_df):,} predictions saved to {output_path}")
        return output_path
    
    def _get_item_column_name(self) -> str:
        """Get item column name from column mapping"""
        return self.column_names.get('item', 'item_id:token')
    
    def _get_user_column_name(self) -> str:
        """Get user column name from column mapping"""
        return self.column_names.get('user', 'user_id:token')


class RandomModel(BaselineModel):
    """
    Random Baseline
    
    Recommends random items with uniform probability.
    - Ignores user preferences
    - Ignores item popularity
    - Ignores context
    - Assigns random scores to all items
    
    Serves as absolute lower bound for performance.
    """
    
    def fit(self):
        """No training needed - just get item list"""
        item_col = self._get_item_column_name()
        self.all_items = sorted(self.train_df[item_col].unique())
        print(f"    Random model: {len(self.all_items)} items available")
        
    def predict(self, top_k: int = 50):
        """Generate random predictions for each user-context pair"""
        # Set seed for reproducibility
        np.random.seed(self.config.get('training', {}).get('seed', 42))
        
        # Get all unique user-context pairs from test set
        user_col = self._get_user_column_name()
        user_contexts = self.test_df[[user_col] + self.context_features].drop_duplicates()
        user_contexts['q_context_id'] = user_contexts.apply(self._create_context_id, axis=1)
        
        print(f"    Generating random predictions for {len(user_contexts):,} queries...")
        
        self.predictions = []
        
        for idx, row in tqdm(user_contexts.iterrows(), total=len(user_contexts),
                            desc="    Random predictions", leave=False):
            user = str(row[user_col])
            context_id = row['q_context_id']
            
            # Generate random scores for all items
            random_scores = np.random.uniform(0, 1, len(self.all_items))
            
            # Get top-K items based on random scores
            top_indices = np.argsort(-random_scores)[:top_k]
            
            for rank, item_idx in enumerate(top_indices):
                self.predictions.append({
                    'user_id:token': user,
                    'item_id:token': str(self.all_items[item_idx]),
                    'q_context_id': context_id,
                    'prediction': float(random_scores[item_idx]),
                    'rank': rank + 1
                })
        
        print(f"    ✓ Generated {len(self.predictions):,} random predictions")


class PopularityModel(BaselineModel):
    """
    Popularity Baseline
    
    Recommends most popular items to everyone.
    - Ignores user preferences (same recommendations for all users)
    - Uses global item popularity (interaction count)
    - Ignores context
    
    Popularity is computed as the number of interactions per item in training data.
    Often a surprisingly strong baseline, especially in domains where popular items
    are generally good recommendations.
    """
    
    def fit(self):
        """Compute item popularity from training data"""
        from collections import Counter
        
        item_col = self._get_item_column_name()
        
        # Count interactions per item
        item_counts = Counter(self.train_df[item_col])
        
        # Get all unique items and their popularity scores
        self.all_items = sorted(item_counts.keys())
        
        # Popularity score = interaction count
        self.popularity_scores = {
            item: count for item, count in item_counts.items()
        }
        
        # For items not in training (shouldn't happen), assign zero popularity
        self.default_score = 0.0
        
        # Statistics
        total_interactions = sum(item_counts.values())
        max_pop = max(self.popularity_scores.values())
        mean_pop = total_interactions / len(self.all_items)
        
        print(f"    Popularity model statistics:")
        print(f"      • Total items: {len(self.all_items):,}")
        print(f"      • Total interactions: {total_interactions:,}")
        print(f"      • Most popular item: {max_pop:,} interactions")
        print(f"      • Mean interactions: {mean_pop:.1f}")
        
    def predict(self, top_k: int = 50):
        """Generate popularity-based predictions for each user-context pair"""
        user_col = self._get_user_column_name()
        
        # Get all unique user-context pairs from test set
        user_contexts = self.test_df[[user_col] + self.context_features].drop_duplicates()
        user_contexts['q_context_id'] = user_contexts.apply(self._create_context_id, axis=1)
        
        print(f"    Generating popularity predictions for {len(user_contexts):,} queries...")
        
        # Pre-compute sorted item list by popularity (same for all users)
        items_by_popularity = sorted(
            self.all_items,
            key=lambda x: self.popularity_scores.get(x, self.default_score),
            reverse=True
        )
        
        # Get top-K most popular items
        top_k_items = items_by_popularity[:top_k]
        top_k_scores = [
            self.popularity_scores.get(item, self.default_score) 
            for item in top_k_items
        ]
        
        self.predictions = []
        
        # For each user-context pair, recommend the same top-K popular items
        for idx, row in tqdm(user_contexts.iterrows(), total=len(user_contexts),
                            desc="    Pop predictions", leave=False):
            user = str(row[user_col])
            context_id = row['q_context_id']
            
            # Add top-K popular items for this query
            for rank, (item, score) in enumerate(zip(top_k_items, top_k_scores)):
                self.predictions.append({
                    'user_id:token': user,
                    'item_id:token': str(item),
                    'q_context_id': context_id,
                    'prediction': float(score),
                    'rank': rank + 1
                })
        
        print(f"    ✓ Generated {len(self.predictions):,} popularity predictions")