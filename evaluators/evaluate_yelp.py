"""
Yelp Dataset Evaluator
======================

Consolidated evaluator for Yelp dataset.

Context features:
- Temporal: hour_of_day, day_of_week, is_weekend
- Social: review_length, user_elite
- Spatial: city, category, price_range

Usage:
    from evaluators import YelpEvaluator
    
    config = {
        'test_path': './data/yelp/yelp_test.csv',
        'train_path': './data/yelp/yelp_train.csv',
        'results_dir': './outputs/yelp',
        'output_dir': './results/yelp/context_metrics',
        'cutoffs': [5, 10, 20],
        'alpha': 0.5
    }
    
    evaluator = YelpEvaluator(config)
    evaluator.run()
"""

# Import the Frappe evaluator as base (same structure)
from .evaluate_frappe import FrappeEvaluator
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


class YelpEvaluator(FrappeEvaluator):
    """
    Yelp evaluator - extends Frappe evaluator with Yelp-specific features.
    Most methods inherited, only context features change.
    """
    
    CONTEXT_FEATURES = [
        'hour_of_day', 'day_of_week', 'is_weekend',
        'review_length', 'user_elite',
        'city', 'category', 'price_range'
    ]
    
    def __init__(self, config):
        # Override context features before calling parent init
        super().__init__(config)
        self.context_features = self.CONTEXT_FEATURES
    
    def load_test_set(self):
        """Load Yelp test set"""
        print("="*70)
        print("LOADING YELP TEST SET")
        print("="*70)
        
        test_path = Path(self.config['test_path'])
        self.test_df = pd.read_csv(test_path)
        
        # Ensure string types
        self.test_df['user_id'] = self.test_df['user_id'].astype(str).str.strip()
        self.test_df['business_id'] = self.test_df['business_id'].astype(str).str.strip()
        
        # Handle price_range NaN
        if 'price_range' in self.test_df.columns:
            self.test_df['price_range'] = self.test_df['price_range'].fillna('2').astype(int).astype(str)
        
        # Create q_context_id
        context_cols = [col for col in self.context_features if col in self.test_df.columns]
        self.test_df['q_context_id'] = (
            self.test_df[context_cols].astype(str).agg('_'.join, axis=1)
        )
        
        # Create query_id
        self.test_df['query_id'] = (
            self.test_df['user_id'] + '_' + self.test_df['q_context_id']
        )
        
        # Store ground truth (binary from stars)
        self.ground_truth = {}
        for _, row in self.test_df.iterrows():
            qid = row['query_id']
            item = str(row['business_id']).strip()
            # Binary relevance from stars (>= 4 is relevant)
            relevance = 1.0 if row['stars'] >= 4 else 0.0
            
            if qid not in self.ground_truth:
                self.ground_truth[qid] = {}
            self.ground_truth[qid][item] = relevance
        
        self.unique_query_ids = self.test_df['query_id'].unique()
        
        print(f"✓ Test set loaded: {self.test_df.shape}")
        print(f"  Unique queries: {len(self.unique_query_ids):,}")
        print(f"  Unique businesses: {self.test_df['business_id'].nunique():,}")
        print(f"  Stars distribution: {self.test_df['stars'].value_counts().sort_index().to_dict()}")
        print()
    
    def load_context_info(self):
        """Load business context information"""
        print("Loading context definitions...")
        
        train_path = Path(self.config['train_path'])
        train_df = pd.read_csv(train_path)
        
        # Ensure string types
        train_df['business_id'] = train_df['business_id'].astype(str).str.strip()
        
        # Handle price_range NaN
        if 'price_range' in train_df.columns:
            train_df['price_range'] = train_df['price_range'].fillna('2').astype(int).astype(str)
        
        # Get unique business-context combinations
        item_cols = ['business_id'] + self.context_features
        self.item_context = train_df[item_cols].groupby('business_id').agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
        ).reset_index()
        
        # Rename for consistency with parent class
        self.item_context = self.item_context.rename(columns={'business_id': 'item'})
        
        print(f"✓ Business contexts loaded: {len(self.item_context)}")
        
        # Compute IDF weights
        self._compute_idf_weights(train_df)
        print()


if __name__ == '__main__':
    config = {
        'test_path': './data/yelp/yelp_test.csv',
        'train_path': './data/yelp/yelp_train.csv',
        'results_dir': './outputs/yelp',
        'output_dir': './results/yelp/context_metrics',
        'cutoffs': [5, 10, 20],
        'alpha': 0.5,
    }
    
    evaluator = YelpEvaluator(config)
    success = evaluator.run()
    
    import sys
    sys.exit(0 if success else 1)