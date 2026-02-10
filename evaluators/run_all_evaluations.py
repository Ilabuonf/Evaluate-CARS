#!/usr/bin/env python3
"""
Master Evaluation Script
=========================

Run all evaluators for all datasets with complete metrics.

Usage:
    python run_all_evaluations.py                    # All datasets
    python run_all_evaluations.py --dataset bgg      # Single dataset
    python run_all_evaluations.py --parallel         # Parallel execution
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import subprocess
from typing import List, Dict

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

# Import evaluators
from evaluate_bgg_complete import CompleteBGGEvaluator
from evaluate_frappe_complete import CompleteFrappeEvaluator
from evaluate_yelp_complete import CompleteYelpEvaluator


class MasterEvaluator:
    """Run all evaluations"""
    
    DATASETS = {
        'bgg': {
            'name': 'BoardGameGeek',
            'evaluator': CompleteBGGEvaluator,
            'config': {
                'test_path': './datasets/bgg/test_df.tsv',
                'context_info_path': './datasets/bgg/context_info.tsv',
                'results_dir': './outputs',
                'output_dir': './results/bgg/complete_metrics',
                'rating_threshold': 7.0
            }
        },
        'frappe': {
            'name': 'Frappe',
            'evaluator': CompleteFrappeEvaluator,
            'config': {
                'test_path': './datasets/frappe/test_df.tsv',
                'train_path': './datasets/frappe/train_df.tsv',
                'results_dir': './outputs',
                'output_dir': './results/frappe/complete_metrics'
            }
        },
        'yelp': {
            'name': 'Yelp',
            'evaluator': CompleteYelpEvaluator,
            'config': {
                'test_path': './datasets/yelp/test_df.tsv',
                'train_path': './datasets/yelp/train_df.tsv',
                'results_dir': './outputs',
                'output_dir': './results/yelp/complete_metrics',
                'stars_threshold': 4.0
            }
        }
    }
    
    def __init__(self, datasets: List[str] = None, parallel: bool = False):
        self.datasets_to_run = datasets or list(self.DATASETS.keys())
        self.parallel = parallel
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def run_single_dataset(self, dataset_key: str) -> bool:
        """Run evaluation for single dataset"""
        if dataset_key not in self.DATASETS:
            print(f"✗ Unknown dataset: {dataset_key}")
            return False
        
        dataset_info = self.DATASETS[dataset_key]
        
        print("\n" + "="*80)
        print(f"EVALUATING {dataset_info['name'].upper()}")
        print("="*80)
        
        try:
            evaluator_class = dataset_info['evaluator']
            config = dataset_info['config']
            
            evaluator = evaluator_class(config)
            success = evaluator.run()
            
            self.results[dataset_key] = {
                'success': success,
                'output_dir': config['output_dir']
            }
            
            return success
            
        except Exception as e:
            print(f"\n✗ {dataset_info['name']} failed: {e}")
            import traceback
            traceback.print_exc()
            
            self.results[dataset_key] = {
                'success': False,
                'error': str(e)
            }
            
            return False
    
    def run_sequential(self):
        """Run datasets sequentially"""
        print("\n" + "="*80)
        print("SEQUENTIAL EVALUATION MODE")
        print("="*80)
        print(f"Datasets: {', '.join(self.datasets_to_run)}")
        print()
        
        for dataset_key in self.datasets_to_run:
            success = self.run_single_dataset(dataset_key)
            if not success:
                print(f"\n⚠ {dataset_key} evaluation failed, continuing...\n")
    
    def run_parallel(self):
        """Run datasets in parallel"""
        print("\n" + "="*80)
        print("PARALLEL EVALUATION MODE")
        print("="*80)
        print(f"Datasets: {', '.join(self.datasets_to_run)}")
        print()
        
        import concurrent.futures
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(self.run_single_dataset, dataset_key): dataset_key
                for dataset_key in self.datasets_to_run
            }
            
            for future in concurrent.futures.as_completed(futures):
                dataset_key = futures[future]
                try:
                    success = future.result()
                    if success:
                        print(f"\n✓ {dataset_key} completed")
                    else:
                        print(f"\n✗ {dataset_key} failed")
                except Exception as e:
                    print(f"\n✗ {dataset_key} error: {e}")
    
    def run(self):
        """Execute evaluations"""
        print("\n" + "="*80)
        print("MASTER EVALUATOR - ALL DATASETS, ALL METRICS")
        print("="*80)
        print(f"Timestamp: {self.timestamp}")
        print(f"Mode: {'Parallel' if self.parallel else 'Sequential'}")
        print()
        
        if self.parallel:
            self.run_parallel()
        else:
            self.run_sequential()
        
        self.print_summary()
    
    def print_summary(self):
        """Print summary of results"""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        for dataset_key in self.datasets_to_run:
            if dataset_key in self.results:
                result = self.results[dataset_key]
                status = "✓ Success" if result['success'] else "✗ Failed"
                print(f"\n{self.DATASETS[dataset_key]['name']:15s} {status}")
                
                if result['success']:
                    print(f"  Output: {result['output_dir']}")
                else:
                    if 'error' in result:
                        print(f"  Error: {result['error']}")
        
        print("\n" + "="*80)
        
        # Overall status
        total = len(self.datasets_to_run)
        successful = sum(1 for r in self.results.values() if r.get('success', False))
        
        print(f"\nResults: {successful}/{total} datasets evaluated successfully")
        
        if successful == total:
            print("\n🎉 ALL EVALUATIONS COMPLETED SUCCESSFULLY!")
        elif successful > 0:
            print(f"\n⚠ {total - successful} dataset(s) failed")
        else:
            print("\n✗ ALL EVALUATIONS FAILED")
        
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Run complete evaluations for all datasets'
    )
    parser.add_argument(
        '--dataset',
        choices=['bgg', 'frappe', 'yelp'],
        help='Evaluate single dataset (default: all)'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run datasets in parallel'
    )
    
    args = parser.parse_args()
    
    datasets = [args.dataset] if args.dataset else None
    
    master = MasterEvaluator(datasets=datasets, parallel=args.parallel)
    master.run()


if __name__ == '__main__':
    main()