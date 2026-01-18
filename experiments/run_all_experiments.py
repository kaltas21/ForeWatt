#!/usr/bin/env python3
"""
Run All Experiments
===================
Main entry point to run all consumption and price model experiments.

This script will:
1. Run all consumption model experiments
2. Run all price model experiments
3. Generate combined summary and comparison plots
4. Save everything to experiments/results/

Usage:
    python experiments/run_all_experiments.py

Output:
    experiments/results/
    ├── consumption/
    │   └── {timestamp}/
    │       ├── data_size_*/
    │       ├── features_*/
    │       ├── hyperparam_*/
    │       ├── baselines/
    │       └── summary.json
    ├── price/
    │   └── {timestamp}/
    │       ├── data_size_*/
    │       ├── features_*/
    │       ├── model_*/
    │       ├── correction_*/
    │       ├── baselines/
    │       └── summary.json
    └── combined_summary_{timestamp}.json
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.config import RESULTS_DIR, RUN_TIMESTAMP
from experiments.run_consumption_experiments import run_all_consumption_experiments
from experiments.run_price_experiments import run_all_price_experiments

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_combined_summary(
    consumption_results: dict,
    price_results: dict
) -> dict:
    """Create a combined summary of all experiments."""

    # Extract best results
    consumption_metrics = {name: r['test_metrics'] for name, r in consumption_results[0].items()}
    price_metrics = {name: r['test_metrics'] for name, r in price_results[0].items()}

    best_consumption = min(consumption_metrics.items(), key=lambda x: x[1]['sMAPE'])
    best_price = min(price_metrics.items(), key=lambda x: x[1]['sMAPE'])

    summary = {
        'timestamp': datetime.now().isoformat(),
        'run_id': RUN_TIMESTAMP,
        'consumption': {
            'n_experiments': len(consumption_results[0]),
            'n_baselines': len(consumption_results[1]),
            'best_model': best_consumption[0],
            'best_metrics': best_consumption[1],
        },
        'price': {
            'n_experiments': len(price_results[0]),
            'n_baselines': len(price_results[1]),
            'best_model': best_price[0],
            'best_metrics': best_price[1],
        },
        'all_consumption_smape': {k: v['sMAPE'] for k, v in consumption_metrics.items()},
        'all_price_smape': {k: v['sMAPE'] for k, v in price_metrics.items()},
    }

    return summary


def main():
    """Main entry point."""
    logger.info("="*80)
    logger.info("FOREWATT EXPERIMENT SUITE")
    logger.info("="*80)
    logger.info(f"Run ID: {RUN_TIMESTAMP}")
    logger.info(f"Output: {RESULTS_DIR}")
    logger.info("="*80)

    start_time = datetime.now()

    # Run consumption experiments
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: CONSUMPTION MODEL EXPERIMENTS")
    logger.info("="*80)

    try:
        consumption_results = run_all_consumption_experiments()
        logger.info("Consumption experiments completed successfully!")
    except Exception as e:
        logger.error(f"Consumption experiments failed: {e}")
        consumption_results = ({}, {})

    # Run price experiments
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: PRICE MODEL EXPERIMENTS")
    logger.info("="*80)

    try:
        price_results = run_all_price_experiments()
        logger.info("Price experiments completed successfully!")
    except Exception as e:
        logger.error(f"Price experiments failed: {e}")
        price_results = ({}, {})

    # Create combined summary
    logger.info("\n" + "="*80)
    logger.info("GENERATING COMBINED SUMMARY")
    logger.info("="*80)

    combined_summary = create_combined_summary(consumption_results, price_results)

    summary_path = RESULTS_DIR / f'combined_summary_{RUN_TIMESTAMP}.json'
    with open(summary_path, 'w') as f:
        json.dump(combined_summary, f, indent=2, default=str)

    # Print final summary
    end_time = datetime.now()
    duration = end_time - start_time

    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT SUITE COMPLETE")
    logger.info("="*80)
    logger.info(f"\nDuration: {duration}")
    logger.info(f"\nConsumption Model:")
    logger.info(f"  Best: {combined_summary['consumption']['best_model']}")
    logger.info(f"  sMAPE: {combined_summary['consumption']['best_metrics']['sMAPE']:.4f}%")
    logger.info(f"  MAE: {combined_summary['consumption']['best_metrics']['MAE']:.2f}")

    logger.info(f"\nPrice Model:")
    logger.info(f"  Best: {combined_summary['price']['best_model']}")
    logger.info(f"  sMAPE: {combined_summary['price']['best_metrics']['sMAPE']:.4f}%")
    logger.info(f"  MAE: {combined_summary['price']['best_metrics']['MAE']:.2f}")

    logger.info(f"\nResults saved to: {RESULTS_DIR}")
    logger.info(f"Combined summary: {summary_path}")

    return combined_summary


if __name__ == "__main__":
    main()
