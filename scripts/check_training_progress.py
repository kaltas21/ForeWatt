"""
Check Training Progress
=======================
Monitors training progress and displays summary of completed runs.

Usage:
    python scripts/check_training_progress.py
"""

import sys
from pathlib import Path
from datetime import datetime
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def check_baseline_progress():
    """Check baseline model training progress."""
    print("\n" + "=" * 80)
    print("BASELINE MODELS TRAINING PROGRESS")
    print("=" * 80)

    log_dir = PROJECT_ROOT / 'reports' / 'baseline' / 'logs'
    results_dir = PROJECT_ROOT / 'reports' / 'baseline' / 'grid_search'

    if not log_dir.exists():
        print("No baseline logs found.")
        return

    # Check overall log files
    overall_logs = list(log_dir.glob('grid_search_run_*.log'))
    if overall_logs:
        latest_log = max(overall_logs, key=lambda p: p.stat().st_mtime)
        print(f"\nLatest log: {latest_log.name}")
        print(f"Last modified: {datetime.fromtimestamp(latest_log.stat().st_mtime)}")

        # Count lines in log
        with open(latest_log, 'r') as f:
            lines = f.readlines()
            print(f"Log lines: {len(lines)}")

            # Check for completion
            if any('GRID SEARCH COMPLETED' in line for line in lines):
                print("Status: ✓ COMPLETED")
            else:
                print("Status: ⏳ IN PROGRESS")

    # Check for results
    result_files = list(results_dir.glob('grid_search_results_*.csv')) if results_dir.exists() else []
    if result_files:
        print(f"\nResults files: {len(result_files)}")
        for result_file in result_files:
            print(f"  - {result_file.name}")

    # Check individual model logs
    targets = ['consumption', 'price_real']
    models = ['catboost', 'xgboost', 'lightgbm', 'prophet']

    print("\nPer-Model Progress:")
    for target in targets:
        for model in models:
            model_log_dir = log_dir / target / model
            if model_log_dir.exists():
                log_files = list(model_log_dir.glob('*.log'))
                print(f"  {target}/{model}: {len(log_files)} configs run")


def check_deep_learning_progress():
    """Check deep learning model training progress."""
    print("\n" + "=" * 80)
    print("DEEP LEARNING MODELS TRAINING PROGRESS")
    print("=" * 80)

    log_dir = PROJECT_ROOT / 'reports' / 'deep_learning' / 'logs'
    results_dir = PROJECT_ROOT / 'reports' / 'deep_learning' / 'grid_search'

    if not log_dir.exists():
        print("No deep learning logs found.")
        return

    # Check overall log files
    overall_logs = list(log_dir.glob('grid_search_run_*.log'))
    if overall_logs:
        latest_log = max(overall_logs, key=lambda p: p.stat().st_mtime)
        print(f"\nLatest log: {latest_log.name}")
        print(f"Last modified: {datetime.fromtimestamp(latest_log.stat().st_mtime)}")

        # Count lines in log
        with open(latest_log, 'r') as f:
            lines = f.readlines()
            print(f"Log lines: {len(lines)}")

            # Check for completion
            if any('GRID SEARCH COMPLETED' in line for line in lines):
                print("Status: ✓ COMPLETED")
            else:
                print("Status: ⏳ IN PROGRESS")

                # Count progress
                completed = sum(1 for line in lines if 'completed successfully' in line)
                failed = sum(1 for line in lines if 'failed:' in line.lower())
                print(f"Completed: {completed} | Failed: {failed}")

    # Check for results
    result_files = list(results_dir.glob('grid_search_results_*.csv')) if results_dir.exists() else []
    if result_files:
        print(f"\nResults files: {len(result_files)}")
        for result_file in result_files:
            print(f"  - {result_file.name}")

    # Check individual model logs
    targets = ['consumption', 'price_real']
    models = ['nhits', 'tft', 'patchtst']

    print("\nPer-Model Progress:")
    for target in targets:
        for model in models:
            model_log_dir = log_dir / target / model
            if model_log_dir.exists():
                log_files = list(model_log_dir.glob('*.log'))
                print(f"  {target}/{model}: {len(log_files)} configs run")


def check_gpu_status():
    """Check GPU status."""
    print("\n" + "=" * 80)
    print("GPU STATUS")
    print("=" * 80)

    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total',
                                '--format=csv,noheader,nounits'],
                               capture_output=True, text=True)

        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) == 4:
                    name, util, mem_used, mem_total = parts
                    print(f"\nGPU {i}: {name}")
                    print(f"  Utilization: {util}%")
                    print(f"  Memory: {mem_used}MB / {mem_total}MB ({float(mem_used)/float(mem_total)*100:.1f}%)")
        else:
            print("nvidia-smi not available or no GPU found")

    except Exception as e:
        print(f"Could not check GPU status: {e}")


def main():
    """Main entry point."""
    print("\n" + "█" * 80)
    print("ForeWatt Training Progress Monitor")
    print("█" * 80)

    check_gpu_status()
    check_baseline_progress()
    check_deep_learning_progress()

    print("\n" + "=" * 80)
    print("To tail logs in real-time:")
    print("  Baseline:       tail -f reports/baseline/logs/grid_search_run_*.log")
    print("  Deep Learning:  tail -f reports/deep_learning/logs/grid_search_run_*.log")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
