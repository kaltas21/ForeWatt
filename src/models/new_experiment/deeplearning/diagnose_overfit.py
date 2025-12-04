"""
Model Overfitting & Distribution Shift Diagnostic Tool
=======================================================
Diagnoses why a model performs well on validation but poorly on test.

Usage:
    python diagnose_overfit.py --config-hash 9626ebea245b
    python diagnose_overfit.py --analyze-all

Author: ForeWatt Team
Date: December 2025
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))


def load_data_with_splits(val_size=0.2, test_size=0.2):
    """Load data and return with split indices."""
    from src.models.new_experiment.deeplearning.feature_preparer_v2 import load_master_v2

    df = load_master_v2()
    n = len(df)

    train_end = int(n * (1 - val_size - test_size))
    val_end = int(n * (1 - test_size))

    splits = {
        'train': (0, train_end),
        'val': (train_end, val_end),
        'test': (val_end, n)
    }

    return df, splits


def analyze_distribution_shift(df: pd.DataFrame, splits: dict, target: str = 'price_real'):
    """Analyze distribution shift between splits."""
    print("\n" + "="*80)
    print("DISTRIBUTION SHIFT ANALYSIS")
    print("="*80)

    train_data = df.iloc[splits['train'][0]:splits['train'][1]]
    val_data = df.iloc[splits['val'][0]:splits['val'][1]]
    test_data = df.iloc[splits['test'][0]:splits['test'][1]]

    print(f"\n📅 DATE RANGES:")
    print(f"   Train: {train_data.index[0].strftime('%Y-%m-%d')} to {train_data.index[-1].strftime('%Y-%m-%d')} ({len(train_data)} samples)")
    print(f"   Val:   {val_data.index[0].strftime('%Y-%m-%d')} to {val_data.index[-1].strftime('%Y-%m-%d')} ({len(val_data)} samples)")
    print(f"   Test:  {test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')} ({len(test_data)} samples)")

    print(f"\n📊 TARGET ({target}) STATISTICS:")
    print("-"*80)
    print(f"{'Split':<8} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12} {'Median':>12}")
    print("-"*80)

    for name, data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
        y = data[target]
        print(f"{name:<8} {y.mean():>12.2f} {y.std():>12.2f} {y.min():>12.2f} {y.max():>12.2f} {y.median():>12.2f}")

    # Calculate shifts
    train_mean = train_data[target].mean()
    train_std = train_data[target].std()
    test_mean = test_data[target].mean()
    test_std = test_data[target].std()
    val_mean = val_data[target].mean()

    print(f"\n⚠️  SHIFT METRICS:")
    print(f"   Mean shift (Test vs Train): {(test_mean - train_mean) / train_mean * 100:+.1f}%")
    print(f"   Std shift (Test vs Train):  {(test_std - train_std) / train_std * 100:+.1f}%")
    print(f"   Mean shift (Test vs Val):   {(test_mean - val_mean) / val_mean * 100:+.1f}%")

    # Severity assessment
    mean_shift = abs((test_mean - train_mean) / train_mean * 100)
    std_shift = abs((test_std - train_std) / train_std * 100)

    if mean_shift > 30 or std_shift > 50:
        severity = "🔴 SEVERE"
    elif mean_shift > 15 or std_shift > 30:
        severity = "🟡 MODERATE"
    else:
        severity = "🟢 MILD"

    print(f"\n   Distribution shift severity: {severity}")

    return {
        'mean_shift': (test_mean - train_mean) / train_mean * 100,
        'std_shift': (test_std - train_std) / train_std * 100,
        'severity': severity
    }


def analyze_feature_shifts(df: pd.DataFrame, splits: dict, features: list = None):
    """Analyze which features shifted most between train and test."""
    print("\n" + "="*80)
    print("FEATURE DISTRIBUTION SHIFT ANALYSIS")
    print("="*80)

    if features is None:
        # Key features to check
        features = [
            'reserve_margin_ratio', 'renewable_saturation', 'spark_spread_proxy',
            'thermal_gap', 'system_short_signal', 'temp_national',
            'price_ptf_lag_24h', 'price_ptf_rolling_mean_24h',
            'consumption_forecast', 'load_factor'
        ]

    train_data = df.iloc[splits['train'][0]:splits['train'][1]]
    test_data = df.iloc[splits['test'][0]:splits['test'][1]]

    shifts = []
    for feat in features:
        if feat not in df.columns:
            continue

        train_mean = train_data[feat].mean()
        test_mean = test_data[feat].mean()

        if abs(train_mean) > 1e-8:
            shift_pct = (test_mean - train_mean) / abs(train_mean) * 100
        else:
            shift_pct = 0

        shifts.append({
            'feature': feat,
            'train_mean': train_mean,
            'test_mean': test_mean,
            'shift_pct': shift_pct
        })

    # Sort by absolute shift
    shifts = sorted(shifts, key=lambda x: abs(x['shift_pct']), reverse=True)

    print(f"\n{'Feature':<35} {'Train':>12} {'Test':>12} {'Shift':>10}")
    print("-"*80)

    for s in shifts[:15]:  # Top 15
        shift_str = f"{s['shift_pct']:+.1f}%"
        if abs(s['shift_pct']) > 20:
            shift_str = f"⚠️  {shift_str}"
        print(f"{s['feature']:<35} {s['train_mean']:>12.4f} {s['test_mean']:>12.4f} {shift_str:>10}")

    return shifts


def analyze_model_config(config_hash: str):
    """Analyze a specific model configuration."""
    metrics_dir = PROJECT_ROOT / 'reports' / 'new_experiment' / 'deeplearning' / 'metrics'
    metrics_file = metrics_dir / f"{config_hash}.json"

    if not metrics_file.exists():
        print(f"❌ Metrics file not found: {metrics_file}")
        return None

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    print("\n" + "="*80)
    print(f"MODEL ANALYSIS: {config_hash}")
    print("="*80)

    print(f"\n📋 CONFIGURATION:")
    print(f"   Model type: {metrics.get('model_type', 'N/A')}")
    print(f"   Target: {metrics.get('target', 'N/A')}")
    print(f"   Feature strategy: {metrics.get('feature_strategy', 'N/A')}")
    print(f"   Feature tier: {metrics.get('feature_tier', 'N/A')}")

    config = metrics.get('config', {})
    print(f"\n   Hyperparameters:")
    for key in ['input_size', 'horizon', 'hidden_size', 'n_blocks', 'learning_rate', 'batch_size', 'max_steps']:
        if key in config:
            print(f"      {key}: {config[key]}")

    print(f"\n📈 PERFORMANCE:")
    val_metrics = metrics.get('validation_metrics', {})
    test_metrics = metrics.get('test_metrics', {})

    print(f"\n   {'Metric':<12} {'Validation':>12} {'Test':>12} {'Gap':>12}")
    print("   " + "-"*50)

    for metric in ['MAE', 'sMAPE', 'MASE']:
        val = val_metrics.get(metric, np.nan)
        test = test_metrics.get(metric, np.nan)
        if not np.isnan(val) and val > 0:
            gap = (test - val) / val * 100
            gap_str = f"{gap:+.1f}%"
        else:
            gap_str = "N/A"
        print(f"   {metric:<12} {val:>12.2f} {test:>12.2f} {gap_str:>12}")

    # Overfitting assessment
    val_smape = val_metrics.get('sMAPE', 0)
    test_smape = test_metrics.get('sMAPE', 0)

    if val_smape > 0:
        gap_ratio = test_smape / val_smape
        if gap_ratio > 2.0:
            status = "🔴 SEVERE OVERFITTING/DRIFT"
        elif gap_ratio > 1.5:
            status = "🟡 MODERATE OVERFITTING/DRIFT"
        elif gap_ratio > 1.2:
            status = "🟢 MILD OVERFITTING"
        else:
            status = "✅ GOOD GENERALIZATION"

        print(f"\n   Assessment: {status}")
        print(f"   Test/Val ratio: {gap_ratio:.2f}x")

    return metrics


def suggest_fixes(shift_analysis: dict, feature_shifts: list):
    """Suggest fixes based on analysis."""
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)

    mean_shift = abs(shift_analysis['mean_shift'])
    std_shift = abs(shift_analysis['std_shift'])

    print("\n1️⃣  DATA STRATEGY:")
    if mean_shift > 30:
        print("   • Use walk-forward validation (runner_walkforward.py)")
        print("   • Consider training on recent data only (--recent-only flag)")
        print("   • The market regime has fundamentally changed")
    if std_shift > 50:
        print("   • Volatility collapsed - model learned high-vol patterns")
        print("   • Consider using relative features instead of absolute")

    print("\n2️⃣  FEATURE ENGINEERING:")
    high_shift_features = [f for f in feature_shifts if abs(f['shift_pct']) > 20]
    if high_shift_features:
        print("   • These features shifted significantly (>20%):")
        for f in high_shift_features[:5]:
            print(f"      - {f['feature']}: {f['shift_pct']:+.1f}%")
        print("   • Consider normalizing these or using ratios")

    print("\n3️⃣  MODEL ARCHITECTURE:")
    print("   • Add dropout (0.1-0.3) for regularization")
    print("   • Reduce model complexity (smaller hidden_size)")
    print("   • Use ensemble of models trained on different periods")

    print("\n4️⃣  QUICK TEST:")
    print("   Run walk-forward validation to get robust estimates:")
    print("   $ python src/models/new_experiment/deeplearning/runner_walkforward.py \\")
    print("       --models nhits --folds 4 --max-configs 3")


def main():
    parser = argparse.ArgumentParser(description='Diagnose model overfitting and distribution shift')
    parser.add_argument('--config-hash', type=str, help='Analyze specific model configuration')
    parser.add_argument('--analyze-all', action='store_true', help='Full distribution shift analysis')
    parser.add_argument('--target', type=str, default='price_real', choices=['price_real', 'consumption'])

    args = parser.parse_args()

    # Load data
    df, splits = load_data_with_splits()

    if args.config_hash:
        # Analyze specific model
        analyze_model_config(args.config_hash)

    # Distribution shift analysis
    shift_analysis = analyze_distribution_shift(df, splits, args.target)
    feature_shifts = analyze_feature_shifts(df, splits)

    # Suggestions
    suggest_fixes(shift_analysis, feature_shifts)


if __name__ == "__main__":
    main()
