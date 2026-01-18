"""
Create Curated Report Plots
============================
Selects and organizes the most important, meaningful plots for the final report.
Each plot has a clear purpose - no summary dashboards.
"""

import shutil
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paths
EXPERIMENTS_DIR = Path(__file__).parent
RESULTS_DIR = EXPERIMENTS_DIR / 'results'
REPORT_DIR = EXPERIMENTS_DIR / 'report_plots'

# Find latest experiment folders
CONSUMPTION_DIR = sorted(RESULTS_DIR.glob('consumption/2026*'))[-1] if list(RESULTS_DIR.glob('consumption/2026*')) else None
PRICE_DIR = sorted(RESULTS_DIR.glob('price/2026*'))[-1] if list(RESULTS_DIR.glob('price/2026*')) else None


def setup_report_structure():
    """Create organized report folder structure."""
    folders = [
        REPORT_DIR / '01_best_model_results' / 'consumption',
        REPORT_DIR / '01_best_model_results' / 'price',
        REPORT_DIR / '02_baseline_comparison',
        REPORT_DIR / '03_data_size_impact',
        REPORT_DIR / '04_feature_ablation',
        REPORT_DIR / '05_feature_importance',
        REPORT_DIR / '06_model_comparison',
        REPORT_DIR / '07_test_period_analysis',
        REPORT_DIR / '08_training_history',
    ]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    return REPORT_DIR


def copy_best_model_plots():
    """Copy plots for the best performing models."""
    print("\n1. Best Model Results")
    print("-" * 40)

    # Consumption best model: hyperparam_higher_lr
    if CONSUMPTION_DIR:
        best_cons = CONSUMPTION_DIR / 'hyperparam_higher_lr'
        if best_cons.exists():
            plots_to_copy = [
                ('predictions_vs_actual.png', 'consumption_predictions_vs_actual.png'),
                ('error_distribution.png', 'consumption_error_distribution.png'),
                ('scatter_actual_vs_pred.png', 'consumption_scatter.png'),
                ('hourly_errors.png', 'consumption_hourly_errors.png'),
            ]
            for src_name, dst_name in plots_to_copy:
                src = best_cons / src_name
                if src.exists():
                    dst = REPORT_DIR / '01_best_model_results' / 'consumption' / dst_name
                    shutil.copy2(src, dst)
                    print(f"  Copied: {dst_name}")

    # Price best model: data_size_4_years
    if PRICE_DIR:
        best_price = PRICE_DIR / 'data_size_4_years'
        if best_price.exists():
            plots_to_copy = [
                ('predictions_vs_actual.png', 'price_predictions_vs_actual.png'),
                ('error_distribution.png', 'price_error_distribution.png'),
                ('scatter_actual_vs_pred.png', 'price_scatter.png'),
                ('hourly_errors.png', 'price_hourly_errors.png'),
            ]
            for src_name, dst_name in plots_to_copy:
                src = best_price / src_name
                if src.exists():
                    dst = REPORT_DIR / '01_best_model_results' / 'price' / dst_name
                    shutil.copy2(src, dst)
                    print(f"  Copied: {dst_name}")


def copy_baseline_comparison():
    """Copy baseline comparison plots."""
    print("\n2. Baseline Comparison")
    print("-" * 40)

    if CONSUMPTION_DIR:
        src = CONSUMPTION_DIR / 'baseline_comparison.png'
        if src.exists():
            dst = REPORT_DIR / '02_baseline_comparison' / 'consumption_baseline_comparison.png'
            shutil.copy2(src, dst)
            print(f"  Copied: consumption_baseline_comparison.png")

    if PRICE_DIR:
        src = PRICE_DIR / 'baseline_comparison.png'
        if src.exists():
            dst = REPORT_DIR / '02_baseline_comparison' / 'price_baseline_comparison.png'
            shutil.copy2(src, dst)
            print(f"  Copied: price_baseline_comparison.png")


def copy_data_size_impact():
    """Copy data size impact plots."""
    print("\n3. Data Size Impact")
    print("-" * 40)

    if CONSUMPTION_DIR:
        src = CONSUMPTION_DIR / 'data_size_comparison.png'
        if src.exists():
            dst = REPORT_DIR / '03_data_size_impact' / 'consumption_data_size_impact.png'
            shutil.copy2(src, dst)
            print(f"  Copied: consumption_data_size_impact.png")

    if PRICE_DIR:
        src = PRICE_DIR / 'data_size_comparison.png'
        if src.exists():
            dst = REPORT_DIR / '03_data_size_impact' / 'price_data_size_impact.png'
            shutil.copy2(src, dst)
            print(f"  Copied: price_data_size_impact.png")


def copy_feature_ablation():
    """Copy feature ablation plots."""
    print("\n4. Feature Ablation")
    print("-" * 40)

    if CONSUMPTION_DIR:
        src = CONSUMPTION_DIR / 'feature_ablation_comparison.png'
        if src.exists():
            dst = REPORT_DIR / '04_feature_ablation' / 'consumption_feature_ablation.png'
            shutil.copy2(src, dst)
            print(f"  Copied: consumption_feature_ablation.png")

    if PRICE_DIR:
        src = PRICE_DIR / 'feature_ablation_comparison.png'
        if src.exists():
            dst = REPORT_DIR / '04_feature_ablation' / 'price_feature_ablation.png'
            shutil.copy2(src, dst)
            print(f"  Copied: price_feature_ablation.png")


def copy_feature_importance():
    """Copy feature importance plots from best models."""
    print("\n5. Feature Importance")
    print("-" * 40)

    if CONSUMPTION_DIR:
        src = CONSUMPTION_DIR / 'hyperparam_higher_lr' / 'feature_importance.png'
        if src.exists():
            dst = REPORT_DIR / '05_feature_importance' / 'consumption_feature_importance.png'
            shutil.copy2(src, dst)
            print(f"  Copied: consumption_feature_importance.png")

    if PRICE_DIR:
        src = PRICE_DIR / 'data_size_4_years' / 'feature_importance.png'
        if src.exists():
            dst = REPORT_DIR / '05_feature_importance' / 'price_feature_importance.png'
            shutil.copy2(src, dst)
            print(f"  Copied: price_feature_importance.png")


def copy_model_comparison():
    """Copy model architecture comparison (for price)."""
    print("\n6. Model Comparison (Price only)")
    print("-" * 40)

    if PRICE_DIR:
        src = PRICE_DIR / 'model_comparison.png'
        if src.exists():
            dst = REPORT_DIR / '06_model_comparison' / 'price_model_comparison.png'
            shutil.copy2(src, dst)
            print(f"  Copied: price_model_comparison.png")

        # Error correction comparison
        src = PRICE_DIR / 'error_correction_comparison.png'
        if src.exists():
            dst = REPORT_DIR / '06_model_comparison' / 'price_error_correction_comparison.png'
            shutil.copy2(src, dst)
            print(f"  Copied: price_error_correction_comparison.png")


def copy_test_period_analysis():
    """Copy test period analysis plot."""
    print("\n7. Test Period Analysis")
    print("-" * 40)

    src = RESULTS_DIR / 'test_period_analysis' / 'test_period_comparison.png'
    if src.exists():
        dst = REPORT_DIR / '07_test_period_analysis' / 'test_period_comparison.png'
        shutil.copy2(src, dst)
        print(f"  Copied: test_period_comparison.png")


def create_training_history_plot():
    """Create training history plot from available data."""
    print("\n8. Training History")
    print("-" * 40)

    # Check if we have training logs
    logs_dir = EXPERIMENTS_DIR / 'logs'
    if not logs_dir.exists():
        logs_dir.mkdir(parents=True, exist_ok=True)

    # Try to find training history in experiment results
    history_found = False

    if CONSUMPTION_DIR:
        best_cons = CONSUMPTION_DIR / 'hyperparam_higher_lr'
        history_file = best_cons / 'training_history.json'
        if history_file.exists():
            with open(history_file) as f:
                history = json.load(f)

            fig, ax = plt.subplots(figsize=(10, 5))

            if 'train_loss' in history:
                ax.plot(history['train_loss'], label='Train Loss', color='#2E86AB')
            if 'val_loss' in history:
                ax.plot(history['val_loss'], label='Validation Loss', color='#E74C3C')

            ax.set_xlabel('Iteration', fontsize=11)
            ax.set_ylabel('Loss (MAE)', fontsize=11)
            ax.set_title('Consumption Model: Training History', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(REPORT_DIR / '08_training_history' / 'consumption_training_history.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Created: consumption_training_history.png")
            history_found = True

    if PRICE_DIR:
        best_price = PRICE_DIR / 'data_size_4_years'
        history_file = best_price / 'training_history.json'
        if history_file.exists():
            with open(history_file) as f:
                history = json.load(f)

            fig, ax = plt.subplots(figsize=(10, 5))

            if 'catboost' in history:
                ax.plot(history['catboost'].get('train_loss', []), label='CatBoost Train', color='#2E86AB')
                ax.plot(history['catboost'].get('val_loss', []), label='CatBoost Val', color='#2E86AB', linestyle='--')
            if 'lightgbm' in history:
                ax.plot(history['lightgbm'].get('train_loss', []), label='LightGBM Train', color='#28A745')
                ax.plot(history['lightgbm'].get('val_loss', []), label='LightGBM Val', color='#28A745', linestyle='--')

            ax.set_xlabel('Iteration', fontsize=11)
            ax.set_ylabel('Loss (MAE)', fontsize=11)
            ax.set_title('Price Model: Training History', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(REPORT_DIR / '08_training_history' / 'price_training_history.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Created: price_training_history.png")
            history_found = True

    if not history_found:
        print("  No training history files found - skipping")


def create_metrics_summary_table():
    """Create a summary metrics table as an image."""
    print("\n9. Creating Metrics Summary Table")
    print("-" * 40)

    # Load summary data
    cons_summary = None
    price_summary = None

    if CONSUMPTION_DIR:
        summary_file = CONSUMPTION_DIR / 'summary.json'
        if summary_file.exists():
            with open(summary_file) as f:
                cons_summary = json.load(f)

    if PRICE_DIR:
        summary_file = PRICE_DIR / 'summary.json'
        if summary_file.exists():
            with open(summary_file) as f:
                price_summary = json.load(f)

    if cons_summary and price_summary:
        # Create a clean metrics table
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Consumption metrics table
        ax = axes[0]
        ax.axis('off')

        cons_best = cons_summary.get('best_experiment', {})
        cons_metrics = cons_best.get('metrics', {})

        table_data = [
            ['Metric', 'Value'],
            ['MAE', f"{cons_metrics.get('MAE', 0):.1f} MWh"],
            ['RMSE', f"{cons_metrics.get('RMSE', 0):.1f} MWh"],
            ['MAPE', f"{cons_metrics.get('MAPE', 0):.2f}%"],
            ['sMAPE', f"{cons_metrics.get('sMAPE', 0):.2f}%"],
            ['R²', f"{cons_metrics.get('R2', 0):.3f}"],
            ['MBE', f"{cons_metrics.get('MBE', 0):.1f} MWh"],
        ]

        table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                        colWidths=[0.4, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)

        # Style header
        for j in range(2):
            table[(0, j)].set_facecolor('#2E86AB')
            table[(0, j)].set_text_props(color='white', fontweight='bold')

        ax.set_title(f'Consumption Model\n(Best: {cons_best.get("name", "N/A")})',
                    fontsize=12, fontweight='bold', pad=20)

        # Price metrics table
        ax = axes[1]
        ax.axis('off')

        price_best = price_summary.get('best_experiment', {})
        price_metrics = price_best.get('metrics', {})

        table_data = [
            ['Metric', 'Value'],
            ['MAE', f"{price_metrics.get('MAE', 0):.1f} TL/MWh"],
            ['RMSE', f"{price_metrics.get('RMSE', 0):.1f} TL/MWh"],
            ['MAPE', f"{price_metrics.get('MAPE', 0):.2f}%"],
            ['sMAPE', f"{price_metrics.get('sMAPE', 0):.2f}%"],
            ['R²', f"{price_metrics.get('R2', 0):.3f}"],
            ['MBE', f"{price_metrics.get('MBE', 0):.1f} TL/MWh"],
        ]

        table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                        colWidths=[0.4, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)

        # Style header
        for j in range(2):
            table[(0, j)].set_facecolor('#E74C3C')
            table[(0, j)].set_text_props(color='white', fontweight='bold')

        ax.set_title(f'Price Model\n(Best: {price_best.get("name", "N/A")})',
                    fontsize=12, fontweight='bold', pad=20)

        plt.suptitle('Best Model Performance Summary', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(REPORT_DIR / 'metrics_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Created: metrics_summary.png")


def print_report_structure():
    """Print the final report folder structure."""
    print("\n" + "=" * 60)
    print("REPORT PLOTS FOLDER STRUCTURE")
    print("=" * 60)

    for folder in sorted(REPORT_DIR.rglob('*')):
        if folder.is_file():
            rel_path = folder.relative_to(REPORT_DIR)
            print(f"  {rel_path}")


def main():
    print("=" * 60)
    print("Creating Curated Report Plots")
    print("=" * 60)

    print(f"\nConsumption results: {CONSUMPTION_DIR}")
    print(f"Price results: {PRICE_DIR}")

    # Setup folder structure
    setup_report_structure()

    # Copy selected plots
    copy_best_model_plots()
    copy_baseline_comparison()
    copy_data_size_impact()
    copy_feature_ablation()
    copy_feature_importance()
    copy_model_comparison()
    copy_test_period_analysis()
    create_training_history_plot()
    create_metrics_summary_table()

    # Print final structure
    print_report_structure()

    print(f"\nReport plots saved to: {REPORT_DIR}")


if __name__ == '__main__':
    main()
