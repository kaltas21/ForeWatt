"""
Horizon-Wise Evaluation for Multi-Horizon Forecasting
======================================================
Evaluates forecasts at each horizon (h=1, 2, ..., H) separately.

Metrics:
- sMAPE (Symmetric MAPE) - Primary metric
- MASE (Mean Absolute Scaled Error) - Scale-independent
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

Author: ForeWatt Team
Date: November 2025
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.evaluate import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    symmetric_mean_absolute_percentage_error,
    mean_absolute_scaled_error
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HorizonWiseEvaluator:
    """
    Evaluator for multi-horizon forecasting.
    """

    def __init__(self, horizon: int = 24):
        """
        Initialize evaluator.

        Args:
            horizon: Maximum forecast horizon
        """
        self.horizon = horizon

    def evaluate_single_horizon(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: np.ndarray,
        h: int
    ) -> Dict[str, float]:
        """
        Evaluate forecast for a single horizon.

        Args:
            y_true: True values for horizon h (n_samples,)
            y_pred: Predictions for horizon h (n_samples,)
            y_train: Training data for MASE calculation
            h: Horizon number (1-indexed)

        Returns:
            Dictionary with metrics
        """
        metrics = {
            'horizon': h,
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': root_mean_squared_error(y_true, y_pred),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred),
            'sMAPE': symmetric_mean_absolute_percentage_error(y_true, y_pred),
            'MASE': mean_absolute_scaled_error(y_true, y_pred, y_train, seasonality=24)
        }

        return metrics

    def evaluate_all_horizons(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: np.ndarray
    ) -> pd.DataFrame:
        """
        Evaluate forecasts for all horizons.

        Args:
            y_true: True values (n_samples, horizon)
            y_pred: Predictions (n_samples, horizon)
            y_train: Training data for MASE

        Returns:
            DataFrame with metrics per horizon
        """
        if len(y_true.shape) == 1:
            # Single horizon
            metrics = self.evaluate_single_horizon(y_true, y_pred, y_train, h=1)
            return pd.DataFrame([metrics])

        # Multi-horizon
        results = []
        for h in range(y_true.shape[1]):
            metrics = self.evaluate_single_horizon(
                y_true[:, h],
                y_pred[:, h],
                y_train,
                h=h+1
            )
            results.append(metrics)

        df = pd.DataFrame(results)

        logger.info(f"\n{'='*80}")
        logger.info("HORIZON-WISE EVALUATION")
        logger.info(f"{'='*80}")
        logger.info(f"\n{df.to_string(index=False)}")
        logger.info(f"{'='*80}\n")

        return df

    def aggregate_metrics(
        self,
        horizon_metrics: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Aggregate metrics across horizons.

        Args:
            horizon_metrics: DataFrame from evaluate_all_horizons

        Returns:
            Dictionary with aggregated metrics
        """
        aggregated = {
            'MAE_mean': horizon_metrics['MAE'].mean(),
            'MAE_std': horizon_metrics['MAE'].std(),
            'RMSE_mean': horizon_metrics['RMSE'].mean(),
            'RMSE_std': horizon_metrics['RMSE'].std(),
            'sMAPE_mean': horizon_metrics['sMAPE'].mean(),
            'sMAPE_std': horizon_metrics['sMAPE'].std(),
            'MASE_mean': horizon_metrics['MASE'].mean(),
            'MASE_std': horizon_metrics['MASE'].std(),

            # By horizon group
            'sMAPE_short_term': horizon_metrics.loc[horizon_metrics['horizon'] <= 6, 'sMAPE'].mean(),
            'sMAPE_day_ahead': horizon_metrics.loc[horizon_metrics['horizon'] == 24, 'sMAPE'].mean() if 24 in horizon_metrics['horizon'].values else np.nan,
            'MASE_short_term': horizon_metrics.loc[horizon_metrics['horizon'] <= 6, 'MASE'].mean(),
            'MASE_day_ahead': horizon_metrics.loc[horizon_metrics['horizon'] == 24, 'MASE'].mean() if 24 in horizon_metrics['horizon'].values else np.nan
        }

        logger.info(f"\n{'='*80}")
        logger.info("AGGREGATED METRICS")
        logger.info(f"{'='*80}")
        logger.info(f"Mean sMAPE: {aggregated['sMAPE_mean']:.4f}% ± {aggregated['sMAPE_std']:.4f}%")
        logger.info(f"Mean MASE: {aggregated['MASE_mean']:.4f} ± {aggregated['MASE_std']:.4f}")
        logger.info(f"Short-term (h=1-6) sMAPE: {aggregated['sMAPE_short_term']:.4f}%")
        if not np.isnan(aggregated['sMAPE_day_ahead']):
            logger.info(f"Day-ahead (h=24) sMAPE: {aggregated['sMAPE_day_ahead']:.4f}%")
        logger.info(f"{'='*80}\n")

        return aggregated

    def plot_horizon_metrics(
        self,
        horizon_metrics: pd.DataFrame,
        save_path: Optional[Path] = None
    ):
        """
        Plot metrics across horizons.

        Args:
            horizon_metrics: DataFrame from evaluate_all_horizons
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        metrics_to_plot = ['MAE', 'RMSE', 'sMAPE', 'MASE']
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

        for idx, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
            ax = axes[idx // 2, idx % 2]

            ax.plot(
                horizon_metrics['horizon'],
                horizon_metrics[metric],
                marker='o',
                linewidth=2,
                markersize=6,
                color=color,
                label=metric
            )

            ax.set_xlabel('Horizon (h)', fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            ax.set_title(f'{metric} by Forecast Horizon', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Highlight specific horizons
            if 24 in horizon_metrics['horizon'].values:
                ax.axvline(x=24, color='red', linestyle='--', alpha=0.5, label='Day-ahead')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_forecast_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        horizon: int = 24,
        n_samples: int = 100,
        save_path: Optional[Path] = None
    ):
        """
        Plot forecast vs actual for selected horizons.

        Args:
            y_true: True values (n_samples, horizon)
            y_pred: Predictions (n_samples, horizon)
            timestamps: Timestamps for samples
            horizon: Forecast horizon
            n_samples: Number of samples to plot
            save_path: Path to save plot
        """
        # Select subset of samples
        n_samples = min(n_samples, len(y_true))
        idx = np.linspace(0, len(y_true)-1, n_samples, dtype=int)

        y_true_subset = y_true[idx]
        y_pred_subset = y_pred[idx]

        # Select specific horizons to plot
        horizons_to_plot = [1, 6, 12, 24] if horizon >= 24 else [1, horizon//2, horizon]
        horizons_to_plot = [h for h in horizons_to_plot if h <= horizon]

        fig, axes = plt.subplots(len(horizons_to_plot), 1, figsize=(15, 4*len(horizons_to_plot)))

        if len(horizons_to_plot) == 1:
            axes = [axes]

        for idx, h in enumerate(horizons_to_plot):
            ax = axes[idx]

            if timestamps is not None:
                x = timestamps[:n_samples]
            else:
                x = np.arange(n_samples)

            ax.plot(x, y_true_subset[:, h-1], label='Actual', linewidth=2, alpha=0.7)
            ax.plot(x, y_pred_subset[:, h-1], label='Forecast', linewidth=2, alpha=0.7)

            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.set_title(f'Horizon h={h}: Forecast vs Actual', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def create_evaluation_report(
        self,
        horizon_metrics: pd.DataFrame,
        aggregated_metrics: Dict[str, float],
        model_name: str,
        target: str,
        save_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Create comprehensive evaluation report.

        Args:
            horizon_metrics: Per-horizon metrics
            aggregated_metrics: Aggregated metrics
            model_name: Name of the model
            target: Target variable
            save_dir: Directory to save report

        Returns:
            Dictionary with all evaluation results
        """
        report = {
            'model': model_name,
            'target': target,
            'horizon_metrics': horizon_metrics.to_dict('records'),
            'aggregated_metrics': aggregated_metrics,
            'summary': {
                'best_horizon': int(horizon_metrics.loc[horizon_metrics['sMAPE'].idxmin(), 'horizon']),
                'worst_horizon': int(horizon_metrics.loc[horizon_metrics['sMAPE'].idxmax(), 'horizon']),
                'best_sMAPE': float(horizon_metrics['sMAPE'].min()),
                'worst_sMAPE': float(horizon_metrics['sMAPE'].max()),
                'mean_sMAPE': float(aggregated_metrics['sMAPE_mean']),
                'std_sMAPE': float(aggregated_metrics['sMAPE_std'])
            }
        }

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Save horizon metrics CSV
            csv_path = save_dir / f'horizon_metrics_{model_name}_{target}.csv'
            horizon_metrics.to_csv(csv_path, index=False)
            logger.info(f"Horizon metrics saved to: {csv_path}")

            # Save aggregated metrics JSON
            import json
            json_path = save_dir / f'aggregated_metrics_{model_name}_{target}.json'
            with open(json_path, 'w') as f:
                json.dump(aggregated_metrics, f, indent=2)
            logger.info(f"Aggregated metrics saved to: {json_path}")

            # Save report JSON
            report_path = save_dir / f'evaluation_report_{model_name}_{target}.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Evaluation report saved to: {report_path}")

        return report


def compare_models_horizon_wise(
    model_results: Dict[str, pd.DataFrame],
    save_path: Optional[Path] = None
):
    """
    Compare multiple models across horizons.

    Args:
        model_results: Dictionary mapping model names to horizon metrics DataFrames
        save_path: Path to save comparison plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # sMAPE comparison
    ax1 = axes[0]
    for model_name, metrics_df in model_results.items():
        ax1.plot(
            metrics_df['horizon'],
            metrics_df['sMAPE'],
            marker='o',
            linewidth=2,
            markersize=6,
            label=model_name
        )

    ax1.set_xlabel('Horizon (h)', fontsize=12)
    ax1.set_ylabel('sMAPE (%)', fontsize=12)
    ax1.set_title('sMAPE by Forecast Horizon', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # MASE comparison
    ax2 = axes[1]
    for model_name, metrics_df in model_results.items():
        ax2.plot(
            metrics_df['horizon'],
            metrics_df['MASE'],
            marker='o',
            linewidth=2,
            markersize=6,
            label=model_name
        )

    ax2.set_xlabel('Horizon (h)', fontsize=12)
    ax2.set_ylabel('MASE', fontsize=12)
    ax2.set_title('MASE by Forecast Horizon', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Naive baseline')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()
