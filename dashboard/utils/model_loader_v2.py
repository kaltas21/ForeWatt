"""
Model and Metrics Loader for New Experiments
Loads models and metrics from ForeWatt/reports/new_experiment/ directories
"""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project root (dashboard is in ForeWatt/dashboard, new experiments in ForeWatt/reports/)
DASHBOARD_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = DASHBOARD_ROOT.parent

# Paths to new experiment directories
NEW_EXPERIMENT_ROOT = PROJECT_ROOT / "reports" / "new_experiment"
BASELINE_METRICS_DIR = NEW_EXPERIMENT_ROOT / "baseline" / "metrics"
BASELINE_MODELS_DIR = NEW_EXPERIMENT_ROOT / "baseline" / "models"
BASELINE_RESULTS_CSV = NEW_EXPERIMENT_ROOT / "baseline" / "results.csv"
DEEPLEARNING_METRICS_DIR = NEW_EXPERIMENT_ROOT / "deeplearning" / "metrics"
DEEPLEARNING_MODELS_DIR = NEW_EXPERIMENT_ROOT / "deeplearning" / "models"
DEEPLEARNING_RESULTS_CSV = NEW_EXPERIMENT_ROOT / "deeplearning" / "results.csv"


def load_results_from_csv(
    target: str = "consumption",
    model_category: Optional[str] = None
) -> pd.DataFrame:
    """
    Load model results from results.csv files.

    Args:
        target: Target variable ('consumption' or 'price_real')
        model_category: Filter by 'baseline' or 'deeplearning' (None for all)

    Returns:
        DataFrame with model results
    """
    all_results = []

    # Load baseline results
    if (model_category is None or model_category == 'baseline') and BASELINE_RESULTS_CSV.exists():
        try:
            df_baseline = pd.read_csv(BASELINE_RESULTS_CSV)
            df_baseline = df_baseline[df_baseline['target'] == target]
            df_baseline = df_baseline[df_baseline['status'] == 'success']
            df_baseline['category'] = 'baseline'
            all_results.append(df_baseline)
        except Exception as e:
            logger.warning(f"Failed to load baseline results: {e}")

    # Load deep learning results
    if (model_category is None or model_category == 'deeplearning') and DEEPLEARNING_RESULTS_CSV.exists():
        try:
            import json
            df_dl = pd.read_csv(DEEPLEARNING_RESULTS_CSV)
            df_dl = df_dl[df_dl['target'] == target]
            df_dl = df_dl[df_dl['status'] == 'success']
            df_dl['category'] = 'deeplearning'

            # Extract config_name from config_json
            def extract_config_name(config_json):
                try:
                    config = json.loads(config_json)
                    return config.get('config_name', config.get('description', 'unknown'))
                except:
                    return 'unknown'

            df_dl['config_name'] = df_dl['config_json'].apply(extract_config_name)
            all_results.append(df_dl)
        except Exception as e:
            logger.warning(f"Failed to load deeplearning results: {e}")

    if not all_results:
        logger.warning(f"No results found for target={target}, category={model_category}")
        return pd.DataFrame()

    df = pd.concat(all_results, ignore_index=True)

    # Rename columns to match expected format
    df = df.rename(columns={
        'test_mae': 'MAE',
        'test_smape': 'sMAPE',
        'test_mase': 'MASE',
        'val_mae': 'val_MAE',
        'val_smape': 'val_sMAPE',
        'val_mase': 'val_MASE',
        'training_time_seconds': 'training_time'
    })

    # Standardize model names
    df['model_name'] = df['model_type'].str.upper()
    df.loc[df['model_type'] == 'nhits', 'model_name'] = 'N-HiTS'
    df.loc[df['model_type'] == 'patchtst', 'model_name'] = 'PatchTST'
    df.loc[df['model_type'] == 'tft', 'model_name'] = 'TFT'

    return df


def load_all_metrics(
    target: str = "consumption",
    model_category: Optional[str] = None
) -> pd.DataFrame:
    """
    Load all model metrics from new experiment directories.
    This function now uses the results.csv files for faster loading.

    Args:
        target: Target variable ('consumption' or 'price_real')
        model_category: Filter by 'baseline' or 'deeplearning' (None for all)

    Returns:
        DataFrame with model metrics indexed by model name
    """
    # Use CSV-based loading for better performance
    df = load_results_from_csv(target=target, model_category=model_category)

    if df.empty:
        logger.warning(f"No metrics found for target={target}, category={model_category}")

    return df


def get_best_models_per_type(
    target: str = "consumption",
    metric: str = "MAE"
) -> pd.DataFrame:
    """
    Get best model for each model type based on specified metric.

    Args:
        target: Target variable
        metric: Metric to optimize ('MAE', 'MASE', 'sMAPE')

    Returns:
        DataFrame with best models per type
    """
    df = load_all_metrics(target=target)

    if df.empty:
        return df

    # Group by model_name and get best config
    best_models = df.loc[df.groupby('model_name')[metric].idxmin()]

    # Sort by metric
    best_models = best_models.sort_values(metric)

    # Select columns that are available
    available_cols = ['model_name', 'config_name', 'MAE', 'sMAPE', 'MASE',
                     'val_MAE', 'val_MASE', 'category', 'n_features',
                     'feature_tier', 'config_hash']

    # Add training_time if available
    if 'training_time' in best_models.columns:
        available_cols.insert(-1, 'training_time')

    return best_models[available_cols]


def load_model_config(model_type: str, config_hash: str, target: str = "consumption") -> Optional[Dict]:
    """
    Load full config for a specific model.

    Args:
        model_type: Model type (catboost, xgboost, lightgbm, nhits, patchtst, tft)
        config_hash: Unique config hash
        target: Target variable

    Returns:
        Config dictionary or None
    """
    # Determine category
    if model_type.lower() in ['catboost', 'xgboost', 'lightgbm']:
        metrics_dir = BASELINE_METRICS_DIR
    else:
        metrics_dir = DEEPLEARNING_METRICS_DIR

    metric_file = metrics_dir / f"{config_hash}.json"

    if not metric_file.exists():
        logger.warning(f"Config file not found: {metric_file}")
        return None

    try:
        with open(metric_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return None


def get_model_summary(target: str = "consumption") -> Dict:
    """
    Get summary statistics for all models.

    Returns:
        Dictionary with summary stats
    """
    df = load_all_metrics(target=target)

    if df.empty:
        return {}

    return {
        'total_models': len(df),
        'baseline_models': len(df[df['category'] == 'baseline']),
        'deep_learning_models': len(df[df['category'] == 'deeplearning']),  # Fixed: was 'deep_learning'
        'best_mae': df['MAE'].min(),
        'best_model': df.loc[df['MAE'].idxmin(), 'model_name'],
        'avg_mae': df['MAE'].mean(),
        'avg_training_time': df['training_time'].mean() if 'training_time' in df.columns else 0,
        'models_under_1000_mae': len(df[df['MAE'] < 1000]),
        'models_mase_under_1': len(df[df['MASE'] < 1.0])
    }


def get_available_models(target: str = "consumption") -> List[str]:
    """
    Get list of available model names.

    Returns:
        List of unique model names
    """
    df = load_all_metrics(target=target)

    if df.empty:
        return []

    return sorted(df['model_name'].unique().tolist())


def get_model_comparison_df(target: str = "consumption") -> pd.DataFrame:
    """
    Get comparison dataframe for best models of each type.

    Returns:
        Formatted DataFrame for display
    """
    best_models = get_best_models_per_type(target=target, metric='MAE')

    if best_models.empty:
        return pd.DataFrame()

    # Format for display
    display_df = best_models[['model_name', 'MAE', 'sMAPE', 'MASE', 'category']].copy()
    display_df.columns = ['Model', 'MAE (MWh)', 'sMAPE (%)', 'MASE', 'Category']
    display_df = display_df.reset_index(drop=True)

    return display_df


def load_feature_importance(model_type: str, config_hash: str, target: str = "consumption") -> Optional[pd.DataFrame]:
    """
    Load feature importance for a specific model.

    Args:
        model_type: Model type
        config_hash: Config hash
        target: Target variable

    Returns:
        DataFrame with feature importance or None
    """
    # Determine category and construct path
    if model_type.lower() in ['catboost', 'xgboost', 'lightgbm']:
        models_dir = BASELINE_MODELS_DIR
        model_dir_name = f"{model_type.lower()}_{target}_{config_hash}"
    else:
        # Deep learning models don't have traditional feature importance
        return None

    feature_importance_file = models_dir / model_dir_name / "feature_importance.csv"

    if not feature_importance_file.exists():
        logger.warning(f"Feature importance file not found: {feature_importance_file}")
        return None

    try:
        return pd.read_csv(feature_importance_file)
    except Exception as e:
        logger.error(f"Failed to load feature importance: {e}")
        return None


if __name__ == "__main__":
    # Test the loader
    print("Testing model loader...")

    # Test load all metrics
    print("\n1. Loading all metrics for consumption:")
    df = load_all_metrics(target="consumption")
    print(f"Found {len(df)} models")
    print(df.head())

    # Test best models
    print("\n2. Best models per type:")
    best = get_best_models_per_type(target="consumption")
    print(best)

    # Test summary
    print("\n3. Model summary:")
    summary = get_model_summary(target="consumption")
    print(summary)

    # Test available models
    print("\n4. Available models:")
    models = get_available_models(target="consumption")
    print(models)
