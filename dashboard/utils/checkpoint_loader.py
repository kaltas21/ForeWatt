"""
Checkpoint Loader for Deep Learning Models
Extract training metrics from PyTorch Lightning checkpoints.
"""
import joblib
import streamlit as st
from pathlib import Path
from typing import Dict, List, Optional
import json


@st.cache_data
def get_available_dl_runs(deeplearning_dir: Path) -> List[Dict]:
    """
    Get list of available deep learning training runs with metrics.

    Args:
        deeplearning_dir: Path to deeplearning directory (parent of models/ and metrics/)

    Returns:
        List of dicts with model info and metrics
    """
    runs = []

    metrics_dir = deeplearning_dir / "metrics"
    if not metrics_dir.exists():
        return runs

    # Read all metric files
    for metric_file in sorted(metrics_dir.glob("*.json")):
        try:
            with open(metric_file) as f:
                metrics_data = json.load(f)

            # Extract basic info
            config_hash = metrics_data.get('config_hash', 'unknown')
            model_type = metrics_data.get('model_type', 'unknown')
            target = metrics_data.get('target', 'unknown')

            # Get config
            config = metrics_data.get('config', {})
            config_name = config.get('config_name', 'Unknown')

            # Create display name
            model_name = f"{model_type}_{target}_{config_hash}"
            display_name = f"{model_type.upper()} - {config_name} ({target})"

            runs.append({
                'model_name': model_name,
                'model_type': model_type.upper(),
                'target': target,
                'config_hash': config_hash,
                'config_name': config_name,
                'display_name': display_name,
                'metrics': metrics_data  # Full metrics data including val/test
            })
        except Exception as e:
            st.warning(f"Error loading {metric_file.name}: {e}")
            continue

    return runs


def load_checkpoint_metrics(model_dir: Path) -> Optional[Dict]:
    """
    Load training metrics from checkpoint file.

    Args:
        model_dir: Path to model directory

    Returns:
        Dictionary with training metrics if available
    """
    checkpoint_dir = model_dir / "model"
    if not checkpoint_dir.exists():
        return None

    # Find checkpoint file
    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoint_files:
        return None

    try:
        # Load checkpoint
        checkpoint = joblib.load(checkpoint_files[0])

        # Try to extract metrics from checkpoint
        # Different formats depending on how the checkpoint was saved
        metrics = {}

        # Check if checkpoint is a dict with model and trainer info
        if isinstance(checkpoint, dict):
            # Look for callback metrics
            if 'callbacks' in checkpoint:
                callbacks = checkpoint['callbacks']
                if isinstance(callbacks, dict):
                    for cb_name, cb_data in callbacks.items():
                        if 'train_loss' in str(cb_data).lower():
                            metrics['has_train_loss'] = True
                        if 'val_loss' in str(cb_data).lower():
                            metrics['has_val_loss'] = True

            # Look for epoch info
            if 'epoch' in checkpoint:
                metrics['epochs'] = checkpoint['epoch'] + 1

            # Look for global step
            if 'global_step' in checkpoint:
                metrics['steps'] = checkpoint['global_step']

        return metrics if metrics else None

    except Exception as e:
        st.error(f"Error loading checkpoint: {e}")
        return None


def extract_training_curve_from_metadata(metadata: Dict) -> Optional[Dict]:
    """
    Extract training curve information from metadata.

    Args:
        metadata: Metadata dictionary

    Returns:
        Dict with training curve data if available
    """
    # Check if metadata contains training history
    if 'training_history' in metadata:
        return metadata['training_history']

    # Try to infer from metrics
    if 'val_mae' in metadata and 'test_mae' in metadata:
        # We have final metrics but not full curves
        return {
            'final_val_mae': metadata.get('val_mae'),
            'final_test_mae': metadata.get('test_mae'),
            'has_full_curve': False
        }

    return None


def get_model_config_summary(config: Dict) -> Dict:
    """
    Extract key configuration parameters for display.

    Args:
        config: Configuration dictionary

    Returns:
        Dict with key parameters
    """
    summary = {}

    # Common parameters
    for key in ['input_size', 'horizon', 'batch_size', 'learning_rate',
                'max_steps', 'feature_strategy', 'config_name']:
        if key in config:
            summary[key] = config[key]

    # Model-specific parameters
    if 'n_blocks' in config:  # N-HiTS
        summary['n_blocks'] = config['n_blocks']
        summary['hidden_size'] = config.get('hidden_size')
    elif 'd_model' in config:  # PatchTST, TFT
        summary['d_model'] = config['d_model']
        summary['n_heads'] = config.get('n_heads')
        summary['n_layers'] = config.get('n_layers')

    return summary
