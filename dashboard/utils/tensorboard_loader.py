"""
TensorBoard Log Loader
Load training metrics from PyTorch Lightning TensorBoard logs.
"""
import streamlit as st
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd

try:
    from tensorboard.backend.event_processing import event_accumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


def get_available_runs(lightning_logs_dir: Path, filter_empty: bool = True) -> List[str]:
    """
    Get list of available training runs.

    Args:
        lightning_logs_dir: Path to lightning_logs directory
        filter_empty: If True, only return versions with actual training data

    Returns:
        List of version names (e.g., ['version_0', 'version_1', ...])
    """
    if not lightning_logs_dir.exists():
        return []

    all_versions = sorted([
        d.name for d in lightning_logs_dir.iterdir()
        if d.is_dir() and d.name.startswith('version_')
    ], key=lambda x: int(x.split('_')[1]))

    if not filter_empty:
        return all_versions

    # Filter to only versions with training/validation loss data
    versions_with_data = []
    for version_name in all_versions:
        version_dir = lightning_logs_dir / version_name
        event_files = list(version_dir.glob("events.out.tfevents.*"))

        if event_files:
            try:
                ea = event_accumulator.EventAccumulator(str(event_files[0]))
                ea.Reload()

                tags = ea.Tags()['scalars']
                # Check if it has training or validation loss
                has_train = any('train' in tag.lower() and 'loss' in tag.lower() for tag in tags)
                has_val = any(('val' in tag.lower() or 'valid' in tag.lower()) and 'loss' in tag.lower() for tag in tags)

                if has_train or has_val:
                    versions_with_data.append(version_name)
            except:
                continue

    return versions_with_data


def load_training_metrics(version_dir: Path) -> Optional[Dict[str, List[Tuple[int, float]]]]:
    """
    Load training and validation metrics from a TensorBoard log directory.

    Args:
        version_dir: Path to version directory (e.g., lightning_logs/version_0)

    Returns:
        Dictionary with metrics:
        {
            'train_loss': [(epoch, loss), ...],
            'val_loss': [(epoch, loss), ...],
            'epochs': number of epochs
        }
    """
    if not TENSORBOARD_AVAILABLE:
        return None

    event_files = list(version_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return None

    try:
        ea = event_accumulator.EventAccumulator(str(event_files[0]))
        ea.Reload()

        metrics = {}

        # Get available scalar tags
        available_tags = ea.Tags()['scalars']

        # Map common tag names to standard names
        train_tags = ['train_loss_epoch', 'train/loss_epoch', 'train_loss', 'train/loss']
        val_tags = ['valid_loss', 'val_loss', 'val/loss', 'ptl/val_loss']

        # Extract training loss
        for tag in train_tags:
            if tag in available_tags:
                data = ea.Scalars(tag)
                metrics['train_loss'] = [(int(d.step), float(d.value)) for d in data]
                break

        # Extract validation loss
        for tag in val_tags:
            if tag in available_tags:
                data = ea.Scalars(tag)
                # Use epoch from event if available, otherwise use step
                if 'epoch' in available_tags:
                    epoch_data = ea.Scalars('epoch')
                    epoch_map = {e.step: int(e.value) for e in epoch_data}
                    metrics['val_loss'] = [
                        (epoch_map.get(d.step, d.step), float(d.value))
                        for d in data
                    ]
                else:
                    metrics['val_loss'] = [(int(d.step), float(d.value)) for d in data]
                break

        # Get number of epochs
        if 'epoch' in available_tags:
            epoch_data = ea.Scalars('epoch')
            metrics['epochs'] = max(int(e.value) for e in epoch_data) + 1
        elif 'train_loss' in metrics:
            metrics['epochs'] = len(metrics['train_loss'])
        else:
            metrics['epochs'] = 0

        return metrics if metrics else None

    except Exception as e:
        st.error(f"Error loading TensorBoard logs: {e}")
        return None


def get_run_info(version_dir: Path) -> Optional[Dict]:
    """
    Get hyperparameters and metadata for a training run.

    Args:
        version_dir: Path to version directory

    Returns:
        Dictionary with run information
    """
    import yaml

    hparams_file = version_dir / "hparams.yaml"
    if not hparams_file.exists():
        return None

    try:
        with open(hparams_file, 'r') as f:
            hparams = yaml.safe_load(f)
        return hparams
    except Exception:
        return None


@st.cache_data
def load_all_runs_summary(lightning_logs_dir: Path) -> pd.DataFrame:
    """
    Load summary of all training runs.

    Args:
        lightning_logs_dir: Path to lightning_logs directory

    Returns:
        DataFrame with columns: version, epochs, final_train_loss, final_val_loss, best_val_loss
    """
    runs = get_available_runs(lightning_logs_dir)

    data = []
    for run_name in runs:
        version_dir = lightning_logs_dir / run_name
        metrics = load_training_metrics(version_dir)

        if metrics:
            row = {
                'version': run_name,
                'epochs': metrics.get('epochs', 0)
            }

            if 'train_loss' in metrics and metrics['train_loss']:
                train_losses = [loss for _, loss in metrics['train_loss']]
                row['final_train_loss'] = train_losses[-1]
                row['best_train_loss'] = min(train_losses)

            if 'val_loss' in metrics and metrics['val_loss']:
                val_losses = [loss for _, loss in metrics['val_loss']]
                row['final_val_loss'] = val_losses[-1]
                row['best_val_loss'] = min(val_losses)

            # Get hyperparameters if available
            hparams = get_run_info(version_dir)
            if hparams:
                row['model'] = hparams.get('model_name', 'Unknown')
                row['target'] = hparams.get('target', 'Unknown')

            data.append(row)

    return pd.DataFrame(data)
