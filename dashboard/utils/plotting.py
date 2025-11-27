"""
ForeWatt Plotting Utilities
Functions for creating interactive visualizations with Plotly.
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple

from .config import PLOT_CONFIG, COLORS, BASELINE_MODELS


def create_forecast_plot(dates: pd.DatetimeIndex,
                        actual: np.ndarray,
                        predictions: Dict[str, np.ndarray],
                        intervals: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
                        title: str = "Electricity Demand Forecast") -> go.Figure:
    """
    Create interactive forecast visualization.

    Args:
        dates: Datetime index
        actual: Actual consumption values
        predictions: Dictionary of model_name -> predictions
        intervals: Optional dict of model_name -> (lower, upper) bounds
        title: Plot title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Add actual values
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual,
        mode='lines',
        name='Actual',
        line=dict(color=COLORS['text'], width=2),
        hovertemplate='<b>Actual</b><br>%{x}<br>%{y:.0f} MWh<extra></extra>'
    ))

    # Add predictions for each model
    for i, (model_name, preds) in enumerate(predictions.items()):
        color = BASELINE_MODELS.get(model_name, {}).get('color', COLORS['primary'])

        fig.add_trace(go.Scatter(
            x=dates,
            y=preds,
            mode='lines',
            name=model_name,
            line=dict(color=color, width=2, dash='dash'),
            hovertemplate=f'<b>{model_name}</b><br>%{{x}}<br>%{{y:.0f}} MWh<extra></extra>'
        ))

        # Add prediction intervals if provided
        if intervals and model_name in intervals:
            lower, upper = intervals[model_name]

            fig.add_trace(go.Scatter(
                x=dates,
                y=upper,
                mode='lines',
                name=f'{model_name} Upper',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))

            fig.add_trace(go.Scatter(
                x=dates,
                y=lower,
                mode='lines',
                name=f'{model_name} Lower',
                line=dict(width=0),
                fillcolor=color.replace('rgb', 'rgba').replace(')', ', 0.2)'),
                fill='tonexty',
                showlegend=False,
                hovertemplate=f'<b>{model_name} 90% PI</b><br>Lower: %{{y:.0f}} MWh<extra></extra>'
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Consumption (MWh)",
        height=PLOT_CONFIG['height'],
        template=PLOT_CONFIG['template'],
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def create_horizon_performance_plot(horizon_df: pd.DataFrame,
                                    metric: str = "MAE",
                                    title: Optional[str] = None) -> go.Figure:
    """
    Create plot showing performance across forecast horizons.

    Args:
        horizon_df: DataFrame with 'Horizon' column and metric columns
        metric: Metric to plot
        title: Optional plot title

    Returns:
        Plotly figure
    """
    if title is None:
        title = f"{metric} by Forecast Horizon"

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=horizon_df['Horizon'],
        y=horizon_df[metric],
        mode='lines+markers',
        name=metric,
        line=dict(color=COLORS['primary'], width=3),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Forecast Horizon (hours ahead)",
        yaxis_title=metric,
        height=PLOT_CONFIG['height'],
        template=PLOT_CONFIG['template'],
        showlegend=False
    )

    return fig


def create_metrics_comparison(metrics_df: pd.DataFrame,
                              metrics: List[str] = ["MAE", "RMSE", "MASE"]) -> go.Figure:
    """
    Create bar chart comparing models across metrics.

    Args:
        metrics_df: DataFrame with models as index and metrics as columns
        metrics: List of metrics to compare

    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=1, cols=len(metrics),
        subplot_titles=metrics,
        horizontal_spacing=0.1
    )

    for i, metric in enumerate(metrics, 1):
        if metric in metrics_df.columns:
            # Sort by metric value
            sorted_df = metrics_df[[metric]].sort_values(metric)

            colors = [BASELINE_MODELS.get(model, {}).get('color', COLORS['primary'])
                     for model in sorted_df.index]

            fig.add_trace(
                go.Bar(
                    x=sorted_df.index,
                    y=sorted_df[metric],
                    name=metric,
                    marker_color=colors,
                    showlegend=False,
                    hovertemplate='<b>%{x}</b><br>%{y:.2f}<extra></extra>'
                ),
                row=1, col=i
            )

    fig.update_layout(
        height=PLOT_CONFIG['height'],
        template=PLOT_CONFIG['template'],
        title_text="Model Comparison"
    )

    return fig


def create_residual_plot(residuals: np.ndarray,
                        dates: pd.DatetimeIndex,
                        model_name: str = "Model") -> go.Figure:
    """
    Create residual plot for error analysis.

    Args:
        residuals: Forecast residuals (actual - predicted)
        dates: Datetime index
        model_name: Name of the model

    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f"{model_name} - Residuals Over Time",
            "Residual Distribution"
        ),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )

    # Time series of residuals
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(color=COLORS['primary'], size=4, opacity=0.6),
            hovertemplate='%{x}<br>Residual: %{y:.0f} MWh<extra></extra>'
        ),
        row=1, col=1
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    # Histogram of residuals
    fig.add_trace(
        go.Histogram(
            x=residuals,
            name='Distribution',
            marker_color=COLORS['primary'],
            nbinsx=50,
            showlegend=False
        ),
        row=2, col=1
    )

    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Residual (MWh)", row=1, col=1)
    fig.update_xaxes(title_text="Residual (MWh)", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)

    fig.update_layout(
        height=PLOT_CONFIG['height'] * 1.2,
        template=PLOT_CONFIG['template'],
        showlegend=False
    )

    return fig


def create_time_series_plot(df: pd.DataFrame,
                           columns: List[str],
                           title: str = "Time Series",
                           yaxis_title: Optional[str] = None) -> go.Figure:
    """
    Create multi-line time series plot.

    Args:
        df: DataFrame with datetime index
        columns: List of columns to plot
        title: Plot title
        yaxis_title: Y-axis label

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    for col in columns:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[col],
                mode='lines',
                name=col,
                hovertemplate=f'<b>{col}</b><br>%{{x}}<br>%{{y:.2f}}<extra></extra>'
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=yaxis_title or "Value",
        height=PLOT_CONFIG['height'],
        template=PLOT_CONFIG['template'],
        hovermode='x unified'
    )

    return fig


def create_correlation_heatmap(corr_matrix: pd.DataFrame,
                               title: str = "Feature Correlation") -> go.Figure:
    """
    Create correlation heatmap.

    Args:
        corr_matrix: Correlation matrix
        title: Plot title

    Returns:
        Plotly figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 8},
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        title=title,
        height=max(500, len(corr_matrix) * 30),
        template=PLOT_CONFIG['template']
    )

    return fig


def create_feature_importance_plot(features, importances=None,
                                  top_n: int = 20,
                                  title: str = "Top Feature Importance") -> go.Figure:
    """
    Create feature importance bar chart.

    Args:
        features: List of feature names OR DataFrame with 'feature' and 'importance' columns
        importances: List of importance values (optional, if features is a DataFrame)
        top_n: Number of top features to show
        title: Plot title

    Returns:
        Plotly figure
    """
    # Handle both DataFrame and list inputs
    if isinstance(features, pd.DataFrame):
        # If it's a DataFrame, use it directly
        importance_df = features
        top_features = importance_df.nlargest(top_n, 'importance')
    else:
        # If lists are provided, create a DataFrame
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': importances
        })
        # Sort by importance and get top N
        top_features = importance_df.nlargest(top_n, 'importance')

    fig = go.Figure(go.Bar(
        x=top_features['importance'],
        y=top_features['feature'],
        orientation='h',
        marker_color=COLORS['primary'],
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=max(400, top_n * 25),
        template=PLOT_CONFIG['template'],
        yaxis=dict(autorange="reversed")
    )

    return fig


def create_box_plot(data: Dict[str, np.ndarray],
                   title: str = "Distribution Comparison",
                   yaxis_title: str = "Value") -> go.Figure:
    """
    Create box plot for comparing distributions.

    Args:
        data: Dictionary of label -> values
        title: Plot title
        yaxis_title: Y-axis label

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    for label, values in data.items():
        fig.add_trace(go.Box(
            y=values,
            name=label,
            boxmean='sd'
        ))

    fig.update_layout(
        title=title,
        yaxis_title=yaxis_title,
        height=PLOT_CONFIG['height'],
        template=PLOT_CONFIG['template']
    )

    return fig


def create_hourly_pattern_plot(hourly_df: pd.DataFrame,
                               title: str = "Hourly Consumption Pattern") -> go.Figure:
    """
    Create plot showing hourly patterns with confidence intervals.

    Args:
        hourly_df: DataFrame with hour as index and mean/std columns
        title: Plot title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    hours = hourly_df.index
    mean_values = hourly_df['mean']
    std_values = hourly_df['std']

    # Add shaded region for standard deviation
    fig.add_trace(go.Scatter(
        x=hours,
        y=mean_values + std_values,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=hours,
        y=mean_values - std_values,
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(31, 119, 180, 0.2)',
        fill='tonexty',
        showlegend=False,
        hoverinfo='skip'
    ))

    # Add mean line
    fig.add_trace(go.Scatter(
        x=hours,
        y=mean_values,
        mode='lines+markers',
        name='Mean',
        line=dict(color=COLORS['primary'], width=3),
        marker=dict(size=8),
        hovertemplate='<b>Hour %{x}</b><br>Mean: %{y:.0f} MWh<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Hour of Day",
        yaxis_title="Consumption (MWh)",
        height=PLOT_CONFIG['height'],
        template=PLOT_CONFIG['template'],
        xaxis=dict(tickmode='linear', tick0=0, dtick=2)
    )

    return fig


def create_scatter_plot(x: np.ndarray, y: np.ndarray,
                       x_label: str = "Predicted",
                       y_label: str = "Actual",
                       title: str = "Predicted vs Actual") -> go.Figure:
    """
    Create scatter plot (typically for predicted vs actual).

    Args:
        x: X values (typically predictions)
        y: Y values (typically actuals)
        x_label: X-axis label
        y_label: Y-axis label
        title: Plot title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Add scatter points
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(color=COLORS['primary'], size=6, opacity=0.5),
        name='Data',
        hovertemplate=f'{x_label}: %{{x:.0f}}<br>{y_label}: %{{y:.0f}}<extra></extra>'
    ))

    # Add diagonal line (perfect prediction)
    min_val = min(np.min(x), np.min(y))
    max_val = max(np.max(x), np.max(y))

    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='Perfect Prediction',
        showlegend=True
    ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=PLOT_CONFIG['height'],
        template=PLOT_CONFIG['template']
    )

    return fig


def create_gauge_chart(value: float, title: str,
                      min_val: float = 0, max_val: float = 100,
                      threshold_good: float = 70,
                      threshold_fair: float = 40) -> go.Figure:
    """
    Create gauge chart for displaying single metric.

    Args:
        value: Current value
        title: Gauge title
        min_val: Minimum value
        max_val: Maximum value
        threshold_good: Threshold for 'good' (green)
        threshold_fair: Threshold for 'fair' (yellow)

    Returns:
        Plotly figure
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': COLORS['primary']},
            'steps': [
                {'range': [min_val, threshold_fair], 'color': COLORS['error']},
                {'range': [threshold_fair, threshold_good], 'color': COLORS['warning']},
                {'range': [threshold_good, max_val], 'color': COLORS['success']}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold_good
            }
        }
    ))

    fig.update_layout(
        height=300,
        template=PLOT_CONFIG['template']
    )

    return fig


def create_split_visualization(df: pd.DataFrame,
                               train_start: str, train_end: str,
                               val_start: str, val_end: str,
                               test_start: str, test_end: str,
                               feature: str = 'consumption',
                               title: str = "Train/Validation/Test Split Visualization") -> go.Figure:
    """
    Create visualization showing train/validation/test data splits.

    Args:
        df: DataFrame with timezone-aware datetime index
        train_start: Training start date (string)
        train_end: Training end date (string)
        val_start: Validation start date (string)
        val_end: Validation end date (string)
        test_start: Test start date (string)
        test_end: Test end date (string)
        feature: Feature to visualize
        title: Plot title

    Returns:
        Plotly figure with colored regions for each split
    """
    # Get timezone from dataframe index if available
    tz = df.index.tz if hasattr(df.index, 'tz') and df.index.tz is not None else None

    def make_tz_aware(date_str: str) -> pd.Timestamp:
        """Convert date string to timezone-aware timestamp matching df index."""
        ts = pd.to_datetime(date_str)
        if tz and ts.tz is None:
            ts = ts.tz_localize(tz)
        return ts

    # Convert date strings to timezone-aware timestamps
    train_start_ts = make_tz_aware(train_start)
    train_end_ts = make_tz_aware(train_end)
    val_start_ts = make_tz_aware(val_start)
    val_end_ts = make_tz_aware(val_end)
    test_start_ts = make_tz_aware(test_start)
    test_end_ts = make_tz_aware(test_end)

    fig = go.Figure()

    # Extract splits with timezone-aware timestamps
    train_df = df.loc[train_start_ts:train_end_ts]
    val_df = df.loc[val_start_ts:val_end_ts]
    test_df = df.loc[test_start_ts:test_end_ts]

    # Add train data
    fig.add_trace(go.Scatter(
        x=train_df.index,
        y=train_df[feature],
        mode='lines',
        name='Train',
        line=dict(color='#2ca02c', width=1.5),
        hovertemplate='<b>Train</b><br>%{x}<br>%{y:.2f}<extra></extra>'
    ))

    # Add validation data
    fig.add_trace(go.Scatter(
        x=val_df.index,
        y=val_df[feature],
        mode='lines',
        name='Validation',
        line=dict(color='#ff7f0e', width=1.5),
        hovertemplate='<b>Validation</b><br>%{x}<br>%{y:.2f}<extra></extra>'
    ))

    # Add test data
    fig.add_trace(go.Scatter(
        x=test_df.index,
        y=test_df[feature],
        mode='lines',
        name='Test',
        line=dict(color='#d62728', width=1.5),
        hovertemplate='<b>Test</b><br>%{x}<br>%{y:.2f}<extra></extra>'
    ))

    # Add vertical lines at split boundaries
    # Use shapes instead of add_vline to avoid timestamp arithmetic issues
    fig.add_shape(
        type="line",
        x0=train_end_ts, x1=train_end_ts,
        y0=0, y1=1,
        yref="paper",
        line=dict(color="gray", width=2, dash="dash")
    )

    fig.add_shape(
        type="line",
        x0=val_end_ts, x1=val_end_ts,
        y0=0, y1=1,
        yref="paper",
        line=dict(color="gray", width=2, dash="dash")
    )

    # Add annotations separately (positioned manually to avoid arithmetic issues)
    fig.add_annotation(
        x=train_end_ts,
        y=1.02,
        yref="paper",
        text="Train/Val Split",
        showarrow=False,
        font=dict(size=10, color="gray")
    )

    fig.add_annotation(
        x=val_end_ts,
        y=1.02,
        yref="paper",
        text="Val/Test Split",
        showarrow=False,
        font=dict(size=10, color="gray")
    )

    # Add shaded regions using shapes to avoid timestamp arithmetic issues
    fig.add_shape(
        type="rect",
        x0=train_start_ts, x1=train_end_ts,
        y0=0, y1=1,
        yref="paper",
        fillcolor="green",
        opacity=0.1,
        layer="below",
        line_width=0
    )

    fig.add_shape(
        type="rect",
        x0=val_start_ts, x1=val_end_ts,
        y0=0, y1=1,
        yref="paper",
        fillcolor="orange",
        opacity=0.1,
        layer="below",
        line_width=0
    )

    fig.add_shape(
        type="rect",
        x0=test_start_ts, x1=test_end_ts,
        y0=0, y1=1,
        yref="paper",
        fillcolor="red",
        opacity=0.1,
        layer="below",
        line_width=0
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=feature.replace('_', ' ').title(),
        height=PLOT_CONFIG['height'],
        template=PLOT_CONFIG['template'],
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def create_learning_curve_plot(train_losses: List[float],
                               val_losses: List[float],
                               title: str = "Training & Validation Loss") -> go.Figure:
    """
    Create learning curve plot showing train and validation losses.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        title: Plot title

    Returns:
        Plotly figure
    """
    epochs = list(range(1, len(train_losses) + 1))

    fig = go.Figure()

    # Training loss
    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_losses,
        mode='lines+markers',
        name='Train Loss',
        line=dict(color=COLORS['primary'], width=2),
        marker=dict(size=6),
        hovertemplate='<b>Train</b><br>Epoch: %{x}<br>Loss: %{y:.4f}<extra></extra>'
    ))

    # Validation loss
    fig.add_trace(go.Scatter(
        x=epochs,
        y=val_losses,
        mode='lines+markers',
        name='Val Loss',
        line=dict(color=COLORS['warning'], width=2),
        marker=dict(size=6),
        hovertemplate='<b>Validation</b><br>Epoch: %{x}<br>Loss: %{y:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=PLOT_CONFIG['height'],
        template=PLOT_CONFIG['template'],
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def create_error_analysis_plot(errors: np.ndarray,
                               dates: pd.DatetimeIndex,
                               title: str = "Prediction Error Analysis") -> go.Figure:
    """
    Create comprehensive error analysis plot.

    Args:
        errors: Prediction errors
        dates: Datetime index
        title: Plot title

    Returns:
        Plotly figure with error distribution and time series
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Error Over Time",
            "Error Distribution",
            "Absolute Error Over Time",
            "Error Statistics by Hour"
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # Error time series
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=errors,
            mode='markers',
            marker=dict(color=COLORS['primary'], size=3, opacity=0.5),
            name='Error',
            hovertemplate='%{x}<br>Error: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    # Error histogram
    fig.add_trace(
        go.Histogram(
            x=errors,
            nbinsx=50,
            marker_color=COLORS['primary'],
            name='Distribution',
            showlegend=False
        ),
        row=1, col=2
    )

    # Absolute error time series
    abs_errors = np.abs(errors)
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=abs_errors,
            mode='markers',
            marker=dict(color=COLORS['error'], size=3, opacity=0.5),
            name='Abs Error',
            hovertemplate='%{x}<br>|Error|: %{y:.2f}<extra></extra>'
        ),
        row=2, col=1
    )

    # Error by hour
    df_errors = pd.DataFrame({'error': errors, 'hour': dates.hour})
    hourly_errors = df_errors.groupby('hour')['error'].agg(['mean', 'std'])

    fig.add_trace(
        go.Bar(
            x=hourly_errors.index,
            y=hourly_errors['mean'].abs(),
            error_y=dict(type='data', array=hourly_errors['std']),
            marker_color=COLORS['secondary'],
            name='Hourly MAE',
            showlegend=False
        ),
        row=2, col=2
    )

    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Error", row=1, col=1)
    fig.update_xaxes(title_text="Error", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="|Error|", row=2, col=1)
    fig.update_xaxes(title_text="Hour", row=2, col=2)
    fig.update_yaxes(title_text="MAE", row=2, col=2)

    fig.update_layout(
        title_text=title,
        height=PLOT_CONFIG['height'] * 1.5,
        template=PLOT_CONFIG['template'],
        showlegend=False
    )

    return fig
