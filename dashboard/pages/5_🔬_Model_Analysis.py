"""
ForeWatt Dashboard - Model Analysis & Training Visualization
Comprehensive page for model analysis, training visualization, and forecasting.
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

# Add parent utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    load_master_data, get_data_summary, VALIDATION_CONFIG,
    TARGET_VARIABLE, PAGE_CONFIG, create_split_visualization,
    create_learning_curve_plot, create_error_analysis_plot,
    create_forecast_plot, create_time_series_plot,
    simulate_predictions, calculate_all_metrics,
    # New experiment loaders
    load_all_metrics, get_best_models_per_type, get_model_summary,
    get_model_comparison_df, load_feature_importance, get_available_models as get_available_models_v2
)

# TensorBoard log loader
from utils.tensorboard_loader import (
    get_available_runs, load_training_metrics, get_run_info,
    load_all_runs_summary, TENSORBOARD_AVAILABLE
)

# Page configuration
st.set_page_config(**PAGE_CONFIG)

st.title("🔬 Model Analysis & Training Visualization")
st.markdown("Deep learning model training visualization and data split analysis.")

st.divider()

# Sidebar menu for user to select analysis type
st.sidebar.header("Analysis Menu")
analysis_type = st.sidebar.radio(
    "Select Analysis Type:",
    [
        "📊 Train/Val/Test Split Visualization",
        "📉 Training & Validation Loss (Deep Learning)",
        "🔍 Deep Learning Model Performance"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("### Instructions")
st.sidebar.info("""
**Train/Val/Test Split**: View data splits with statistics

**Training & Validation Loss**: View deep learning training curves from TensorBoard logs

**DL Model Performance**: Compare all 90 deep learning models with metrics
""")

try:
    # Load data
    with st.spinner("Loading data..."):
        df = load_master_data()
        data_summary = get_data_summary(df)

    if df.empty:
        st.error("No data available. Please ensure the master dataset is accessible.")
        st.stop()

    # Helper function to make timestamps timezone-aware
    def make_tz_aware(date_input) -> pd.Timestamp:
        """Convert date input to timezone-aware timestamp matching df index."""
        tz = df.index.tz if hasattr(df.index, 'tz') and df.index.tz is not None else 'Europe/Istanbul'
        if isinstance(date_input, str):
            ts = pd.to_datetime(date_input)
        else:
            ts = pd.Timestamp(date_input)
        if ts.tz is None:
            ts = ts.tz_localize(tz)
        return ts

    # ===================================================================
    # TRAIN/VAL/TEST SPLIT VISUALIZATION
    # ===================================================================
    if analysis_type == "📊 Train/Val/Test Split Visualization":
        st.markdown("## Train/Validation/Test Data Split")
        st.markdown("""
        This visualization shows how the data is split for model training and evaluation.
        - **Train** (2020-2022): 3 years for model training
        - **Validation** (2023): 1 year for hyperparameter tuning
        - **Test** (2024): 1 year for final model evaluation
        """)

        # Feature selector
        col1, col2 = st.columns([2, 1])

        with col1:
            feature_to_plot = st.selectbox(
                "Select feature to visualize:",
                [TARGET_VARIABLE] + [col for col in df.columns if col != TARGET_VARIABLE][:20],
                help="Choose which feature to display across train/val/test splits"
            )

        with col2:
            show_statistics = st.checkbox("Show Statistics", value=True)

        # Create split visualization
        fig = create_split_visualization(
            df,
            train_start=VALIDATION_CONFIG['train_start'],
            train_end=VALIDATION_CONFIG['train_end'],
            val_start=VALIDATION_CONFIG['val_start'],
            val_end=VALIDATION_CONFIG['val_end'],
            test_start=VALIDATION_CONFIG['test_start'],
            test_end=VALIDATION_CONFIG['test_end'],
            feature=feature_to_plot,
            title=f"{feature_to_plot.replace('_', ' ').title()} - Train/Validation/Test Split"
        )
        st.plotly_chart(fig, use_container_width=True)

        if show_statistics:
            st.markdown("### Split Statistics")

            # Extract splits with timezone-aware timestamps
            train_start_ts = make_tz_aware(VALIDATION_CONFIG['train_start'])
            train_end_ts = make_tz_aware(VALIDATION_CONFIG['train_end'])
            val_start_ts = make_tz_aware(VALIDATION_CONFIG['val_start'])
            val_end_ts = make_tz_aware(VALIDATION_CONFIG['val_end'])
            test_start_ts = make_tz_aware(VALIDATION_CONFIG['test_start'])
            test_end_ts = make_tz_aware(VALIDATION_CONFIG['test_end'])

            train_df = df.loc[train_start_ts:train_end_ts]
            val_df = df.loc[val_start_ts:val_end_ts]
            test_df = df.loc[test_start_ts:test_end_ts]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### 🟢 Train Set (2020-2022)")
                st.metric("Records", f"{len(train_df):,}")
                st.metric("Mean", f"{train_df[feature_to_plot].mean():.2f}")
                st.metric("Std Dev", f"{train_df[feature_to_plot].std():.2f}")
                st.metric("Min", f"{train_df[feature_to_plot].min():.2f}")
                st.metric("Max", f"{train_df[feature_to_plot].max():.2f}")

            with col2:
                st.markdown("#### 🟠 Validation Set (2023)")
                st.metric("Records", f"{len(val_df):,}")
                st.metric("Mean", f"{val_df[feature_to_plot].mean():.2f}")
                st.metric("Std Dev", f"{val_df[feature_to_plot].std():.2f}")
                st.metric("Min", f"{val_df[feature_to_plot].min():.2f}")
                st.metric("Max", f"{val_df[feature_to_plot].max():.2f}")

            with col3:
                st.markdown("#### 🔴 Test Set (2024)")
                st.metric("Records", f"{len(test_df):,}")
                st.metric("Mean", f"{test_df[feature_to_plot].mean():.2f}")
                st.metric("Std Dev", f"{test_df[feature_to_plot].std():.2f}")
                st.metric("Min", f"{test_df[feature_to_plot].min():.2f}")
                st.metric("Max", f"{test_df[feature_to_plot].max():.2f}")

            # Distribution comparison
            st.markdown("### Distribution Comparison")

            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig_dist = make_subplots(
                rows=1, cols=3,
                subplot_titles=("Train Distribution", "Validation Distribution", "Test Distribution")
            )

            fig_dist.add_trace(
                go.Histogram(x=train_df[feature_to_plot], name="Train", marker_color='#2ca02c', nbinsx=50),
                row=1, col=1
            )
            fig_dist.add_trace(
                go.Histogram(x=val_df[feature_to_plot], name="Validation", marker_color='#ff7f0e', nbinsx=50),
                row=1, col=2
            )
            fig_dist.add_trace(
                go.Histogram(x=test_df[feature_to_plot], name="Test", marker_color='#d62728', nbinsx=50),
                row=1, col=3
            )

            fig_dist.update_layout(
                height=400,
                showlegend=False,
                template='plotly_white'
            )
            fig_dist.update_xaxes(title_text=feature_to_plot)
            fig_dist.update_yaxes(title_text="Frequency")

            st.plotly_chart(fig_dist, use_container_width=True)

    # ===================================================================
    # TRAINING & VALIDATION LOSS (keep this section)
    # ===================================================================
    elif analysis_type == "📉 Training & Validation Loss (Deep Learning)":
        st.markdown("## Interactive Demand Forecasting")
        st.markdown("Generate and visualize electricity demand forecasts with prediction intervals.")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Allow forecasting for any date (past, present, or future)
            import datetime
            today = datetime.date.today()
            test_start_naive = pd.to_datetime(VALIDATION_CONFIG['test_start']).date()
            # Get actual data end date
            data_end_date = df.index.max().date()

            forecast_date = st.date_input(
                "Forecast Date",
                value=min(today, data_end_date),  # Default to latest available data
                min_value=test_start_naive,
                max_value=data_end_date,  # Limit to available data
                help=f"Select date to generate forecast for (data available until {data_end_date})"
            )

        with col2:
            forecast_horizon = st.slider(
                "Forecast Horizon (hours)",
                min_value=1,
                max_value=24,
                value=24,
                help="Number of hours ahead to forecast"
            )

        with col3:
            # Get available models from new experiments
            available_models = get_available_models_v2(target='consumption')
            if not available_models:
                st.error("No trained models found")
                st.stop()

            model_choice = st.selectbox(
                "Select Model",
                available_models,
                help="Choose forecasting model"
            )

        if st.button("🚀 Generate Forecast", use_container_width=True):
            with st.spinner(f"Generating {forecast_horizon}h forecast using {model_choice}..."):
                # Get test data for the selected date with timezone awareness
                forecast_start = make_tz_aware(forecast_date)
                forecast_end = forecast_start + pd.Timedelta(hours=forecast_horizon-1)

                # Create forecast timestamps
                forecast_timestamps = pd.date_range(
                    start=forecast_start,
                    end=forecast_end,
                    freq='H'
                )

                # Try to get actual data if available
                try:
                    test_actual = df.loc[forecast_start:forecast_end, TARGET_VARIABLE]
                    has_actual = len(test_actual) > 0
                except:
                    has_actual = False
                    test_actual = None

                # Generate predictions (in real implementation, use actual model)
                # For now, simulate based on historical average
                if has_actual and len(test_actual) > 0:
                    # Use actual data to simulate predictions
                    # Create a temporary DataFrame for simulation
                    temp_df = pd.DataFrame({TARGET_VARIABLE: test_actual.values}, index=test_actual.index)
                    result = simulate_predictions(model_choice, temp_df, n_samples=len(test_actual))
                    predictions = result['predictions']
                    dates_for_plot = result['dates']
                    actual_values = result['actual']
                else:
                    # No actual data (future forecast) - use historical average
                    st.info(f"📅 **Future Forecast**: No actual data available for {forecast_date}. Showing prediction only.")
                    historical_avg = df[TARGET_VARIABLE].mean()
                    historical_std = df[TARGET_VARIABLE].std()
                    # Simulate predictions based on historical patterns
                    predictions = np.random.normal(historical_avg, historical_std * 0.1, forecast_horizon)
                    dates_for_plot = forecast_timestamps
                    actual_values = None

                # Create prediction intervals (90%)
                std_error = 800
                lower_bound = predictions - 1.645 * std_error
                upper_bound = predictions + 1.645 * std_error

                # Create forecast plot
                if actual_values is not None:
                    fig = create_forecast_plot(
                        dates=dates_for_plot,
                        actual=actual_values,
                        predictions={model_choice: predictions},
                        intervals={model_choice: (lower_bound, upper_bound)},
                        title=f"Demand Forecast - {model_choice} ({forecast_date})"
                    )
                else:
                    # Plot predictions only (no actual data)
                    import plotly.graph_objects as go
                    fig = go.Figure()

                    # Add prediction intervals
                    fig.add_trace(go.Scatter(
                        x=dates_for_plot,
                        y=upper_bound,
                        mode='lines',
                        name='Upper 90% PI',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ))

                    fig.add_trace(go.Scatter(
                        x=dates_for_plot,
                        y=lower_bound,
                        mode='lines',
                        name='Lower 90% PI',
                        line=dict(width=0),
                        fillcolor='rgba(31, 119, 180, 0.2)',
                        fill='tonexty',
                        showlegend=False,
                        hoverinfo='skip'
                    ))

                    # Add predictions
                    fig.add_trace(go.Scatter(
                        x=dates_for_plot,
                        y=predictions,
                        mode='lines',
                        name=f'{model_choice} Forecast',
                        line=dict(color='#1f77b4', width=2, dash='dash'),
                        hovertemplate=f'<b>{model_choice}</b><br>%{{x}}<br>%{{y:.0f}} MWh<extra></extra>'
                    ))

                    fig.update_layout(
                        title=f"Demand Forecast - {model_choice} ({forecast_date})",
                        xaxis_title="Date",
                        yaxis_title="Consumption (MWh)",
                        height=500,
                        template='plotly_white',
                        hovermode='x unified'
                    )

                st.plotly_chart(fig, use_container_width=True)

                # Calculate metrics only if actual data exists
                if actual_values is not None:
                    metrics = calculate_all_metrics(actual_values, predictions)

                    st.markdown("### Forecast Metrics")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("MAE", f"{metrics['MAE']:.2f} MWh")
                    with col2:
                        st.metric("RMSE", f"{metrics['RMSE']:.2f} MWh")
                    with col3:
                        st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
                    with col4:
                        st.metric("R²", f"{metrics['R2']:.3f}")

                    # Error analysis
                    errors = actual_values - predictions
                    fig_error = create_error_analysis_plot(
                        errors,
                        dates_for_plot,
                        title=f"Error Analysis - {model_choice}"
                    )
                    st.plotly_chart(fig_error, use_container_width=True)
                else:
                    st.markdown("### Forecast Summary")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Predicted Mean", f"{predictions.mean():.2f} MWh")
                    with col2:
                        st.metric("Predicted Min", f"{predictions.min():.2f} MWh")
                    with col3:
                        st.metric("Predicted Max", f"{predictions.max():.2f} MWh")

                    st.info("💡 **Note**: Metrics cannot be calculated without actual data. This is a future forecast.")

    # ===================================================================
    # PRICE ESTIMATION (INTERACTIVE)
    # ===================================================================
    elif analysis_type == "💰 Price Estimation (Interactive)":
        st.markdown("## Interactive Price Estimation")
        st.markdown("Generate and visualize electricity price forecasts.")

        # Check if price data is available
        price_cols = [col for col in df.columns if 'price' in col.lower()]

        if not price_cols:
            st.warning("Price data not available in the current dataset.")
            st.stop()

        col1, col2, col3 = st.columns(3)

        with col1:
            price_feature = st.selectbox(
                "Select Price Feature",
                price_cols,
                help="Choose which price metric to forecast"
            )

        with col2:
            # Allow forecasting for any date (past, present, or future)
            import datetime
            today = datetime.date.today()
            test_start_naive = pd.to_datetime(VALIDATION_CONFIG['test_start']).date()
            # Get actual data end date
            data_end_date = df.index.max().date()

            forecast_date = st.date_input(
                "Forecast Date",
                value=min(today, data_end_date),  # Default to latest available data
                min_value=test_start_naive,
                max_value=data_end_date,  # Limit to available data
                help=f"Select date to generate price forecast for (data available until {data_end_date})"
            )

        with col3:
            forecast_horizon = st.slider(
                "Forecast Horizon (hours)",
                min_value=1,
                max_value=24,
                value=24,
                help="Number of hours ahead to forecast"
            )

        if st.button("🚀 Generate Price Forecast", use_container_width=True):
            with st.spinner(f"Generating {forecast_horizon}h price forecast..."):
                forecast_start = make_tz_aware(forecast_date)
                forecast_end = forecast_start + pd.Timedelta(hours=forecast_horizon-1)

                test_actual = df.loc[forecast_start:forecast_end, price_feature]

                if len(test_actual) > 0:
                    # Simulate price predictions
                    # Create a temporary DataFrame for simulation
                    temp_df = pd.DataFrame({TARGET_VARIABLE: test_actual.values}, index=test_actual.index)
                    result = simulate_predictions("CatBoost", temp_df, n_samples=len(test_actual))
                    predictions = result['predictions']

                    # Create forecast plot
                    fig = create_forecast_plot(
                        dates=test_actual.index,
                        actual=test_actual.values,
                        predictions={"Price Model": predictions},
                        title=f"{price_feature.replace('_', ' ').title()} Forecast ({forecast_date})"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Calculate metrics
                    metrics = calculate_all_metrics(test_actual.values, predictions)

                    st.markdown("### Forecast Metrics")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("MAE", f"{metrics['MAE']:.2f}")
                    with col2:
                        st.metric("RMSE", f"{metrics['RMSE']:.2f}")
                    with col3:
                        st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
                    with col4:
                        st.metric("R²", f"{metrics['R2']:.3f}")

                else:
                    st.error("No data available for selected date range.")

    # ===================================================================
    # TRAINING & VALIDATION LOSS
    # ===================================================================
    elif analysis_type == "📉 Training & Validation Loss (Deep Learning)":
        st.markdown("## Deep Learning Model Training Analysis")
        st.markdown("Analyze deep learning model configurations and training metrics from the new experiment.")

        # Get deep learning directory
        from utils.checkpoint_loader import get_available_dl_runs, get_model_config_summary
        deeplearning_dir = Path(__file__).parent.parent.parent / "reports" / "new_experiment" / "deeplearning"

        if not deeplearning_dir.exists():
            st.error("❌ Deep learning directory not found")
            st.stop()

        # Get available deep learning runs (reads from metrics/ directory)
        dl_runs = get_available_dl_runs(deeplearning_dir)

        if not dl_runs:
            st.warning("⚠️ No deep learning training runs found")
            st.stop()

        st.success(f"✅ Found {len(dl_runs)} deep learning model configurations")

        # Filter by target
        target_filter = st.radio(
            "Filter by target:",
            ["All", "consumption", "price_real"],
            horizontal=True
        )

        filtered_runs = [r for r in dl_runs if target_filter == "All" or r['target'] == target_filter]

        st.info(f"📊 Showing {len(filtered_runs)} models for target: {target_filter}")

        # Select run
        col1, col2 = st.columns([2, 1])

        with col1:
            selected_idx = st.selectbox(
                "Select Model Configuration",
                range(len(filtered_runs)),
                format_func=lambda i: filtered_runs[i]['display_name'],
                help="Choose a deep learning model to analyze"
            )

            selected_run = filtered_runs[selected_idx]

        with col2:
            st.markdown("**Model Info:**")
            st.caption(f"**Type:** {selected_run['model_type']}")
            st.caption(f"**Target:** {selected_run['target']}")
            st.caption(f"**Config:** {selected_run['config_name']}")

        # Display model configuration
        st.markdown("### ⚙️ Model Configuration")

        metrics_data = selected_run['metrics']
        config = metrics_data.get('config', {})
        config_summary = get_model_config_summary(config)

        cols = st.columns(4)
        col_idx = 0
        for key, value in config_summary.items():
            with cols[col_idx % 4]:
                st.metric(key.replace('_', ' ').title(), str(value))
            col_idx += 1

        # Display training metrics from metrics data
        st.markdown("### 📊 Training & Validation Metrics")

        col1, col2, col3 = st.columns(3)

        validation_metrics = metrics_data.get('validation_metrics', {})
        test_metrics = metrics_data.get('test_metrics', {})

        with col1:
            st.markdown("**Validation Performance**")
            if 'MAE' in validation_metrics:
                st.metric("Val MAE", f"{validation_metrics['MAE']:.2f}")
            if 'sMAPE' in validation_metrics:
                st.metric("Val sMAPE", f"{validation_metrics['sMAPE']:.2f}%")
            if 'MASE' in validation_metrics:
                st.metric("Val MASE", f"{validation_metrics['MASE']:.3f}")

        with col2:
            st.markdown("**Test Performance**")
            if 'MAE' in test_metrics:
                st.metric("Test MAE", f"{test_metrics['MAE']:.2f}")
            if 'sMAPE' in test_metrics:
                st.metric("Test sMAPE", f"{test_metrics['sMAPE']:.2f}%")
            if 'MASE' in test_metrics:
                st.metric("Test MASE", f"{test_metrics['MASE']:.3f}")

        with col3:
            st.markdown("**Training Info**")
            if 'training_time_seconds' in metrics_data:
                training_time = metrics_data['training_time_seconds']
                if training_time < 60:
                    time_str = f"{training_time:.1f}s"
                elif training_time < 3600:
                    time_str = f"{training_time/60:.1f}min"
                else:
                    time_str = f"{training_time/3600:.1f}h"
                st.metric("Training Time", time_str)
            if 'status' in metrics_data:
                status_emoji = "✅" if metrics_data['status'] == 'success' else "❌"
                st.metric("Status", f"{status_emoji} {metrics_data['status']}")

        # Generalization analysis
        if 'MASE' in validation_metrics and 'MASE' in test_metrics:
            st.markdown("### 📈 Generalization Analysis")
            val_mase = validation_metrics['MASE']
            test_mase = test_metrics['MASE']
            gap = test_mase - val_mase
            gap_pct = (gap / val_mase) * 100 if val_mase > 0 else 0

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Val → Test Gap", f"{gap:.3f}", f"{gap_pct:+.1f}%")

                if gap_pct > 20:
                    st.warning("⚠️ Significant performance degradation on test set")
                elif gap_pct > 10:
                    st.info("ℹ️ Moderate performance drop on test set")
                else:
                    st.success("✅ Good generalization to test set")

            with col2:
                # Baseline comparison (MASE < 1 means better than naive)
                if val_mase < 1.0 and test_mase < 1.0:
                    st.success("✅ Outperforms naive forecast on both val & test")
                elif val_mase < 1.0 or test_mase < 1.0:
                    st.info("ℹ️ Outperforms naive forecast on one split")
                else:
                    st.warning("⚠️ Underperforms naive forecast")

        # Learning rate from config
        if 'learning_rate' in config:
            st.markdown("### 📉 Learning Rate Configuration")

            lr = config['learning_rate']
            max_steps = config.get('max_steps', 1000)

            # Show configured learning rate
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Initial LR", f"{lr:.4f}")
            with col2:
                st.metric("Max Steps", f"{max_steps:,}")
            with col3:
                batch_size = config.get('batch_size', 'N/A')
                st.metric("Batch Size", batch_size)

            # Create LR schedule visualization
            # Use cosine annealing schedule (more realistic than exponential decay)
            steps_array = np.arange(0, max_steps)

            # Cosine annealing: lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))
            lr_min = lr * 0.1  # Minimum LR is 10% of initial
            lr_schedule = lr_min + 0.5 * (lr - lr_min) * (1 + np.cos(np.pi * steps_array / max_steps))

            import plotly.graph_objects as go
            fig_lr = go.Figure()

            fig_lr.add_trace(go.Scatter(
                x=steps_array,
                y=lr_schedule,
                mode='lines',
                name='Learning Rate',
                line=dict(color='#1f77b4', width=2),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.1)'
            ))

            # Add initial and final LR markers
            fig_lr.add_hline(y=lr, line_dash="dash", line_color="green",
                           annotation_text=f"Initial: {lr:.4f}",
                           annotation_position="right")
            fig_lr.add_hline(y=lr_min, line_dash="dash", line_color="red",
                           annotation_text=f"Min: {lr_min:.4f}",
                           annotation_position="right")

            fig_lr.update_layout(
                title=f"Learning Rate Schedule (Cosine Annealing)",
                xaxis_title="Training Step",
                yaxis_title="Learning Rate",
                height=400,
                template='plotly_white',
                showlegend=False
            )

            st.plotly_chart(fig_lr, use_container_width=True)
            st.caption("ℹ️ Visualization uses cosine annealing schedule. Actual schedule depends on PyTorch Lightning scheduler configuration.")

    # ===================================================================
    # ERROR ANALYSIS
    # ===================================================================
    elif analysis_type == "🎯 Error Analysis":
        st.markdown("## Comprehensive Error Analysis")
        st.markdown("Analyze prediction errors across different dimensions.")

        # Select test period
        col1, col2 = st.columns(2)

        with col1:
            # Use timezone-naive dates for date_input widget
            test_start_naive = pd.to_datetime(VALIDATION_CONFIG['test_start']).date()
            test_end_naive = pd.to_datetime(VALIDATION_CONFIG['test_end']).date()

            test_start_date = st.date_input(
                "Test Start Date",
                value=test_start_naive,
                min_value=test_start_naive,
                max_value=test_end_naive
            )

        with col2:
            test_days = st.slider("Number of Days", min_value=7, max_value=90, value=30)

        if st.button("📊 Analyze Errors", use_container_width=True):
            with st.spinner("Analyzing prediction errors..."):
                test_start = make_tz_aware(test_start_date)
                test_end = test_start + pd.Timedelta(days=test_days)

                test_data = df.loc[test_start:test_end, TARGET_VARIABLE]

                if len(test_data) > 0:
                    # Simulate predictions
                    temp_df = pd.DataFrame({TARGET_VARIABLE: test_data.values}, index=test_data.index)
                    result = simulate_predictions("CatBoost", temp_df, n_samples=len(test_data))
                    predictions = result['predictions']
                    errors = result['actual'] - predictions

                    # Create comprehensive error analysis
                    fig = create_error_analysis_plot(
                        errors,
                        test_data.index,
                        title=f"Error Analysis ({test_start_date} to {test_end.date()})"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Error statistics
                    st.markdown("### Error Statistics")

                    col1, col2, col3, col4, col5 = st.columns(5)

                    with col1:
                        st.metric("Mean Error", f"{errors.mean():.2f} MWh")
                    with col2:
                        st.metric("Std Error", f"{errors.std():.2f} MWh")
                    with col3:
                        st.metric("MAE", f"{np.abs(errors).mean():.2f} MWh")
                    with col4:
                        st.metric("Max Error", f"{np.abs(errors).max():.2f} MWh")
                    with col5:
                        mape = np.mean(np.abs(errors / test_data.values)) * 100
                        st.metric("MAPE", f"{mape:.2f}%")

                else:
                    st.error("No data available for selected period.")

    # ===================================================================
    # MODEL PERFORMANCE COMPARISON
    # ===================================================================
    elif analysis_type == "🔍 Deep Learning Model Performance":
        st.markdown("## Model Performance Comparison")
        st.markdown("Compare all trained models organized by target type and model category.")

        # Main tabs for Demand vs Price
        main_tabs = st.tabs(["⚡ Demand Forecasting Models", "💰 Price Estimation Models"])

        # ============================================================
        # DEMAND FORECASTING MODELS
        # ============================================================
        with main_tabs[0]:
            st.markdown("### Electricity Demand (Consumption) Forecasting")
            target_choice = 'consumption'

            # Get model summary
            summary = get_model_summary(target=target_choice)

            if summary:
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Configurations", summary['total_models'])
                with col2:
                    st.metric("Baseline Models", summary['baseline_models'])
                with col3:
                    st.metric("Deep Learning Models", summary['deep_learning_models'])
                with col4:
                    st.metric("Models MASE < 1.0", summary['models_mase_under_1'])

            st.divider()

            # Metric selector for sorting and comparison
            col1, col2 = st.columns([1, 3])
            with col1:
                metric_choice_demand = st.selectbox(
                    "📊 Sort & Compare By:",
                    options=['MAE', 'sMAPE', 'MASE', 'val_MAE', 'val_MASE'],
                    index=0,
                    key='demand_metric_selector',
                    help="Choose which error metric to use for sorting and comparison"
                )
            with col2:
                metric_descriptions = {
                    'MAE': '**MAE** (Mean Absolute Error) - Average prediction error in MWh. Lower is better.',
                    'sMAPE': '**sMAPE** (Symmetric MAPE) - Percentage error measure. Lower is better.',
                    'MASE': '**MASE** (Mean Absolute Scaled Error) - Scaled relative to naive baseline. < 1.0 is good.',
                    'val_MAE': '**Validation MAE** - MAE on validation set (2023 data). Lower is better.',
                    'val_MASE': '**Validation MASE** - MASE on validation set. < 1.0 is good.'
                }
                st.info(metric_descriptions[metric_choice_demand])

            st.divider()

            # Separate sections for Baseline and Deep Learning
            model_category_tabs = st.tabs(["🎯 Baseline Models", "🧠 Deep Learning Models", "📊 All Models"])

            # Load ALL model configurations for consumption (90 total)
            all_models = load_all_metrics(target=target_choice)

            # Sort by selected metric
            if not all_models.empty and metric_choice_demand in all_models.columns:
                all_models = all_models.sort_values(by=metric_choice_demand)

            # -------------------- BASELINE MODELS --------------------
            with model_category_tabs[0]:
                st.markdown("#### Baseline Model Performance (CatBoost, LightGBM, XGBoost)")

                baseline_models = all_models[all_models['category'] == 'baseline']

                if not baseline_models.empty:
                    # Overview metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        best_baseline = baseline_models.nsmallest(1, metric_choice_demand).iloc[0]
                        metric_value = best_baseline[metric_choice_demand]
                        metric_unit = "MWh" if "MAE" in metric_choice_demand else ("%" if "sMAPE" in metric_choice_demand else "")
                        st.metric(f"🏆 Best Baseline ({metric_choice_demand})",
                                 best_baseline['model_name'],
                                 f"{metric_choice_demand}: {metric_value:.2f} {metric_unit}")
                    with col2:
                        avg_value = baseline_models[metric_choice_demand].mean()
                        st.metric(f"Average {metric_choice_demand}", f"{avg_value:.2f} {metric_unit}")
                    with col3:
                        total_configs = len(baseline_models)
                        st.metric("Total Configurations", f"{total_configs}")

                    st.markdown(f"##### Detailed Metrics (Sorted by {metric_choice_demand})")
                    baseline_display = baseline_models[['model_name', 'config_name', 'MAE', 'sMAPE', 'MASE',
                                                        'val_MAE', 'val_MASE', 'n_features', 'training_time']].copy()
                    baseline_display.columns = ['Model', 'Configuration', 'Test MAE', 'Test sMAPE (%)', 'Test MASE',
                                                'Val MAE', 'Val MASE', 'Features', 'Train Time (s)']

                    # Round values
                    for col in ['Test MAE', 'Test sMAPE (%)', 'Test MASE', 'Val MAE', 'Val MASE', 'Train Time (s)']:
                        baseline_display[col] = baseline_display[col].round(3)

                    # Map metric choice to display column name
                    metric_to_display_col = {
                        'MAE': 'Test MAE',
                        'sMAPE': 'Test sMAPE (%)',
                        'MASE': 'Test MASE',
                        'val_MAE': 'Val MAE',
                        'val_MASE': 'Val MASE'
                    }
                    highlight_col = metric_to_display_col.get(metric_choice_demand, 'Test MAE')

                    st.dataframe(
                        baseline_display.style.background_gradient(
                            subset=[highlight_col],
                            cmap='RdYlGn_r'
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

                    # Visualization
                    import plotly.graph_objects as go
                    fig = go.Figure()

                    # Group by model name and get best config per model for visualization
                    viz_data = baseline_models.groupby('model_name')[metric_choice_demand].min().reset_index()

                    fig.add_trace(go.Bar(
                        name=metric_choice_demand,
                        x=viz_data['model_name'],
                        y=viz_data[metric_choice_demand],
                        text=viz_data[metric_choice_demand].round(2),
                        textposition='outside',
                        marker_color='#1f77b4'
                    ))

                    y_axis_label = f"{metric_choice_demand} ({metric_unit})" if metric_unit else metric_choice_demand
                    fig.update_layout(
                        title=f"Baseline Models - {metric_choice_demand} Comparison",
                        xaxis_title="Model",
                        yaxis_title=y_axis_label,
                        height=400,
                        template='plotly_white'
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No baseline models found for consumption")

            # -------------------- DEEP LEARNING MODELS --------------------
            with model_category_tabs[1]:
                st.markdown("#### Deep Learning Model Performance (N-HiTS, PatchTST, TFT)")

                dl_models = all_models[all_models['category'] == 'deeplearning']  # Fixed: was 'deep_learning'

                if not dl_models.empty:
                    # Overview metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        best_dl = dl_models.nsmallest(1, metric_choice_demand).iloc[0]
                        metric_value_dl = best_dl[metric_choice_demand]
                        st.metric(f"🏆 Best DL ({metric_choice_demand})",
                                 best_dl['model_name'],
                                 f"{metric_choice_demand}: {metric_value_dl:.2f} {metric_unit}")
                    with col2:
                        avg_value_dl = dl_models[metric_choice_demand].mean()
                        st.metric(f"Average {metric_choice_demand}", f"{avg_value_dl:.2f} {metric_unit}")
                    with col3:
                        total_configs_dl = len(dl_models)
                        st.metric("Total Configurations", f"{total_configs_dl}")

                    st.markdown(f"##### Detailed Metrics (Sorted by {metric_choice_demand})")
                    dl_display = dl_models[['model_name', 'config_name', 'MAE', 'sMAPE', 'MASE',
                                           'val_MAE', 'val_MASE', 'training_time']].copy()
                    dl_display.columns = ['Model', 'Configuration', 'Test MAE', 'Test sMAPE (%)', 'Test MASE',
                                         'Val MAE', 'Val MASE', 'Train Time (s)']

                    # Round values
                    for col in ['Test MAE', 'Test sMAPE (%)', 'Test MASE', 'Val MAE', 'Val MASE', 'Train Time (s)']:
                        dl_display[col] = dl_display[col].round(3)

                    st.dataframe(
                        dl_display.style.background_gradient(
                            subset=[highlight_col],
                            cmap='RdYlGn_r'
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

                    # Visualization
                    import plotly.graph_objects as go
                    fig = go.Figure()

                    # Group by model name and get best config per model for visualization
                    viz_data_dl = dl_models.groupby('model_name')[metric_choice_demand].min().reset_index()

                    fig.add_trace(go.Bar(
                        name=metric_choice_demand,
                        x=viz_data_dl['model_name'],
                        y=viz_data_dl[metric_choice_demand],
                        text=viz_data_dl[metric_choice_demand].round(2),
                        textposition='outside',
                        marker_color='#ff7f0e'
                    ))

                    fig.update_layout(
                        title=f"Deep Learning Models - {metric_choice_demand} Comparison",
                        xaxis_title="Model",
                        yaxis_title=y_axis_label,
                        height=400,
                        template='plotly_white'
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Performance note
                    if dl_models['MASE'].min() > 1.0:
                        st.warning("⚠️ **Note**: All deep learning models have MASE > 1.0, meaning they underperform the naive seasonal baseline. Consider investigating hyperparameters, feature engineering, or training procedures.")
                else:
                    st.warning("No deep learning models found for consumption")

            # -------------------- ALL MODELS COMBINED --------------------
            with model_category_tabs[2]:

                st.markdown("#### All Demand Forecasting Models")

                if not all_models.empty:
                    st.markdown("##### Combined Comparison")

                    # Format for display
                    display_df = all_models[['model_name', 'category', 'MAE', 'sMAPE', 'MASE',
                                            'val_MAE', 'val_MASE']].copy()
                    display_df.columns = ['Model', 'Category', 'Test MAE', 'Test sMAPE (%)',
                                         'Test MASE', 'Val MAE', 'Val MASE']

                    # Round numeric columns
                    for col in ['Test MAE', 'Test sMAPE (%)', 'Test MASE', 'Val MAE', 'Val MASE']:
                        display_df[col] = display_df[col].round(3)

                    st.dataframe(
                        display_df.style.background_gradient(
                            subset=['Test MAE', 'Test MASE'],
                            cmap='RdYlGn_r'
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

                    # Comparison visualization
                    comparison_df = all_models[['model_name', 'category', 'MAE', 'sMAPE', 'MASE']].copy()

                    from plotly.subplots import make_subplots
                    import plotly.graph_objects as go

                    fig = make_subplots(
                        rows=1, cols=3,
                        subplot_titles=("MAE Comparison", "sMAPE Comparison", "MASE Comparison")
                    )

                    # Color by category
                    colors = ['#1f77b4' if cat == 'baseline' else '#ff7f0e'
                             for cat in comparison_df['category']]

                    # MAE
                    fig.add_trace(
                        go.Bar(x=comparison_df['model_name'], y=comparison_df['MAE'],
                               marker_color=colors, name='MAE',
                               text=comparison_df['MAE'].round(1),
                               textposition='outside',
                               showlegend=False),
                        row=1, col=1
                    )

                    # sMAPE
                    fig.add_trace(
                        go.Bar(x=comparison_df['model_name'], y=comparison_df['sMAPE'],
                               marker_color=colors, name='sMAPE',
                               text=comparison_df['sMAPE'].round(2),
                               textposition='outside',
                               showlegend=False),
                        row=1, col=2
                    )

                    # MASE with reference line
                    mase_colors = ['#2ca02c' if x < 1.0 else '#d62728' for x in comparison_df['MASE']]
                    fig.add_trace(
                        go.Bar(x=comparison_df['model_name'], y=comparison_df['MASE'],
                               marker_color=mase_colors, name='MASE',
                               text=comparison_df['MASE'].round(3),
                               textposition='outside',
                               showlegend=False),
                        row=1, col=3
                    )

                    fig.update_layout(
                        height=500,
                        template='plotly_white'
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Best models summary
                    col1, col2 = st.columns(2)
                    with col1:
                        best_overall = all_models.nsmallest(1, metric_choice_demand).iloc[0]
                        best_value = best_overall[metric_choice_demand]
                        st.success(f"🏆 **Best Overall ({metric_choice_demand})**: {best_overall['model_name']} ({metric_choice_demand}: {best_value:.2f} {metric_unit})")
                    with col2:
                        # Show config info
                        st.info(f"📋 **Configuration**: {best_overall['config_name']}")
                else:
                    st.warning("No models found for consumption")

        # ============================================================
        # PRICE ESTIMATION MODELS
        # ============================================================
        with main_tabs[1]:
            st.markdown("### Electricity Price (PTF) Estimation")
            target_choice = 'price_real'

            # Get model summary for price
            summary_price = get_model_summary(target=target_choice)

            if summary_price:
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Configurations", summary_price['total_models'])
                with col2:
                    st.metric("Baseline Models", summary_price['baseline_models'])
                with col3:
                    st.metric("Deep Learning Models", summary_price['deep_learning_models'])
                with col4:
                    st.metric("Models MASE < 1.0", summary_price['models_mase_under_1'])

            st.divider()

            # Metric selector for sorting and comparison
            col1, col2 = st.columns([1, 3])
            with col1:
                metric_choice_price = st.selectbox(
                    "📊 Sort & Compare By:",
                    options=['MAE', 'sMAPE', 'MASE', 'val_MAE', 'val_MASE'],
                    index=0,
                    key='price_metric_selector',
                    help="Choose which error metric to use for sorting and comparison"
                )
            with col2:
                metric_descriptions_price = {
                    'MAE': '**MAE** (Mean Absolute Error) - Average prediction error in TL/MWh. Lower is better.',
                    'sMAPE': '**sMAPE** (Symmetric MAPE) - Percentage error measure. Lower is better.',
                    'MASE': '**MASE** (Mean Absolute Scaled Error) - Scaled relative to naive baseline. < 1.0 is good.',
                    'val_MAE': '**Validation MAE** - MAE on validation set (2023 data). Lower is better.',
                    'val_MASE': '**Validation MASE** - MASE on validation set. < 1.0 is good.'
                }
                st.info(metric_descriptions_price[metric_choice_price])

            st.divider()

            # Separate sections for Baseline and Deep Learning
            price_category_tabs = st.tabs(["🎯 Baseline Models", "🧠 Deep Learning Models", "📊 All Models"])

            # Load ALL model configurations for price (90 total)
            all_price_models = load_all_metrics(target=target_choice)

            # Sort by selected metric
            if not all_price_models.empty and metric_choice_price in all_price_models.columns:
                all_price_models = all_price_models.sort_values(by=metric_choice_price)

            # -------------------- BASELINE MODELS (PRICE) --------------------
            with price_category_tabs[0]:
                st.markdown("#### Baseline Model Performance (CatBoost, LightGBM, XGBoost)")

                baseline_price = all_price_models[all_price_models['category'] == 'baseline']

                if not baseline_price.empty:
                    # Overview metrics
                    metric_unit_price = "TL/MWh" if "MAE" in metric_choice_price else ("%" if "sMAPE" in metric_choice_price else "")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        best_baseline_p = baseline_price.nsmallest(1, metric_choice_price).iloc[0]
                        metric_value_p = best_baseline_p[metric_choice_price]
                        st.metric(f"🏆 Best Baseline ({metric_choice_price})",
                                 best_baseline_p['model_name'],
                                 f"{metric_choice_price}: {metric_value_p:.2f} {metric_unit_price}")
                    with col2:
                        avg_value_p = baseline_price[metric_choice_price].mean()
                        st.metric(f"Average {metric_choice_price}", f"{avg_value_p:.2f} {metric_unit_price}")
                    with col3:
                        total_configs_p = len(baseline_price)
                        st.metric("Total Configurations", f"{total_configs_p}")

                    # Map metric choice to display column name for price
                    metric_to_display_col_price = {
                        'MAE': 'Test MAE',
                        'sMAPE': 'Test sMAPE (%)',
                        'MASE': 'Test MASE',
                        'val_MAE': 'Val MAE',
                        'val_MASE': 'Val MASE'
                    }
                    highlight_col_price = metric_to_display_col_price.get(metric_choice_price, 'Test MAE')

                    st.markdown(f"##### Detailed Metrics (Sorted by {metric_choice_price})")
                    baseline_p_display = baseline_price[['model_name', 'config_name', 'MAE', 'sMAPE', 'MASE',
                                                         'val_MAE', 'val_MASE', 'n_features', 'training_time']].copy()
                    baseline_p_display.columns = ['Model', 'Configuration', 'Test MAE', 'Test sMAPE (%)', 'Test MASE',
                                                  'Val MAE', 'Val MASE', 'Features', 'Train Time (s)']

                    for col in ['Test MAE', 'Test sMAPE (%)', 'Test MASE', 'Val MAE', 'Val MASE', 'Train Time (s)']:
                        baseline_p_display[col] = baseline_p_display[col].round(3)

                    st.dataframe(
                        baseline_p_display.style.background_gradient(
                            subset=[highlight_col_price],
                            cmap='RdYlGn_r'
                        ),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.warning("No baseline models found for price")

            # -------------------- DEEP LEARNING MODELS (PRICE) --------------------
            with price_category_tabs[1]:
                st.markdown("#### Deep Learning Model Performance (N-HiTS, PatchTST, TFT)")

                dl_price = all_price_models[all_price_models['category'] == 'deeplearning']  # Fixed: was 'deep_learning'

                if not dl_price.empty:
                    # Overview metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        best_dl_p = dl_price.nsmallest(1, metric_choice_price).iloc[0]
                        metric_value_dl_p = best_dl_p[metric_choice_price]
                        st.metric(f"🏆 Best DL ({metric_choice_price})",
                                 best_dl_p['model_name'],
                                 f"{metric_choice_price}: {metric_value_dl_p:.2f} {metric_unit_price}")
                    with col2:
                        avg_value_dl_p = dl_price[metric_choice_price].mean()
                        st.metric(f"Average {metric_choice_price}", f"{avg_value_dl_p:.2f} {metric_unit_price}")
                    with col3:
                        total_configs_dl_p = len(dl_price)
                        st.metric("Total Configurations", f"{total_configs_dl_p}")

                    st.markdown(f"##### Detailed Metrics (Sorted by {metric_choice_price})")
                    dl_p_display = dl_price[['model_name', 'config_name', 'MAE', 'sMAPE', 'MASE',
                                            'val_MAE', 'val_MASE', 'training_time']].copy()
                    dl_p_display.columns = ['Model', 'Configuration', 'Test MAE', 'Test sMAPE (%)', 'Test MASE',
                                           'Val MAE', 'Val MASE', 'Train Time (s)']

                    for col in ['Test MAE', 'Test sMAPE (%)', 'Test MASE', 'Val MAE', 'Val MASE', 'Train Time (s)']:
                        dl_p_display[col] = dl_p_display[col].round(3)

                    st.dataframe(
                        dl_p_display.style.background_gradient(
                            subset=[highlight_col_price],
                            cmap='RdYlGn_r'
                        ),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.warning("No deep learning models found for price")

            # -------------------- ALL MODELS (PRICE) --------------------
            with price_category_tabs[2]:
                st.markdown(f"#### All Price Estimation Models (Sorted by {metric_choice_price})")

                if not all_price_models.empty:
                    display_p_df = all_price_models[['model_name', 'category', 'MAE', 'sMAPE', 'MASE',
                                                     'val_MAE', 'val_MASE']].copy()
                    display_p_df.columns = ['Model', 'Category', 'Test MAE', 'Test sMAPE (%)',
                                           'Test MASE', 'Val MAE', 'Val MASE']

                    for col in ['Test MAE', 'Test sMAPE (%)', 'Test MASE', 'Val MAE', 'Val MASE']:
                        display_p_df[col] = display_p_df[col].round(3)

                    st.dataframe(
                        display_p_df.style.background_gradient(
                            subset=[highlight_col_price],
                            cmap='RdYlGn_r'
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

                    # Best models summary
                    col1, col2 = st.columns(2)
                    with col1:
                        best_overall_price = all_price_models.nsmallest(1, metric_choice_price).iloc[0]
                        best_value_price = best_overall_price[metric_choice_price]
                        st.success(f"🏆 **Best Overall ({metric_choice_price})**: {best_overall_price['model_name']} ({metric_choice_price}: {best_value_price:.2f} {metric_unit_price})")
                    with col2:
                        # Show config info
                        st.info(f"📋 **Configuration**: {best_overall_price['config_name']}")
                else:
                    st.warning("No models found for price")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.exception(e)
