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

st.markdown("## 🔬 Model Analysis")

# Use tabs instead of sidebar menu for cleaner navigation
tab1, tab2, tab3 = st.tabs([
    "📊 Train/Val/Test Split",
    "📉 Training Loss",
    "🔍 Performance"
])


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
    with tab1:
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
    # TRAINING & VALIDATION LOSS (from lightning_logs)
    # ===================================================================
    with tab2:
        st.markdown("## Training & Validation Loss Curves")
        st.markdown("""
        Visualize the training progress of deep learning models. These curves show how the model
        learned over time - **training loss** should decrease, and **validation loss** indicates generalization.
        """)

        # Check if tensorboard is available
        if not TENSORBOARD_AVAILABLE:
            st.error("❌ TensorBoard not installed. Install with: `pip install tensorboard`")
            st.stop()

        # Get lightning_logs directory
        lightning_logs_dir = Path(__file__).parent.parent.parent / "lightning_logs"

        if not lightning_logs_dir.exists():
            st.error(f"❌ Lightning logs directory not found at: {lightning_logs_dir}")
            st.info("💡 Training logs are created automatically when you train deep learning models.")
            st.stop()

        # Get available runs
        available_runs = get_available_runs(lightning_logs_dir, filter_empty=True)

        if not available_runs:
            st.warning("⚠️ No training runs with loss data found in lightning_logs/")
            st.info("💡 Make sure you have trained deep learning models. The logs are created during training.")
            st.stop()

        # Run selector
        col1, col2 = st.columns([2, 1])

        with col1:
            selected_run = st.selectbox(
                "Select Training Run",
                available_runs,
                help="Choose a training run to visualize"
            )

        with col2:
            # Load run info (hyperparameters)
            version_dir = lightning_logs_dir / selected_run
            run_info = get_run_info(version_dir)
            if run_info:
                st.markdown("**Model Info:**")
                model_name = run_info.get('model_name', 'Unknown')
                target = run_info.get('target', 'Unknown')
                st.caption(f"Model: {model_name}")
                st.caption(f"Target: {target}")

        # Load training metrics
        metrics = load_training_metrics(version_dir)

        if metrics is None:
            st.error("❌ Could not load training metrics from this run")
            st.stop()

        # Display training curves
        st.markdown("### 📈 Training & Validation Loss Curves")

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Loss Over Epochs", "Loss Comparison"),
            specs=[[{"type": "scatter"}, {"type": "bar"}]]
        )

        # Extract data
        train_loss_data = metrics.get('train_loss', [])
        val_loss_data = metrics.get('val_loss', [])
        num_epochs = metrics.get('epochs', 0)

        # Plot training loss
        if train_loss_data:
            epochs_train = [x[0] for x in train_loss_data]
            losses_train = [x[1] for x in train_loss_data]

            fig.add_trace(
                go.Scatter(
                    x=epochs_train,
                    y=losses_train,
                    mode='lines+markers',
                    name='Training Loss',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=6),
                    hovertemplate='Epoch %{x}<br>Train Loss: %{y:.4f}<extra></extra>'
                ),
                row=1, col=1
            )

        # Plot validation loss
        if val_loss_data:
            epochs_val = [x[0] for x in val_loss_data]
            losses_val = [x[1] for x in val_loss_data]

            fig.add_trace(
                go.Scatter(
                    x=epochs_val,
                    y=losses_val,
                    mode='lines+markers',
                    name='Validation Loss',
                    line=dict(color='#ff7f0e', width=2),
                    marker=dict(size=6),
                    hovertemplate='Epoch %{x}<br>Val Loss: %{y:.4f}<extra></extra>'
                ),
                row=1, col=1
            )

        # Bar chart comparison (final values)
        final_train = losses_train[-1] if train_loss_data else 0
        final_val = losses_val[-1] if val_loss_data else 0
        best_train = min(losses_train) if train_loss_data else 0
        best_val = min(losses_val) if val_loss_data else 0

        fig.add_trace(
            go.Bar(
                x=['Final Train', 'Final Val', 'Best Train', 'Best Val'],
                y=[final_train, final_val, best_train, best_val],
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                text=[f'{v:.4f}' for v in [final_train, final_val, best_train, best_val]],
                textposition='outside',
                showlegend=False,
                hovertemplate='%{x}: %{y:.4f}<extra></extra>'
            ),
            row=1, col=2
        )

        fig.update_layout(
            height=450,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Loss Value", row=1, col=2)

        st.plotly_chart(fig, use_container_width=True)

        # Summary metrics
        st.markdown("### 📊 Training Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Epochs", num_epochs)

        with col2:
            if train_loss_data:
                improvement = ((losses_train[0] - losses_train[-1]) / losses_train[0]) * 100
                st.metric("Train Loss Improvement", f"{improvement:.1f}%",
                         delta=f"{losses_train[0]:.4f} → {losses_train[-1]:.4f}")

        with col3:
            if val_loss_data:
                st.metric("Best Val Loss", f"{best_val:.4f}",
                         delta=f"Epoch {epochs_val[losses_val.index(best_val)]}")

        with col4:
            if train_loss_data and val_loss_data:
                gap = final_val - final_train
                gap_pct = (gap / final_train) * 100 if final_train > 0 else 0
                st.metric("Train-Val Gap", f"{gap:.4f}", delta=f"{gap_pct:+.1f}%")

        # Overfitting analysis
        if train_loss_data and val_loss_data and len(losses_train) > 1:
            st.markdown("### 🔍 Overfitting Analysis")

            col1, col2 = st.columns(2)

            with col1:
                # Check if validation loss increased while training decreased
                train_decreasing = losses_train[-1] < losses_train[0]
                val_increasing = losses_val[-1] > min(losses_val) if len(losses_val) > 1 else False

                if train_decreasing and val_increasing:
                    st.warning("⚠️ **Potential Overfitting Detected**")
                    st.caption("Training loss decreased but validation loss increased from its best value.")
                elif final_val > final_train * 2:
                    st.warning("⚠️ **Large Train-Val Gap**")
                    st.caption("Validation loss is much higher than training loss, suggesting overfitting.")
                else:
                    st.success("✅ **Good Generalization**")
                    st.caption("Training and validation losses are reasonably close.")

            with col2:
                # Learning status
                if train_loss_data and len(losses_train) > 2:
                    recent_change = (losses_train[-1] - losses_train[-2]) / losses_train[-2] * 100
                    if abs(recent_change) < 1:
                        st.info("📊 **Converged**")
                        st.caption("Loss has stabilized (< 1% change in last epoch)")
                    elif recent_change < 0:
                        st.success("📈 **Still Learning**")
                        st.caption(f"Loss decreased by {abs(recent_change):.1f}% in last epoch")
                    else:
                        st.warning("📉 **Loss Increasing**")
                        st.caption(f"Loss increased by {recent_change:.1f}% in last epoch")

        # Multi-run comparison
        st.markdown("### 📊 Compare Multiple Runs")

        compare_runs = st.multiselect(
            "Select runs to compare",
            available_runs,
            default=[selected_run] if selected_run else [],
            max_selections=5,
            help="Select up to 5 runs to compare their training curves"
        )

        if len(compare_runs) > 1:
            fig_compare = go.Figure()

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

            for i, run_name in enumerate(compare_runs):
                run_dir = lightning_logs_dir / run_name
                run_metrics = load_training_metrics(run_dir)

                if run_metrics and 'train_loss' in run_metrics:
                    epochs = [x[0] for x in run_metrics['train_loss']]
                    losses = [x[1] for x in run_metrics['train_loss']]

                    fig_compare.add_trace(go.Scatter(
                        x=epochs,
                        y=losses,
                        mode='lines',
                        name=f'{run_name} (train)',
                        line=dict(color=colors[i % len(colors)], width=2),
                    ))

                if run_metrics and 'val_loss' in run_metrics:
                    epochs = [x[0] for x in run_metrics['val_loss']]
                    losses = [x[1] for x in run_metrics['val_loss']]

                    fig_compare.add_trace(go.Scatter(
                        x=epochs,
                        y=losses,
                        mode='lines',
                        name=f'{run_name} (val)',
                        line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                    ))

            fig_compare.update_layout(
                title="Training Loss Comparison Across Runs",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                height=400,
                template='plotly_white',
                legend=dict(orientation="h", yanchor="bottom", y=-0.3)
            )

            st.plotly_chart(fig_compare, use_container_width=True)

    # ===================================================================
    # MODEL PERFORMANCE COMPARISON
    # ===================================================================
    with tab3:
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
                    # Calculate overfitting metric (Val→Test gap)
                    baseline_models['Overfit %'] = ((baseline_models['MAE'] - baseline_models['val_MAE']) / baseline_models['val_MAE'] * 100)

                    baseline_display = baseline_models[['model_name', 'config_name', 'val_MAE', 'MAE', 'Overfit %',
                                                        'sMAPE', 'MASE', 'val_MASE', 'n_features', 'training_time']].copy()
                    baseline_display.columns = ['Model', 'Configuration', 'Val MAE', 'Test MAE', 'Overfit %',
                                                'Test sMAPE (%)', 'Test MASE', 'Val MASE', 'Features', 'Train Time (s)']

                    # Round values
                    for col in ['Val MAE', 'Test MAE', 'Test sMAPE (%)', 'Test MASE', 'Val MASE', 'Train Time (s)']:
                        baseline_display[col] = baseline_display[col].round(3)
                    baseline_display['Overfit %'] = baseline_display['Overfit %'].round(1)

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

                    # Calculate overfitting metric (Val→Test gap)
                    dl_models['Overfit %'] = ((dl_models['MAE'] - dl_models['val_MAE']) / dl_models['val_MAE'] * 100)

                    dl_display = dl_models[['model_name', 'config_name', 'val_MAE', 'MAE', 'Overfit %',
                                           'sMAPE', 'MASE', 'val_MASE', 'training_time']].copy()
                    dl_display.columns = ['Model', 'Configuration', 'Val MAE', 'Test MAE', 'Overfit %',
                                         'Test sMAPE (%)', 'Test MASE', 'Val MASE', 'Train Time (s)']

                    # Round values
                    for col in ['Val MAE', 'Test MAE', 'Test sMAPE (%)', 'Test MASE', 'Val MASE', 'Train Time (s)']:
                        dl_display[col] = dl_display[col].round(3)
                    dl_display['Overfit %'] = dl_display['Overfit %'].round(1)

                    st.dataframe(
                        dl_display.style.background_gradient(
                            subset=[highlight_col],
                            cmap='RdYlGn_r'
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

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
