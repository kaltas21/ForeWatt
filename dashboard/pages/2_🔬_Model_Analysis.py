"""
ForeWatt Dashboard - Unified Model Analysis & Training Visualization
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
    PAGE_CONFIG, create_split_visualization,
    create_learning_curve_plot, create_error_analysis_plot,
    create_forecast_plot, create_time_series_plot,
    simulate_predictions, calculate_all_metrics,
    load_all_metrics, get_best_models_per_type, get_model_summary,
    get_model_comparison_df, load_feature_importance, get_available_models as get_available_models_v2,
    PROFESSIONAL_CSS, HIDE_OPTIMIZATION_NAV_CSS, get_model_colors, render_page_header
)

# TensorBoard log loader
from utils.tensorboard_loader import (
    get_available_runs, load_training_metrics, get_run_info,
    load_all_runs_summary, TENSORBOARD_AVAILABLE
)

# Page configuration
st.set_page_config(**PAGE_CONFIG)

# Apply professional styling
st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)

# Check session state for model selection
if 'selected_model' not in st.session_state or st.session_state.selected_model is None:
    st.warning("Please select a model type first.")
    if st.button("Go to Home", use_container_width=True):
        st.switch_page("Home.py")
    st.stop()

# Get model type from session state
MODEL_TYPE = st.session_state.selected_model

# Hide Optimization Architecture from sidebar for consumption mode
if MODEL_TYPE == 'consumption':
    st.markdown(HIDE_OPTIMIZATION_NAV_CSS, unsafe_allow_html=True)

# Configure based on model type
colors = get_model_colors(MODEL_TYPE)
if MODEL_TYPE == 'consumption':
    TARGET = "consumption"
    ICON = "⚡"
    TITLE = "Consumption Model Analysis"
    COLOR = colors['primary']
    UNIT = "MWh"
    TARGET_LABEL = "Electricity Consumption"
else:
    TARGET = "price_real"
    ICON = "💰"
    TITLE = "Price Model Analysis"
    COLOR = colors['primary']
    UNIT = "TL/MWh"
    TARGET_LABEL = "Electricity Price (PTF)"

# Header with back button
col1, col2 = st.columns([6, 1])
with col1:
    render_page_header(ICON, TITLE, "Training curves, performance metrics, and model comparison", COLOR)
with col2:
    if st.button("← Back", key="back_menu_model", use_container_width=True):
        st.switch_page("Home.py")

# Use tabs for navigation
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

    # Check if target column exists
    if TARGET not in df.columns:
        st.error(f"Target column '{TARGET}' not found in dataset.")
        st.stop()

    # Helper function to make timestamps timezone-aware
    def make_tz_aware(date_input) -> pd.Timestamp:
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
        st.markdown(f"### Train/Validation/Test Data Split - {TARGET_LABEL}")
        st.markdown("""
        This visualization shows how the data is split for model training and evaluation.
        - **Train** (2020-2022): 3 years for model training
        - **Validation** (2023): 1 year for hyperparameter tuning
        - **Test** (2024): 1 year for final model evaluation
        """)

        # Feature selector - prioritize target-related features
        if MODEL_TYPE == 'consumption':
            priority_features = [col for col in df.columns if 'consumption' in col.lower()]
        else:
            priority_features = [col for col in df.columns if 'price' in col.lower() or 'ptf' in col.lower()]

        other_features = [col for col in df.columns if col not in priority_features][:15]
        all_features = [TARGET] + [f for f in priority_features if f != TARGET][:10] + other_features

        col1, col2 = st.columns([2, 1])

        with col1:
            feature_to_plot = st.selectbox(
                "Select feature to visualize:",
                all_features,
                help="Choose which feature to display across train/val/test splits"
            )

        with col2:
            show_statistics = st.checkbox("Show Statistics", value=True, key="split_stats")

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
            st.markdown("#### Split Statistics")

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
                st.markdown("##### 🟢 Train Set (2020-2022)")
                st.metric("Records", f"{len(train_df):,}")
                st.metric("Mean", f"{train_df[feature_to_plot].mean():.2f} {UNIT}")
                st.metric("Std Dev", f"{train_df[feature_to_plot].std():.2f}")

            with col2:
                st.markdown("##### 🟠 Validation Set (2023)")
                st.metric("Records", f"{len(val_df):,}")
                st.metric("Mean", f"{val_df[feature_to_plot].mean():.2f} {UNIT}")
                st.metric("Std Dev", f"{val_df[feature_to_plot].std():.2f}")

            with col3:
                st.markdown("##### 🔴 Test Set (2024)")
                st.metric("Records", f"{len(test_df):,}")
                st.metric("Mean", f"{test_df[feature_to_plot].mean():.2f} {UNIT}")
                st.metric("Std Dev", f"{test_df[feature_to_plot].std():.2f}")

            # Distribution comparison
            st.markdown("#### Distribution Comparison")

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

            fig_dist.update_layout(height=400, showlegend=False, template='plotly_white')
            fig_dist.update_xaxes(title_text=f"{feature_to_plot} ({UNIT})")
            fig_dist.update_yaxes(title_text="Frequency")

            st.plotly_chart(fig_dist, use_container_width=True)

    # ===================================================================
    # TRAINING & VALIDATION LOSS
    # ===================================================================
    with tab2:
        st.markdown(f"### Training & Validation Loss Curves - {TARGET_LABEL}")
        st.markdown("""
        Visualize the training progress of deep learning models. These curves show how the model
        learned over time - **training loss** should decrease, and **validation loss** indicates generalization.
        """)

        if not TENSORBOARD_AVAILABLE:
            st.error("TensorBoard not installed. Install with: `pip install tensorboard`")
            st.stop()

        lightning_logs_dir = Path(__file__).parent.parent.parent / "lightning_logs"

        if not lightning_logs_dir.exists():
            st.error(f"Lightning logs directory not found at: {lightning_logs_dir}")
            st.info("Training logs are created automatically when you train deep learning models.")
            st.stop()

        available_runs = get_available_runs(lightning_logs_dir, filter_empty=True)

        # Filter for model type if possible
        model_type_runs = [run for run in available_runs if MODEL_TYPE in run.lower() or TARGET.split('_')[0] in run.lower()]
        if not model_type_runs:
            model_type_runs = available_runs

        if not model_type_runs:
            st.warning("No training runs with loss data found in lightning_logs/")
            st.info("Make sure you have trained deep learning models. The logs are created during training.")
            st.stop()

        col1, col2 = st.columns([2, 1])

        with col1:
            selected_run = st.selectbox(
                "Select Training Run",
                model_type_runs,
                help="Choose a training run to visualize"
            )

        with col2:
            version_dir = lightning_logs_dir / selected_run
            run_info = get_run_info(version_dir)
            if run_info:
                st.markdown("**Model Info:**")
                model_name = run_info.get('model_name', 'Unknown')
                target = run_info.get('target', 'Unknown')
                st.caption(f"Model: {model_name}")
                st.caption(f"Target: {target}")

        metrics = load_training_metrics(version_dir)

        if metrics is None:
            st.error("Could not load training metrics from this run")
            st.stop()

        st.markdown("#### Training & Validation Loss Curves")

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Loss Over Epochs", "Loss Comparison"),
            specs=[[{"type": "scatter"}, {"type": "bar"}]]
        )

        train_loss_data = metrics.get('train_loss', [])
        val_loss_data = metrics.get('val_loss', [])
        num_epochs = metrics.get('epochs', 0)

        if train_loss_data:
            epochs_train = [x[0] for x in train_loss_data]
            losses_train = [x[1] for x in train_loss_data]

            fig.add_trace(
                go.Scatter(
                    x=epochs_train, y=losses_train, mode='lines+markers',
                    name='Training Loss', line=dict(color='#1f77b4', width=2), marker=dict(size=6)
                ),
                row=1, col=1
            )

        if val_loss_data:
            epochs_val = [x[0] for x in val_loss_data]
            losses_val = [x[1] for x in val_loss_data]

            fig.add_trace(
                go.Scatter(
                    x=epochs_val, y=losses_val, mode='lines+markers',
                    name='Validation Loss', line=dict(color='#ff7f0e', width=2), marker=dict(size=6)
                ),
                row=1, col=1
            )

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
                textposition='outside', showlegend=False
            ),
            row=1, col=2
        )

        fig.update_layout(
            height=450, template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Loss Value", row=1, col=2)

        st.plotly_chart(fig, use_container_width=True)

        # Summary metrics
        st.markdown("#### Training Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Epochs", num_epochs)

        with col2:
            if train_loss_data:
                improvement = ((losses_train[0] - losses_train[-1]) / losses_train[0]) * 100
                st.metric("Train Loss Improvement", f"{improvement:.1f}%")

        with col3:
            if val_loss_data:
                st.metric("Best Val Loss", f"{best_val:.4f}")

        with col4:
            if train_loss_data and val_loss_data:
                gap = final_val - final_train
                st.metric("Train-Val Gap", f"{gap:.4f}")

        # Multi-run comparison
        st.markdown("#### Compare Multiple Runs")

        compare_runs = st.multiselect(
            "Select runs to compare",
            model_type_runs,
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
                        x=epochs, y=losses, mode='lines',
                        name=f'{run_name} (train)', line=dict(color=colors[i % len(colors)], width=2)
                    ))

                if run_metrics and 'val_loss' in run_metrics:
                    epochs = [x[0] for x in run_metrics['val_loss']]
                    losses = [x[1] for x in run_metrics['val_loss']]
                    fig_compare.add_trace(go.Scatter(
                        x=epochs, y=losses, mode='lines',
                        name=f'{run_name} (val)', line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                    ))

            fig_compare.update_layout(
                title="Training Loss Comparison Across Runs",
                xaxis_title="Epoch", yaxis_title="Loss",
                height=400, template='plotly_white',
                legend=dict(orientation="h", yanchor="bottom", y=-0.3)
            )
            st.plotly_chart(fig_compare, use_container_width=True)

    # ===================================================================
    # MODEL PERFORMANCE COMPARISON
    # ===================================================================
    with tab3:
        st.markdown(f"### {TARGET_LABEL} Model Performance Comparison")
        st.markdown("Compare all trained models organized by model category.")

        # Get model summary
        summary = get_model_summary(target=TARGET)

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

        # Metric selector
        col1, col2 = st.columns([1, 3])
        with col1:
            metric_choice = st.selectbox(
                "Sort & Compare By:",
                options=['MAE', 'sMAPE', 'MASE', 'val_MAE', 'val_MASE'],
                index=0,
                help="Choose which error metric to use for sorting and comparison"
            )
        with col2:
            metric_descriptions = {
                'MAE': f'**MAE** (Mean Absolute Error) - Average prediction error in {UNIT}. Lower is better.',
                'sMAPE': '**sMAPE** (Symmetric MAPE) - Percentage error measure. Lower is better.',
                'MASE': '**MASE** (Mean Absolute Scaled Error) - Scaled relative to naive baseline. < 1.0 is good.',
                'val_MAE': f'**Validation MAE** - MAE on validation set (2023 data). Lower is better.',
                'val_MASE': '**Validation MASE** - MASE on validation set. < 1.0 is good.'
            }
            st.info(metric_descriptions[metric_choice])

        st.divider()

        # Model category tabs
        model_category_tabs = st.tabs(["🎯 Baseline Models", "🧠 Deep Learning Models", "📊 All Models"])

        # Load all models
        all_models = load_all_metrics(target=TARGET)

        if not all_models.empty and metric_choice in all_models.columns:
            all_models = all_models.sort_values(by=metric_choice)

        metric_to_display_col = {
            'MAE': 'Test MAE', 'sMAPE': 'Test sMAPE (%)', 'MASE': 'Test MASE',
            'val_MAE': 'Val MAE', 'val_MASE': 'Val MASE'
        }
        highlight_col = metric_to_display_col.get(metric_choice, 'Test MAE')
        metric_unit = UNIT if "MAE" in metric_choice else ("%" if "sMAPE" in metric_choice else "")

        # Baseline Models
        with model_category_tabs[0]:
            st.markdown("#### Baseline Model Performance (CatBoost, LightGBM, XGBoost)")

            baseline_models = all_models[all_models['category'] == 'baseline'] if not all_models.empty else pd.DataFrame()

            if not baseline_models.empty:
                col1, col2, col3 = st.columns(3)
                with col1:
                    best_baseline = baseline_models.nsmallest(1, metric_choice).iloc[0]
                    st.metric(f"Best Baseline ({metric_choice})",
                             best_baseline['model_name'],
                             f"{metric_choice}: {best_baseline[metric_choice]:.2f} {metric_unit}")
                with col2:
                    st.metric(f"Average {metric_choice}", f"{baseline_models[metric_choice].mean():.2f} {metric_unit}")
                with col3:
                    st.metric("Total Configurations", f"{len(baseline_models)}")

                st.markdown(f"##### Detailed Metrics (Sorted by {metric_choice})")

                baseline_models = baseline_models.copy()
                baseline_models['Overfit %'] = ((baseline_models['MAE'] - baseline_models['val_MAE']) / baseline_models['val_MAE'] * 100)

                baseline_display = baseline_models[['model_name', 'config_name', 'val_MAE', 'MAE', 'Overfit %',
                                                    'sMAPE', 'MASE', 'val_MASE', 'n_features', 'training_time']].copy()
                baseline_display.columns = ['Model', 'Configuration', 'Val MAE', 'Test MAE', 'Overfit %',
                                            'Test sMAPE (%)', 'Test MASE', 'Val MASE', 'Features', 'Train Time (s)']

                for col in ['Val MAE', 'Test MAE', 'Test sMAPE (%)', 'Test MASE', 'Val MASE', 'Train Time (s)']:
                    baseline_display[col] = baseline_display[col].round(3)
                baseline_display['Overfit %'] = baseline_display['Overfit %'].round(1)

                st.dataframe(
                    baseline_display.style.background_gradient(subset=[highlight_col], cmap='RdYlGn_r'),
                    use_container_width=True, hide_index=True
                )
            else:
                st.warning(f"No baseline models found for {TARGET_LABEL}. Train some models first!")

        # Deep Learning Models
        with model_category_tabs[1]:
            st.markdown("#### Deep Learning Model Performance (N-HiTS, PatchTST, TFT)")

            dl_models = all_models[all_models['category'] == 'deeplearning'] if not all_models.empty else pd.DataFrame()

            if not dl_models.empty:
                col1, col2, col3 = st.columns(3)
                with col1:
                    best_dl = dl_models.nsmallest(1, metric_choice).iloc[0]
                    st.metric(f"Best DL ({metric_choice})",
                             best_dl['model_name'],
                             f"{metric_choice}: {best_dl[metric_choice]:.2f} {metric_unit}")
                with col2:
                    st.metric(f"Average {metric_choice}", f"{dl_models[metric_choice].mean():.2f} {metric_unit}")
                with col3:
                    st.metric("Total Configurations", f"{len(dl_models)}")

                st.markdown(f"##### Detailed Metrics (Sorted by {metric_choice})")

                dl_models = dl_models.copy()
                dl_models['Overfit %'] = ((dl_models['MAE'] - dl_models['val_MAE']) / dl_models['val_MAE'] * 100)

                dl_display = dl_models[['model_name', 'config_name', 'val_MAE', 'MAE', 'Overfit %',
                                       'sMAPE', 'MASE', 'val_MASE', 'training_time']].copy()
                dl_display.columns = ['Model', 'Configuration', 'Val MAE', 'Test MAE', 'Overfit %',
                                     'Test sMAPE (%)', 'Test MASE', 'Val MASE', 'Train Time (s)']

                for col in ['Val MAE', 'Test MAE', 'Test sMAPE (%)', 'Test MASE', 'Val MASE', 'Train Time (s)']:
                    dl_display[col] = dl_display[col].round(3)
                dl_display['Overfit %'] = dl_display['Overfit %'].round(1)

                st.dataframe(
                    dl_display.style.background_gradient(subset=[highlight_col], cmap='RdYlGn_r'),
                    use_container_width=True, hide_index=True
                )

                if dl_models['MASE'].min() > 1.0:
                    st.warning("**Note**: All deep learning models have MASE > 1.0, meaning they underperform the naive seasonal baseline.")
            else:
                st.warning(f"No deep learning models found for {TARGET_LABEL}. Train some models first!")

        # All Models
        with model_category_tabs[2]:
            st.markdown(f"#### All {TARGET_LABEL} Models (Sorted by {metric_choice})")

            if not all_models.empty:
                display_df = all_models[['model_name', 'category', 'MAE', 'sMAPE', 'MASE',
                                        'val_MAE', 'val_MASE']].copy()
                display_df.columns = ['Model', 'Category', 'Test MAE', 'Test sMAPE (%)',
                                     'Test MASE', 'Val MAE', 'Val MASE']

                for col in ['Test MAE', 'Test sMAPE (%)', 'Test MASE', 'Val MAE', 'Val MASE']:
                    display_df[col] = display_df[col].round(3)

                st.dataframe(
                    display_df.style.background_gradient(subset=[highlight_col], cmap='RdYlGn_r'),
                    use_container_width=True, hide_index=True
                )

                # Best model summary
                col1, col2 = st.columns(2)
                with col1:
                    best_overall = all_models.nsmallest(1, metric_choice).iloc[0]
                    st.success(f"**Best Overall ({metric_choice})**: {best_overall['model_name']} ({metric_choice}: {best_overall[metric_choice]:.2f} {metric_unit})")
                with col2:
                    st.info(f"**Configuration**: {best_overall['config_name']}")
            else:
                st.warning(f"No models found for {TARGET_LABEL}. Train some models first!")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.exception(e)
