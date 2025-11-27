"""
ForeWatt Dashboard - Forecast Page
Generate and visualize 1-24h ahead electricity demand forecasts.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    load_master_data, split_train_test, get_available_models,
    load_best_model, simulate_predictions, TARGET_VARIABLE,
    PAGE_CONFIG, BASELINE_MODELS, create_forecast_plot,
    calculate_all_metrics, format_metric_value, create_residual_plot,
    calculate_residuals, create_scatter_plot
)

# Page configuration
st.set_page_config(**PAGE_CONFIG)

st.title("📈 Electricity Forecasting")
st.markdown("Generate 1-24 hour ahead forecasts for demand and price using trained models with prediction intervals.")

st.divider()

# Load data
@st.cache_data
def load_forecast_data():
    """Load and prepare data for forecasting."""
    df = load_master_data()
    train_df, test_df = split_train_test(df)
    return df, train_df, test_df

try:
    with st.spinner("Loading data and models..."):
        df, train_df, test_df = load_forecast_data()
        available_models = get_available_models()

    if df.empty:
        st.error("No data available. Please ensure the master dataset is accessible.")
        st.stop()

    # Sidebar controls
    st.sidebar.header("Forecast Configuration")

    # Target selection
    st.sidebar.subheader("0. Forecast Target")
    forecast_target = st.sidebar.selectbox(
        "Select what to forecast:",
        ["⚡ Electricity Demand (Consumption)", "💰 Electricity Price"],
        index=0,
        help="Choose between forecasting electricity consumption or price"
    )

    # Determine target variable and available models based on selection
    if "Demand" in forecast_target:
        target_var = TARGET_VARIABLE  # 'consumption'
        target_type = 'consumption'
        target_label = "Demand (MWh)"
        unit = "MWh"
    else:
        # Check if price columns exist
        price_cols = [col for col in df.columns if 'price' in col.lower()]
        if price_cols:
            target_var = 'price_real' if 'price_real' in price_cols else price_cols[0]
            target_type = 'price_real'
            target_label = "Price (TRY/MWh)"
            unit = "TRY/MWh"
        else:
            st.sidebar.warning("Price data not available. Defaulting to demand forecasting.")
            target_var = TARGET_VARIABLE
            target_type = 'consumption'
            target_label = "Demand (MWh)"
            unit = "MWh"

    # Model selection
    st.sidebar.subheader("1. Select Models")

    if not available_models:
        st.sidebar.warning("No trained models found. Using simulation mode.")
        model_names = list(BASELINE_MODELS.keys())
    else:
        # available_models is now a list from model_loader_v2
        model_names = available_models if isinstance(available_models, list) else list(available_models.keys())

    selected_models = st.sidebar.multiselect(
        "Choose models to compare:",
        model_names,
        default=[model_names[0]] if model_names else [],
        help="Select one or more models to generate forecasts"
    )

    # Forecast horizon
    st.sidebar.subheader("2. Forecast Horizon")
    forecast_hours = st.sidebar.slider(
        "Hours ahead:",
        min_value=1,
        max_value=24,
        value=24,
        help="Number of hours to forecast (1-24)"
    )

    # Date range for visualization
    st.sidebar.subheader("3. Evaluation Period")
    eval_period = st.sidebar.selectbox(
        "Select evaluation period:",
        ["Last 7 days", "Last 30 days", "Full test set (2024)"],
        index=0
    )

    # Show prediction intervals
    show_intervals = st.sidebar.checkbox(
        "Show 90% prediction intervals",
        value=True,
        help="Display confidence bands around predictions"
    )

    # Map evaluation period to data
    if eval_period == "Last 7 days":
        eval_data = test_df.last("7D")
    elif eval_period == "Last 30 days":
        eval_data = test_df.last("30D")
    else:
        eval_data = test_df

    # Generate forecasts button
    generate_forecast = st.sidebar.button(
        "🔮 Generate Forecast",
        type="primary",
        use_container_width=True
    )

    # Main content
    if not selected_models:
        st.info("👈 Please select at least one model from the sidebar to generate forecasts.")
        st.stop()

    if generate_forecast or 'forecasts' in st.session_state:
        with st.spinner("Generating forecasts..."):
            # Generate or retrieve forecasts
            if generate_forecast:
                forecasts = {}
                metrics_results = {}

                for model_name in selected_models:
                    # Simulate predictions (in production, load actual models)
                    result = simulate_predictions(
                        model_name,
                        eval_data,
                        n_samples=min(len(eval_data), 168)  # Max 1 week
                    )

                    if result:
                        forecasts[model_name] = result

                        # Calculate metrics
                        metrics = calculate_all_metrics(
                            result['actual'],
                            result['predictions'],
                            train_df[target_var].values if target_var in train_df.columns else train_df[TARGET_VARIABLE].values
                        )
                        metrics_results[model_name] = metrics

                # Store in session state
                st.session_state.forecasts = forecasts
                st.session_state.metrics_results = metrics_results
                st.session_state.forecast_hours = forecast_hours
            else:
                forecasts = st.session_state.forecasts
                metrics_results = st.session_state.metrics_results

            # Display results
            st.markdown("## Forecast Results")

            # Metrics overview
            st.markdown("### Performance Metrics")

            if metrics_results:
                cols = st.columns(len(selected_models))

                for i, model_name in enumerate(selected_models):
                    if model_name in metrics_results:
                        metrics = metrics_results[model_name]

                        with cols[i]:
                            st.markdown(f"#### {model_name}")
                            st.metric("MAE", f"{metrics['MAE']:.0f} {unit}")
                            st.metric("RMSE", f"{metrics['RMSE']:.0f} {unit}")
                            st.metric("MASE", f"{metrics['MASE']:.3f}")
                            st.metric("R²", f"{metrics['R²']:.3f}")

            st.divider()

            # Forecast visualization
            st.markdown("### Forecast Visualization")

            if forecasts:
                # Prepare data for plotting
                first_forecast = forecasts[list(forecasts.keys())[0]]
                dates = first_forecast['dates']
                actual = first_forecast['actual']

                predictions_dict = {
                    model: forecasts[model]['predictions'][:len(dates)]
                    for model in selected_models
                    if model in forecasts
                }

                intervals_dict = None
                if show_intervals:
                    intervals_dict = {
                        model: (
                            forecasts[model]['lower_bound'][:len(dates)],
                            forecasts[model]['upper_bound'][:len(dates)]
                        )
                        for model in selected_models
                        if model in forecasts
                    }

                # Create forecast plot
                fig = create_forecast_plot(
                    dates,
                    actual[:len(dates)],
                    predictions_dict,
                    intervals=intervals_dict,
                    title=f"{target_label} Forecast - Next {forecast_hours} Hours"
                )

                st.plotly_chart(fig, use_container_width=True)

                # Download forecast data
                if st.checkbox("Show forecast data table"):
                    forecast_df = pd.DataFrame({
                        'datetime': dates,
                        'actual': actual[:len(dates)]
                    })

                    for model in selected_models:
                        if model in forecasts:
                            forecast_df[f'{model}_prediction'] = forecasts[model]['predictions'][:len(dates)]
                            if show_intervals:
                                forecast_df[f'{model}_lower'] = forecasts[model]['lower_bound'][:len(dates)]
                                forecast_df[f'{model}_upper'] = forecasts[model]['upper_bound'][:len(dates)]

                    st.dataframe(forecast_df, use_container_width=True)

                    # Download button
                    csv = forecast_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download forecast data (CSV)",
                        data=csv,
                        file_name=f"forewatt_forecast_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

            st.divider()

            # Model comparison
            if len(selected_models) > 1:
                st.markdown("### Model Comparison")

                tab1, tab2, tab3 = st.tabs(["📊 Metrics", "📉 Residuals", "🎯 Predicted vs Actual"])

                with tab1:
                    st.markdown("#### Performance Metrics Comparison")

                    metrics_comparison = pd.DataFrame(metrics_results).T
                    metrics_comparison = metrics_comparison.sort_values('MAE')

                    # Format for display
                    display_df = metrics_comparison.copy()
                    for col in display_df.columns:
                        if col in ['MAE', 'RMSE']:
                            display_df[col] = display_df[col].apply(lambda x: f"{x:.0f}")
                        else:
                            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")

                    st.dataframe(display_df, use_container_width=True)

                    # Find best model
                    best_model = metrics_comparison['MAE'].idxmin()
                    best_mae = metrics_comparison.loc[best_model, 'MAE']

                    st.success(f"🏆 Best performing model: **{best_model}** (MAE: {best_mae:.0f} {unit})")

                with tab2:
                    st.markdown("#### Residual Analysis")

                    selected_model_residual = st.selectbox(
                        "Select model for residual analysis:",
                        selected_models,
                        key="residual_model"
                    )

                    if selected_model_residual in forecasts:
                        forecast_data = forecasts[selected_model_residual]
                        residuals = calculate_residuals(
                            forecast_data['actual'],
                            forecast_data['predictions']
                        )

                        fig = create_residual_plot(
                            residuals,
                            forecast_data['dates'],
                            selected_model_residual
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Mean Residual", f"{np.mean(residuals):.2f} {unit}")

                        with col2:
                            st.metric("Std Dev", f"{np.std(residuals):.2f} {unit}")

                        with col3:
                            st.metric("Max Abs Error", f"{np.max(np.abs(residuals)):.0f} {unit}")

                with tab3:
                    st.markdown("#### Predicted vs Actual Scatter Plot")

                    selected_model_scatter = st.selectbox(
                        "Select model:",
                        selected_models,
                        key="scatter_model"
                    )

                    if selected_model_scatter in forecasts:
                        forecast_data = forecasts[selected_model_scatter]

                        fig = create_scatter_plot(
                            forecast_data['predictions'],
                            forecast_data['actual'],
                            x_label=f"Predicted ({unit})",
                            y_label=f"Actual ({unit})",
                            title=f"{selected_model_scatter} - Predicted vs Actual ({target_label})"
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Calculate R²
                        r2 = metrics_results[selected_model_scatter]['R²']
                        st.info(f"**R² Score:** {r2:.3f} - Indicates how well predictions match actual values (1.0 = perfect)")

    else:
        # Initial state - show example
        st.info("👈 Configure your forecast settings in the sidebar and click 'Generate Forecast' to begin.")

        st.markdown("### How to Use This Page")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### 📋 Steps:
            1. **Select models** - Choose one or more models to compare
            2. **Set forecast horizon** - Choose 1-24 hours ahead
            3. **Pick evaluation period** - Select test data range
            4. **Generate forecast** - Click the button to run predictions
            """)

        with col2:
            st.markdown("""
            #### 💡 Features:
            - **Multi-model comparison** - Compare multiple models side-by-side
            - **Prediction intervals** - 90% confidence bands
            - **Performance metrics** - MAE, RMSE, MASE, R²
            - **Residual analysis** - Understand forecast errors
            - **Data export** - Download forecasts as CSV
            """)

        st.divider()

        st.markdown("### Available Models")

        if available_models:
            # Handle both list and dict formats
            if isinstance(available_models, list):
                for model_name in available_models:
                    st.markdown(f"- ✅ **{model_name}**")
            else:
                for model_name, meta in available_models.items():
                    with st.expander(f"**{model_name}** - {meta['description']}"):
                        st.markdown(f"""
                        - **Type:** {meta['type'].replace('_', ' ').title()}
                        - **Trained runs:** {meta['num_runs']}
                        - **Status:** {'✅ Available' if meta['available'] else '❌ Unavailable'}
                        """)
        else:
            st.warning("No trained models found. Please ensure models are trained and accessible.")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.exception(e)
