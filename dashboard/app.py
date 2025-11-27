"""
ForeWatt Dashboard - Home Page
Energy demand forecasting platform for Turkey's electricity market.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_master_data, get_data_summary,
    TARGET_VARIABLE, PAGE_CONFIG, COLORS, create_time_series_plot,
    # New experiment loaders
    get_best_models_per_type, get_model_summary
)

# Page configuration
st.set_page_config(**PAGE_CONFIG)

# Sidebar - Configuration
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # Target selection
    target_variable = st.selectbox(
        "Target Variable",
        ["consumption", "price_real"],
        index=0,
        help="Select the target variable for forecasting"
    )

    st.divider()

    st.markdown("## 🔄 Data Controls")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Refresh Data", use_container_width=True, help="Reload data from source"):
            st.cache_data.clear()
            st.rerun()

    with col2:
        if st.button("🗑️ Clear Cache", use_container_width=True, help="Clear all cached data"):
            st.cache_data.clear()
            st.success("Cache cleared!")

    # Show last update time
    import datetime
    from pathlib import Path
    master_data_path = Path("../data/gold/master/master_v2_fundamental.csv")
    if master_data_path.exists():
        last_modified = datetime.datetime.fromtimestamp(master_data_path.stat().st_mtime)
        st.caption(f"**Last Updated:**  \n{last_modified.strftime('%Y-%m-%d %H:%M:%S')}")

    st.divider()
    st.markdown("### ℹ️ Data Source")
    st.info("""
    Dashboard reads from:
    - Master data (parquet)
    - New experiment results (CSV)

    **For real-time data:**
    - Connect to InfluxDB
    - Set up auto-refresh
    - Or manually refresh data
    """)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">⚡ ForeWatt</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">1-24 Hour Ahead Electricity Demand Forecasting for Turkey</div>',
    unsafe_allow_html=True
)

# Introduction
st.markdown("""
### Welcome to ForeWatt

A fully reproducible, open-source platform for **electricity demand forecasting** with:
- **1-24 hour ahead predictions** with calibrated uncertainty intervals
- **Multiple state-of-the-art models** (LightGBM, CatBoost, XGBoost, Prophet)
- **5 years of historical data** (2020-2024) with 106 engineered features
- **Comprehensive evaluation** on 2024 test set

**Team**: Koç University COMP 491 Fall 2025 (Zeynep Öykü Aslan, Kaan Altaş, Zeliha Paycı)
""")

st.divider()

# Load data
with st.spinner("Loading data..."):
    try:
        df = load_master_data()
        data_summary = get_data_summary(df)

        # Overview metrics
        st.markdown("### 📊 Data Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Records",
                f"{data_summary['total_rows']:,}",
                help="Total hourly records in dataset"
            )

        with col2:
            st.metric(
                "Features",
                data_summary['num_features'],
                help="Engineered features including lags, rolling stats, calendar, weather, and prices"
            )

        with col3:
            start_date = data_summary['date_range'][0].strftime('%Y-%m-%d')
            end_date = data_summary['date_range'][1].strftime('%Y-%m-%d')
            st.metric(
                "Date Range",
                f"{start_date}",
                f"to {end_date}",
                help="Full date range of available data"
            )

        with col4:
            st.metric(
                "Data Quality",
                f"{(1 - data_summary['missing_values'] / (data_summary['total_rows'] * data_summary['num_features'])) * 100:.1f}%",
                help="Percentage of non-missing values"
            )

        st.divider()

        # Model performance from new experiments
        st.markdown(f"### 🏆 Model Performance (Best Configurations) - {target_variable.replace('_', ' ').title()}")

        # Load best models for selected target
        best_models = get_best_models_per_type(target=target_variable, metric='MAE')

        if not best_models.empty:
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("#### Top 3 Models")

                # Display top 3 models
                top_models = best_models.nsmallest(3, 'MAE')

                medals = ["🥇", "🥈", "🥉"]
                for i, (idx, row) in enumerate(top_models.iterrows()):
                    medal = medals[i] if i < 3 else "🔹"
                    model_name = row['model_name']
                    mae = row['MAE']
                    mase = row['MASE']
                    category = row['category'].replace('_', ' ').title()

                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #000;">{medal} {model_name}</h4>
                        <p style="color: #000;"><strong>MAE:</strong> {mae:.0f} MWh</p>
                        <p style="color: #000;"><strong>MASE:</strong> {mase:.3f}</p>
                        <p style="color: #666; font-size: 0.9rem;">{category}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

            with col2:
                st.markdown("#### All Models Comparison")

                # Display metrics table
                display_df = best_models[['model_name', 'MAE', 'sMAPE', 'MASE', 'category']].copy()
                display_df.columns = ['Model', 'MAE (MWh)', 'sMAPE (%)', 'MASE', 'Category']

                # Format values
                display_df['MAE (MWh)'] = display_df['MAE (MWh)'].apply(lambda x: f"{x:.0f}")
                display_df['sMAPE (%)'] = display_df['sMAPE (%)'].apply(lambda x: f"{x:.2f}")
                display_df['MASE'] = display_df['MASE'].apply(lambda x: f"{x:.3f}")

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    height=200
                )

                st.info("""
                **Metrics Guide:**
                - **MAE** (Mean Absolute Error): Average prediction error in MWh
                - **sMAPE** (Symmetric MAPE): Percentage error measure
                - **MASE** (Mean Absolute Scaled Error): Relative to naive baseline (<1 is good)
                """)
        else:
            st.warning("Model metrics not available. Please ensure models are trained in the new experiment directories.")

        st.divider()

        # Recent consumption data
        st.markdown("### 📈 Recent Consumption Trends")

        # Get last 7 days
        recent_data = df.last('7D')[[TARGET_VARIABLE]]

        if not recent_data.empty:
            fig = create_time_series_plot(
                recent_data,
                [TARGET_VARIABLE],
                title="Last 7 Days - Hourly Electricity Consumption",
                yaxis_title="Consumption (MWh)"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Average (7d)",
                    f"{recent_data[TARGET_VARIABLE].mean():.0f} MWh"
                )

            with col2:
                st.metric(
                    "Peak (7d)",
                    f"{recent_data[TARGET_VARIABLE].max():.0f} MWh"
                )

            with col3:
                st.metric(
                    "Min (7d)",
                    f"{recent_data[TARGET_VARIABLE].min():.0f} MWh"
                )

            with col4:
                st.metric(
                    "Std Dev (7d)",
                    f"{recent_data[TARGET_VARIABLE].std():.0f} MWh"
                )

        st.divider()

        # Navigation guide
        st.markdown("### 🧭 Dashboard Navigation")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### 📈 **Forecast**
            - Generate 1-24h ahead predictions
            - Compare multiple models
            - View prediction intervals
            - Analyze forecast accuracy

            #### 📊 **Data Explorer**
            - Explore historical consumption patterns
            - Analyze feature distributions
            - Discover correlations
            - Identify seasonal patterns
            """)

        with col2:
            st.markdown("""
            #### 🔬 **Model Comparison**
            - Compare all 180 trained models
            - View all 90 baseline configurations
            - View all 90 deep learning configurations
            - Evaluate metrics and feature importance

            #### 🔬 **Model Analysis**
            - Deep learning training visualization
            - Training & validation loss curves
            - Learning rate schedules
            - Train/val/test split analysis
            """)

        st.divider()

        # System status
        st.markdown("### 🔧 System Status")

        # Get model summary
        model_summary = get_model_summary(target=target_variable)

        col1, col2, col3 = st.columns(3)

        with col1:
            data_status = "🟢 Operational" if not df.empty else "🔴 Unavailable"
            st.markdown(f"**Data Pipeline:** {data_status}")

        with col2:
            total_models = model_summary.get('total_models', 0)
            models_status = "🟢 Operational" if total_models > 0 else "🔴 No models"
            st.markdown(f"**Models:** {models_status}")
            st.caption(f"{total_models} configurations trained")

        with col3:
            models_status = "🟢 Ready" if not best_models.empty else "🟡 Limited"
            st.markdown(f"**New Experiments:** {models_status}")
            if model_summary:
                st.caption(f"Baseline: {model_summary.get('baseline_models', 0)} | DL: {model_summary.get('deep_learning_models', 0)}")

        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.9rem;">
            <p>ForeWatt Dashboard v1.0 | Koç University COMP 491 Fall 2025</p>
            <p>For questions or issues, please contact the development team.</p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error loading dashboard: {str(e)}")
        st.exception(e)
