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

# Page configuration - Home page with custom title
st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Default target variable (used throughout the app)
target_variable = "consumption"

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
    '<div class="sub-header">24-Hour Ahead Electricity Demand & Price Forecasting for Turkey</div>',
    unsafe_allow_html=True
)

# Introduction
st.markdown("""
### Welcome to ForeWatt

An open-source platform for **electricity demand and price forecasting** in Turkey.

**Forecasting**
- 24-hour ahead predictions with 90% prediction intervals
- 180 trained models: Baseline (CatBoost, XGBoost, LightGBM) + Deep Learning (N-HiTS, PatchTST, TFT)
- Live forecasts updated hourly via EPİAŞ API

**Data**
- 5+ years of hourly data (2020-2025)
- 100+ engineered features: lags, rolling statistics, calendar, weather, and price signals

**Dashboard**
- **Real-Time Forecast**: Live predictions with confidence intervals
- **Data Explorer**: Interactive visualization of historical trends
- **Model Analysis**: Training curves, performance metrics, and model comparison

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

        col1, col2, col3 = st.columns(3)

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

        st.divider()

        # Model performance - Best models for Demand and Price
        st.markdown("### 🏆 Best Model Performance")

        col1, col2 = st.columns(2)

        # Best model for Demand (consumption)
        with col1:
            st.markdown("#### ⚡ Demand Forecasting")
            best_demand = get_best_models_per_type(target='consumption', metric='MAE')
            if not best_demand.empty:
                best = best_demand.nsmallest(1, 'MAE').iloc[0]
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #000;">🥇 {best['model_name']}</h4>
                    <p style="color: #000;"><strong>Configuration:</strong> {best['config_name']}</p>
                    <p style="color: #000;"><strong>MAE:</strong> {best['MAE']:.0f} MWh</p>
                    <p style="color: #000;"><strong>MASE:</strong> {best['MASE']:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("No demand models available.")

        # Best model for Price
        with col2:
            st.markdown("#### 💰 Price Estimation")
            best_price = get_best_models_per_type(target='price_real', metric='MAE')
            if not best_price.empty:
                best_p = best_price.nsmallest(1, 'MAE').iloc[0]
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: #000;">🥇 {best_p['model_name']}</h4>
                    <p style="color: #000;"><strong>Configuration:</strong> {best_p['config_name']}</p>
                    <p style="color: #000;"><strong>MAE:</strong> {best_p['MAE']:.2f} TL/MWh</p>
                    <p style="color: #000;"><strong>MASE:</strong> {best_p['MASE']:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("No price models available.")

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
