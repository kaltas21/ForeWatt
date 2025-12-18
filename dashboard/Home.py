"""
ForeWatt Dashboard - Home Page
Interactive model selection: Choose between Consumption or Price forecasting.
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_master_data, get_data_summary,
    PAGE_CONFIG, get_best_models_per_type, get_model_summary,
    HIDE_SIDEBAR_CSS, PROFESSIONAL_CSS
)

# Page configuration
st.set_page_config(
    page_title="ForeWatt",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state for model selection
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None

# Hide sidebar completely on home page
st.markdown(HIDE_SIDEBAR_CSS, unsafe_allow_html=True)

# Apply Windows 2000 Theme
st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)


def show_selection_page():
    """Display the model selection page."""
    # Header
    st.markdown('<div class="main-header">⚡ ForeWatt</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">24-Hour Ahead Electricity Forecasting for Turkey</div>',
        unsafe_allow_html=True
    )

    st.markdown('<hr>', unsafe_allow_html=True)

    # Selection prompt
    st.markdown('<div class="selection-header">Select Forecasting Model:</div>', unsafe_allow_html=True)

    # Two columns for selection
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        # Get consumption stats
        consumption_stats = None
        try:
            consumption_stats = get_model_summary(target='consumption')
        except:
            pass

        st.markdown("""
        <div class="model-card consumption-card">
            <span class="card-icon">⚡</span>
            <div class="card-title">Consumption</div>
            <div class="card-subtitle">Electricity Demand Forecasting</div>
            <div class="card-description">
                Predict hourly electricity consumption across Turkey's grid
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Select Consumption", key="select_consumption", use_container_width=True):
            st.session_state.selected_model = 'consumption'
            st.rerun()

        if consumption_stats:
            st.markdown(f"""
            <div class="stats-box">
                <div class="stats-row">
                    <span class="stats-label">Available Models:</span>
                    <span class="stats-value">{consumption_stats['total_models']}</span>
                </div>
                <div class="stats-row">
                    <span class="stats-label">Best MAE:</span>
                    <span class="stats-value">{consumption_stats['best_mae']:.0f} MWh</span>
                </div>
                <div class="stats-row">
                    <span class="stats-label">Top Performer:</span>
                    <span class="stats-value">{consumption_stats['best_model']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        # Get price stats
        price_stats = None
        try:
            price_stats = get_model_summary(target='price_real')
        except:
            pass

        st.markdown("""
        <div class="model-card price-card">
            <span class="card-icon">💰</span>
            <div class="card-title">Price</div>
            <div class="card-subtitle">Electricity Price Estimation</div>
            <div class="card-description">
                Forecast hourly PTF electricity prices for trading decisions
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Select Price", key="select_price", use_container_width=True):
            st.session_state.selected_model = 'price'
            st.rerun()

        if price_stats:
            st.markdown(f"""
            <div class="stats-box">
                <div class="stats-row">
                    <span class="stats-label">Available Models:</span>
                    <span class="stats-value">{price_stats['total_models']}</span>
                </div>
                <div class="stats-row">
                    <span class="stats-label">Best MAE:</span>
                    <span class="stats-value">{price_stats['best_mae']:.2f} TL/MWh</span>
                </div>
                <div class="stats-row">
                    <span class="stats-label">Top Performer:</span>
                    <span class="stats-value">{price_stats['best_model']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)

    # About section
    st.markdown("""
    <div class="about-section">
        <div class="about-title">ℹ️ About ForeWatt</div>
        <div class="about-text">
            An open-source machine learning platform for electricity demand and price forecasting in Turkey.
        </div>
        <div class="feature-grid">
            <div class="feature-item">
                <div class="feature-number">180+</div>
                <div class="feature-label">TRAINED MODELS</div>
            </div>
            <div class="feature-item">
                <div class="feature-number">5+</div>
                <div class="feature-label">YEARS OF DATA</div>
            </div>
            <div class="feature-item">
                <div class="feature-number">100+</div>
                <div class="feature-label">FEATURES</div>
            </div>
        </div>
        <div class="team-badge">
            Koc University COMP 491 - Fall 2025
        </div>
    </div>
    """, unsafe_allow_html=True)


def show_dashboard_menu():
    """Display the dashboard menu after model selection."""
    model_type = st.session_state.selected_model

    # Model-specific configuration
    if model_type == 'consumption':
        icon = "⚡"
        title = "Consumption Forecasting"
        color = "#008000"
        target = 'consumption'
        unit = "MWh"
    else:
        icon = "💰"
        title = "Price Estimation"
        color = "#ff8000"
        target = 'price_real'
        unit = "TL/MWh"

    # Header with back button
    col1, col2 = st.columns([5, 1])

    with col1:
        st.markdown(f"""
        <div class="page-header">
            <span class="page-header-icon">{icon}</span>
            <span class="page-header-title">{title}</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if st.button("Back", key="back_to_menu", use_container_width=True):
            st.session_state.selected_model = None
            st.rerun()

    # Quick stats
    try:
        summary = get_model_summary(target=target)
        if summary:
            st.markdown('<div class="section-header">MODEL OVERVIEW</div>', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{summary['total_models']}</div>
                    <div class="metric-label">TOTAL MODELS</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{summary['baseline_models']}</div>
                    <div class="metric-label">BASELINE</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{summary['deep_learning_models']}</div>
                    <div class="metric-label">DEEP LEARNING</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{summary['best_mae']:.1f}</div>
                    <div class="metric-label">BEST MAE ({unit})</div>
                </div>
                """, unsafe_allow_html=True)
    except:
        pass

    st.markdown('<hr>', unsafe_allow_html=True)

    # Navigation cards
    st.markdown('<div class="section-header">DASHBOARD MODULES</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="nav-card" style="border-left: 4px solid {color};">
            <h3>📊 Data Explorer</h3>
            <p>Explore historical data, patterns, and feature analysis</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Data Explorer", key="open_data_explorer", use_container_width=True):
            st.switch_page("pages/1_📊_Data_Explorer.py")

    with col2:
        st.markdown(f"""
        <div class="nav-card" style="border-left: 4px solid {color};">
            <h3>🔬 Model Analysis</h3>
            <p>Training curves, performance metrics, model comparison</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Model Analysis", key="open_model_analysis", use_container_width=True):
            st.switch_page("pages/2_🔬_Model_Analysis.py")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown(f"""
        <div class="nav-card" style="border-left: 4px solid {color};">
            <h3>📈 Real-Time Forecasts</h3>
            <p>Live predictions with EPIAS data integration</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Real-Time", key="open_realtime", use_container_width=True):
            st.switch_page("pages/3_📈_Real_Time.py")

    with col4:
        st.markdown(f"""
        <div class="nav-card" style="border-left: 4px solid {color};">
            <h3>🚨 Anomaly Detection</h3>
            <p>Monitor anomalies and forecast deviations</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Anomaly Monitor", key="open_anomaly", use_container_width=True):
            st.switch_page("pages/4_🚨_Anomaly_Monitor.py")

    # Fifth card - Optimization Architecture (Price only)
    if model_type == 'price':
        col5, col6 = st.columns(2)

        with col5:
            st.markdown(f"""
            <div class="nav-card" style="border-left: 4px solid {color};">
                <h3>🏗️ Optimization Architecture</h3>
                <p>Model optimization journey and architecture flowcharts</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Open Architecture", key="open_architecture", use_container_width=True):
                st.switch_page("pages/5_🏗️_Optimization_Architecture.py")

    # Footer
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("""
    <div class="footer">
        ForeWatt v1.0 | Koc University COMP 491 Fall 2025
    </div>
    """, unsafe_allow_html=True)


# Main logic
if st.session_state.selected_model is None:
    show_selection_page()
else:
    show_dashboard_menu()
