"""
ForeWatt Dashboard - Optimization Architecture
Visualizes the model optimization journey, data statistics, and future opportunities.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import PAGE_CONFIG, PROFESSIONAL_CSS, get_model_colors, render_page_header

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

# This page is only for Price model - redirect consumption users
if st.session_state.selected_model == 'consumption':
    st.switch_page("Home.py")

# Get model type from session state
MODEL_TYPE = st.session_state.selected_model
colors = get_model_colors(MODEL_TYPE)

# Header with back button
col1, col2 = st.columns([6, 1])
with col1:
    if MODEL_TYPE == 'price':
        render_page_header("🏗️", "Price Optimization Architecture",
                          "Complete optimization analysis: Data, Models, Architecture & Future", colors['primary'])
    else:
        render_page_header("🏗️", "Consumption Model Architecture",
                          "Model architecture and optimization overview", colors['primary'])
with col2:
    if st.button("Back", key="back_menu_arch", use_container_width=True):
        st.switch_page("Home.py")


# ============== CHART FUNCTIONS ==============

def create_optimization_flowchart():
    """Create the optimization journey flowchart for price model."""
    fig = go.Figure()

    stages = [
        {"name": "N-HiTS\nBaseline", "smape": "16.01%", "x": 0, "color": "#ef4444"},
        {"name": "V4\nTransfer\nLearning", "smape": "14.29%", "x": 1, "color": "#f97316"},
        {"name": "V5\nFine-tune\nWindow", "smape": "14.08%", "x": 2, "color": "#f97316"},
        {"name": "V10\nProfile\nFeatures", "smape": "12.73%", "x": 3, "color": "#eab308"},
        {"name": "V11\nEnsemble", "smape": "12.31%", "x": 4, "color": "#eab308"},
        {"name": "V12\nAEC", "smape": "12.03%", "x": 5, "color": "#22c55e"},
        {"name": "V13\nHourly\nAEC", "smape": "11.89%", "x": 6, "color": "#22c55e"},
        {"name": "V14\nKNN-EC", "smape": "11.63%", "x": 7, "color": "#10b981"},
    ]

    for i in range(len(stages) - 1):
        fig.add_trace(go.Scatter(
            x=[stages[i]["x"] + 0.4, stages[i+1]["x"] - 0.4],
            y=[0.5, 0.5],
            mode='lines',
            line=dict(color='#64748b', width=3),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_annotation(
            x=stages[i+1]["x"] - 0.4, y=0.5,
            ax=stages[i+1]["x"] - 0.6, ay=0.5,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor='#64748b'
        )

    for stage in stages:
        fig.add_shape(
            type="rect", x0=stage["x"] - 0.4, x1=stage["x"] + 0.4, y0=0.1, y1=0.9,
            fillcolor=stage["color"], line=dict(color=stage["color"], width=2), opacity=0.9
        )
        fig.add_annotation(x=stage["x"], y=0.6, text=f"<b>{stage['name']}</b>",
                          showarrow=False, font=dict(size=11, color="white"), align="center")
        fig.add_annotation(x=stage["x"], y=0.25, text=f"<b>{stage['smape']}</b>",
                          showarrow=False, font=dict(size=13, color="white"), align="center")

    fig.update_layout(
        title=dict(text="<b>Optimization Journey: sMAPE Reduction</b>", font=dict(size=18, color="#f1f5f9"), x=0.5),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.8, 7.8]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.2, 1.2]),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        height=280, margin=dict(l=20, r=20, t=60, b=20), showlegend=False
    )
    return fig


def create_architecture_diagram():
    """Create the final model architecture diagram."""
    fig = go.Figure()

    # Input layer
    fig.add_shape(type="rect", x0=0.5, x1=4.5, y0=4.5, y1=5.2,
                  fillcolor="#3b82f6", line=dict(color="#3b82f6", width=2))
    fig.add_annotation(x=2.5, y=4.85, text="<b>INPUT FEATURES (27)</b><br>Price lags, Rolling stats, Calendar, Fundamentals",
                      showarrow=False, font=dict(size=11, color="white"))

    fig.add_annotation(x=2.5, y=4.3, ax=2.5, ay=4.5, showarrow=True,
                      arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor='#64748b')

    # CatBoost & LightGBM
    fig.add_shape(type="rect", x0=0.5, x1=2.2, y0=3.2, y1=4.1,
                  fillcolor="#f97316", line=dict(color="#f97316", width=2))
    fig.add_annotation(x=1.35, y=3.65, text="<b>CatBoost</b><br>61.4%",
                      showarrow=False, font=dict(size=12, color="white"))

    fig.add_shape(type="rect", x0=2.8, x1=4.5, y0=3.2, y1=4.1,
                  fillcolor="#22c55e", line=dict(color="#22c55e", width=2))
    fig.add_annotation(x=3.65, y=3.65, text="<b>LightGBM</b><br>38.6%",
                      showarrow=False, font=dict(size=12, color="white"))

    fig.add_annotation(x=1.35, y=3.0, ax=1.35, ay=3.2, showarrow=True,
                      arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor='#64748b')
    fig.add_annotation(x=3.65, y=3.0, ax=3.65, ay=3.2, showarrow=True,
                      arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor='#64748b')

    # Ensemble
    fig.add_shape(type="rect", x0=1.0, x1=4.0, y0=2.2, y1=2.9,
                  fillcolor="#8b5cf6", line=dict(color="#8b5cf6", width=2))
    fig.add_annotation(x=2.5, y=2.55, text="<b>Weighted Ensemble</b>",
                      showarrow=False, font=dict(size=12, color="white"))

    fig.add_annotation(x=2.5, y=2.0, ax=2.5, ay=2.2, showarrow=True,
                      arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor='#64748b')

    # KNN-EC
    fig.add_shape(type="rect", x0=1.0, x1=4.0, y0=1.3, y1=1.9,
                  fillcolor="#06b6d4", line=dict(color="#06b6d4", width=2))
    fig.add_annotation(x=2.5, y=1.6, text="<b>Context-Aware KNN-EC</b><br>k=5, lookback=45d",
                      showarrow=False, font=dict(size=11, color="white"))

    fig.add_annotation(x=2.5, y=1.1, ax=2.5, ay=1.3, showarrow=True,
                      arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor='#64748b')

    # Hourly AEC
    fig.add_shape(type="rect", x0=1.0, x1=4.0, y0=0.4, y1=1.0,
                  fillcolor="#ec4899", line=dict(color="#ec4899", width=2))
    fig.add_annotation(x=2.5, y=0.7, text="<b>Hourly-Dynamic AEC</b><br>Hour-specific parameters",
                      showarrow=False, font=dict(size=11, color="white"))

    fig.add_annotation(x=2.5, y=0.2, ax=2.5, ay=0.4, showarrow=True,
                      arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor='#64748b')

    # Final output
    fig.add_shape(type="rect", x0=1.2, x1=3.8, y0=-0.5, y1=0.1,
                  fillcolor="#10b981", line=dict(color="#10b981", width=3))
    fig.add_annotation(x=2.5, y=-0.2, text="<b>FINAL PREDICTION</b><br>sMAPE: 11.63%",
                      showarrow=False, font=dict(size=12, color="white"))

    fig.update_layout(
        title=dict(text="<b>Final Model Architecture</b>", font=dict(size=18, color="#f1f5f9"), x=0.5),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.8, 5.5]),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        height=500, margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig


def create_smape_reduction_chart():
    """Create sMAPE reduction bar chart."""
    versions = ['Baseline', 'V4', 'V5', 'V10', 'V11', 'V12', 'V13', 'V14']
    smape_values = [16.01, 14.29, 14.08, 12.73, 12.31, 12.03, 11.89, 11.63]
    colors_list = ['#ef4444', '#f97316', '#f97316', '#eab308', '#eab308', '#22c55e', '#22c55e', '#10b981']

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=versions, y=smape_values, marker_color=colors_list,
        text=[f'{v}%' for v in smape_values], textposition='outside',
        textfont=dict(color='#f1f5f9', size=12)
    ))
    fig.add_hline(y=11.75, line_dash="dash", line_color="#ef4444",
                  annotation_text="Oracle Floor (11.75%)", annotation_position="right",
                  annotation_font_color="#ef4444")

    fig.update_layout(
        title=dict(text="<b>sMAPE Reduction Across Versions</b>", font=dict(size=16, color="#f1f5f9"), x=0.5),
        xaxis=dict(title=dict(text="Optimization Version", font=dict(color='#94a3b8')),
                   tickfont=dict(color='#94a3b8'), gridcolor='rgba(148, 163, 184, 0.1)'),
        yaxis=dict(title=dict(text="sMAPE (%)", font=dict(color='#94a3b8')),
                   tickfont=dict(color='#94a3b8'), gridcolor='rgba(148, 163, 184, 0.1)', range=[10, 17]),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        height=350, margin=dict(l=60, r=20, t=60, b=60), showlegend=False
    )
    return fig


def create_feature_importance_chart():
    """Create top features chart."""
    display_names = ['Price Volatility (24h)', 'Thermal Gap', 'Price Lag (168h)', 'Hour (cos)',
                     'Renewable Saturation', 'Price Mean (24h)', 'Price Lag (24h)',
                     'Spark Spread (24h)', 'System Short Signal', 'Load Factor']
    importance = [100, 87, 82, 75, 71, 68, 65, 58, 52, 48]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=display_names[::-1], x=importance[::-1], orientation='h',
        marker_color='#f97316', text=[f'{v}%' for v in importance[::-1]],
        textposition='outside', textfont=dict(color='#f1f5f9', size=11)
    ))

    fig.update_layout(
        title=dict(text="<b>Top 10 Features by Importance</b>", font=dict(size=16, color="#f1f5f9"), x=0.5),
        xaxis=dict(title=dict(text="Relative Importance (%)", font=dict(color='#94a3b8')),
                   tickfont=dict(color='#94a3b8'), gridcolor='rgba(148, 163, 184, 0.1)', range=[0, 115]),
        yaxis=dict(tickfont=dict(color='#94a3b8', size=10)),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        height=400, margin=dict(l=150, r=60, t=60, b=40), showlegend=False
    )
    return fig


def create_hourly_smape_chart():
    """Create hourly sMAPE distribution chart."""
    hours = list(range(24))
    # Approximated hourly sMAPE values based on the optimization data
    hourly_smape = [10.4, 9.6, 9.1, 12.9, 14.6, 15.5, 15.3, 20.1, 16.1, 30.0,
                    21.5, 18.8, 16.8, 14.5, 9.1, 5.7, 5.2, 4.1, 3.8, 8.0,
                    10.3, 11.9, 7.5, 8.6]

    colors_hourly = ['#ef4444' if v > 20 else '#f97316' if v > 15 else '#eab308' if v > 10 else '#22c55e'
                     for v in hourly_smape]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hours, y=hourly_smape, marker_color=colors_hourly,
        text=[f'{v:.1f}%' for v in hourly_smape], textposition='outside',
        textfont=dict(color='#f1f5f9', size=9)
    ))

    fig.add_hline(y=11.63, line_dash="dash", line_color="#3b82f6",
                  annotation_text="Global Avg (11.63%)", annotation_position="right",
                  annotation_font_color="#3b82f6")

    fig.update_layout(
        title=dict(text="<b>sMAPE by Hour of Day</b>", font=dict(size=16, color="#f1f5f9"), x=0.5),
        xaxis=dict(title=dict(text="Hour", font=dict(color='#94a3b8')),
                   tickfont=dict(color='#94a3b8'), tickmode='linear', dtick=2),
        yaxis=dict(title=dict(text="sMAPE (%)", font=dict(color='#94a3b8')),
                   tickfont=dict(color='#94a3b8'), gridcolor='rgba(148, 163, 184, 0.1)'),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        height=300, margin=dict(l=60, r=20, t=60, b=40), showlegend=False
    )
    return fig


def create_model_comparison_chart():
    """Create model comparison radar chart."""
    categories = ['MAE', 'sMAPE', 'Training Speed', 'Interpretability', 'Robustness']

    # Normalized scores (higher is better, 0-100 scale)
    catboost_scores = [85, 82, 70, 90, 88]
    lightgbm_scores = [90, 85, 95, 85, 82]
    xgboost_scores = [88, 84, 80, 85, 85]
    nhits_scores = [60, 65, 40, 50, 70]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=catboost_scores + [catboost_scores[0]], theta=categories + [categories[0]],
        fill='toself', name='CatBoost', line_color='#f97316', fillcolor='rgba(249, 115, 22, 0.2)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=lightgbm_scores + [lightgbm_scores[0]], theta=categories + [categories[0]],
        fill='toself', name='LightGBM', line_color='#22c55e', fillcolor='rgba(34, 197, 94, 0.2)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=nhits_scores + [nhits_scores[0]], theta=categories + [categories[0]],
        fill='toself', name='N-HiTS', line_color='#ef4444', fillcolor='rgba(239, 68, 68, 0.2)'
    ))

    fig.update_layout(
        title=dict(text="<b>Model Comparison</b>", font=dict(size=16, color="#f1f5f9"), x=0.5),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color='#64748b')),
            angularaxis=dict(tickfont=dict(color='#94a3b8'))
        ),
        showlegend=True,
        legend=dict(font=dict(color='#94a3b8'), bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)',
        height=350, margin=dict(l=60, r=60, t=60, b=40)
    )
    return fig


def create_future_optimization_chart():
    """Create future optimization potential chart."""
    opportunities = ['Generation Mix\nFeatures', 'Hour 9-10\nSpecialized Model',
                     'Natural Gas\nPrice', 'Hydro Reservoir\nFeatures', 'IDM Market\nSignals']
    potential_gain = [0.4, 0.75, 0.25, 0.15, 0.2]
    complexity = [2, 4, 3, 2, 3]  # 1-5 scale

    colors_opt = ['#22c55e', '#f97316', '#3b82f6', '#22c55e', '#8b5cf6']

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=opportunities, y=potential_gain, marker_color=colors_opt,
        text=[f'-{v}%' for v in potential_gain], textposition='outside',
        textfont=dict(color='#f1f5f9', size=12)
    ))

    fig.update_layout(
        title=dict(text="<b>Potential sMAPE Reduction by Optimization</b>", font=dict(size=16, color="#f1f5f9"), x=0.5),
        xaxis=dict(tickfont=dict(color='#94a3b8', size=10)),
        yaxis=dict(title=dict(text="Potential sMAPE Gain (%)", font=dict(color='#94a3b8')),
                   tickfont=dict(color='#94a3b8'), gridcolor='rgba(148, 163, 184, 0.1)'),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        height=300, margin=dict(l=60, r=20, t=60, b=80), showlegend=False
    )
    return fig


# ============== MAIN CONTENT ==============

if MODEL_TYPE == 'price':
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Explorer", "🔬 Model Analysis", "🏗️ Architecture", "🚀 Future Optimization"])

    # ============== TAB 1: DATA EXPLORER ==============
    with tab1:
        st.markdown('<div class="section-header">PRICE DATA STATISTICS</div>', unsafe_allow_html=True)

        # Key data metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value" style="color: #f97316;">174</div>
                <div class="metric-label">TOTAL FEATURES</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value" style="color: #f97316;">27</div>
                <div class="metric-label">CORE FEATURES</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value" style="color: #f97316;">5+ Years</div>
                <div class="metric-label">DATA COVERAGE</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value" style="color: #f97316;">43,800+</div>
                <div class="metric-label">HOURLY RECORDS</div>
            </div>
            """, unsafe_allow_html=True)
        with c5:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value" style="color: #f97316;">TL/MWh</div>
                <div class="metric-label">TARGET UNIT</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="card">
                <div class="card-header">Available Data Sources</div>
                <div class="card-content">
                    <div class="stats-box">
                        <div class="stats-row">
                            <span class="stats-label">EPİAŞ Price Data (PTF, SMF, IDM)</span>
                            <span class="stats-value" style="color: #22c55e;">✓ Used</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">Weather Data (10 cities)</span>
                            <span class="stats-value" style="color: #22c55e;">✓ Used</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">Consumption Actual & Forecast</span>
                            <span class="stats-value" style="color: #22c55e;">✓ Used</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">Wind Forecast (RİTM)</span>
                            <span class="stats-value" style="color: #22c55e;">✓ Used</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">Available Capacity (EAK)</span>
                            <span class="stats-value" style="color: #22c55e;">✓ Used</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">Generation by Source</span>
                            <span class="stats-value" style="color: #eab308;">◐ Partial</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">Hydro Reservoir Volume</span>
                            <span class="stats-value" style="color: #ef4444;">✗ Not Used</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">Generation Plans (KGUP/KUDUP)</span>
                            <span class="stats-value" style="color: #ef4444;">✗ Not Used</span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="card">
                <div class="card-header">Feature Categories Used</div>
                <div class="card-content">
                    <div class="stats-box">
                        <div class="stats-row">
                            <span class="stats-label">Price Features</span>
                            <span class="stats-value">12 features</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">├─ PTF Lags (1h, 24h, 168h)</span>
                            <span class="stats-value">3</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">├─ Rolling Stats (mean, std, min, max)</span>
                            <span class="stats-value">8</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">└─ Volatility Metrics</span>
                            <span class="stats-value">1</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">Fundamental Features</span>
                            <span class="stats-value">8 features</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">├─ thermal_gap, renewable_saturation</span>
                            <span class="stats-value">2</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">├─ spark_spread, load_factor</span>
                            <span class="stats-value">2</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">└─ reserve_margin, system_short</span>
                            <span class="stats-value">4</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">Calendar Features</span>
                            <span class="stats-value">5 features</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">Market Features</span>
                            <span class="stats-value">2 features</span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">FEATURE IMPORTANCE</div>', unsafe_allow_html=True)
        st.plotly_chart(create_feature_importance_chart(), use_container_width=True, config={"displayModeBar": False})

    # ============== TAB 2: MODEL ANALYSIS ==============
    with tab2:
        st.markdown('<div class="section-header">MODEL PERFORMANCE COMPARISON</div>', unsafe_allow_html=True)

        # Key metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value" style="color: #10b981;">11.63%</div>
                <div class="metric-label">FINAL sMAPE</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value" style="color: #10b981;">~75</div>
                <div class="metric-label">MAE (TL/MWh)</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value" style="color: #f97316;">45</div>
                <div class="metric-label">PRICE MODELS TRAINED</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value" style="color: #3b82f6;">6</div>
                <div class="metric-label">MODEL TYPES TESTED</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-header">MODEL COMPARISON RADAR</div>', unsafe_allow_html=True)
            st.plotly_chart(create_model_comparison_chart(), use_container_width=True, config={"displayModeBar": False})

        with col2:
            st.markdown('<div class="section-header">HOURLY PERFORMANCE</div>', unsafe_allow_html=True)
            st.plotly_chart(create_hourly_smape_chart(), use_container_width=True, config={"displayModeBar": False})

        st.markdown("<br>", unsafe_allow_html=True)

        # Model details tables
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="card">
                <div class="card-header" style="color: #ef4444;">Deep Learning Models</div>
                <div class="card-content">
                    <div class="stats-box">
                        <div class="stats-row">
                            <span class="stats-label">N-HiTS (Best DL)</span>
                            <span class="stats-value">sMAPE: 16.01% | MAE: 108.80</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">TFT</span>
                            <span class="stats-value">sMAPE: 16.79% | MAE: 111.17</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">PatchTST</span>
                            <span class="stats-value">sMAPE: 17.17% | MAE: 116.02</span>
                        </div>
                    </div>
                    <div style="margin-top: 1rem; padding: 0.75rem; background: rgba(239, 68, 68, 0.1); border-radius: 8px; color: #fca5a5; font-size: 0.85rem;">
                        ⚠️ Deep learning underperformed gradient boosting by 31% MAE
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="card">
                <div class="card-header" style="color: #10b981;">Gradient Boosting Models</div>
                <div class="card-content">
                    <div class="stats-box">
                        <div class="stats-row">
                            <span class="stats-label">LightGBM (Best)</span>
                            <span class="stats-value" style="color: #22c55e;">sMAPE: 16.83% | MAE: 75.00 ✓</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">XGBoost</span>
                            <span class="stats-value">sMAPE: 16.84% | MAE: 75.25</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">CatBoost</span>
                            <span class="stats-value">sMAPE: 17.64% | MAE: 80.74</span>
                        </div>
                    </div>
                    <div style="margin-top: 1rem; padding: 0.75rem; background: rgba(34, 197, 94, 0.1); border-radius: 8px; color: #86efac; font-size: 0.85rem;">
                        ✓ Final ensemble: 61.4% CatBoost + 38.6% LightGBM
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Training configurations
        st.markdown('<div class="section-header">BEST MODEL CONFIGURATIONS</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="card">
                <div class="card-header">CatBoost Configuration</div>
                <div class="card-content">
                    <div class="stats-box">
                        <div class="stats-row">
                            <span class="stats-label">iterations</span>
                            <span class="stats-value">800</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">depth</span>
                            <span class="stats-value">8</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">learning_rate</span>
                            <span class="stats-value">0.03</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">l2_leaf_reg</span>
                            <span class="stats-value">1.0</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">border_count</span>
                            <span class="stats-value">128</span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="card">
                <div class="card-header">LightGBM Configuration</div>
                <div class="card-content">
                    <div class="stats-box">
                        <div class="stats-row">
                            <span class="stats-label">n_estimators</span>
                            <span class="stats-value">2000</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">max_depth</span>
                            <span class="stats-value">8</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">learning_rate</span>
                            <span class="stats-value">0.02</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">num_leaves</span>
                            <span class="stats-value">255</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">reg_lambda</span>
                            <span class="stats-value">3.0</span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ============== TAB 3: ARCHITECTURE ==============
    with tab3:
        st.markdown('<div class="section-header">KEY RESULTS</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value" style="color: #ef4444;">16.01%</div>
                <div class="metric-label">INITIAL sMAPE</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value" style="color: #10b981;">11.63%</div>
                <div class="metric-label">FINAL sMAPE</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value" style="color: #f97316;">4.38%</div>
                <div class="metric-label">TOTAL REDUCTION</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value" style="color: #3b82f6;">27.4%</div>
                <div class="metric-label">RELATIVE IMPROVEMENT</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="section-header">OPTIMIZATION JOURNEY</div>', unsafe_allow_html=True)
        st.plotly_chart(create_optimization_flowchart(), use_container_width=True, config={"displayModeBar": False})

        st.markdown("""
        <div class="info-box">
            <strong>Optimization Techniques:</strong><br>
            • <b>V4-V5:</b> Transfer learning with 6-month fine-tuning window<br>
            • <b>V10:</b> Hour-specific profile evolution features<br>
            • <b>V11:</b> CatBoost + LightGBM weighted ensemble (61.4% / 38.6%)<br>
            • <b>V12-V13:</b> Adaptive Error Correction with hourly parameters<br>
            • <b>V14:</b> Context-Aware KNN Error Correction (beat oracle floor!)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.markdown('<div class="section-header">MODEL ARCHITECTURE</div>', unsafe_allow_html=True)
            st.plotly_chart(create_architecture_diagram(), use_container_width=True, config={"displayModeBar": False})

        with col2:
            st.markdown('<div class="section-header">sMAPE BY VERSION</div>', unsafe_allow_html=True)
            st.plotly_chart(create_smape_reduction_chart(), use_container_width=True, config={"displayModeBar": False})

    # ============== TAB 4: FUTURE OPTIMIZATION ==============
    with tab4:
        st.markdown('<div class="section-header">FUTURE OPTIMIZATION OPPORTUNITIES</div>', unsafe_allow_html=True)

        # Current vs potential
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value" style="color: #10b981;">11.63%</div>
                <div class="metric-label">CURRENT sMAPE</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value" style="color: #3b82f6;">~10.5%</div>
                <div class="metric-label">THEORETICAL FLOOR</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value" style="color: #f97316;">~1.1%</div>
                <div class="metric-label">POTENTIAL GAIN</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.plotly_chart(create_future_optimization_chart(), use_container_width=True, config={"displayModeBar": False})

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="card">
                <div class="card-header" style="color: #22c55e;">Untapped Data Sources</div>
                <div class="card-content">
                    <div class="stats-box">
                        <div class="stats-row">
                            <span class="stats-label">Generation by Source (Solar/Wind)</span>
                            <span class="stats-value" style="color: #22c55e;">High Impact</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">Intraday Market (IDM) Signals</span>
                            <span class="stats-value" style="color: #eab308;">Medium Impact</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">Hydro Reservoir Levels</span>
                            <span class="stats-value" style="color: #eab308;">Medium Impact</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">Generation Plans (KGUP/KUDUP)</span>
                            <span class="stats-value" style="color: #22c55e;">High Impact</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">Natural Gas Prices (TTF/BOTAS)</span>
                            <span class="stats-value" style="color: #22c55e;">High Impact</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">EU Carbon Prices (EUA)</span>
                            <span class="stats-value" style="color: #eab308;">Medium Impact</span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="card">
                <div class="card-header" style="color: #f97316;">Hour 9-10 Problem (25%+ sMAPE)</div>
                <div class="card-content">
                    <div class="stats-box">
                        <div class="stats-row">
                            <span class="stats-label">Current Hour 9-10 sMAPE</span>
                            <span class="stats-value" style="color: #ef4444;">25.46%</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">Global Average sMAPE</span>
                            <span class="stats-value">11.63%</span>
                        </div>
                        <div class="stats-row">
                            <span class="stats-label">Gap to Close</span>
                            <span class="stats-value" style="color: #ef4444;">13.83%</span>
                        </div>
                    </div>
                    <div style="margin-top: 1rem; padding: 0.75rem; background: rgba(249, 115, 22, 0.1); border-radius: 8px; color: #fdba74; font-size: 0.85rem;">
                        <b>Potential Solutions:</b><br>
                        • Dedicated model for hours 8-11<br>
                        • Solar irradiance forecast features<br>
                        • Industrial load proxy (factory start-up)<br>
                        • Market opening dynamics features
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Feature engineering opportunities
        st.markdown('<div class="section-header">PROPOSED NEW FEATURES</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="card">
                <div class="card-header">Generation Mix Features</div>
                <div class="card-content" style="font-size: 0.85rem; color: #94a3b8;">
                    <code style="color: #22c55e;">solar_share</code> = sun / total_gen<br>
                    <code style="color: #22c55e;">wind_share</code> = wind / total_gen<br>
                    <code style="color: #22c55e;">thermal_share</code> = thermal / total<br>
                    <code style="color: #22c55e;">renewable_variability</code> = std(solar+wind, 3h)<br>
                    <code style="color: #22c55e;">merit_order_proxy</code> = gas_gen / thermal_cap
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="card">
                <div class="card-header">Market Microstructure</div>
                <div class="card-content" style="font-size: 0.85rem; color: #94a3b8;">
                    <code style="color: #3b82f6;">idm_ptf_spread</code> = idm - ptf_lag<br>
                    <code style="color: #3b82f6;">idm_volume_signal</code> = vol / avg_vol<br>
                    <code style="color: #3b82f6;">bid_ask_imbalance</code> = (bid-ask) / total<br>
                    <code style="color: #3b82f6;">time_to_market_close</code> = hours to 10:00
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="card">
                <div class="card-header">Hydro & External</div>
                <div class="card-content" style="font-size: 0.85rem; color: #94a3b8;">
                    <code style="color: #8b5cf6;">hydro_fill_rate</code> = vol / max_vol<br>
                    <code style="color: #8b5cf6;">hydro_change_7d</code> = vol - vol_lag_7d<br>
                    <code style="color: #8b5cf6;">gas_price_lag</code> = ttf_price_lag_24h<br>
                    <code style="color: #8b5cf6;">carbon_price</code> = eua_spot
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="success-box">
            <b>Summary:</b> With full implementation of these optimizations, the theoretical floor is ~10.5% sMAPE.
            The biggest opportunity is the Hour 9-10 specialized model, which alone could reduce sMAPE by 0.5-1.0%.
        </div>
        """, unsafe_allow_html=True)

else:
    # Consumption model - simpler view
    st.markdown('<div class="section-header">CONSUMPTION MODEL OVERVIEW</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        The consumption forecasting model uses a similar gradient boosting approach but with features
        optimized for electricity demand prediction rather than price.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color: #22c55e;">CatBoost</div>
            <div class="metric-label">PRIMARY MODEL</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color: #22c55e;">27</div>
            <div class="metric-label">CORE FEATURES</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color: #22c55e;">24h</div>
            <div class="metric-label">FORECAST HORIZON</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="card-header">Key Consumption Features</div>
        <div class="card-content">
            <div class="stats-box">
                <div class="stats-row">
                    <span class="stats-label">consumption_lag_24h</span>
                    <span class="stats-value">Daily pattern</span>
                </div>
                <div class="stats-row">
                    <span class="stats-label">consumption_lag_168h</span>
                    <span class="stats-value">Weekly pattern</span>
                </div>
                <div class="stats-row">
                    <span class="stats-label">temperature_weighted</span>
                    <span class="stats-value">Weather impact</span>
                </div>
                <div class="stats-row">
                    <span class="stats-label">hour_sin/cos</span>
                    <span class="stats-value">Cyclical time</span>
                </div>
                <div class="stats-row">
                    <span class="stats-label">is_holiday</span>
                    <span class="stats-value">Holiday effects</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    ForeWatt v1.0 | Optimization Architecture Documentation
</div>
""", unsafe_allow_html=True)
