"""
ForeWatt Dashboard - Shared Professional Styles
Centralized styling for consistent look across all pages.
"""

# Professional Dark Theme CSS
PROFESSIONAL_CSS = """
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Styles */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Page Header */
    .page-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.5rem;
    }

    .page-header-icon {
        font-size: 2rem;
    }

    .page-header-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f1f5f9;
        letter-spacing: -0.02em;
        margin: 0;
    }

    .page-header-subtitle {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    /* Section Headers */
    .section-header {
        color: #94a3b8;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.1);
    }

    .section-title {
        color: #f1f5f9;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    /* Card Styles */
    .card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.1);
        margin-bottom: 1rem;
    }

    .card-header {
        color: #f1f5f9;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }

    .card-content {
        color: #94a3b8;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.1);
        text-align: center;
    }

    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 500;
    }

    /* Stats Box */
    .stats-box {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.1);
    }

    .stats-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.1);
    }

    .stats-row:last-child {
        border-bottom: none;
    }

    .stats-label {
        color: #94a3b8;
        font-size: 0.8rem;
        font-weight: 500;
    }

    .stats-value {
        color: #f1f5f9;
        font-size: 0.9rem;
        font-weight: 600;
    }

    /* Info Box */
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        color: #93c5fd;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    .info-box strong {
        color: #bfdbfe;
    }

    /* Warning Box */
    .warning-box {
        background: rgba(251, 191, 36, 0.1);
        border: 1px solid rgba(251, 191, 36, 0.2);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        color: #fcd34d;
        font-size: 0.9rem;
    }

    /* Success Box */
    .success-box {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.2);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        color: #86efac;
        font-size: 0.9rem;
    }

    /* Button Overrides */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        border-radius: 10px;
        transition: all 0.3s ease;
        font-size: 0.95rem;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 10px 30px -10px rgba(59, 130, 246, 0.5);
        transform: translateY(-2px);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 41, 59, 0.5);
        border-radius: 8px;
        padding: 10px 20px;
        color: #94a3b8;
        font-weight: 500;
        border: 1px solid rgba(148, 163, 184, 0.1);
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        border-color: rgba(59, 130, 246, 0.3);
    }

    /* Selectbox Styling */
    .stSelectbox [data-baseweb="select"] {
        background-color: #1e293b;
        border-color: rgba(148, 163, 184, 0.2);
    }

    .stSelectbox [data-baseweb="select"]:hover {
        border-color: rgba(148, 163, 184, 0.4);
    }

    /* Metric Override */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f1f5f9;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-size: 0.85rem;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.8rem;
    }

    /* Expander Styling */
    .streamlit-expanderHeader {
        background-color: rgba(30, 41, 59, 0.5);
        border-radius: 8px;
        color: #f1f5f9;
        font-weight: 500;
    }

    /* Divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(148, 163, 184, 0.2), transparent);
        margin: 2rem 0;
    }

    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(148, 163, 184, 0.15), transparent);
        margin: 1.5rem 0;
    }

    /* Dataframe Styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem 0;
        color: #475569;
        font-size: 0.8rem;
    }

    .footer a {
        color: #60a5fa;
        text-decoration: none;
    }

    /* Badge */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
    }

    .badge-primary {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    .badge-success {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }

    .badge-warning {
        background: rgba(251, 191, 36, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }

    .badge-danger {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* Chart container */
    .chart-container {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.1);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.1);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* Navigation Cards */
    .nav-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.1);
        margin-bottom: 0.75rem;
        transition: all 0.3s ease;
    }

    .nav-card:hover {
        border-color: rgba(148, 163, 184, 0.25);
        box-shadow: 0 10px 40px -10px rgba(0, 0, 0, 0.4);
        transform: translateX(4px);
    }

    .nav-card h3 {
        color: #f1f5f9;
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
        font-weight: 600;
    }

    .nav-card p {
        color: #64748b;
        margin: 0;
        font-size: 0.875rem;
        line-height: 1.4;
    }

    /* Model Selection Cards */
    .model-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.1);
        text-align: center;
        transition: all 0.3s ease;
    }

    .model-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.4);
        border-color: rgba(148, 163, 184, 0.2);
    }

    .consumption-card {
        border-top: 3px solid #22c55e;
    }

    .price-card {
        border-top: 3px solid #f97316;
    }

    .card-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }

    .card-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 0.5rem;
    }

    .card-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 0.5rem;
    }

    .card-description {
        font-size: 0.85rem;
        color: #64748b;
    }

    /* About Section */
    .about-section {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.1);
    }

    .about-title {
        color: #f1f5f9;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .about-text {
        color: #94a3b8;
        font-size: 0.95rem;
        line-height: 1.6;
        margin-bottom: 1.25rem;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-bottom: 1.25rem;
    }

    .feature-item {
        background: rgba(15, 23, 42, 0.5);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(148, 163, 184, 0.08);
        text-align: center;
    }

    .feature-number {
        font-size: 1.5rem;
        font-weight: 700;
        color: #60a5fa;
        margin-bottom: 0.25rem;
    }

    .feature-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    .team-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(99, 102, 241, 0.1);
        color: #a5b4fc;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }

    /* Main Header */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.25rem;
    }

    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        color: #64748b;
        margin-bottom: 2rem;
    }

    .selection-header {
        font-size: 1.25rem;
        text-align: center;
        color: #e2e8f0;
        margin: 1.5rem 0;
        font-weight: 500;
    }

    /* Page Title */
    .page-title {
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }

    .page-subtitle {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    /* Responsive */
    @media (max-width: 768px) {
        .feature-grid {
            grid-template-columns: 1fr;
        }
        .main-header {
            font-size: 2rem;
        }
    }
</style>
"""

# CSS to hide the Optimization Architecture page from sidebar (for consumption mode)
HIDE_OPTIMIZATION_NAV_CSS = """
<style>
    /* Hide the Optimization Architecture page from sidebar navigation */
    [data-testid="stSidebarNav"] li a[href*="Optimization"] {
        display: none !important;
    }
    [data-testid="stSidebarNav"] ul li:has(a[href*="Optimization"]) {
        display: none !important;
    }
</style>
"""

# CSS to completely hide sidebar (for home page)
HIDE_SIDEBAR_CSS = """
<style>
    /* Hide sidebar completely */
    [data-testid="stSidebar"] {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        min-width: 0 !important;
        max-width: 0 !important;
    }

    /* Hide the collapse control / hamburger button */
    [data-testid="collapsedControl"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* Hide sidebar section */
    section[data-testid="stSidebar"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* Hide sidebar navigation */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }

    /* Hide sidebar content */
    [data-testid="stSidebarContent"] {
        display: none !important;
    }

    /* Remove sidebar toggle button */
    button[kind="header"] {
        display: none !important;
    }

    /* Expand main content to full width */
    .main .block-container {
        max-width: 100% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
</style>
"""

# Plotly dark theme template
PLOTLY_DARK_TEMPLATE = {
    'layout': {
        'paper_bgcolor': 'rgba(15, 23, 42, 0)',
        'plot_bgcolor': 'rgba(30, 41, 59, 0.5)',
        'font': {'color': '#94a3b8', 'family': 'Inter, sans-serif'},
        'title': {'font': {'color': '#f1f5f9', 'size': 16}},
        'xaxis': {
            'gridcolor': 'rgba(148, 163, 184, 0.1)',
            'linecolor': 'rgba(148, 163, 184, 0.2)',
            'tickfont': {'color': '#94a3b8'},
            'title': {'font': {'color': '#94a3b8'}}
        },
        'yaxis': {
            'gridcolor': 'rgba(148, 163, 184, 0.1)',
            'linecolor': 'rgba(148, 163, 184, 0.2)',
            'tickfont': {'color': '#94a3b8'},
            'title': {'font': {'color': '#94a3b8'}}
        },
        'legend': {
            'bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': '#94a3b8'}
        }
    }
}

# Color schemes for consumption and price
CONSUMPTION_COLORS = {
    'primary': '#22c55e',
    'secondary': '#4ade80',
    'accent': '#86efac',
    'gradient': 'linear-gradient(135deg, #22c55e 0%, #16a34a 100%)'
}

PRICE_COLORS = {
    'primary': '#f97316',
    'secondary': '#fb923c',
    'accent': '#fdba74',
    'gradient': 'linear-gradient(135deg, #f97316 0%, #ea580c 100%)'
}


def get_model_colors(model_type: str) -> dict:
    """Get color scheme based on model type."""
    if model_type == 'consumption':
        return CONSUMPTION_COLORS
    return PRICE_COLORS


def apply_professional_style():
    """Apply professional styling to the page."""
    import streamlit as st
    st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)


def hide_sidebar():
    """Hide the sidebar completely."""
    import streamlit as st
    st.markdown(HIDE_SIDEBAR_CSS, unsafe_allow_html=True)


def render_page_header(icon: str, title: str, subtitle: str = None, color: str = None):
    """Render a professional page header."""
    import streamlit as st

    style = f'color: {color};' if color else ''
    html = f'''
    <div class="page-header">
        <span class="page-header-icon">{icon}</span>
        <h1 class="page-header-title" style="{style}">{title}</h1>
    </div>
    '''
    if subtitle:
        html += f'<p class="page-header-subtitle">{subtitle}</p>'

    st.markdown(html, unsafe_allow_html=True)


def render_metric_card(value: str, label: str, color: str = None):
    """Render a styled metric card."""
    import streamlit as st

    value_style = f'color: {color};' if color else ''
    st.markdown(f'''
    <div class="metric-card">
        <div class="metric-value" style="{value_style}">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    ''', unsafe_allow_html=True)


def render_info_box(content: str):
    """Render an info box."""
    import streamlit as st
    st.markdown(f'<div class="info-box">{content}</div>', unsafe_allow_html=True)


def render_success_box(content: str):
    """Render a success box."""
    import streamlit as st
    st.markdown(f'<div class="success-box">{content}</div>', unsafe_allow_html=True)


def render_warning_box(content: str):
    """Render a warning box."""
    import streamlit as st
    st.markdown(f'<div class="warning-box">{content}</div>', unsafe_allow_html=True)


def render_section_header(title: str):
    """Render a section header."""
    import streamlit as st
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def render_divider():
    """Render a custom divider."""
    import streamlit as st
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
