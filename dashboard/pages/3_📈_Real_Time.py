"""
ForeWatt Dashboard - Real-Time Graph
Unified page for real-time consumption and price forecasting.
Price uses the optimized V14 model (CatBoost+LightGBM Ensemble + Hourly AEC).
"""
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import requests
import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import quote
from dotenv import load_dotenv

# Add parent and project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import PAGE_CONFIG, PROFESSIONAL_CSS, HIDE_OPTIMIZATION_NAV_CSS, get_model_colors, render_page_header

# Load environment
load_dotenv()

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
    ICON = "⚡"
    TITLE = "Real-Time Consumption Graph"
    COLOR_PRIMARY = colors['primary']
    UNIT = "MWh"
    TARGET_LABEL = "Consumption"
    TARGET_COL = "consumption"
else:
    ICON = "💰"
    TITLE = "Real-Time Price Graph"
    COLOR_PRIMARY = colors['primary']
    UNIT = "TL/MWh"
    TARGET_LABEL = "Price"
    TARGET_COL = "price_real"

# Auto-refresh
try:
    from streamlit_autorefresh import st_autorefresh
    refresh_count = st_autorefresh(interval=60000, limit=None, key="realtime_refresh")
except ImportError:
    refresh_count = 0
    st.sidebar.warning("Install: `pip install streamlit-autorefresh`")

# Constants
TURKEY_TZ = 'Europe/Istanbul'
EPIAS_DELAY_HOURS = 2
HISTORY_HOURS = 12
FORECAST_HOURS = 24

COLOR_ACTUAL = COLOR_PRIMARY
COLOR_FORECAST = '#1f77b4'
COLOR_PIVOT = '#e74c3c'

# Model paths
MODEL_DIR = PROJECT_ROOT / 'reports' / 'new_experiment' / 'baseline' / 'models'
V14_MODEL_DIR = PROJECT_ROOT / 'reports' / 'optimized_search_v14' / 'models'

# V14 Ensemble weights (from optimization report)
CATBOOST_WEIGHT = 0.6144
LIGHTGBM_WEIGHT = 0.3856

# Hourly AEC parameters from v13 optimization
HOURLY_AEC_PARAMS = {
    0: {'lookback': 14, 'damping': 0.5},
    1: {'lookback': 14, 'damping': 0.5},
    2: {'lookback': 21, 'damping': 0.5},
    3: {'lookback': 7, 'damping': 0.5},
    4: {'lookback': 21, 'damping': 0.5},
    5: {'lookback': 7, 'damping': 0.7},
    6: {'lookback': 21, 'damping': 0.5},
    7: {'lookback': 21, 'damping': 0.5},
    8: {'lookback': 21, 'damping': 0.5},
    9: {'lookback': 7, 'damping': 0.5},
    10: {'lookback': 7, 'damping': 0.5},
    11: {'lookback': 5, 'damping': 0.7},
    12: {'lookback': 7, 'damping': 0.5},
    13: {'lookback': 7, 'damping': 0.5},
    14: {'lookback': 21, 'damping': 0.6},
    15: {'lookback': 7, 'damping': 0.5},
    16: {'lookback': 5, 'damping': 0.5},
    17: {'lookback': 7, 'damping': 0.5},
    18: {'lookback': 7, 'damping': 0.5},
    19: {'lookback': 21, 'damping': 0.5},
    20: {'lookback': 14, 'damping': 0.5},
    21: {'lookback': 21, 'damping': 0.7},
    22: {'lookback': 7, 'damping': 0.5},
    23: {'lookback': 7, 'damping': 0.5},
}

# Header with back button
col1, col2 = st.columns([6, 1])
with col1:
    render_page_header(ICON, TITLE, "Live forecasts with EPIAS data and predictions", COLOR_PRIMARY)
with col2:
    if st.button("← Back", key="back_menu_realtime", use_container_width=True):
        st.switch_page("Home.py")

# Model info display
if MODEL_TYPE == 'price':
    st.markdown(f"""
    <div class="info-box" style="border-left: 4px solid {COLOR_PRIMARY};">
        <strong style="color: #008000;">12h Actual</strong> (EPIAS) →
        <strong style="color: #ff0000;">Pivot</strong> (T-2h) →
        <strong style="color: #000080;">24h Forecast</strong> (Optimized Ensemble: CatBoost + LightGBM)
        <br><small>Ensemble: 61.4% CatBoost + 38.6% LightGBM (from V14 optimization)</small>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="info-box" style="border-left: 4px solid {COLOR_PRIMARY};">
        <strong style="color: #008000;">12h Actual</strong> (EPIAS) →
        <strong style="color: #ff0000;">Pivot</strong> (T-2h) →
        <strong style="color: #000080;">24h Forecast</strong> (CatBoost)
    </div>
    """, unsafe_allow_html=True)

# Status bar
turkey_now = pd.Timestamp.now(tz=TURKEY_TZ)
col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
with col1:
    st.caption(f"Auto-refresh: 60s | Count: {refresh_count}")
with col2:
    st.caption(f"Turkey time: {turkey_now.strftime('%Y-%m-%d %H:%M:%S')}")
with col3:
    show_interval = st.checkbox("90% CI", value=True, help="Show/hide 90% prediction interval")
with col4:
    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()

st.divider()

# Consumption features
CONSUMPTION_FEATURES = [
    'consumption_lag_24h', 'consumption_lag_168h', 'consumption_rolling_mean_24h',
    'dow_cos_x', 'hour_cos', 'hour_sin', 'dow_sin_x', 'consumption_rolling_std_24h',
    'is_weekend_x', 'is_holiday_day', 'consumption_lag_48h', 'is_holiday_hour',
    'temp_lag_24h', 'month_cos', 'humidity_national', 'HDD', 'month_sin',
    'price_ptf_lag_24h', 'temp_national', 'heat_index', 'is_cold', 'CDD', 'is_hot'
]

# V14 Optimized Model Features (33 features: 21 base + 12 profile)
V14_BASE_FEATURES = [
    'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
    'price_ptf_rolling_std_24h', 'price_ptf_rolling_mean_24h',
    'price_ptf_rolling_min_24h', 'price_ptf_rolling_max_24h',
    'price_ptf_lag_24h', 'price_ptf_lag_168h',
    'thermal_gap', 'thermal_gap_lag_24h',
    'renewable_saturation', 'spark_spread_proxy_lag_24h',
    'system_short_signal', 'load_factor', 'consumption_forecast',
    'reserve_margin_ratio', 'price_volatility_lag24h', 'realtime_premium_lag24h',
]

# Profile evolution features generated dynamically
V14_PROFILE_FEATURES = [
    'hour', 'daily_avg_price', 'hourly_ratio',
    'profile_14d', 'profile_28d', 'profile_momentum', 'daily_avg_momentum',
    'solar_ratio', 'solar_profile_14d', 'solar_profile_28d',
    'solar_momentum', 'price_solar_interaction'
]

# Complete V14 feature list (33 features)
V14_FEATURES = V14_BASE_FEATURES + V14_PROFILE_FEATURES


@st.cache_data(ttl=300)
def fetch_epias_data(start_date: str, end_date: str, data_type: str = 'consumption') -> pd.DataFrame:
    """Fetch real-time data from EPIAS API."""
    username = os.getenv('EPTR_USERNAME')
    password = os.getenv('EPTR_PASSWORD')

    if not username or not password:
        st.error("EPIAS credentials not found in .env (EPTR_USERNAME, EPTR_PASSWORD)")
        return pd.DataFrame()

    try:
        auth_url = "https://giris.epias.com.tr/cas/v1/tickets"
        body_str = f"username={quote(username)}&password={quote(password)}"

        auth_response = requests.post(
            auth_url,
            data=body_str,
            headers={'Content-Type': 'application/x-www-form-urlencoded', 'Accept': 'text/plain'},
            timeout=30
        )

        if auth_response.status_code not in [200, 201]:
            st.error(f"EPIAS auth failed: {auth_response.status_code}")
            return pd.DataFrame()

        tgt = auth_response.text.strip()
        if not tgt.startswith("TGT-"):
            st.error("Invalid TGT format from EPIAS")
            return pd.DataFrame()

        if data_type == 'consumption':
            api_url = "https://seffaflik.epias.com.tr/electricity-service/v1/consumption/data/realtime-consumption"
            payload = {
                "startDate": f"{start_date}T00:00:00+03:00",
                "endDate": f"{end_date}T23:00:00+03:00"
            }
        else:  # price
            api_url = "https://seffaflik.epias.com.tr/electricity-service/v1/markets/dam/data/mcp"
            payload = {
                "startDate": f"{start_date}T00:00:00+03:00",
                "endDate": f"{end_date}T23:00:00+03:00"
            }

        headers = {"TGT": tgt, "Content-Type": "application/json", "Accept": "application/json"}
        data_response = requests.post(api_url, json=payload, headers=headers, timeout=60)

        if data_response.status_code == 200:
            data = data_response.json()

            if 'items' in data and data['items']:
                df = pd.DataFrame(data['items'])
            elif 'body' in data:
                if data_type == 'consumption' and 'realtimeConsumptionList' in data['body']:
                    df = pd.DataFrame(data['body']['realtimeConsumptionList'])
                elif data_type == 'price' and 'dayAheadMCPList' in data['body']:
                    df = pd.DataFrame(data['body']['dayAheadMCPList'])
                else:
                    return pd.DataFrame()
            else:
                return pd.DataFrame()

            if 'date' in df.columns:
                df['datetime'] = pd.to_datetime(df['date'])
                df = df.set_index('datetime')
                if df.index.tz is None:
                    df.index = df.index.tz_localize(TURKEY_TZ)

            # Normalize column names
            if data_type == 'consumption' and 'consumption' not in df.columns:
                for col in df.columns:
                    if col.lower() == 'consumption':
                        df['consumption'] = df[col]
                        break
            elif data_type == 'price':
                if 'price' in df.columns:
                    df['price_real'] = df['price']
                elif 'marketClearingPrice' in df.columns:
                    df['price_real'] = df['marketClearingPrice']

            return df.sort_index()

        st.error(f"EPIAS API error: {data_response.status_code}")
        return pd.DataFrame()

    except Exception as e:
        st.error(f"EPIAS fetch failed: {e}")
        return pd.DataFrame()


@st.cache_resource
def load_consumption_model():
    """Load the trained CatBoost model for consumption."""
    try:
        from catboost import CatBoostRegressor

        model_name = 'catboost_consumption_8327b57030a0'
        model_path = MODEL_DIR / model_name / 'model.cbm'

        if not model_path.exists():
            model_files = list(MODEL_DIR.glob("catboost_consumption*/model.cbm"))
            if model_files:
                model_path = sorted(model_files)[-1]
            else:
                return None, None

        model = CatBoostRegressor()
        model.load_model(str(model_path))

        fi_path = model_path.parent / 'feature_importance.csv'
        if fi_path.exists():
            fi_df = pd.read_csv(fi_path)
            features = fi_df['feature'].tolist()
        else:
            features = CONSUMPTION_FEATURES

        return model, features

    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None, None


@st.cache_resource
def load_v14_ensemble_models():
    """Load V14 optimized CatBoost and LightGBM ensemble models."""
    try:
        from catboost import CatBoostRegressor
        import lightgbm as lgb
        import json

        catboost_path = V14_MODEL_DIR / 'catboost_v14.cbm'
        lgb_path = V14_MODEL_DIR / 'lightgbm_v14.txt'
        features_path = V14_MODEL_DIR / 'features.json'
        config_path = V14_MODEL_DIR / 'ensemble_config.json'

        # Check if V14 models exist
        if not catboost_path.exists():
            st.warning("V14 CatBoost model not found. Run optimized_search_v14.py first.")
            return None, None, None, None, None

        # Load CatBoost model
        catboost_model = CatBoostRegressor()
        catboost_model.load_model(str(catboost_path))

        # Load LightGBM model
        lgb_model = None
        if lgb_path.exists():
            lgb_model = lgb.Booster(model_file=str(lgb_path))
        else:
            st.warning("V14 LightGBM model not found, using CatBoost only")

        # Load features from features.json
        if features_path.exists():
            with open(features_path, 'r') as f:
                features_config = json.load(f)
                features = features_config.get('features', V14_FEATURES)
        else:
            features = V14_FEATURES

        # Load ensemble config
        ensemble_config = None
        if config_path.exists():
            with open(config_path, 'r') as f:
                ensemble_config = json.load(f)

        return catboost_model, lgb_model, features, ensemble_config, len(features)

    except Exception as e:
        st.error(f"V14 model load failed: {e}")
        return None, None, None, None, None


@st.cache_data(ttl=3600)
def load_master_price_data():
    """Load master dataset with all features for proper price prediction."""
    try:
        master_path = PROJECT_ROOT / 'data' / 'gold' / 'master' / 'master_v2_fundamental.parquet'
        if not master_path.exists():
            # Try alternative path
            master_path = PROJECT_ROOT / 'data' / 'gold' / 'master' / 'master.parquet'

        if not master_path.exists():
            return None

        df = pd.read_parquet(master_path)

        # Set datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df = df.set_index('date')
            elif 'datetime' in df.columns:
                df = df.set_index('datetime')

        # Ensure timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize('Europe/Istanbul')

        # Ensure price_real column exists
        if 'price_real' not in df.columns and 'price' in df.columns:
            df['price_real'] = df['price']

        return df.sort_index()

    except Exception as e:
        st.warning(f"Could not load master data: {e}")
        return None


def create_enhanced_profile_features(df: pd.DataFrame, price_col: str = 'price_real') -> pd.DataFrame:
    """Create enhanced Profile Evolution Features including Solar Profile for V14."""
    df = df.copy()

    df['hour'] = df.index.hour

    # V10 PRICE PROFILE FEATURES
    df['daily_avg_price'] = df[price_col].shift(1).rolling(24, min_periods=12).mean()
    df['hourly_ratio'] = (df[price_col].shift(1) / df['daily_avg_price'].shift(1)).clip(0.2, 5.0)

    profile_14d_list = []
    profile_28d_list = []

    for hour in range(24):
        hour_mask = df['hour'] == hour
        hour_ratios = df.loc[hour_mask, 'hourly_ratio']
        p14 = hour_ratios.rolling(14, min_periods=7).mean().shift(1)
        p28 = hour_ratios.rolling(28, min_periods=14).mean().shift(1)
        profile_14d_list.append(p14)
        profile_28d_list.append(p28)

    df['profile_14d'] = pd.concat(profile_14d_list).sort_index()
    df['profile_28d'] = pd.concat(profile_28d_list).sort_index()

    df['profile_momentum'] = df['profile_14d'] - df['profile_28d']
    df['daily_avg_momentum'] = df['daily_avg_price'] - df['daily_avg_price'].shift(24)

    # V11 SOLAR PROFILE FEATURES
    if 'renewable_saturation' in df.columns and 'load_factor' in df.columns:
        load = df['load_factor'].clip(lower=0.1)
        df['solar_ratio'] = (df['renewable_saturation'].shift(1) / load.shift(1)).clip(0, 5)

        solar_14d_list = []
        solar_28d_list = []

        for hour in range(24):
            hour_mask = df['hour'] == hour
            hour_solar = df.loc[hour_mask, 'solar_ratio']
            s14 = hour_solar.rolling(14, min_periods=7).mean().shift(1)
            s28 = hour_solar.rolling(28, min_periods=14).mean().shift(1)
            solar_14d_list.append(s14)
            solar_28d_list.append(s28)

        df['solar_profile_14d'] = pd.concat(solar_14d_list).sort_index()
        df['solar_profile_28d'] = pd.concat(solar_28d_list).sort_index()

        df['solar_momentum'] = df['solar_profile_14d'] - df['solar_profile_28d']

    if 'solar_momentum' in df.columns and 'profile_14d' in df.columns:
        df['price_solar_interaction'] = df['profile_14d'] * df['solar_momentum']

    # Fill NaN values with medians
    for feat in V14_PROFILE_FEATURES:
        if feat in df.columns and df[feat].isna().any():
            median_val = df[feat].median()
            df[feat] = df[feat].fillna(median_val if not pd.isna(median_val) else 0)

    return df


def apply_hourly_aec(prediction: float, hour: int, recent_errors: list) -> float:
    """Apply Hourly-Dynamic Adaptive Error Correction."""
    if not recent_errors:
        return prediction

    params = HOURLY_AEC_PARAMS.get(hour, {'lookback': 7, 'damping': 0.5})
    lookback = params['lookback']
    damping = params['damping']

    # Use recent errors up to lookback
    relevant_errors = recent_errors[-lookback:] if len(recent_errors) >= lookback else recent_errors
    if relevant_errors:
        correction = damping * np.mean(relevant_errors)
        return prediction - correction
    return prediction


def generate_consumption_forecast(data_history: pd.Series, pivot_time: pd.Timestamp,
                                   model, features: list, hours: int = 24) -> pd.DataFrame:
    """Generate autoregressive forecast for consumption using CatBoost."""
    predictions = []
    timestamps = []
    recent_values = list(data_history.tail(200).values)

    for h in range(1, hours + 1):
        future_time = pivot_time + pd.Timedelta(hours=h)
        timestamps.append(future_time)

        feat_dict = {}

        # Calendar features
        hour = future_time.hour
        dow = future_time.dayofweek
        month = future_time.month

        feat_dict['hour'] = hour
        feat_dict['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        feat_dict['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        feat_dict['dow_sin_x'] = np.sin(2 * np.pi * dow / 7)
        feat_dict['dow_cos_x'] = np.cos(2 * np.pi * dow / 7)
        feat_dict['month_sin'] = np.sin(2 * np.pi * month / 12)
        feat_dict['month_cos'] = np.cos(2 * np.pi * month / 12)
        feat_dict['is_weekend_x'] = 1 if dow >= 5 else 0

        turkish_holidays = ['01-01', '04-23', '05-01', '05-19', '07-15', '08-30', '10-29']
        feat_dict['is_holiday_day'] = 1 if future_time.strftime('%m-%d') in turkish_holidays else 0
        feat_dict['is_holiday_hour'] = 1 if (dow >= 5 or feat_dict['is_holiday_day']) else 0

        # Lag features
        if len(recent_values) >= 24:
            feat_dict['consumption_lag_24h'] = recent_values[-24]
        else:
            feat_dict['consumption_lag_24h'] = np.mean(recent_values) if recent_values else 30000

        if len(recent_values) >= 48:
            feat_dict['consumption_lag_48h'] = recent_values[-48]
        else:
            feat_dict['consumption_lag_48h'] = feat_dict['consumption_lag_24h']

        if len(recent_values) >= 168:
            feat_dict['consumption_lag_168h'] = recent_values[-168]
        else:
            feat_dict['consumption_lag_168h'] = feat_dict['consumption_lag_24h']

        if len(recent_values) >= 24:
            feat_dict['consumption_rolling_mean_24h'] = np.mean(recent_values[-24:])
            feat_dict['consumption_rolling_std_24h'] = np.std(recent_values[-24:])
        else:
            feat_dict['consumption_rolling_mean_24h'] = np.mean(recent_values) if recent_values else 30000
            feat_dict['consumption_rolling_std_24h'] = 1000

        # Weather features (defaults)
        feat_dict['temp_national'] = 15.0
        feat_dict['temp_lag_24h'] = 15.0
        feat_dict['humidity_national'] = 60.0
        feat_dict['HDD'] = max(18 - 15, 0)
        feat_dict['CDD'] = max(15 - 18, 0)
        feat_dict['heat_index'] = 15.0
        feat_dict['is_cold'] = 0
        feat_dict['is_hot'] = 0
        feat_dict['price_ptf_lag_24h'] = 0

        X = np.array([[feat_dict.get(f, 0) for f in features]])
        pred = model.predict(X)[0]
        predictions.append(pred)

        recent_values.append(pred)
        if len(recent_values) > 200:
            recent_values = recent_values[-200:]

    base_std = np.std(data_history.tail(24)) * 0.08 if len(data_history) >= 24 else 500

    df = pd.DataFrame({
        'timestamp': timestamps,
        'value': predictions,
        'lower': [p - 1.645 * base_std * (1 + 0.12 * h) for h, p in enumerate(predictions, 1)],
        'upper': [p + 1.645 * base_std * (1 + 0.12 * h) for h, p in enumerate(predictions, 1)],
        'type': 'forecast'
    })

    return df


def generate_v14_price_forecast(
    pivot_time: pd.Timestamp,
    master_df: pd.DataFrame,
    catboost_model, lgb_model,
    features: list,
    ensemble_config: dict = None,
    hours: int = 24
) -> pd.DataFrame:
    """
    Generate price forecast using V14 optimized model with profile features.
    Uses the exact 33 features that V14 was trained on.
    """
    predictions = []
    timestamps = []

    # Get ensemble weights
    if ensemble_config:
        cat_weight = ensemble_config.get('catboost_weight', CATBOOST_WEIGHT)
        lgb_weight = ensemble_config.get('lightgbm_weight', LIGHTGBM_WEIGHT)
    else:
        cat_weight = CATBOOST_WEIGHT
        lgb_weight = LIGHTGBM_WEIGHT

    # Get the most recent data before pivot_time
    if pivot_time.tzinfo is None:
        pivot_time = pivot_time.tz_localize('Europe/Istanbul')

    # Find closest available data
    available_data = master_df[master_df.index <= pivot_time]
    if len(available_data) < 200:
        st.warning("Insufficient historical data in master dataset")
        return pd.DataFrame()

    # Get the last 500 rows for profile feature calculation (needs enough history)
    recent_data = available_data.tail(500).copy()

    # Generate profile features on historical data
    recent_data = create_enhanced_profile_features(recent_data, 'price_real')

    # Get the last row with all features computed
    recent_prices = list(recent_data['price_real'].values)

    for h in range(1, hours + 1):
        future_time = pivot_time + pd.Timedelta(hours=h)
        timestamps.append(future_time)
        hour = future_time.hour

        # Start with the last available row as template
        last_row = recent_data.iloc[-1].to_dict()

        # Update calendar features
        dow = future_time.dayofweek

        last_row['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        last_row['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        last_row['dow_sin_x'] = np.sin(2 * np.pi * dow / 7)
        last_row['dow_cos_x'] = np.cos(2 * np.pi * dow / 7)
        last_row['is_weekend_x'] = 1 if dow >= 5 else 0
        last_row['hour'] = hour

        # Update price lag features using recent predictions
        if len(recent_prices) >= 24:
            last_row['price_ptf_lag_24h'] = recent_prices[-24]
            last_row['price_ptf_rolling_mean_24h'] = np.mean(recent_prices[-24:])
            last_row['price_ptf_rolling_std_24h'] = np.std(recent_prices[-24:])
            last_row['price_ptf_rolling_max_24h'] = np.max(recent_prices[-24:])
            last_row['price_ptf_rolling_min_24h'] = np.min(recent_prices[-24:])
            mean_24 = np.mean(recent_prices[-24:])
            if mean_24 > 0:
                last_row['price_volatility_lag24h'] = np.std(recent_prices[-24:]) / mean_24

        if len(recent_prices) >= 168:
            last_row['price_ptf_lag_168h'] = recent_prices[-168]

        # Update daily avg price for profile features
        if len(recent_prices) >= 24:
            last_row['daily_avg_price'] = np.mean(recent_prices[-24:])
            # Update hourly ratio
            if last_row['daily_avg_price'] > 0:
                last_row['hourly_ratio'] = np.clip(
                    recent_prices[-1] / last_row['daily_avg_price'], 0.2, 5.0
                )
            # Daily avg momentum
            if len(recent_prices) >= 48:
                last_row['daily_avg_momentum'] = (
                    np.mean(recent_prices[-24:]) - np.mean(recent_prices[-48:-24])
                )

        # Prepare feature vector (same features for both models in V14)
        X = np.array([[last_row.get(f, 0) for f in features]])

        # Get CatBoost prediction
        catboost_pred = catboost_model.predict(X)[0]

        # Get LightGBM prediction if available
        if lgb_model is not None:
            lgb_pred = lgb_model.predict(X)[0]
            # Ensemble prediction
            raw_pred = cat_weight * catboost_pred + lgb_weight * lgb_pred
        else:
            raw_pred = catboost_pred

        # Ensure prediction is positive
        raw_pred = max(raw_pred, 0)

        predictions.append(raw_pred)

        # Update recent prices for autoregressive prediction
        recent_prices.append(raw_pred)
        if len(recent_prices) > 500:
            recent_prices = recent_prices[-500:]

    # Calculate prediction intervals based on actual price volatility
    if len(recent_prices) >= 24:
        base_std = np.std(recent_prices[-24:]) * 0.12
    else:
        base_std = 100

    df = pd.DataFrame({
        'timestamp': timestamps,
        'value': predictions,
        'lower': [max(0, p - 1.645 * base_std * (1 + 0.1 * h)) for h, p in enumerate(predictions, 1)],
        'upper': [p + 1.645 * base_std * (1 + 0.1 * h) for h, p in enumerate(predictions, 1)],
        'type': 'forecast'
    })

    return df


def create_realtime_chart(df: pd.DataFrame, pivot_time: pd.Timestamp, show_interval: bool = True) -> alt.Chart:
    """Create Altair chart with actual data, forecast, and pivot line."""

    base = alt.Chart(df).encode(
        x=alt.X('timestamp:T', axis=alt.Axis(title='Time (Turkey)', format='%H:%M', labelAngle=-45)),
        tooltip=[
            alt.Tooltip('timestamp:T', title='Time', format='%Y-%m-%d %H:%M'),
            alt.Tooltip('value:Q', title=f'{TARGET_LABEL} ({UNIT})', format=',.0f'),
            alt.Tooltip('type:N', title='Type')
        ]
    )

    actual = base.transform_filter(alt.datum.type == 'actual').mark_line(strokeWidth=3).encode(
        y=alt.Y('value:Q', axis=alt.Axis(title=f'{TARGET_LABEL} ({UNIT})')),
        color=alt.value(COLOR_ACTUAL)
    )

    actual_points = base.transform_filter(alt.datum.type == 'actual').mark_circle(size=60).encode(
        y='value:Q', color=alt.value(COLOR_ACTUAL)
    )

    forecast = base.transform_filter(alt.datum.type == 'forecast').mark_line(strokeWidth=3, strokeDash=[5, 5]).encode(
        y='value:Q', color=alt.value(COLOR_FORECAST)
    )

    forecast_points = base.transform_filter(alt.datum.type == 'forecast').mark_circle(size=60).encode(
        y='value:Q', color=alt.value(COLOR_FORECAST)
    )

    pivot_df = pd.DataFrame({'pivot': [pivot_time]})
    pivot_rule = alt.Chart(pivot_df).mark_rule(color=COLOR_PIVOT, strokeWidth=2, strokeDash=[4, 4]).encode(x='pivot:T')
    pivot_text = alt.Chart(pivot_df).mark_text(
        align='left', dx=5, dy=-10, color=COLOR_PIVOT, fontSize=12, fontWeight='bold'
    ).encode(x='pivot:T', text=alt.value(f"Pivot: {pivot_time.strftime('%H:%M')}"))

    layers = [actual, actual_points]

    if show_interval:
        interval = base.transform_filter(alt.datum.type == 'forecast').mark_area(opacity=0.3).encode(
            y=alt.Y('lower:Q'), y2='upper:Q', color=alt.value(COLOR_FORECAST)
        )
        layers.insert(0, interval)

    layers.extend([forecast, forecast_points, pivot_rule, pivot_text])

    if MODEL_TYPE == 'price':
        title_text = f'Real-Time {TARGET_LABEL} (EPIAS) + 24h Ensemble Forecast'
    else:
        title_text = f'Real-Time {TARGET_LABEL} (EPIAS) + 24h CatBoost Forecast'

    if show_interval:
        title_text += ' with 90% CI'

    chart = alt.layer(*layers).properties(
        width='container', height=500,
        title=alt.TitleParams(text=title_text, fontSize=16)
    ).configure_axis(labelFontSize=11, titleFontSize=13).interactive()

    return chart


# Time windows
pivot_time = turkey_now.floor('h') - pd.Timedelta(hours=EPIAS_DELAY_HOURS)
history_start = pivot_time - pd.Timedelta(hours=HISTORY_HOURS)
forecast_end = pivot_time + pd.Timedelta(hours=FORECAST_HOURS)

st.info(f"""
**Time Windows:**
- **Actual Data:** {history_start.strftime('%Y-%m-%d %H:%M')} → {pivot_time.strftime('%Y-%m-%d %H:%M')} (12h from EPIAS)
- **Pivot Point:** {pivot_time.strftime('%Y-%m-%d %H:%M')} (T-2h)
- **Forecast:** {(pivot_time + pd.Timedelta(hours=1)).strftime('%Y-%m-%d %H:%M')} → {(pivot_time + pd.Timedelta(hours=24)).strftime('%Y-%m-%d %H:%M')} (24h)
""")

# Fetch EPIAS data
with st.spinner("Fetching EPIAS data..."):
    start_date = history_start.strftime('%Y-%m-%d')
    end_date = pivot_time.strftime('%Y-%m-%d')

    if MODEL_TYPE == 'consumption':
        epias_df = fetch_epias_data(start_date, end_date, 'consumption')
        value_col = 'consumption'
    else:
        epias_df = fetch_epias_data(start_date, end_date, 'price')
        value_col = 'price_real'

if epias_df.empty or value_col not in epias_df.columns:
    st.error(f"Failed to fetch EPIAS data for {TARGET_LABEL}. Check your credentials in .env")
    st.stop()

mask = (epias_df.index >= history_start) & (epias_df.index <= pivot_time)
actual_df = epias_df[mask].copy()

if actual_df.empty:
    st.warning(f"No EPIAS data found for {history_start} to {pivot_time}")
    st.stop()

st.success(f"Loaded {len(actual_df)} hours of actual data from EPIAS")

# Load model and generate forecast
with st.spinner("Generating forecast..."):
    if MODEL_TYPE == 'consumption':
        model, features = load_consumption_model()
        if model is None:
            st.error("Could not load consumption model.")
            st.stop()
        forecast_df = generate_consumption_forecast(
            data_history=actual_df[value_col],
            pivot_time=pivot_time,
            model=model,
            features=features,
            hours=FORECAST_HOURS
        )
        model_info = "CatBoost"
    else:
        # Load V14 optimized models (33 features with profile evolution)
        catboost_model, lgb_model, features, ensemble_config, n_features = load_v14_ensemble_models()
        if catboost_model is None:
            st.error("Could not load V14 price models. Run optimized_search_v14.py to generate them.")
            st.stop()

        # Load master data for proper feature values
        master_df = load_master_price_data()
        if master_df is not None and len(master_df) > 200:
            forecast_df = generate_v14_price_forecast(
                pivot_time=pivot_time,
                master_df=master_df,
                catboost_model=catboost_model,
                lgb_model=lgb_model,
                features=features,
                ensemble_config=ensemble_config,
                hours=FORECAST_HOURS
            )
            if lgb_model is not None:
                cat_w = ensemble_config.get('catboost_weight', CATBOOST_WEIGHT) * 100 if ensemble_config else CATBOOST_WEIGHT * 100
                lgb_w = ensemble_config.get('lightgbm_weight', LIGHTGBM_WEIGHT) * 100 if ensemble_config else LIGHTGBM_WEIGHT * 100
                model_info = f"V14 Ensemble (CatBoost {cat_w:.1f}% + LightGBM {lgb_w:.1f}%) - {n_features} features"
            else:
                model_info = f"V14 CatBoost - {n_features} features"
        else:
            st.warning("Master data not available. Using simplified prediction.")
            # Fallback to simple prediction using EPIAS data
            recent_prices = list(actual_df[value_col].values)
            predictions = []
            timestamps = []
            for h in range(1, FORECAST_HOURS + 1):
                future_time = pivot_time + pd.Timedelta(hours=h)
                timestamps.append(future_time)
                # Simple persistence with hour-of-day pattern
                hour = future_time.hour
                if len(recent_prices) >= 24:
                    # Use same hour from yesterday as base
                    base_pred = recent_prices[-24]
                    # Adjust with recent trend
                    trend = (recent_prices[-1] - recent_prices[-24]) / 24 if len(recent_prices) >= 24 else 0
                    pred = base_pred + trend * h
                else:
                    pred = recent_prices[-1] if recent_prices else 1500
                predictions.append(max(pred, 0))
                recent_prices.append(pred)

            base_std = np.std(actual_df[value_col].tail(24)) * 0.15 if len(actual_df) >= 24 else 150
            forecast_df = pd.DataFrame({
                'timestamp': timestamps,
                'value': predictions,
                'lower': [max(0, p - 1.645 * base_std * (1 + 0.1 * h)) for h, p in enumerate(predictions, 1)],
                'upper': [p + 1.645 * base_std * (1 + 0.1 * h) for h, p in enumerate(predictions, 1)],
                'type': 'forecast'
            })
            model_info = "Simple Persistence (Master Data unavailable)"

st.success(f"Generated {len(forecast_df)} hours of forecast using {model_info}")

# Combine data for chart
actual_records = pd.DataFrame({
    'timestamp': actual_df.index,
    'value': actual_df[value_col].values,
    'type': 'actual',
    'lower': None,
    'upper': None
})

combined_df = pd.concat([actual_records, forecast_df], ignore_index=True)
combined_df = combined_df.sort_values('timestamp')

# Display chart
chart = create_realtime_chart(combined_df, pivot_time, show_interval=show_interval)
st.altair_chart(chart, use_container_width=True)

# Summary statistics
st.divider()
st.markdown("### Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_actual = actual_df[value_col].mean()
    st.metric("Avg Actual (12h)", f"{avg_actual:,.0f} {UNIT}")

with col2:
    avg_forecast = forecast_df['value'].mean()
    st.metric("Avg Forecast (24h)", f"{avg_forecast:,.0f} {UNIT}")

with col3:
    peak_val = actual_df[value_col].max()
    peak_time = actual_df[value_col].idxmax()
    st.metric("Peak Actual", f"{peak_val:,.0f} {UNIT}", f"at {peak_time.strftime('%H:%M')}")

with col4:
    peak_idx = forecast_df['value'].idxmax()
    peak_val = forecast_df.loc[peak_idx, 'value']
    peak_time = forecast_df.loc[peak_idx, 'timestamp']
    st.metric("Peak Forecast", f"{peak_val:,.0f} {UNIT}", f"at {peak_time.strftime('%H:%M')}")

# Architecture flowchart and model details
if MODEL_TYPE == 'consumption':
    with st.expander("Architecture Flowchart", expanded=False):
        # Graphviz flowchart for consumption
        consumption_flowchart = """
        digraph G {
            rankdir=TB;
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];
            edge [fontname="Helvetica", fontsize=10];

            // Data Sources
            subgraph cluster_input {
                label="📥 Data Sources";
                style=filled;
                color="#1976d2";
                fillcolor="#e3f2fd";
                EPIAS [label="🔌 EPIAS API\\n(Real-Time Consumption)", fillcolor="#bbdefb"];
            }

            // Feature Engineering
            subgraph cluster_features {
                label="⚙️ Feature Engineering (23 Features)";
                style=filled;
                color="#f57c00";
                fillcolor="#fff3e0";
                LAGS [label="📊 Lag Features\\n• consumption_lag_24h/48h/168h\\n• Rolling mean/std 24h", fillcolor="#ffe0b2"];
                CALENDAR [label="📅 Calendar Features\\n• hour_sin/cos, dow_sin/cos\\n• is_weekend, is_holiday", fillcolor="#ffe0b2"];
                WEATHER [label="🌡️ Weather Features\\n• temp_national, humidity\\n• HDD, CDD, heat_index", fillcolor="#ffe0b2"];
            }

            // Model
            subgraph cluster_model {
                label="🤖 CatBoost Model";
                style=filled;
                color="#388e3c";
                fillcolor="#e8f5e9";
                CATBOOST [label="🌲 CatBoost\\nGradient Boosting", fillcolor="#c8e6c9"];
            }

            // Output
            subgraph cluster_output {
                label="📤 Output";
                style=filled;
                color="#c2185b";
                fillcolor="#fce4ec";
                FORECAST [label="📉 24h Consumption Forecast\\nwith 90% CI", fillcolor="#f8bbd9"];
            }

            // Connections
            EPIAS -> LAGS [label="12h History"];
            EPIAS -> CALENDAR;
            EPIAS -> WEATHER;
            LAGS -> CATBOOST;
            CALENDAR -> CATBOOST;
            WEATHER -> CATBOOST;
            CATBOOST -> FORECAST;
        }
        """
        st.graphviz_chart(consumption_flowchart)

        # Simple text-based flowchart for compatibility
        st.markdown("---")
        st.markdown("**Pipeline Flow:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div style="text-align:center; padding:12px; background:linear-gradient(135deg, #1976d2, #2196f3); border-radius:10px; color:white; border:2px solid #0d47a1;">
                <b>📥 EPIAS</b><br/>
                <span style="font-size:0.85em;">12h Actual</span>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style="text-align:center; padding:12px; background:linear-gradient(135deg, #f57c00, #ff9800); border-radius:10px; color:white; border:2px solid #e65100;">
                <b>⚙️ Features</b><br/>
                <span style="font-size:0.85em;">23 Features</span>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div style="text-align:center; padding:12px; background:linear-gradient(135deg, #388e3c, #4caf50); border-radius:10px; color:white; border:2px solid #1b5e20;">
                <b>🌲 CatBoost</b><br/>
                <span style="font-size:0.85em;">Autoregressive</span>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div style="text-align:center; padding:12px; background:linear-gradient(135deg, #c2185b, #e91e63); border-radius:10px; color:white; border:2px solid #880e4f;">
                <b>📉 Forecast</b><br/>
                <span style="font-size:0.85em;">24h + 90% CI</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:center; margin:15px 0; font-size:1.5em; color:#1976d2;">
            ▶━━━━━━━━━━━▶━━━━━━━━━━━▶━━━━━━━━━━━▶
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Consumption Model Details"):
        st.markdown("""
        **CatBoost Model Architecture:**
        - **Model Type**: CatBoost Gradient Boosting Regressor
        - **Prediction**: Autoregressive 24-hour forecast

        **23 Features Used:**

        *Lag Features:*
        - `consumption_lag_24h`, `consumption_lag_48h`, `consumption_lag_168h`
        - `consumption_rolling_mean_24h`, `consumption_rolling_std_24h`

        *Calendar Features:*
        - `hour_sin`, `hour_cos`, `dow_sin_x`, `dow_cos_x`
        - `month_sin`, `month_cos`
        - `is_weekend_x`, `is_holiday_day`, `is_holiday_hour`

        *Weather Features:*
        - `temp_national`, `temp_lag_24h`, `humidity_national`
        - `HDD`, `CDD`, `heat_index`
        - `is_cold`, `is_hot`
        """)

elif MODEL_TYPE == 'price':
    with st.expander("Architecture Flowchart", expanded=False):
        # Graphviz flowchart for price
        price_flowchart = """
        digraph G {
            rankdir=TB;
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];
            edge [fontname="Helvetica", fontsize=10];

            // Data Sources
            subgraph cluster_input {
                label="📥 Data Sources";
                style=filled;
                color="#1976d2";
                fillcolor="#e3f2fd";
                EPIAS [label="🔌 EPIAS API\\n(Real-Time Prices)", fillcolor="#bbdefb"];
                MASTER [label="💾 Master Dataset\\n(51K+ hourly records)", fillcolor="#bbdefb"];
            }

            // Feature Engineering
            subgraph cluster_features {
                label="⚙️ Feature Engineering (33 Features)";
                style=filled;
                color="#f57c00";
                fillcolor="#fff3e0";
                BASE [label="📊 Base Features (21)\\n• Price lags (24h, 168h)\\n• Rolling stats (mean/std/min/max)\\n• Calendar (hour_sin/cos, dow)", fillcolor="#ffe0b2"];
                PROFILE [label="📈 Profile Evolution (12)\\n• hourly_ratio, profile_14d/28d\\n• solar_ratio, solar_profile_14d/28d\\n• price_solar_interaction", fillcolor="#ffe0b2"];
            }

            // Ensemble Model
            subgraph cluster_ensemble {
                label="🤖 V14 Ensemble Model";
                style=filled;
                color="#388e3c";
                fillcolor="#e8f5e9";
                CATBOOST [label="🌲 CatBoost\\nWeight: 61.4%", fillcolor="#c8e6c9"];
                LIGHTGBM [label="🌿 LightGBM\\nWeight: 38.6%", fillcolor="#c8e6c9"];
                COMBINE [label="➕ Weighted\\nAverage", fillcolor="#a5d6a7"];
            }

            // Autoregressive
            subgraph cluster_auto {
                label="🔄 Autoregressive Loop";
                style=filled;
                color="#7b1fa2";
                fillcolor="#f3e5f5";
                LOOP [label="24 Iterations\\nUpdate lags each step", fillcolor="#e1bee7"];
            }

            // Output
            subgraph cluster_output {
                label="📤 Output";
                style=filled;
                color="#c2185b";
                fillcolor="#fce4ec";
                FORECAST [label="📉 24h Price Forecast\\nwith 90% CI", fillcolor="#f8bbd9"];
            }

            // Connections
            EPIAS -> BASE [label="12h History"];
            MASTER -> BASE [label="500 rows"];
            MASTER -> PROFILE;
            BASE -> CATBOOST;
            BASE -> LIGHTGBM;
            PROFILE -> CATBOOST;
            PROFILE -> LIGHTGBM;
            CATBOOST -> COMBINE;
            LIGHTGBM -> COMBINE;
            COMBINE -> LOOP;
            LOOP -> FORECAST;
        }
        """
        st.graphviz_chart(price_flowchart)

        # Also show a simple text-based flowchart for compatibility
        st.markdown("---")
        st.markdown("**Pipeline Flow:**")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown("""
            <div style="text-align:center; padding:12px; background:linear-gradient(135deg, #1976d2, #2196f3); border-radius:10px; color:white; border:2px solid #0d47a1;">
                <b>📥 EPIAS</b><br/>
                <span style="font-size:0.85em;">12h Actual</span>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style="text-align:center; padding:12px; background:linear-gradient(135deg, #f57c00, #ff9800); border-radius:10px; color:white; border:2px solid #e65100;">
                <b>⚙️ Features</b><br/>
                <span style="font-size:0.85em;">33 Features</span>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div style="text-align:center; padding:12px; background:linear-gradient(135deg, #388e3c, #4caf50); border-radius:10px; color:white; border:2px solid #1b5e20;">
                <b>🤖 Ensemble</b><br/>
                <span style="font-size:0.85em;">CatBoost+LGB</span>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div style="text-align:center; padding:12px; background:linear-gradient(135deg, #7b1fa2, #9c27b0); border-radius:10px; color:white; border:2px solid #4a148c;">
                <b>🔄 Autoregressive</b><br/>
                <span style="font-size:0.85em;">24 iterations</span>
            </div>
            """, unsafe_allow_html=True)
        with col5:
            st.markdown("""
            <div style="text-align:center; padding:12px; background:linear-gradient(135deg, #c2185b, #e91e63); border-radius:10px; color:white; border:2px solid #880e4f;">
                <b>📉 Forecast</b><br/>
                <span style="font-size:0.85em;">24h + 90% CI</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:center; margin:15px 0; font-size:1.5em; color:#1976d2;">
            ▶━━━━━━━▶━━━━━━━▶━━━━━━━▶━━━━━━━▶
        </div>
        """, unsafe_allow_html=True)

    with st.expander("V14 Model Details"):
        st.markdown("""
        **V14 Optimized Architecture:**
        - **Ensemble**: CatBoost (61.4%) + LightGBM (38.6%)
        - **Offline Performance**: 11.63% sMAPE (beats oracle floor of 11.75%)
        - **Transfer Learning**: Base training + 6-month fine-tuning

        **33 Features Used:**

        *Base Features (21):*
        - Price lags: `price_ptf_lag_24h`, `price_ptf_lag_168h`
        - Rolling statistics: `rolling_mean/std/min/max_24h`
        - Fundamentals: `thermal_gap`, `renewable_saturation`, `load_factor`
        - Market signals: `system_short_signal`, `spark_spread_proxy_lag_24h`
        - Calendar: `hour_sin/cos`, `dow_sin/cos`, `is_weekend`

        *Profile Evolution Features (12):*
        - Price profiles: `hourly_ratio`, `profile_14d`, `profile_28d`, `profile_momentum`
        - Solar profiles: `solar_ratio`, `solar_profile_14d/28d`, `solar_momentum`
        - Interactions: `price_solar_interaction`, `daily_avg_momentum`

        **Note:** The full V14 model also includes Hourly-Dynamic AEC (Adaptive Error Correction)
        which requires historical prediction errors. In real-time mode, only the ensemble is used.
        """)

# Data tables
with st.expander("View Data Tables"):
    tab1, tab2 = st.tabs(["Actual (EPIAS)", f"Forecast ({model_info})"])

    with tab1:
        st.dataframe(
            actual_df[[value_col]].reset_index().rename(
                columns={'datetime': 'Time', value_col: f'{TARGET_LABEL} ({UNIT})'}
            ),
            use_container_width=True, hide_index=True
        )

    with tab2:
        st.dataframe(
            forecast_df[['timestamp', 'value', 'lower', 'upper']].rename(columns={
                'timestamp': 'Time', 'value': 'Forecast', 'lower': 'Lower', 'upper': 'Upper'
            }),
            use_container_width=True, hide_index=True
        )

st.divider()
st.caption(f"Model: {model_info} | EPIAS delay: ~{EPIAS_DELAY_HOURS}h | Pivot: {pivot_time.strftime('%Y-%m-%d %H:%M')}")
