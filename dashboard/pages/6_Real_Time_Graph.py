"""
ForeWatt Dashboard - Real-Time Consumption Graph
=================================================
Architecture:
- EPIAS API: Fetch last 12h of actual consumption (available up to ~2h ago)
- Pivot Point: Latest EPIAS data point (T-2h)
- CatBoost: Generate 24h forecast from pivot point
- Auto-refresh: streamlit-autorefresh (every 60s)
"""
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import requests
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import quote
from dotenv import load_dotenv

# Add parent and project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import PAGE_CONFIG

# Load environment
load_dotenv()

# Page configuration
st.set_page_config(**PAGE_CONFIG)

# ============================================================================
# Auto-refresh Configuration
# ============================================================================
try:
    from streamlit_autorefresh import st_autorefresh
    refresh_count = st_autorefresh(interval=60000, limit=None, key="realtime_refresh")
except ImportError:
    refresh_count = 0
    st.sidebar.warning("Install: `pip install streamlit-autorefresh`")

# ============================================================================
# Constants
# ============================================================================
TURKEY_TZ = 'Europe/Istanbul'
EPIAS_DELAY_HOURS = 2  # EPIAS data is available ~2 hours delayed
HISTORY_HOURS = 12     # Show 12 hours of actual data
FORECAST_HOURS = 24    # Predict 24 hours ahead

COLOR_ACTUAL = '#2ca02c'    # Green for actual
COLOR_FORECAST = '#1f77b4'  # Blue for forecast
COLOR_PIVOT = '#e74c3c'     # Red for pivot line

# Model path
MODEL_DIR = PROJECT_ROOT / 'reports' / 'new_experiment' / 'baseline' / 'models'
BEST_MODEL = 'catboost_consumption_8327b57030a0'

# CatBoost features (must match training)
CATBOOST_FEATURES = [
    'consumption_lag_24h', 'consumption_lag_168h', 'consumption_rolling_mean_24h',
    'dow_cos_x', 'hour_cos', 'hour_sin', 'dow_sin_x', 'consumption_rolling_std_24h',
    'is_weekend_x', 'is_holiday_day', 'consumption_lag_48h', 'is_holiday_hour',
    'temp_lag_24h', 'month_cos', 'humidity_national', 'HDD', 'month_sin',
    'price_ptf_lag_24h', 'temp_national', 'heat_index', 'is_cold', 'CDD', 'is_hot'
]


# ============================================================================
# EPIAS API Functions
# ============================================================================

@st.cache_data(ttl=300)  # Cache 5 minutes
def fetch_epias_consumption(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch real-time consumption data from EPIAS API."""
    username = os.getenv('EPTR_USERNAME')
    password = os.getenv('EPTR_PASSWORD')

    if not username or not password:
        st.error("EPIAS credentials not found in .env (EPTR_USERNAME, EPTR_PASSWORD)")
        return pd.DataFrame()

    try:
        # Step 1: Get TGT
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

        # Step 2: Fetch consumption data
        consumption_url = "https://seffaflik.epias.com.tr/electricity-service/v1/consumption/data/realtime-consumption"

        payload = {
            "startDate": f"{start_date}T00:00:00+03:00",
            "endDate": f"{end_date}T23:00:00+03:00"
        }

        headers = {"TGT": tgt, "Content-Type": "application/json", "Accept": "application/json"}

        data_response = requests.post(consumption_url, json=payload, headers=headers, timeout=60)

        if data_response.status_code == 200:
            data = data_response.json()

            if 'items' in data and data['items']:
                df = pd.DataFrame(data['items'])
            elif 'body' in data and 'realtimeConsumptionList' in data['body']:
                df = pd.DataFrame(data['body']['realtimeConsumptionList'])
            else:
                return pd.DataFrame()

            # Parse datetime
            if 'date' in df.columns:
                df['datetime'] = pd.to_datetime(df['date'])
                df = df.set_index('datetime')
                if df.index.tz is None:
                    df.index = df.index.tz_localize(TURKEY_TZ)

            # Ensure consumption column
            if 'consumption' not in df.columns:
                for col in df.columns:
                    if col.lower() == 'consumption':
                        df['consumption'] = df[col]
                        break

            return df.sort_index()

        st.error(f"EPIAS API error: {data_response.status_code}")
        return pd.DataFrame()

    except Exception as e:
        st.error(f"EPIAS fetch failed: {e}")
        return pd.DataFrame()


# ============================================================================
# CatBoost Prediction
# ============================================================================

@st.cache_resource
def load_catboost_model():
    """Load the trained CatBoost model."""
    try:
        from catboost import CatBoostRegressor

        model_path = MODEL_DIR / BEST_MODEL / 'model.cbm'
        if not model_path.exists():
            # Try any catboost model
            model_files = list(MODEL_DIR.glob("catboost_consumption*/model.cbm"))
            if model_files:
                model_path = sorted(model_files)[-1]
            else:
                return None, None

        model = CatBoostRegressor()
        model.load_model(str(model_path))

        # Load feature columns
        fi_path = model_path.parent / 'feature_importance.csv'
        if fi_path.exists():
            fi_df = pd.read_csv(fi_path)
            features = fi_df['feature'].tolist()
        else:
            features = CATBOOST_FEATURES

        return model, features

    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None, None


def generate_forecast(consumption_history: pd.Series, pivot_time: pd.Timestamp,
                      model, features: list, hours: int = 24) -> pd.DataFrame:
    """Generate autoregressive forecast using CatBoost for 24 hours AFTER pivot_time."""

    predictions = []
    timestamps = []

    # Build recent consumption list for lag features
    recent_values = list(consumption_history.tail(200).values)

    # Generate 24 hours AFTER pivot (h=1 to h=24)
    # If pivot is 13:00 → 14:00, 15:00, ..., 13:00 next day (24 data points)
    for h in range(1, hours + 1):
        future_time = pivot_time + pd.Timedelta(hours=h)
        timestamps.append(future_time)

        # Calculate features for this hour
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

        # Holiday features (basic)
        turkish_holidays = ['01-01', '04-23', '05-01', '05-19', '07-15', '08-30', '10-29']
        feat_dict['is_holiday_day'] = 1 if future_time.strftime('%m-%d') in turkish_holidays else 0
        feat_dict['is_holiday_hour'] = 1 if (dow >= 5 or feat_dict['is_holiday_day']) else 0

        # Consumption lag features
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

        # Rolling stats
        if len(recent_values) >= 24:
            feat_dict['consumption_rolling_mean_24h'] = np.mean(recent_values[-24:])
            feat_dict['consumption_rolling_std_24h'] = np.std(recent_values[-24:])
        else:
            feat_dict['consumption_rolling_mean_24h'] = np.mean(recent_values) if recent_values else 30000
            feat_dict['consumption_rolling_std_24h'] = 1000

        # Weather features (use defaults - could integrate Open-Meteo here)
        feat_dict['temp_national'] = 15.0
        feat_dict['temp_lag_24h'] = 15.0
        feat_dict['humidity_national'] = 60.0
        feat_dict['HDD'] = max(18 - 15, 0)
        feat_dict['CDD'] = max(15 - 18, 0)
        feat_dict['heat_index'] = 15.0
        feat_dict['is_cold'] = 0
        feat_dict['is_hot'] = 0
        feat_dict['price_ptf_lag_24h'] = 0

        # Build feature vector
        X = np.array([[feat_dict.get(f, 0) for f in features]])

        # Predict
        pred = model.predict(X)[0]
        predictions.append(pred)

        # Update recent values for next iteration (autoregressive)
        recent_values.append(pred)
        if len(recent_values) > 200:
            recent_values = recent_values[-200:]

    # Calculate 90% prediction intervals (z=1.645 for 90% CI)
    # Wider intervals for further horizons
    base_std = np.std(consumption_history.tail(24)) * 0.08 if len(consumption_history) >= 24 else 500

    df = pd.DataFrame({
        'timestamp': timestamps,
        'value': predictions,
        'lower': [p - 1.645 * base_std * (1 + 0.12 * h) for h, p in enumerate(predictions, 1)],
        'upper': [p + 1.645 * base_std * (1 + 0.12 * h) for h, p in enumerate(predictions, 1)],
        'type': 'forecast'
    })

    return df


# ============================================================================
# Chart Creation
# ============================================================================

def create_realtime_chart(df: pd.DataFrame, pivot_time: pd.Timestamp, show_interval: bool = True) -> alt.Chart:
    """Create Altair chart with actual data, forecast, optional 90% CI, and pivot line."""

    # Base encoding
    base = alt.Chart(df).encode(
        x=alt.X('timestamp:T',
                axis=alt.Axis(title='Time (Turkey)', format='%H:%M', labelAngle=-45)),
        tooltip=[
            alt.Tooltip('timestamp:T', title='Time', format='%Y-%m-%d %H:%M'),
            alt.Tooltip('value:Q', title='Consumption (MWh)', format=',.0f'),
            alt.Tooltip('type:N', title='Type')
        ]
    )

    # Actual data (green solid line)
    actual = base.transform_filter(
        alt.datum.type == 'actual'
    ).mark_line(strokeWidth=3).encode(
        y=alt.Y('value:Q', axis=alt.Axis(title='Consumption (MWh)')),
        color=alt.value(COLOR_ACTUAL)
    )

    actual_points = base.transform_filter(
        alt.datum.type == 'actual'
    ).mark_circle(size=60).encode(
        y='value:Q',
        color=alt.value(COLOR_ACTUAL)
    )

    # Forecast data (blue dashed line)
    forecast = base.transform_filter(
        alt.datum.type == 'forecast'
    ).mark_line(strokeWidth=3, strokeDash=[5, 5]).encode(
        y='value:Q',
        color=alt.value(COLOR_FORECAST)
    )

    forecast_points = base.transform_filter(
        alt.datum.type == 'forecast'
    ).mark_circle(size=60).encode(
        y='value:Q',
        color=alt.value(COLOR_FORECAST)
    )

    # Pivot line (vertical rule)
    pivot_df = pd.DataFrame({'pivot': [pivot_time]})
    pivot_rule = alt.Chart(pivot_df).mark_rule(
        color=COLOR_PIVOT, strokeWidth=2, strokeDash=[4, 4]
    ).encode(x='pivot:T')

    pivot_text = alt.Chart(pivot_df).mark_text(
        align='left', dx=5, dy=-10, color=COLOR_PIVOT, fontSize=12, fontWeight='bold'
    ).encode(
        x='pivot:T',
        text=alt.value(f"Pivot: {pivot_time.strftime('%H:%M')}")
    )

    # Build layers list
    layers = [actual, actual_points]

    # Add 90% prediction interval if enabled
    if show_interval:
        interval = base.transform_filter(
            alt.datum.type == 'forecast'
        ).mark_area(opacity=0.3).encode(
            y=alt.Y('lower:Q'),
            y2='upper:Q',
            color=alt.value(COLOR_FORECAST)
        )
        layers.insert(0, interval)  # Add first so it's behind

    # Add forecast and pivot
    layers.extend([forecast, forecast_points, pivot_rule, pivot_text])

    # Chart title
    title_text = 'Real-Time Consumption (EPIAS) + 24h CatBoost Forecast'
    if show_interval:
        title_text += ' with 90% CI'

    # Combine layers
    chart = alt.layer(*layers).properties(
        width='container',
        height=500,
        title=alt.TitleParams(text=title_text, fontSize=16)
    ).configure_axis(
        labelFontSize=11,
        titleFontSize=13
    ).interactive()

    return chart


# ============================================================================
# Main Page
# ============================================================================

st.markdown("# Real-Time Consumption Graph")
st.markdown("**12h Actual (EPIAS)** → **Pivot (T-2h)** → **24h Forecast (CatBoost)**")

# Status bar
col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
with col1:
    st.caption(f"Auto-refresh: 60s | Count: {refresh_count}")
with col2:
    turkey_now = pd.Timestamp.now(tz=TURKEY_TZ)
    st.caption(f"Turkey time: {turkey_now.strftime('%Y-%m-%d %H:%M:%S')}")
with col3:
    show_interval = st.checkbox("90% CI", value=True, help="Show/hide 90% prediction interval")
with col4:
    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()

st.divider()

# Calculate time windows
# Pivot = T-2h (where EPIAS data ends, forecast starts here)
pivot_time = turkey_now.floor('h') - pd.Timedelta(hours=EPIAS_DELAY_HOURS)
history_start = pivot_time - pd.Timedelta(hours=HISTORY_HOURS)
forecast_end = pivot_time + pd.Timedelta(hours=FORECAST_HOURS)

st.info(f"""
**Time Windows (Dynamic - updates every hour):**
- **Actual Data:** {history_start.strftime('%Y-%m-%d %H:%M')} → {pivot_time.strftime('%Y-%m-%d %H:%M')} (12h from EPIAS)
- **Pivot Point:** {pivot_time.strftime('%Y-%m-%d %H:%M')} (T-2h, last actual data)
- **Forecast:** {(pivot_time + pd.Timedelta(hours=1)).strftime('%Y-%m-%d %H:%M')} → {(pivot_time + pd.Timedelta(hours=24)).strftime('%Y-%m-%d %H:%M')} (CatBoost, 24 hours after pivot)
- **Current Time:** {turkey_now.strftime('%Y-%m-%d %H:%M')} (auto-refresh every 60s)
""")

# Fetch EPIAS data
with st.spinner("Fetching EPIAS consumption data..."):
    start_date = history_start.strftime('%Y-%m-%d')
    end_date = pivot_time.strftime('%Y-%m-%d')
    epias_df = fetch_epias_consumption(start_date, end_date)

if epias_df.empty or 'consumption' not in epias_df.columns:
    st.error("Failed to fetch EPIAS data. Check your credentials in .env")
    st.stop()

# Filter to our window (up to pivot point)
mask = (epias_df.index >= history_start) & (epias_df.index <= pivot_time)
actual_df = epias_df[mask].copy()

if actual_df.empty:
    st.warning(f"No EPIAS data found for {history_start} to {pivot_time}")
    st.stop()

st.success(f"Loaded {len(actual_df)} hours of actual consumption from EPIAS (up to {pivot_time.strftime('%H:%M')})")

# Load model and generate forecast
with st.spinner("Generating CatBoost forecast..."):
    model, features = load_catboost_model()

if model is None:
    st.error("Could not load CatBoost model. Check model path.")
    st.stop()

# Generate forecast
forecast_df = generate_forecast(
    consumption_history=actual_df['consumption'],
    pivot_time=pivot_time,
    model=model,
    features=features,
    hours=FORECAST_HOURS
)

st.success(f"Generated {len(forecast_df)} hours of CatBoost forecast")

# Combine data for chart
actual_records = pd.DataFrame({
    'timestamp': actual_df.index,
    'value': actual_df['consumption'].values,
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
    avg_actual = actual_df['consumption'].mean()
    st.metric("Avg Actual (12h)", f"{avg_actual:,.0f} MWh")

with col2:
    avg_forecast = forecast_df['value'].mean()
    st.metric("Avg Forecast (24h)", f"{avg_forecast:,.0f} MWh")

with col3:
    peak_val = actual_df['consumption'].max()
    peak_time = actual_df['consumption'].idxmax()
    st.metric("Peak Actual", f"{peak_val:,.0f} MWh", f"at {peak_time.strftime('%H:%M')}")

with col4:
    peak_idx = forecast_df['value'].idxmax()
    peak_val = forecast_df.loc[peak_idx, 'value']
    peak_time = forecast_df.loc[peak_idx, 'timestamp']
    st.metric("Peak Forecast", f"{peak_val:,.0f} MWh", f"at {peak_time.strftime('%H:%M')}")

# Data tables
with st.expander("View Data Tables"):
    tab1, tab2 = st.tabs(["Actual (EPIAS)", "Forecast (CatBoost)"])

    with tab1:
        st.dataframe(
            actual_df[['consumption']].reset_index().rename(columns={'datetime': 'Time', 'consumption': 'Consumption (MWh)'}),
            use_container_width=True, hide_index=True
        )

    with tab2:
        st.dataframe(
            forecast_df[['timestamp', 'value', 'lower', 'upper']].rename(columns={
                'timestamp': 'Time', 'value': 'Forecast', 'lower': 'Lower', 'upper': 'Upper'
            }),
            use_container_width=True, hide_index=True
        )

# Footer
st.divider()
st.caption(f"Model: {BEST_MODEL} | EPIAS delay: ~{EPIAS_DELAY_HOURS}h | Pivot: {pivot_time.strftime('%Y-%m-%d %H:%M')}")
