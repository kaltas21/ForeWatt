# ForeWatt Dashboard

Interactive web dashboard for visualizing electricity demand forecasts, model performance, and data exploration.

## Features

### 🏠 Home Page
- Project overview and key statistics
- Model performance rankings
- Recent consumption trends
- System status monitoring

### 📈 Forecast Page
- Generate 1-24 hour ahead predictions
- Multi-model comparison
- 90% prediction intervals
- Interactive visualizations
- Forecast data export

### 🔬 Model Comparison
- Side-by-side model comparison
- Performance metrics (MAE, RMSE, MASE, sMAPE, R²)
- Horizon-wise analysis
- Skill scores vs baseline
- Model strengths & weaknesses

### 📊 Data Explorer
- Historical consumption patterns
- Hourly and daily pattern analysis
- Seasonal trends
- Feature correlation analysis
- 106 engineered features exploration
- Data quality monitoring

### 📉 Performance Monitor
- Historical performance tracking
- Error distribution analysis
- Rolling metric visualization
- Error breakdown by time factors
- Backtesting results
- Performance reports

## Quick Start

### Prerequisites
- Python 3.11+
- Trained models in `../mlruns/` directory
- Master dataset in `../data/gold/master/` directory

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the dashboard:
```bash
streamlit run app.py
```

3. Open your browser to `http://localhost:8501`

### Docker Deployment

```bash
# Build the image
docker build -t forewatt-dashboard .

# Run the container
docker run -p 8501:8501 -v $(pwd)/../data:/app/data -v $(pwd)/../mlruns:/app/mlruns forewatt-dashboard
```

## Project Structure

```
dashboard/
├── app.py                  # Home page (main entry point)
├── pages/                  # Multi-page app pages
│   ├── 1_📈_Forecast.py
│   ├── 2_🔬_Model_Comparison.py
│   ├── 3_📊_Data_Explorer.py
│   └── 4_📉_Performance.py
├── utils/                  # Shared utilities
│   ├── __init__.py
│   ├── config.py          # Configuration and constants
│   ├── data_loader.py     # Data loading utilities
│   ├── model_loader.py    # MLflow model loading
│   ├── metrics.py         # Metrics calculation
│   └── plotting.py        # Plotly visualizations
├── .streamlit/
│   └── config.toml        # Streamlit configuration
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Configuration

### Streamlit Theme

Edit `.streamlit/config.toml` to customize colors and appearance:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f8f9fa"
textColor = "#212529"
```

### Data Paths

Update paths in `utils/config.py`:

```python
# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
MASTER_DATA = GOLD_DIR / "master" / "master_v1_2025-11-12_a567fe49.parquet"

# Model paths
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
```

## Usage Guide

### Generating Forecasts

1. Navigate to **📈 Forecast** page
2. Select one or more models from the sidebar
3. Set forecast horizon (1-24 hours)
4. Choose evaluation period
5. Click **Generate Forecast**
6. View results, metrics, and residual analysis
7. Download predictions as CSV

### Comparing Models

1. Navigate to **🔬 Model Comparison** page
2. View overall performance metrics
3. Compare models across different metrics
4. Analyze horizon-wise performance
5. Check skill scores vs baseline
6. Export comparison results

### Exploring Data

1. Navigate to **📊 Data Explorer** page
2. Select time period to analyze
3. View consumption patterns (hourly, daily, seasonal)
4. Explore feature groups and correlations
5. Check data quality statistics
6. Export filtered data

### Monitoring Performance

1. Navigate to **📉 Performance Monitor** page
2. Select model and analysis period
3. View error distribution and trends
4. Analyze performance by time factors
5. Review backtesting summary
6. Download performance report

## Technical Details

### Model Loading

Models are loaded from MLflow tracking server located at `../mlruns/`. The dashboard automatically:
- Discovers available experiments
- Loads best model runs based on MAE
- Caches models for performance
- Falls back to simulation if models unavailable

### Data Loading

Data is loaded from the gold layer (`../data/gold/master/`):
- Parquet format for efficiency
- Cached with Streamlit's `@st.cache_data`
- Automatic datetime index handling
- Timezone-aware timestamps (Europe/Istanbul)
- Train/val/test split (2020-2022 train, 2023 val, 2024 test)

### Real-Time Data Updates

The dashboard reads from static parquet files by default. For real-time data flow:

**Option 1: Manual Refresh**
- Click **"🔄 Refresh Data"** button in the sidebar
- This clears the cache and reloads data from files
- Best for periodic manual updates

**Option 2: Auto-Refresh**
Add to your Streamlit code:
```python
# In config.toml
[server]
fileWatcherType = "auto"  # Auto-reload on file changes
```

**Option 3: InfluxDB Integration**
For continuous real-time data:
1. Start InfluxDB service: `docker-compose up influxdb`
2. Modify `data_loader.py` to query InfluxDB
3. Add polling mechanism with `st.rerun()` timer
```python
import time
import streamlit as st

# Add to top of page
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()

# Auto-refresh every 5 minutes
if time.time() - st.session_state.last_update > 300:
    st.cache_data.clear()
    st.rerun()
```

**Option 4: Scheduled Data Pipeline**
Set up cron job to regenerate master dataset:
```bash
# Update data every hour
0 * * * * cd /path/to/ForeWatt && python src/features/merge_features.py
```

### Performance Optimization

- **Caching**: All data and model loading is cached
- **Lazy loading**: Pages load data only when accessed
- **Sample limits**: Large datasets are sampled for visualization
- **Efficient formats**: Parquet for data, MLflow for models

## Troubleshooting

### "No data available"
- Ensure `../data/gold/master/master_v1_2025-11-12_a567fe49.parquet` exists
- Check file permissions
- Verify DATA_DIR path in config.py

### "No model metrics available"
- Ensure MLflow experiments are in `../mlruns/`
- Check MLRUNS_DIR path in config.py
- Models will fall back to simulation mode

### Import errors
- Run `pip install -r requirements.txt`
- Ensure Python 3.11+ is installed
- Check virtual environment activation

### Slow performance
- Reduce date range in filters
- Clear Streamlit cache (press 'C' in browser)
- Limit number of features visualized
- Use smaller test sets for analysis

## Development

### Adding New Pages

1. Create file in `pages/` directory with format: `N_emoji_PageName.py`
2. Import utilities from `utils/`
3. Use consistent page configuration
4. Add to navigation guide in Home page

### Adding New Utilities

1. Add functions to appropriate utility module
2. Update `utils/__init__.py` exports
3. Document function parameters and returns
4. Use Streamlit caching decorators

### Customizing Plots

All plots use utilities from `plotting.py`:
- `create_forecast_plot()` - Time series with intervals
- `create_metrics_comparison()` - Bar charts
- `create_time_series_plot()` - Multi-line plots
- `create_correlation_heatmap()` - Heatmaps
- And more...

## Contributing

This dashboard is part of the ForeWatt project (Koç University COMP 491 Fall 2025).

**Team:**
- Zeynep Öykü Aslan
- Kaan Altaş
- Zeliha Paycı

## License

Open source - see main project LICENSE file.

## Version

Dashboard v1.0 - January 2025
