"""
Section 2.2: Feature Engineering and Master Data
=================================================
Generates Section 2.2 of the ForeWatt Technical Report covering
feature engineering pipeline and master dataset construction.

Author: ForeWatt Team
Date: January 2026
"""

import os
import sys

# Add Report directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils_docx import (
    create_document,
    add_heading,
    add_paragraph,
    add_figure,
    add_table,
    save_section,
    FigureCounter,
    TableCounter,
    REPORT_DIR
)


def generate_section(doc, figure_counter, table_counter):
    """
    Generate Section 2.2: Feature Engineering and Master Data.

    Args:
        doc: Document object to add content to
        figure_counter: FigureCounter instance for figure numbering
        table_counter: TableCounter instance for table numbering

    Returns:
        tuple: (doc, figure_counter, table_counter)
    """

    # Section heading
    add_heading(doc, "2.2 Feature Engineering and Master Data", level=2)

    # Introduction paragraph
    intro_text = (
        "The transformation of raw data into predictive features represents a critical stage "
        "in the ForeWatt pipeline. A comprehensive feature engineering framework was developed "
        "to extract meaningful signals from heterogeneous data sources, including historical "
        "consumption patterns, electricity market dynamics, meteorological observations, and "
        "calendar information. The feature engineering pipeline is implemented through a modular "
        "architecture consisting of specialized generators for lag features, rolling window "
        "statistics, calendar encodings, and weather-derived indicators. These components "
        "operate on data organized according to a medallion architecture, processing information "
        "from the Silver layer (cleaned and normalized) to produce Gold layer assets "
        "(machine learning-ready features)."
    )
    add_paragraph(doc, intro_text)

    # Lag features paragraph
    lag_text = (
        "Lag features constitute the foundation of the autoregressive modeling approach "
        "employed in ForeWatt. For the consumption forecasting model, lags are computed at "
        "multiple temporal horizons to capture patterns operating at different time scales. "
        "Short-term lags at 1, 2, 3, and 6 hours capture immediate consumption inertia and "
        "autoregressive dependencies that reflect the physical constraints of electricity "
        "demand. The 12-hour and 24-hour lags capture diurnal patterns and day-over-day "
        "changes, while the 48-hour and 168-hour (weekly) lags encode longer-term seasonal "
        "cycles. Temperature lag features are similarly constructed at 1, 2, 3, 24, and 168 "
        "hour intervals to account for the delayed response of electricity demand to weather "
        "conditions. Price lag features at 24-hour and 168-hour horizons provide economic "
        "context by reflecting historical market conditions."
    )
    add_paragraph(doc, lag_text)

    # Rolling features paragraph
    rolling_text = (
        "Rolling window statistics extend the feature set by computing aggregate measures "
        "over sliding windows of 24 hours and 168 hours. For each window size, the mean, "
        "standard deviation, minimum, and maximum values are calculated for consumption, "
        "temperature, and price variables. These statistics capture both the central tendency "
        "and variability of the underlying processes. Derived features are computed from "
        "these rolling statistics, including the consumption range (maximum minus minimum) "
        "as a measure of daily volatility, the coefficient of variation (standard deviation "
        "divided by mean) as a normalized variability indicator, and the diurnal temperature "
        "range to characterize weather stability. The rolling standard deviation of price "
        "over 24 hours serves as a proxy for market volatility, which is particularly "
        "important for price prediction during periods of supply uncertainty."
    )
    add_paragraph(doc, rolling_text)

    # Calendar features paragraph
    calendar_text = (
        "Calendar features encode temporal structure and institutional patterns that "
        "influence electricity consumption and prices. Basic temporal features include "
        "day of week, day of month, month, and week of year. Binary indicators distinguish "
        "weekends from weekdays, reflecting the systematic reduction in commercial and "
        "industrial demand during non-working days. Holiday features are constructed from "
        "a comprehensive calendar of Turkish official holidays, religious observances, and "
        "half-day holidays. The holiday encoding includes both day-level and hour-level "
        "flags, with special handling for half-day holidays where morning hours may exhibit "
        "normal demand patterns. Cyclical encodings using sine and cosine transformations "
        "are applied to day of week and month features to preserve the periodic nature of "
        "these variables and avoid artificial discontinuities at period boundaries."
    )
    add_paragraph(doc, calendar_text)

    # Weather features paragraph
    weather_text = (
        "Weather features are derived from temperature, humidity, and wind speed observations "
        "collected from twelve major Turkish cities. A population-weighted national temperature "
        "index is computed to represent aggregate weather conditions across the country. "
        "Heating Degree Days (HDD) and Cooling Degree Days (CDD) are calculated using "
        "standard formulas with base temperatures calibrated for Turkish climate conditions. "
        "The heat index combines temperature and humidity to characterize perceived "
        "temperature during summer months when air conditioning loads are significant. "
        "Binary indicators flag extreme temperature conditions (hot and cold thresholds) "
        "that trigger non-linear demand responses. These weather features capture the "
        "thermosensitive component of electricity demand, which can account for substantial "
        "variation during extreme weather events."
    )
    add_paragraph(doc, weather_text)

    # Add figure
    fig_num = figure_counter.next()
    add_figure(
        doc,
        os.path.join("DesignFigures", "18_ml_model_architecture.jpg"),
        f"Figure {fig_num}. Machine learning model architecture showing feature flow and model ensemble",
        width_inches=5.5
    )

    # Price model features paragraph
    price_features_text = (
        "The price forecasting model employs a distinct feature set designed to capture "
        "market dynamics and supply-demand fundamentals. Market signal features include "
        "thermal gap, which measures the difference between thermal generation capacity "
        "and actual thermal output, indicating the availability of dispatchable generation. "
        "Renewable saturation captures the proportion of demand met by variable renewable "
        "sources, which influences price volatility through merit order effects. The spark "
        "spread proxy approximates the profitability of gas-fired generation using historical "
        "price relationships. System short signals indicate periods when scheduled generation "
        "may be insufficient to meet forecasted demand, typically associated with price spikes. "
        "Load factor and reserve margin ratio characterize system utilization and capacity "
        "adequacy, respectively."
    )
    add_paragraph(doc, price_features_text)

    # Profile evolution features paragraph
    profile_text = (
        "Profile evolution features capture the time-varying nature of hourly price patterns. "
        "The daily average price and hourly ratio relative to the daily average encode the "
        "typical shape of the daily price curve. Profile features computed over 14-day and "
        "28-day windows capture the recent evolution of these patterns, while profile momentum "
        "measures the rate of change in the profile shape. Solar-specific features account "
        "for the growing penetration of photovoltaic generation in Turkey, including solar "
        "ratio, solar profiles, and price-solar interaction terms. These features enable "
        "the model to adapt to structural changes in the price formation process as the "
        "generation mix evolves."
    )
    add_paragraph(doc, profile_text)

    # Consumption features table
    table_num_consumption = table_counter.next()
    consumption_headers = ["Feature Name", "Description"]
    consumption_data = [
        ["consumption_lag_24h", "Consumption value from 24 hours prior"],
        ["consumption_lag_48h", "Consumption value from 48 hours prior"],
        ["consumption_lag_168h", "Consumption value from one week prior"],
        ["consumption_rolling_mean_24h", "24-hour rolling mean of consumption"],
        ["consumption_rolling_std_24h", "24-hour rolling standard deviation"],
        ["temp_national", "Population-weighted national temperature"],
        ["humidity_national", "Population-weighted national humidity"],
        ["HDD", "Heating Degree Days"],
        ["CDD", "Cooling Degree Days"],
        ["heat_index", "Combined temperature-humidity index"],
        ["is_hot", "Binary indicator for hot conditions"],
        ["is_cold", "Binary indicator for cold conditions"],
        ["temp_lag_24h", "Temperature from 24 hours prior"],
        ["hour_sin, hour_cos", "Cyclical encoding of hour of day"],
        ["dow_sin, dow_cos", "Cyclical encoding of day of week"],
        ["month_sin, month_cos", "Cyclical encoding of month"],
        ["is_weekend", "Binary weekend indicator"],
        ["is_holiday_day", "Binary day-level holiday indicator"],
        ["is_holiday_hour", "Binary hour-level holiday indicator"],
        ["price_ptf_lag_24h", "Day-ahead price from 24 hours prior"],
    ]
    add_table(
        doc,
        consumption_data,
        consumption_headers,
        caption=f"Table {table_num_consumption}. Consumption Model Features (23 total)"
    )

    # Price features table
    table_num_price = table_counter.next()
    price_headers = ["Feature Name", "Description"]
    price_data = [
        ["hour_sin, hour_cos", "Cyclical encoding of hour of day"],
        ["dow_sin, dow_cos", "Cyclical encoding of day of week"],
        ["is_weekend", "Binary weekend indicator"],
        ["price_ptf_rolling_std_24h", "24-hour rolling price volatility"],
        ["price_ptf_rolling_mean_24h", "24-hour rolling price average"],
        ["price_ptf_rolling_min_24h", "24-hour rolling price minimum"],
        ["price_ptf_rolling_max_24h", "24-hour rolling price maximum"],
        ["price_ptf_lag_24h", "Day-ahead price from 24 hours prior"],
        ["price_ptf_lag_168h", "Day-ahead price from one week prior"],
        ["thermal_gap", "Thermal capacity minus thermal generation"],
        ["thermal_gap_lag_24h", "Thermal gap from 24 hours prior"],
        ["renewable_saturation", "Renewable generation as fraction of demand"],
        ["spark_spread_proxy_lag_24h", "Historical spark spread proxy"],
        ["system_short_signal", "Binary indicator for system shortage"],
        ["load_factor", "Actual demand divided by peak capacity"],
        ["consumption_forecast", "Forecasted consumption for target hour"],
        ["reserve_margin_ratio", "Available reserve as fraction of demand"],
        ["price_volatility_lag24h", "Lagged price volatility measure"],
        ["realtime_premium_lag24h", "Lagged real-time price premium"],
        ["hour", "Hour of day (integer)"],
        ["daily_avg_price", "Daily average price level"],
        ["hourly_ratio", "Current hour price to daily average ratio"],
        ["profile_14d, profile_28d", "Hourly price profiles over rolling windows"],
        ["profile_momentum", "Rate of change in price profile"],
        ["daily_avg_momentum", "Rate of change in daily average price"],
        ["solar_ratio", "Solar generation as fraction of capacity"],
        ["solar_profile_14d, solar_profile_28d", "Solar generation profiles"],
        ["solar_momentum", "Rate of change in solar profile"],
        ["price_solar_interaction", "Interaction between price and solar features"],
    ]
    add_table(
        doc,
        price_data,
        price_headers,
        caption=f"Table {table_num_price}. Price Model Features (33 total)"
    )

    # Master dataset paragraph
    master_text = (
        "The master dataset is produced by merging all feature components through a "
        "systematic process implemented in the feature merger module. Features from "
        "different sources are aligned on timestamp and joined using inner merges to "
        "ensure temporal consistency. The resulting master dataset is stored in the "
        "Gold layer at data/gold/master/ in Apache Parquet format for efficient storage "
        "and retrieval. The current production version (v2) contains approximately 52.5 "
        "megabytes of data spanning from January 2020 to October 2025, comprising over "
        "50,000 hourly observations with all engineered features. A versioning scheme "
        "based on creation date and feature hash ensures reproducibility and enables "
        "tracking of dataset evolution across model iterations. Accompanying metadata "
        "files in JSON format document the feature list, data date range, missing value "
        "statistics, and creation timestamp for each version of the master dataset."
    )
    add_paragraph(doc, master_text)

    return doc, figure_counter, table_counter


if __name__ == "__main__":
    # Standalone testing
    print("Generating Section 2.2: Feature Engineering and Master Data...")

    # Create document and counters
    doc = create_document()
    figure_counter = FigureCounter(start=2)  # Assuming Figure 1 was in previous section
    table_counter = TableCounter(start=1)

    # Generate section
    doc, figure_counter, table_counter = generate_section(doc, figure_counter, table_counter)

    # Save standalone section
    output_path = save_section(doc, "section_2_2_features.docx")

    print(f"\nSection 2.2 generated successfully!")
    print(f"Output: {output_path}")
    print(f"Figures used: {figure_counter.current() - 2}")
    print(f"Tables used: {table_counter.current() - 1}")
