"""
Section 2.1: Data Collection and Data System
=============================================
Generates Section 2.1 of the ForeWatt Technical Report.

This section covers the data collection infrastructure, including:
- EPIAS (Turkish Electricity Market) data via eptr2 library
- Open-Meteo Weather API integration
- EVDS (Central Bank) macroeconomic indicators
- Medallion data architecture (Bronze, Silver, Gold layers)
- Data validation and quality assurance processes

Author: ForeWatt Team
Date: January 2026
"""

import os
import sys

# Add Report directory to path for imports
REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPORT_DIR)

from utils_docx import (
    create_document,
    add_heading,
    add_paragraph,
    add_figure,
    save_section,
    FigureCounter
)


def generate_section(doc, figure_counter):
    """
    Generate Section 2.1: Data Collection and Data System.

    Args:
        doc: Document object from python-docx
        figure_counter: FigureCounter instance for figure numbering

    Returns:
        doc: Modified document with section content added
    """

    # Section heading
    add_heading(doc, "2.1 Data Collection and Data System", level=2)

    # Introduction paragraph
    intro_text = (
        "The ForeWatt forecasting platform was built upon a comprehensive data collection "
        "infrastructure designed to capture the multifaceted factors influencing electricity "
        "consumption and prices in the Turkish market. The data system integrates information "
        "from three primary sources: the Turkish Electricity Market Transparency Platform "
        "(EPIAS), the Open-Meteo Weather API, and the Electronic Data Delivery System (EVDS) "
        "of the Central Bank of the Republic of Turkey. This section describes the data "
        "sources, collection methodologies, and the medallion architecture employed for "
        "data processing and storage."
    )
    add_paragraph(doc, intro_text)

    # EPIAS Data Source
    epias_text = (
        "The primary data source for electricity market variables was the EPIAS Transparency "
        "Platform, accessed through the eptr2 Python library with authenticated API credentials. "
        "The EPIAS data pipeline was designed to fetch twelve distinct datasets covering the "
        "full spectrum of market operations. Real-time consumption data, representing the actual "
        "hourly electricity load measured by TEIAS (Turkish Electricity Transmission Corporation), "
        "served as the target variable for consumption forecasting with values typically ranging "
        "between 20,000 and 50,000 MW. Day-ahead consumption forecasts published by TEIAS provided "
        "baseline predictions against which model improvements could be measured. Market price "
        "data included the Day-Ahead Market Clearing Price (PTF), the System Marginal Price (SMF) "
        "from the balancing power market, the Intraday Market weighted average price, and the "
        "overall Weighted Average Price across all market segments. Generation data encompassed "
        "real-time generation by source and plant, available capacity declarations (EAK), day-ahead "
        "generation plans (KGUP), and bilateral contract plans (KUDUP). Wind generation forecasts "
        "from the RITM system and hydroelectric reservoir status from DSI (State Hydraulic Works) "
        "completed the EPIAS data collection, providing supply-side fundamentals critical for "
        "price forecasting."
    )
    add_paragraph(doc, epias_text)

    # Weather Data Source
    weather_text = (
        "Meteorological data was sourced from the Open-Meteo Weather API, which provides free "
        "access to historical and forecast weather information with hourly resolution. A demand-side "
        "weather pipeline was implemented to fetch data for the ten largest Turkish cities by "
        "population, collectively representing approximately 49 percent of the national population. "
        "The selected cities included Istanbul with a population weight of 18.3 percent, Ankara "
        "at 6.9 percent, Izmir at 5.2 percent, Bursa at 3.8 percent, Antalya at 3.2 percent, "
        "Konya at 2.7 percent, Adana at 2.7 percent, Sanliurfa at 2.6 percent, Gaziantep at 2.6 "
        "percent, and Kocaeli at 2.5 percent. Eight weather variables were collected for each "
        "city: temperature at 2 meters height, relative humidity, precipitation amount, rainfall "
        "specifically, cloud cover percentage, wind speed at 10 meters, surface atmospheric "
        "pressure, and apparent temperature representing the perceived temperature accounting "
        "for humidity and wind effects. Population-weighted national aggregates were computed "
        "from city-level data to create representative weather features for the entire country. "
        "The API implementation incorporated automatic request caching and exponential backoff "
        "retry logic to handle rate limiting, with a 7-second delay between city requests to "
        "respect API usage policies."
    )
    add_paragraph(doc, weather_text)

    # EVDS Macroeconomic Data
    evds_text = (
        "Macroeconomic indicators were obtained from the EVDS system operated by the Central Bank "
        "of the Republic of Turkey. Given that Turkey imports approximately 70 percent of its "
        "energy requirements, primarily natural gas and oil, exchange rate fluctuations have a "
        "substantial impact on electricity generation costs and market prices. The fetched series "
        "included the Consumer Price Index (TUFE) and Producer Price Index (UFE) for inflation "
        "monitoring, the M2 money supply as a monetary policy indicator, and Turkish Lira deposit "
        "interest rates. Additionally, daily foreign exchange rates were collected including "
        "USD/TRY, EUR/TRY, and gold prices in Turkish Lira (XAU/TRY). Derived features were "
        "computed from the raw exchange rate data, including a weighted FX basket calculated as "
        "the simple average of USD and EUR rates, 7-day and 30-day momentum indicators representing "
        "the rate of change, and a 30-day rolling standard deviation capturing exchange rate "
        "volatility. Daily FX data was converted to hourly frequency through forward-fill "
        "replication to enable alignment with the hourly electricity market data."
    )
    add_paragraph(doc, evds_text)

    # Medallion Architecture
    medallion_text = (
        "Data management followed the medallion architecture pattern, organizing data into three "
        "progressively refined layers: Bronze, Silver, and Gold. The Bronze layer stored raw data "
        "exactly as received from source APIs, preserving the original format and enabling "
        "reproducibility. Separate subdirectories maintained data from each source including "
        "EPIAS market data, demand-side weather observations, macroeconomic indicators, and "
        "calendar information. The Silver layer contained cleaned and validated data after "
        "applying quality assurance transformations. Validation procedures included duplicate "
        "record removal, timestamp chronological ordering verification, range validation to "
        "identify physically implausible values such as consumption below 10,000 MW or above "
        "60,000 MW, and flagging of columns with greater than 10 percent missing values. All "
        "datetime columns were standardized to the Europe/Istanbul timezone to ensure temporal "
        "consistency across datasets. The Gold layer housed feature-engineered datasets ready "
        "for model training, including lag features at 24-hour, 48-hour, and 168-hour intervals, "
        "rolling statistics such as means and standard deviations, and the master dataset that "
        "merged all data sources into a unified feature matrix."
    )
    add_paragraph(doc, medallion_text)

    # Add figure for medallion architecture
    fig_num = figure_counter.next()
    add_figure(
        doc,
        "DesignFigures/14_medallion_data_architecture.jpg",
        f"Figure {fig_num}. Medallion data architecture showing Bronze, Silver, and Gold layers "
        "with data flow from external sources through cleaning and feature engineering stages.",
        width_inches=5.5
    )

    # Storage and Format
    storage_text = (
        "All data was persisted in Apache Parquet format with Snappy compression, selected for "
        "its efficient columnar storage, fast read performance, and native support for complex "
        "data types including timezone-aware timestamps. Files were organized with monthly "
        "partitioning for forecast archives and date-range naming conventions for training "
        "datasets. Dual-format export to CSV was maintained as a secondary format for human "
        "readability and compatibility with spreadsheet applications. The complete master "
        "dataset spanning from January 2020 to October 2025 contained approximately 51,000 "
        "hourly records with 80 features, stored in a 52.5 megabyte Parquet file. The data "
        "pipeline supported incremental updates through skip-existing logic that detected "
        "previously downloaded data and avoided redundant API calls, significantly reducing "
        "fetch times for routine updates while ensuring data completeness for model retraining."
    )
    add_paragraph(doc, storage_text)

    return doc


if __name__ == "__main__":
    """
    Generate Section 2.1 as a standalone document for testing.
    """
    # Create document
    doc = create_document()

    # Initialize figure counter
    figure_counter = FigureCounter(start=1)

    # Generate section content
    doc = generate_section(doc, figure_counter)

    # Save document
    output_filename = "section_2_1_data_collection.docx"
    save_section(doc, output_filename)

    print(f"Section 2.1 generated successfully!")
    print(f"Output: Report/{output_filename}")
    print(f"Figures used: {figure_counter.current() - 1}")
