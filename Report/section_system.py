"""
Section 2.5 (System Architecture) and Section 2.6 (Frontend & Dashboard)
for the ForeWatt Technical Report.

This script generates the system architecture and frontend sections describing
the cloud deployment, API design, and React dashboard implementation.
"""

import os
import sys

# Add Report directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils_docx import (
    create_document, add_heading, add_paragraph, add_figure,
    add_table, save_section, get_figure_path, FigureCounter
)


def generate_section(doc, figure_counter):
    """
    Generate Section 2.5 (System Architecture) and Section 2.6 (Frontend & Dashboard).

    Args:
        doc: python-docx Document object
        figure_counter: FigureCounter instance for consistent figure numbering

    Returns:
        doc: Updated Document object
    """

    # =========================================================================
    # Section 2.5: System Architecture
    # =========================================================================

    add_heading(doc, "2.5 System Architecture", level=2)

    # Introduction paragraph
    add_paragraph(doc, (
        "The ForeWatt platform was designed with a cloud-native architecture to ensure "
        "scalability, reliability, and cost-effectiveness for production deployment. The "
        "system architecture follows modern microservices principles while maintaining "
        "simplicity through a serverless deployment model. The backend API was built "
        "using FastAPI, a high-performance Python web framework, and is deployed on "
        "Google Cloud Run, which provides automatic scaling, pay-per-use pricing, and "
        "seamless integration with other Google Cloud services."
    ))

    # Complete system architecture figure
    fig_num = figure_counter.next()
    add_figure(
        doc,
        get_figure_path("DesignFigures", "05_complete_system_architecture.jpg"),
        f"Figure {fig_num}. Complete system architecture showing data flow from external sources through the ML pipeline to the frontend dashboard.",
        width_inches=5.5
    )

    # FastAPI application paragraph
    add_paragraph(doc, (
        "The FastAPI application serves as the central API gateway, exposing multiple "
        "endpoints for forecast generation, data retrieval, and system monitoring. The "
        "primary endpoints include POST /forecast for triggering hourly forecast generation, "
        "GET /api/realtime/{model} for fetching real-time forecasts from Firestore, "
        "GET /history/{model} for retrieving historical data from Parquet archives, and "
        "GET /api/anomaly/{model} for accessing anomaly detection results. Additional "
        "endpoints such as GET /api/aggregates/day-type/{model} and GET /api/aggregates/hourly/{model} "
        "provide pre-computed statistics for efficient dashboard loading. The API was designed "
        "with RESTful principles, utilizing query parameters for date filtering and pagination "
        "while returning JSON responses with consistent data structures."
    ))

    # Cloud deployment figure
    fig_num = figure_counter.next()
    add_figure(
        doc,
        get_figure_path("DesignFigures", "11_cloud_deployment_architecture.jpg"),
        f"Figure {fig_num}. Cloud deployment architecture on Google Cloud Platform showing the interaction between Cloud Run, Cloud Scheduler, Firestore, and Cloud Storage.",
        width_inches=5.5
    )

    # Cold start optimization paragraph
    add_paragraph(doc, (
        "To optimize cold start performance on Cloud Run, several strategies were implemented. "
        "The forecast pipeline and Firestore client were designed with lazy loading patterns, "
        "where expensive imports and model loading are deferred until the first request "
        "requires them. This approach reduces the initial container startup time from "
        "approximately 15 seconds to under 3 seconds. GZIP compression middleware was "
        "enabled with a minimum response size of 1000 bytes, achieving 80-90% reduction "
        "in response payload sizes for large forecast datasets. Cross-Origin Resource "
        "Sharing (CORS) middleware was configured to allow requests from the Firebase-hosted "
        "dashboard domain while maintaining security for production deployments."
    ))

    # Docker containerization paragraph
    add_paragraph(doc, (
        "The application was containerized using Docker with a Python 3.11-slim base image "
        "to minimize the container footprint while providing all necessary dependencies. "
        "The Dockerfile was structured to leverage layer caching by copying requirements.txt "
        "separately before the application code, ensuring that dependency installation is "
        "cached across builds. System dependencies including build-essential and libgomp1 "
        "were installed to support the CatBoost and LightGBM machine learning libraries. "
        "The trained model files were bundled directly into the container image to eliminate "
        "model loading latency during inference, with the models directory containing the "
        "CatBoost and LightGBM model artifacts for both price and consumption forecasting."
    ))

    # Hourly forecast sequence figure
    fig_num = figure_counter.next()
    add_figure(
        doc,
        get_figure_path("DesignFigures", "13_hourly_forecast_generation_sequence.jpg"),
        f"Figure {fig_num}. Sequence diagram illustrating the hourly forecast generation flow triggered by Cloud Scheduler.",
        width_inches=5.5
    )

    # Cloud Scheduler and Firestore paragraph
    add_paragraph(doc, (
        "Google Cloud Scheduler was configured to trigger the forecast pipeline every hour "
        "by sending a POST request to the /forecast endpoint. Upon receiving the trigger, "
        "the ForecastPipeline class orchestrates the complete forecasting workflow. The "
        "pipeline first loads the latest market data from the master Parquet file stored "
        "in Google Cloud Storage, then generates 24-hour ahead forecasts using both the "
        "price and consumption models. The forecast results were stored in multiple locations "
        "to optimize for different access patterns. Google Cloud Firestore was used for "
        "real-time data access, with forecasts stored under the forecasts/latest/{type}/current "
        "document path for immediate retrieval by the dashboard. Historical forecasts were "
        "archived in Parquet format organized by month (data/forecasts/{type}/{YYYY-MM}.parquet) "
        "for efficient batch processing and backtesting operations."
    ))

    # Storage architecture paragraph
    add_paragraph(doc, (
        "The storage architecture followed a medallion pattern with bronze, silver, and gold "
        "data layers. The bronze layer contained raw data fetched from EPIAS, Open-Meteo, and "
        "EVDS APIs. The silver layer stored cleaned and validated data with standardized "
        "schemas. The gold layer housed the feature-engineered master dataset used for model "
        "training and inference, stored as master_v2_fundamental.parquet with a total size "
        "of approximately 52.5 megabytes. The ForecastStorage class provided abstraction over "
        "the storage layer, implementing methods for saving forecasts with automatic monthly "
        "partitioning and retrieving historical forecasts with date range filtering. Query "
        "result caching with a 30-minute time-to-live was implemented to reduce redundant "
        "data fetching and improve API response times for frequently accessed endpoints."
    ))

    # Local development paragraph
    add_paragraph(doc, (
        "For local development and testing, a Docker Compose configuration was provided "
        "that orchestrates multiple services including the scheduler, API, dashboard, MLflow "
        "for experiment tracking, and InfluxDB for optional time-series storage. The scheduler "
        "service was configured to run the forecast pipeline at configurable intervals, while "
        "the MLflow service provided a user interface for tracking model experiments and "
        "comparing performance across different model versions. This development environment "
        "enabled rapid iteration on the forecasting models and API endpoints before deployment "
        "to the production Cloud Run environment."
    ))

    # =========================================================================
    # Section 2.6: Frontend & Dashboard
    # =========================================================================

    add_heading(doc, "2.6 Frontend & Dashboard", level=2)

    # Introduction paragraph
    add_paragraph(doc, (
        "The ForeWatt dashboard was developed as a modern single-page application using "
        "React 19 with TypeScript for type safety and improved developer experience. The "
        "build system was configured with Vite, a next-generation frontend tooling that "
        "provides fast hot module replacement during development and optimized production "
        "builds. Styling was implemented using Tailwind CSS, a utility-first CSS framework "
        "that enabled rapid UI development with consistent design patterns. The dashboard "
        "was deployed on Firebase Hosting, which provides global content delivery network "
        "distribution and automatic SSL certificate management."
    ))

    # Frontend architecture figure
    fig_num = figure_counter.next()
    add_figure(
        doc,
        get_figure_path("DesignFigures", "01_react_frontend_architecture.jpg"),
        f"Figure {fig_num}. React frontend architecture showing the component hierarchy and data flow patterns.",
        width_inches=5.5
    )

    # Dashboard views paragraph
    add_paragraph(doc, (
        "The dashboard was organized into six primary views accessible through a sidebar "
        "navigation component. The RealTime view displays the current 24-hour forecast with "
        "12 hours of historical actuals and 12 hours of predictions, visualized using "
        "interactive ECharts line charts with confidence interval bands. The Historical view "
        "enables exploration of past data with customizable date range selection and comparison "
        "between actual values and model forecasts. The Anomaly view presents detected anomalies "
        "using scatter plots that highlight data points with high anomaly scores based on "
        "statistical deviation from expected patterns. The Compare view facilitates period-over-period "
        "analysis through weekday versus weekend comparisons and hourly pattern visualizations. "
        "The Alerts view provides a management interface for system notifications and forecast "
        "threshold alerts, with role-based access control restricting certain features to "
        "administrator users."
    ))

    # Real-time dashboard screenshot
    fig_num = figure_counter.next()
    add_figure(
        doc,
        get_figure_path("DashboardScreenshots", "01_realtime_consumption_forecast.png"),
        f"Figure {fig_num}. Real-time consumption forecast view showing the 24-hour horizon with actual data, predictions, and confidence intervals.",
        width_inches=5.5
    )

    # UI features paragraph
    add_paragraph(doc, (
        "The user interface was designed with a glassmorphism aesthetic featuring "
        "semi-transparent panels with backdrop blur effects, providing a modern and "
        "professional appearance. A theme toggle was implemented to support both dark and "
        "light modes, with the dark mode set as the default for reduced eye strain during "
        "extended monitoring sessions. Bilingual support was provided through a LanguageContext "
        "provider that manages translations between English and Turkish, enabling seamless "
        "language switching without page reload. The interface was designed to be fully "
        "responsive, adapting layouts for desktop, tablet, and mobile screen sizes through "
        "Tailwind CSS responsive utility classes."
    ))

    # Historical view screenshot
    fig_num = figure_counter.next()
    add_figure(
        doc,
        get_figure_path("DashboardScreenshots", "02_historical_analysis_consumption.png"),
        f"Figure {fig_num}. Historical analysis view displaying consumption data over a selected time period with actual versus forecast comparison.",
        width_inches=5.5
    )

    # AI Chatbot paragraph
    add_paragraph(doc, (
        "An AI-powered chatbot was integrated into the dashboard using the Google Gemini "
        "2.0 Flash model through the @google/genai SDK. The chatbot was configured with a "
        "domain-specific system prompt that establishes context about the ForeWatt platform, "
        "EPIAS data characteristics, and typical electricity consumption and price patterns "
        "in Turkey. When the chat session is initiated, current dashboard context including "
        "the active model type, latest actual values, forecast predictions, and summary "
        "statistics is injected into the conversation to enable contextual responses. The "
        "chatbot interface supports markdown rendering with syntax-highlighted code blocks "
        "through the react-markdown library with remark-gfm plugin for GitHub Flavored Markdown "
        "support. A fallback mock response system was implemented to provide basic functionality "
        "when the Gemini API key is not configured."
    ))

    # AI chatbot screenshot
    fig_num = figure_counter.next()
    add_figure(
        doc,
        get_figure_path("DashboardScreenshots", "07_realtime_price_forecast_with_ai_chatbot.png"),
        f"Figure {fig_num}. Real-time price forecast view with the AI chatbot panel open, demonstrating the contextual conversation capabilities.",
        width_inches=5.5
    )

    # Visualization and data management paragraph
    add_paragraph(doc, (
        "Data visualization was implemented using ECharts, a powerful charting library that "
        "provides interactive features including zooming, panning, and data point tooltips. "
        "The RealTimeChart component renders actual data as solid lines and forecast data as "
        "dashed lines, with a pivot marker indicating the transition point between historical "
        "actuals and predictions. Confidence interval bands were rendered using stacked area "
        "series with semi-transparent fills and dashed border lines. The charts automatically "
        "adapt to theme changes through a MutationObserver that monitors the document's dark "
        "mode class and reinitializes charts with appropriate color schemes. State management "
        "was handled through React hooks with Context providers for global state including "
        "authentication status, language preferences, and current theme. API data fetching "
        "was implemented with automatic refresh intervals of 60 seconds to keep the real-time "
        "view updated with the latest forecast data."
    ))

    return doc


if __name__ == "__main__":
    # Create standalone document for this section
    doc = create_document()
    figure_counter = FigureCounter(start=15)  # Continue from previous sections

    # Generate the section
    doc = generate_section(doc, figure_counter)

    # Save the document
    output_file = "section_2_5_2_6_system_frontend.docx"
    save_section(doc, output_file)

    print(f"\nSection 2.5 (System Architecture) and Section 2.6 (Frontend & Dashboard) generated successfully!")
    print(f"Output: {output_file}")
    print(f"Figures used: {figure_counter.current() - 15}")
