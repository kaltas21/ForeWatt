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
    save_section, get_figure_path, FigureCounter
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
        "using FastAPI, a high-performance Python web framework that leverages asynchronous "
        "programming patterns for optimal throughput, and is deployed on Google Cloud Run, "
        "which provides automatic scaling, pay-per-use pricing, and seamless integration "
        "with other Google Cloud services. The architecture was designed to handle the "
        "complete lifecycle of electricity market forecasting, from data ingestion through "
        "external APIs to real-time forecast delivery to end users."
    ))

    # System context diagram
    fig_num = figure_counter.next()
    add_figure(
        doc,
        get_figure_path("DesignFigures", "15_system_context_diagram.jpg"),
        f"Figure {fig_num}. System context diagram illustrating the high-level interactions between ForeWatt and external systems including EPIAS, weather APIs, and end users.",
        width_inches=5.5
    )

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
        "provide pre-computed statistics for efficient dashboard loading, enabling rapid "
        "visualization of weekday versus weekend consumption patterns and hourly demand profiles. "
        "The API was designed with RESTful principles, utilizing query parameters for date filtering "
        "and pagination while returning JSON responses with consistent data structures that "
        "facilitate seamless integration with the React frontend."
    ))

    # FastAPI endpoint architecture
    fig_num = figure_counter.next()
    add_figure(
        doc,
        get_figure_path("DesignFigures", "17_fastapi_endpoint_architecture.jpg"),
        f"Figure {fig_num}. FastAPI endpoint architecture showing the routing structure and handler organization for forecast and data retrieval operations.",
        width_inches=5.5
    )

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
        "To optimize cold start performance on Cloud Run, several strategies were implemented "
        "to minimize container initialization latency. The forecast pipeline and Firestore client "
        "were designed with lazy loading patterns, where expensive imports and model loading are "
        "deferred until the first request requires them. This approach reduces the initial container "
        "startup time from approximately 15 seconds to under 3 seconds, significantly improving "
        "user experience for the first request after container scaling events. GZIP compression "
        "middleware was enabled with a minimum response size of 1000 bytes, achieving 80-90 percent "
        "reduction in response payload sizes for large forecast datasets containing multiple days "
        "of hourly predictions. Cross-Origin Resource Sharing (CORS) middleware was configured to "
        "allow requests from the Firebase-hosted dashboard domain while maintaining security for "
        "production deployments by rejecting unauthorized origins."
    ))

    # Backend services dataflow
    fig_num = figure_counter.next()
    add_figure(
        doc,
        get_figure_path("DesignFigures", "16_backend_services_dataflow.jpg"),
        f"Figure {fig_num}. Backend services dataflow diagram depicting the movement of data between API handlers, storage layers, and external data sources.",
        width_inches=5.5
    )

    # Docker containerization paragraph
    add_paragraph(doc, (
        "The application was containerized using Docker with a Python 3.11-slim base image "
        "to minimize the container footprint while providing all necessary dependencies for "
        "machine learning inference operations. The Dockerfile was structured to leverage layer "
        "caching by copying requirements.txt separately before the application code, ensuring "
        "that dependency installation is cached across builds and reducing deployment times "
        "during iterative development cycles. System dependencies including build-essential "
        "and libgomp1 were installed to support the CatBoost and LightGBM machine learning "
        "libraries, which require OpenMP for parallel tree evaluation. The trained model files "
        "were bundled directly into the container image to eliminate model loading latency during "
        "inference, with the models directory containing the CatBoost and LightGBM model artifacts "
        "for both price and consumption forecasting at a combined size of approximately 15 megabytes."
    ))

    # Hourly forecast sequence figure
    fig_num = figure_counter.next()
    add_figure(
        doc,
        get_figure_path("DesignFigures", "13_hourly_forecast_generation_sequence.jpg"),
        f"Figure {fig_num}. Sequence diagram illustrating the hourly forecast generation flow triggered by Cloud Scheduler.",
        width_inches=5.5
    )

    # Hourly forecast dataflow
    fig_num = figure_counter.next()
    add_figure(
        doc,
        get_figure_path("DesignFigures", "03_hourly_forecast_dataflow_sequence.jpg"),
        f"Figure {fig_num}. Detailed dataflow sequence for hourly forecast generation showing data fetching, feature engineering, and prediction steps.",
        width_inches=5.5
    )

    # Cloud Scheduler and Firestore paragraph
    add_paragraph(doc, (
        "Google Cloud Scheduler was configured to trigger the forecast pipeline every hour "
        "by sending a POST request to the /forecast endpoint with appropriate authentication "
        "headers. Upon receiving the trigger, the ForecastPipeline class orchestrates the "
        "complete forecasting workflow through a series of coordinated steps. The pipeline "
        "first loads the latest market data from the master Parquet file stored in Google "
        "Cloud Storage, then fetches real-time EPIAS data for the most recent hours to ensure "
        "predictions incorporate the latest market conditions. Feature engineering transformations "
        "are applied to construct the input vectors required by the trained models, and 24-hour "
        "ahead forecasts are generated using both the price and consumption models. The forecast "
        "results were stored in multiple locations to optimize for different access patterns. "
        "Google Cloud Firestore was used for real-time data access, with forecasts stored under "
        "the forecasts/latest/{type}/current document path for immediate retrieval by the "
        "dashboard with sub-millisecond read latency."
    ))

    # Real-time data retrieval sequence
    fig_num = figure_counter.next()
    add_figure(
        doc,
        get_figure_path("DesignFigures", "06_realtime_data_retrieval_sequence.jpg"),
        f"Figure {fig_num}. Sequence diagram for real-time data retrieval showing the interaction between frontend, API, and Firestore.",
        width_inches=5.5
    )

    # Multi-layer cache architecture
    fig_num = figure_counter.next()
    add_figure(
        doc,
        get_figure_path("DesignFigures", "12_multi_layer_cache_architecture_sequence.jpg"),
        f"Figure {fig_num}. Multi-layer caching architecture sequence showing cache lookup, population, and invalidation patterns.",
        width_inches=5.5
    )

    # Storage architecture paragraph
    add_paragraph(doc, (
        "The storage architecture followed a medallion pattern with bronze, silver, and gold "
        "data layers to ensure data quality and traceability throughout the processing pipeline. "
        "The bronze layer contained raw data fetched from EPIAS, Open-Meteo, and EVDS APIs in "
        "their original formats with timestamps recording ingestion time. The silver layer stored "
        "cleaned and validated data with standardized schemas, including null handling, outlier "
        "filtering, and timezone normalization to Europe/Istanbul. The gold layer housed the "
        "feature-engineered master dataset used for model training and inference, stored as "
        "master_v2_fundamental.parquet with a total size of approximately 52.5 megabytes "
        "containing over two years of hourly observations. Historical forecasts were archived "
        "in Parquet format organized by month using the pattern data/forecasts/{type}/{YYYY-MM}.parquet "
        "for efficient batch processing and backtesting operations. The ForecastStorage class "
        "provided abstraction over the storage layer, implementing methods for saving forecasts "
        "with automatic monthly partitioning and retrieving historical forecasts with date range "
        "filtering. Query result caching with a 30-minute time-to-live was implemented using "
        "an in-memory LRU cache to reduce redundant data fetching and improve API response times "
        "for frequently accessed endpoints such as the real-time forecast view."
    ))

    # Local development paragraph
    add_paragraph(doc, (
        "For local development and testing, a Docker Compose configuration was provided "
        "that orchestrates multiple services including the scheduler, API, dashboard, MLflow "
        "for experiment tracking, and InfluxDB for optional time-series storage with retention "
        "policies. The scheduler service was configured to run the forecast pipeline at "
        "configurable intervals specified through environment variables, enabling rapid iteration "
        "on the forecasting logic without requiring cloud deployment. The MLflow service provided "
        "a web interface for tracking model experiments and comparing performance across different "
        "model versions, storing artifacts locally or in cloud storage depending on configuration. "
        "This development environment enabled rapid iteration on the forecasting models and API "
        "endpoints before deployment to the production Cloud Run environment, with environment "
        "parity ensuring consistent behavior between local and production systems."
    ))

    # =========================================================================
    # Section 2.6: Frontend & Dashboard
    # =========================================================================

    add_heading(doc, "2.6 Frontend & Dashboard", level=2)

    # Introduction paragraph
    add_paragraph(doc, (
        "The ForeWatt dashboard was developed as a modern single-page application using "
        "React 19 with TypeScript for type safety and improved developer experience through "
        "compile-time error detection and enhanced IDE support. The build system was configured "
        "with Vite, a next-generation frontend tooling framework that provides fast hot module "
        "replacement during development with sub-second update times and optimized production "
        "builds with tree-shaking and code splitting. Styling was implemented using Tailwind CSS, "
        "a utility-first CSS framework that enabled rapid UI development with consistent design "
        "patterns through a predefined color palette and spacing scale. The dashboard was deployed "
        "on Firebase Hosting, which provides global content delivery network distribution across "
        "edge locations and automatic SSL certificate management for secure HTTPS connections."
    ))

    # Frontend architecture figure
    fig_num = figure_counter.next()
    add_figure(
        doc,
        get_figure_path("DesignFigures", "01_react_frontend_architecture.jpg"),
        f"Figure {fig_num}. React frontend architecture showing the component hierarchy, context providers, and data flow patterns.",
        width_inches=5.5
    )

    # Home View paragraph
    add_paragraph(doc, (
        "The Home view serves as the landing page of the ForeWatt dashboard, presenting users "
        "with a clean interface for selecting between the consumption and price forecasting models. "
        "Two prominently displayed cards provide model selection functionality, each featuring an "
        "icon representing the model type, a description of the forecasting capability, and the "
        "current best performance metric displayed in a highlighted section. The consumption model "
        "card displays the Mean Absolute Error of 892 MWh, while the price model card shows the "
        "corresponding MAE of 78.5 TL/MWh. Upon selection, users are automatically navigated to "
        "the RealTime view with the selected model pre-configured, streamlining the workflow for "
        "users who wish to immediately view current forecasts. The view was designed with responsive "
        "layouts that adapt to screen sizes ranging from mobile devices to large desktop monitors, "
        "with the two model cards transitioning from a stacked vertical arrangement on mobile to "
        "a side-by-side horizontal layout on larger screens."
    ))

    # RealTime View paragraph
    add_paragraph(doc, (
        "The RealTime view constitutes the primary interface for monitoring current forecasts, "
        "displaying a 24-hour horizon that combines 12 hours of historical actual data from EPIAS "
        "with 12 hours of model predictions. The view opens with an AI-generated insight card at "
        "the top, providing a natural language summary of current market conditions and notable "
        "forecast patterns generated through integration with the Gemini language model. Four key "
        "performance indicator cards display summary statistics including average actual value, "
        "average forecast value, peak actual with timestamp, and peak forecast with timestamp, "
        "enabling rapid assessment of demand levels and timing of peak events. The main interactive "
        "chart renders actual data as solid lines and forecast data as dashed lines, with a pivot "
        "marker clearly indicating the transition point between historical actuals and predictions. "
        "Confidence interval bands were rendered using stacked area series with semi-transparent "
        "fills and dashed border lines at the upper and lower bounds, which can be toggled on or "
        "off through a control switch. A tabular data section below the chart presents the same "
        "information in a sortable table format with columns for timestamp, data type, value, and "
        "confidence interval bounds, with options to copy data to clipboard or export as CSV format."
    ))

    # Real-time dashboard screenshot
    fig_num = figure_counter.next()
    add_figure(
        doc,
        get_figure_path("DashboardScreenshots", "01_realtime_consumption_forecast.png"),
        f"Figure {fig_num}. Real-time consumption forecast view showing the 24-hour horizon with actual data, predictions, and confidence intervals.",
        width_inches=5.5
    )

    # Historical View paragraph
    add_paragraph(doc, (
        "The Historical view enables exploration of past data with customizable date range "
        "selection, supporting both preset ranges and custom date inputs. Preset range options "
        "include Latest 3 Days, Latest 7 Days, Latest 15 Days, Latest 1 Month, Latest 3 Months, "
        "Latest 6 Months, and Latest 1 Year, with the default selection set to 7 days for optimal "
        "initial loading performance. Custom range selection was implemented through datetime-local "
        "input fields that allow users to specify precise start and end timestamps down to the "
        "minute level. The main visualization displays actual EPIAS data as solid lines and model "
        "forecasts as dashed lines, with toggle controls enabling users to show or hide either "
        "series independently for focused analysis. Summary statistics cards display key metrics "
        "including minimum, maximum, mean, standard deviation, and count for the selected period. "
        "Two supplementary charts provide additional analytical perspectives: an hourly patterns "
        "bar chart showing average values grouped by hour of day to reveal diurnal consumption "
        "cycles, and a value distribution histogram showing the frequency distribution of values "
        "with percentile markers at P25, P50, and P75 positions."
    ))

    # Historical view screenshot
    fig_num = figure_counter.next()
    add_figure(
        doc,
        get_figure_path("DashboardScreenshots", "02_historical_analysis_consumption.png"),
        f"Figure {fig_num}. Historical analysis view displaying consumption data over a selected time period with actual versus forecast comparison.",
        width_inches=5.5
    )

    # Anomaly View paragraph
    add_paragraph(doc, (
        "The Anomaly view presents statistical anomaly detection results for monitoring "
        "unusual patterns in electricity consumption and price data. The detection algorithm "
        "employs a configurable sigma threshold approach where data points exceeding the mean "
        "by more than the specified number of standard deviations are flagged as anomalies. "
        "Users can adjust the sensitivity through a dropdown selector offering 1.5x (high "
        "sensitivity), 2x (normal), 2.5x (low sensitivity), and 3x (very low) threshold options, "
        "enabling adaptation to different operational contexts and tolerance for false positives. "
        "Summary cards display the anomaly rate as a percentage, total anomaly count, maximum "
        "deviation observed in sigma units, and total data points analyzed. The main visualization "
        "renders a time series line chart with the actual data and horizontal dashed lines "
        "indicating the upper and lower bounds, while detected anomalies are highlighted as "
        "prominent scatter points with red coloring and shadow effects for visibility. A sidebar "
        "panel lists the top anomalies ranked by deviation magnitude, showing the timestamp, "
        "value, and sigma multiple for each detected event to facilitate investigation of "
        "specific incidents."
    ))

    # Anomaly view screenshot
    fig_num = figure_counter.next()
    add_figure(
        doc,
        get_figure_path("DashboardScreenshots", "03_anomaly_monitor_consumption.png"),
        f"Figure {fig_num}. Anomaly detection view displaying statistical anomalies in consumption data with configurable sensitivity threshold.",
        width_inches=5.5
    )

    # Compare View paragraph
    add_paragraph(doc, (
        "The Compare view facilitates period-over-period analysis through overlaid time series "
        "visualizations that enable identification of trends and pattern changes between comparable "
        "time periods. Three comparison presets were implemented: Day-over-Day comparing today "
        "versus yesterday, Week-over-Week comparing the current week against the previous week, "
        "and Month-over-Month comparing the current month against the prior month. Metric cards "
        "display comparative statistics including mean difference, peak difference, and volatility "
        "measurements for both periods, with percentage change indicators highlighting increases "
        "or decreases. The main overlay chart renders both time series on aligned axes, with the "
        "current period shown as a solid line with area fill and the comparison period as a dashed "
        "line without fill for visual distinction. A secondary day-type comparison chart compares "
        "weekday versus weekend consumption profiles using hourly average data, displaying the "
        "percentage difference between weekday and weekend means to quantify the day-type effect "
        "on demand patterns. A statistics summary table provides side-by-side comparison of "
        "minimum, maximum, and standard deviation values for both periods with calculated "
        "percentage differences."
    ))

    # Compare view screenshot
    fig_num = figure_counter.next()
    add_figure(
        doc,
        get_figure_path("DashboardScreenshots", "04_compare_periods_week_over_week.png"),
        f"Figure {fig_num}. Period comparison view showing week-over-week analysis with overlay charts and comparative statistics.",
        width_inches=5.5
    )

    # Alerts View paragraph
    add_paragraph(doc, (
        "The Alerts view provides a comprehensive management interface for configuring "
        "threshold-based alerts on forecast values, enabling proactive notification when "
        "predicted or actual values exceed operational limits. The configuration tab displays "
        "alert rules as cards, each containing the alert title, description, severity level "
        "(critical, warning, or info), enable/disable toggle, and editable threshold value "
        "with associated unit and comparison direction. Four default alert configurations were "
        "provided: Price Spike Alert triggering when PTF price exceeds 2200 TL/MWh, Negative/Low "
        "Price Alert triggering below 100 TL/MWh, Demand Surge Alert triggering above 48000 MWh "
        "consumption, and Low Consumption Alert triggering below 25000 MWh. Threshold values can "
        "be edited inline by clicking on the value and entering a new number, with save and "
        "cancel controls appearing for confirmation. The severity indicator is displayed as "
        "a colored badge with appropriate styling for critical (red), warning (amber), and "
        "info (blue) levels. Access to the Alerts view was restricted to administrator users "
        "through role-based access control implemented in the AuthContext provider."
    ))

    # Alerts configuration screenshot
    fig_num = figure_counter.next()
    add_figure(
        doc,
        get_figure_path("DashboardScreenshots", "05_alerts_configuration.png"),
        f"Figure {fig_num}. Alert configuration view showing threshold settings for price and consumption alerts with severity levels.",
        width_inches=5.5
    )

    # Alerts History paragraph
    add_paragraph(doc, (
        "The Alerts History tab provides a chronological log of triggered alerts, enabling "
        "operators to review past events and track system performance over time. Each alert "
        "entry displays the alert title, detailed message describing the trigger condition, "
        "severity badge, alert type classification, and timestamp in localized format. Unread "
        "alerts are visually distinguished with a subtle background highlight and an animated "
        "pulse indicator, ensuring new events are immediately noticeable. Action buttons allow "
        "users to mark individual alerts as read or delete them from the history log. A refresh "
        "button triggers re-evaluation of current data against configured thresholds to identify "
        "any new alerts that should be generated. The data status indicator shows the timestamp "
        "of the most recent data against which alerts were checked, providing transparency about "
        "the currency of alert evaluations. When no alerts have been triggered, a success state "
        "is displayed with a checkmark icon and reassuring message indicating all systems are "
        "operating within normal parameters."
    ))

    # Alerts history screenshot
    fig_num = figure_counter.next()
    add_figure(
        doc,
        get_figure_path("DashboardScreenshots", "06_alerts_history.png"),
        f"Figure {fig_num}. Alert history view displaying a log of triggered alerts with status tracking and management controls.",
        width_inches=5.5
    )

    # AI Chatbot paragraph
    add_paragraph(doc, (
        "An AI-powered chatbot was integrated into the dashboard using the Google Gemini "
        "2.0 Flash model through the @google/genai SDK, providing natural language interaction "
        "capabilities for data exploration and insight generation. The chatbot was configured "
        "with a domain-specific system prompt that establishes context about the ForeWatt platform, "
        "EPIAS data characteristics, typical electricity consumption patterns in Turkey, and "
        "the structure of available forecast data. When the chat session is initiated through "
        "the header button, current dashboard context including the active model type, latest "
        "actual values, forecast predictions, and summary statistics is automatically injected "
        "into the conversation to enable contextual responses that reference the specific data "
        "currently being viewed. The chatbot interface supports markdown rendering with syntax-"
        "highlighted code blocks through the react-markdown library with remark-gfm plugin for "
        "GitHub Flavored Markdown support, enabling formatted responses with tables, lists, and "
        "code examples. The chat panel can be expanded to full-screen mode for extended "
        "conversations, and a fallback mock response system was implemented to provide basic "
        "functionality when the Gemini API key is not configured in the environment."
    ))

    # AI chatbot screenshot
    fig_num = figure_counter.next()
    add_figure(
        doc,
        get_figure_path("DashboardScreenshots", "07_realtime_price_forecast_with_ai_chatbot.png"),
        f"Figure {fig_num}. Real-time price forecast view with the AI chatbot panel open, demonstrating contextual conversation capabilities.",
        width_inches=5.5
    )

    # UI features and visualization paragraph
    add_paragraph(doc, (
        "The user interface was designed with a glassmorphism aesthetic featuring "
        "semi-transparent panels with backdrop blur effects, providing a modern and "
        "professional appearance that differentiates ForeWatt from conventional data "
        "dashboards. A theme toggle was implemented to support both dark and light modes, "
        "with the dark mode set as the default for reduced eye strain during extended "
        "monitoring sessions in control room environments. Bilingual support was provided "
        "through a LanguageContext provider that manages translations between English and "
        "Turkish, enabling seamless language switching without page reload through React "
        "context-based state management. The interface was designed to be fully responsive, "
        "adapting layouts for desktop, tablet, and mobile screen sizes through Tailwind CSS "
        "responsive utility classes with breakpoints at 640px, 768px, 1024px, and 1280px."
    ))

    # Data visualization and state management paragraph
    add_paragraph(doc, (
        "Data visualization was implemented using ECharts, a powerful charting library that "
        "provides interactive features including zooming, panning, data point tooltips, and "
        "chart export capabilities through its built-in toolbox. The RealTimeChart component "
        "renders actual data as solid lines and forecast data as dashed lines, with a pivot "
        "marker indicating the transition point between historical actuals and predictions at "
        "the T-2h boundary accounting for EPIAS data publication delay. Confidence interval "
        "bands were rendered using stacked area series with semi-transparent fills and dashed "
        "border lines at the upper and lower confidence bounds. The charts automatically adapt "
        "to theme changes through a MutationObserver that monitors the document's dark mode "
        "class and reinitializes charts with appropriate color schemes when the theme toggles. "
        "State management was handled through React hooks with Context providers for global "
        "state including authentication status through AuthContext, language preferences through "
        "LanguageContext, and real-time data through prop drilling from the App component. API "
        "data fetching was implemented with automatic refresh intervals of 60 seconds to keep "
        "the real-time view updated with the latest forecast data from the backend API."
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
