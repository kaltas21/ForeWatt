"""
Section 2.4: Current Price and Consumption Models' Experiments and Results

This script generates Section 2.4 of the ForeWatt Technical Report, covering
the experimental methodology, ablation studies, and performance results for
both the consumption and price forecasting models.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils_docx import (
    create_document,
    add_heading,
    add_paragraph,
    add_paragraph_with_runs,
    add_figure,
    add_table,
    add_page_break,
    save_section,
    get_figure_path,
    FigureCounter,
    TableCounter,
    REPORT_DIR
)


def generate_section(doc, figure_counter, table_counter):
    """
    Generate Section 2.4: Current Price and Consumption Models' Experiments and Results.

    Args:
        doc: Document object
        figure_counter: FigureCounter instance for figure numbering
        table_counter: TableCounter instance for table numbering

    Returns:
        tuple: (doc, figure_counter, table_counter) with updated counters
    """

    # Section 2.4 Main Heading
    add_heading(doc, "2.4 Current Price and Consumption Models' Experiments and Results", level=2)

    # Introduction paragraph
    intro_text = (
        "This section presents the comprehensive experimental evaluation conducted to develop and optimize "
        "the ForeWatt forecasting models. A systematic approach was employed to investigate the impact of "
        "training data volume, feature engineering decisions, model architecture choices, and hyperparameter "
        "configurations on prediction accuracy. The experiments were conducted using a rigorous train-validation-test "
        "split methodology, with the test period spanning from June 2024 to October 2025, comprising 12,429 hours "
        "of unseen data. The symmetric Mean Absolute Percentage Error (sMAPE) was selected as the primary evaluation "
        "metric due to its bounded nature and stability when dealing with values approaching zero, which is particularly "
        "relevant for electricity price forecasting where near-zero prices occasionally occur."
    )
    add_paragraph(doc, intro_text)

    # 2.4.1 Consumption Model Experiments
    add_heading(doc, "2.4.1 Consumption Model Experiments", level=3)

    # Data size impact
    data_size_text = (
        "The first experimental dimension investigated was the impact of training data volume on consumption "
        "prediction accuracy. Four configurations were evaluated, ranging from one year to four years of historical "
        "data. The results demonstrated a clear pattern of diminishing returns as additional training data was incorporated. "
        "Training with one year of data (2023) yielded a sMAPE of 2.27%, which improved to 2.16% with two years of data. "
        "The inclusion of a third year further reduced the error to 2.08%, while extending to four years of training data "
        "produced only a marginal improvement to 2.07%. This analysis revealed that consumption patterns in the Turkish "
        "electricity market are relatively stable over time, with three to four years of historical data providing "
        "sufficient coverage of seasonal variations without introducing excessive noise from outdated patterns."
    )
    add_paragraph(doc, data_size_text)

    # Add data size impact figure for consumption
    fig_num = figure_counter.next()
    add_figure(
        doc,
        get_figure_path("ExperimentPlots", "consumption_data_size_impact.png"),
        f"Figure {fig_num}. Impact of training data size on consumption model performance.",
        width_inches=5.0
    )

    # Feature ablation study
    feature_ablation_text = (
        "A feature ablation study was conducted to understand the contribution of each feature group to the overall "
        "prediction accuracy. The consumption model utilizes 23 features organized into three primary groups: lag-based "
        "features capturing historical consumption patterns, weather-related features encoding meteorological conditions, "
        "and calendar features representing temporal patterns. When evaluated in isolation, lag features alone achieved "
        "a sMAPE of 3.03%, demonstrating their fundamental importance in capturing the autoregressive nature of electricity "
        "consumption. Weather features alone proved insufficient, yielding a sMAPE of 12.95%, while calendar features alone "
        "achieved 10.61%. The most significant finding was that combining lag and calendar features produced the best "
        "performance at 1.96% sMAPE, outperforming the complete feature set (2.07%). This counterintuitive result suggests "
        "that weather information may already be implicitly encoded within the historical consumption patterns, and its "
        "explicit inclusion introduces additional noise that marginally degrades prediction accuracy."
    )
    add_paragraph(doc, feature_ablation_text)

    # Add feature importance figure for consumption
    fig_num = figure_counter.next()
    add_figure(
        doc,
        get_figure_path("ExperimentPlots", "consumption_feature_importance.png"),
        f"Figure {fig_num}. Feature importance ranking for the consumption forecasting model.",
        width_inches=5.5
    )

    # Hyperparameter tuning
    hyperparameter_text = (
        "Hyperparameter optimization was performed on the CatBoost model to identify the optimal configuration. "
        "The baseline model employed a learning rate of 0.03, tree depth of 5, and 1,000 iterations. Various "
        "configurations were systematically evaluated, including adjustments to tree depth (3, 5, and 8), learning "
        "rate (0.01, 0.03, and 0.10), number of iterations (1,000 and 2,000), and regularization parameters. "
        "The experiments revealed that a higher learning rate of 0.10 achieved the best performance, reducing "
        "sMAPE from 2.07% to 1.95%. Deeper trees (depth 8) provided slight improvements but at the cost of "
        "increased computational complexity, while shallower trees (depth 3) exhibited underfitting with a sMAPE "
        "of 2.33%. The final optimized consumption model achieved a sMAPE of 1.95%, corresponding to a Mean Absolute "
        "Error (MAE) of 808.5 MWh and an R-squared value of 0.969, indicating that the model explains 96.9% of the "
        "variance in actual consumption values."
    )
    add_paragraph(doc, hyperparameter_text)

    # Add predictions vs actual figure for consumption
    fig_num = figure_counter.next()
    add_figure(
        doc,
        get_figure_path("ExperimentPlots", "consumption_predictions_vs_actual.png"),
        f"Figure {fig_num}. Consumption model predictions versus actual values over the test period.",
        width_inches=5.5
    )

    # 2.4.2 Price Model Experiments
    add_heading(doc, "2.4.2 Price Model Experiments", level=3)

    # Data size impact for price
    price_data_text = (
        "The price model exhibited different characteristics regarding training data requirements compared to the "
        "consumption model. Unlike consumption patterns, which demonstrated diminishing returns after three years, "
        "price prediction continued to benefit from additional historical data across all tested configurations. "
        "Training with one year of data produced a sMAPE of 12.34%, which unexpectedly increased slightly to 12.43% "
        "with two years, suggesting that the two-year period may have captured an anomalous market regime. However, "
        "extending to three years reduced the error to 11.82%, and the full four-year dataset achieved the best "
        "performance at 11.71% sMAPE. This finding indicates that electricity prices in the Turkish market exhibit "
        "longer-term dependencies and regime changes that require extended historical context to model effectively."
    )
    add_paragraph(doc, price_data_text)

    # Model architecture comparison
    architecture_text = (
        "A comprehensive evaluation of model architectures was conducted to determine the optimal approach for price "
        "forecasting. Three configurations were compared: CatBoost alone, LightGBM alone, and an ensemble combining "
        "both algorithms. CatBoost achieved a sMAPE of 11.94%, demonstrating its strength as a single model, while "
        "LightGBM produced a sMAPE of 12.46% with significantly faster training times. The ensemble approach, which "
        "combines predictions from both models using optimized weights of 61.4% CatBoost and 38.6% LightGBM, achieved "
        "the best performance at 11.71% sMAPE. The ensemble's superiority can be attributed to the complementary "
        "error patterns of the two algorithms, with each model capturing different aspects of the price dynamics, "
        "thereby reducing the overall prediction variance when combined."
    )
    add_paragraph(doc, architecture_text)

    # Add model comparison figure
    fig_num = figure_counter.next()
    add_figure(
        doc,
        get_figure_path("ExperimentPlots", "price_model_comparison.png"),
        f"Figure {fig_num}. Comparison of price model architectures: CatBoost, LightGBM, and Ensemble.",
        width_inches=5.0
    )

    # Error correction methods
    error_correction_text = (
        "A novel error correction framework was developed and evaluated to address systematic prediction biases in the "
        "price model. Four configurations were compared: raw ensemble predictions without correction, Simple Adaptive "
        "Error Correction (AEC) that learns systematic biases from validation errors, K-Nearest Neighbors Error Correction "
        "(KNN-EC) that leverages similar historical contexts to estimate expected errors, and a hybrid approach combining "
        "both methods with equal weighting. The raw ensemble achieved a sMAPE of 12.58%, which was reduced to 11.95% "
        "with Simple AEC and 11.94% with KNN-EC. The hybrid approach, combining 50% Simple AEC and 50% KNN-EC corrections, "
        "achieved the best performance at 11.71% sMAPE, representing a 6.9% relative improvement over the uncorrected "
        "predictions. This hybrid methodology was designated as CHybrid V14 and was selected as the final production model."
    )
    add_paragraph(doc, error_correction_text)

    # Add CHybrid architecture figure
    fig_num = figure_counter.next()
    add_figure(
        doc,
        get_figure_path("DesignFigures", "04_chybrid_error_correction_architecture.jpg"),
        f"Figure {fig_num}. CHybrid error correction architecture combining Simple AEC and KNN-EC methods.",
        width_inches=5.5
    )

    # Add feature importance figure for price
    fig_num = figure_counter.next()
    add_figure(
        doc,
        get_figure_path("ExperimentPlots", "price_feature_importance.png"),
        f"Figure {fig_num}. Feature importance ranking for the price forecasting model.",
        width_inches=5.5
    )

    # Price feature analysis
    price_features_text = (
        "Unlike the consumption model where weather features degraded performance, the price model benefited from all "
        "33 features in its feature set. Feature ablation revealed that price lag features alone achieved a sMAPE of "
        "13.51%, while market signal features alone performed better at 12.84%, indicating the high predictive value "
        "of features such as thermal gap, renewable saturation, and spark spread proxy. Calendar features contributed "
        "a sMAPE of 13.00% when used in isolation. Combining price lags with market signals improved performance to "
        "12.13%, and the complete feature set achieved the best result at 11.71%. These findings highlight the importance "
        "of incorporating domain-specific market fundamentals for electricity price forecasting."
    )
    add_paragraph(doc, price_features_text)

    # Add predictions vs actual figure for price
    fig_num = figure_counter.next()
    add_figure(
        doc,
        get_figure_path("ExperimentPlots", "price_predictions_vs_actual.png"),
        f"Figure {fig_num}. Price model predictions versus actual values over the test period.",
        width_inches=5.5
    )

    # 2.4.3 Baseline Comparisons
    add_heading(doc, "2.4.3 Baseline Comparisons", level=3)

    baseline_text = (
        "To establish the value of the machine learning approach, both models were compared against standard baseline "
        "methods commonly employed in time series forecasting. For consumption prediction, the naive baseline using "
        "the previous day's value at the same hour (Lag-24h) achieved a sMAPE of 5.69%, while the seasonal naive "
        "baseline using the previous week's value (Lag-168h) achieved 5.48%. Rolling mean and hourly mean baselines "
        "performed significantly worse at 15.22% and 13.37% respectively. The optimized consumption model at 1.95% "
        "sMAPE represents a 2.8-fold improvement over the best baseline. For price prediction, the naive baseline "
        "achieved 21.96% sMAPE and the seasonal naive achieved 21.60%, while rolling mean and hourly mean baselines "
        "yielded 25.38% and 42.16% respectively. The price model at 11.71% sMAPE represents a 1.8-fold improvement "
        "over the seasonal naive baseline. These results demonstrate the substantial value of the machine learning "
        "approach for both consumption and price forecasting in the Turkish electricity market."
    )
    add_paragraph(doc, baseline_text)

    # Add baseline comparison table
    tbl_num = table_counter.next()
    baseline_headers = ["Method", "Consumption sMAPE", "Price sMAPE", "Improvement Factor"]
    baseline_data = [
        ["Naive (Lag-24h)", "5.69%", "21.96%", "Baseline"],
        ["Seasonal Naive (Lag-168h)", "5.48%", "21.60%", "Baseline"],
        ["Rolling Mean (24h)", "15.22%", "25.38%", "-"],
        ["Hourly Mean", "13.37%", "42.16%", "-"],
        ["Our ML Model", "1.95%", "11.71%", "2.8x / 1.8x"]
    ]
    add_table(doc, baseline_data, baseline_headers, caption=f"Table {tbl_num}. Baseline comparison for consumption and price models.")

    # 2.4.4 Final Model Performance
    add_heading(doc, "2.4.4 Final Model Performance", level=3)

    final_performance_text = (
        "The final optimized models demonstrate excellent performance across all evaluation metrics. The consumption "
        "model achieves a test sMAPE of 1.95%, a MAE of 808.5 MWh, and an R-squared of 0.969, indicating highly accurate "
        "predictions with 96.9% of variance explained. The error distribution analysis reveals that 50% of predictions "
        "have an absolute error of 536.5 MWh or less, and 90% have an error of 1,825.7 MWh or less. The slight negative "
        "Mean Bias Error of -341.6 MWh indicates a minor tendency toward under-prediction. The price model achieves a "
        "test sMAPE of 11.71%, a MAE of 48.2 TL/MWh, and an R-squared of 0.871. Price prediction is inherently more "
        "challenging due to market volatility, but 90% of predictions fall within 113 TL/MWh of the actual price. "
        "The hybrid error correction approach successfully reduced systematic biases, achieving a final sMAPE that "
        "surpasses the theoretical oracle floor of 11.75%, demonstrating the effectiveness of the developed methodology."
    )
    add_paragraph(doc, final_performance_text)

    # Add consumption model metrics table
    tbl_num = table_counter.next()
    consumption_headers = ["Metric", "Value", "Interpretation"]
    consumption_data = [
        ["sMAPE", "1.95%", "Primary accuracy metric"],
        ["MAE", "808.5 MWh", "Average absolute error"],
        ["RMSE", "1,194.8 MWh", "Penalizes large errors"],
        ["MASE", "0.76", "Better than naive (< 1.0)"],
        ["R-squared", "0.969", "96.9% variance explained"],
        ["MBE", "-341.6 MWh", "Slight under-prediction"]
    ]
    add_table(doc, consumption_data, consumption_headers, caption=f"Table {tbl_num}. Final consumption model performance metrics.")

    # Add price model metrics table
    tbl_num = table_counter.next()
    price_headers = ["Metric", "Value", "Interpretation"]
    price_data = [
        ["sMAPE", "11.71%", "Primary accuracy metric"],
        ["MAE", "48.2 TL/MWh", "Average absolute error"],
        ["RMSE", "69.0 TL/MWh", "Penalizes large errors"],
        ["MASE", "0.36", "Better than naive (< 1.0)"],
        ["R-squared", "0.871", "87.1% variance explained"],
        ["MBE", "+2.5 TL/MWh", "Slight over-prediction"]
    ]
    add_table(doc, price_data, price_headers, caption=f"Table {tbl_num}. Final price model performance metrics.")

    # Best model summary table
    tbl_num = table_counter.next()
    summary_headers = ["Model", "Configuration", "Test sMAPE", "Test MAE", "Test R-squared"]
    summary_data = [
        ["Consumption", "CatBoost V2, lr=0.1", "1.95%", "808.5 MWh", "0.969"],
        ["Price", "CHybrid V14, 4yr data", "11.71%", "48.2 TL/MWh", "0.871"]
    ]
    add_table(doc, summary_data, summary_headers, caption=f"Table {tbl_num}. Best model performance summary.")

    # Summary paragraph
    summary_text = (
        "In summary, the experimental evaluation has yielded production-ready forecasting models for both electricity "
        "consumption and price prediction. The consumption model benefits from a streamlined feature set combining lag "
        "and calendar features with a higher learning rate, while the price model requires a more comprehensive approach "
        "incorporating market fundamentals, ensemble learning, and hybrid error correction. Both models significantly "
        "outperform standard baseline methods, demonstrating the value of the developed machine learning pipeline for "
        "electricity market forecasting in Turkey. The test period of 17 months (12,429 hours) provides robust evidence "
        "of model generalization across varying market conditions, seasonal patterns, and demand profiles."
    )
    add_paragraph(doc, summary_text)

    return doc, figure_counter, table_counter


if __name__ == "__main__":
    # Create standalone document for this section
    doc = create_document()

    # Initialize counters (assuming this is section 2.4, previous sections may have used some figures/tables)
    # For standalone testing, start from 1
    figure_counter = FigureCounter(start=1)
    table_counter = TableCounter(start=1)

    # Generate the section
    doc, figure_counter, table_counter = generate_section(doc, figure_counter, table_counter)

    # Save the document
    output_file = "section_2_4_experiments.docx"
    save_section(doc, output_file)

    print(f"\nSection 2.4 generated successfully!")
    print(f"Figures used: {figure_counter.current() - 1}")
    print(f"Tables used: {table_counter.current() - 1}")
