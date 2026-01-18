"""
Section 2.3: Model Selection
ForeWatt Technical Report

This script generates Section 2.3 covering deep learning experiments,
overfitting analysis, and decision rationale for gradient boosting models.
"""

import os
import sys

# Add Report directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils_docx import (
    create_document,
    add_heading,
    add_paragraph,
    add_table,
    save_section,
    TableCounter
)


def generate_section(doc, table_counter):
    """
    Generate Section 2.3: Model Selection.

    Args:
        doc: Document object to add content to
        table_counter: TableCounter instance for table numbering

    Returns:
        tuple: (doc, table_counter) for chaining
    """

    # Section heading
    add_heading(doc, "2.3 Model Selection", level=2)

    # Paragraph 1: Introduction to deep learning experiments
    add_paragraph(
        doc,
        "A comprehensive evaluation of deep learning architectures was conducted "
        "to assess their suitability for electricity market forecasting in the Turkish context. "
        "A total of 90 experiments were performed across three state-of-the-art transformer-based "
        "models: Temporal Fusion Transformer (TFT), Neural Hierarchical Interpolation for Time "
        "Series (N-HITS), and PatchTST. These architectures were selected due to their documented "
        "success in various time series forecasting benchmarks and their ability to capture "
        "complex temporal dependencies. The experimental framework encompassed both price and "
        "consumption forecasting tasks, with hyperparameter optimization performed through "
        "systematic grid search across learning rates, hidden dimensions, attention heads, and "
        "dropout rates. The complete experimental suite required 1.67 hours of computation time "
        "on GPU-accelerated infrastructure."
    )

    # Paragraph 2: Results summary
    add_paragraph(
        doc,
        "The experimental results revealed significant performance variations across model "
        "architectures and forecasting targets. For electricity price forecasting, the N-HITS "
        "architecture achieved the best performance among deep learning models with a symmetric "
        "Mean Absolute Percentage Error (sMAPE) of 16.01% and a Mean Absolute Scaled Error (MASE) "
        "of 0.67. The TFT model followed closely with a sMAPE of 16.79%, while PatchTST exhibited "
        "the highest error at 17.17%. For consumption forecasting, the ranking differed notably, "
        "with TFT achieving the lowest sMAPE of 11.07%, followed by N-HITS at 12.22% and PatchTST "
        "at 12.40%. These results are summarized in Table " + str(table_counter.current()) + ", "
        "which presents the performance metrics for each deep learning architecture across both "
        "forecasting tasks."
    )

    # Table: Deep Learning Model Comparison
    table_num = table_counter.next()
    headers = ["Model", "Price sMAPE", "Consumption sMAPE", "Price MASE", "Consumption MASE"]
    data = [
        ["N-HITS", "16.01%", "12.22%", "0.67", "2.59"],
        ["TFT", "16.79%", "11.07%", "0.69", "2.33"],
        ["PatchTST", "17.17%", "12.40%", "0.72", "2.63"]
    ]
    add_table(
        doc,
        data,
        headers,
        caption=f"Table {table_num}. Deep Learning Model Performance Comparison"
    )

    # Paragraph 3: Overfitting analysis
    add_paragraph(
        doc,
        "A critical examination of the MASE values revealed substantial evidence of overfitting "
        "in the consumption forecasting models. The MASE metric, which compares model performance "
        "against a naive seasonal baseline, indicated that all deep learning models performed "
        "significantly worse than this simple benchmark for consumption prediction. Specifically, "
        "MASE values exceeding 1.0 signify that the model fails to outperform a naive forecast "
        "that simply repeats the value from the previous seasonal period. With consumption MASE "
        "values ranging from 2.33 to 2.63 across all architectures, the deep learning models "
        "demonstrated performance degradation of 133% to 163% compared to the naive baseline. "
        "This pattern strongly suggests that the models memorized training data patterns rather "
        "than learning generalizable temporal relationships."
    )

    # Paragraph 4: Data limitations
    add_paragraph(
        doc,
        "The observed overfitting behavior was attributed primarily to the limited volume of "
        "training data available for the Turkish electricity market. The dataset encompassed "
        "approximately four to five years of hourly observations, which, while substantial for "
        "traditional machine learning approaches, proved insufficient for the high-capacity "
        "deep learning architectures evaluated. Transformer-based models typically require "
        "extensive training corpora to effectively learn attention patterns and temporal "
        "dependencies without resorting to memorization. The significant gap between training "
        "and test performance metrics, particularly for consumption forecasting, confirmed "
        "that the models were unable to generalize beyond the training distribution. Furthermore, "
        "the relatively stable consumption patterns in the training period did not adequately "
        "prepare the models for the variability encountered in the test set."
    )

    # Paragraph 5: Decision rationale for gradient boosting - part 1
    add_paragraph(
        doc,
        "Based on the comprehensive analysis of deep learning performance limitations, gradient "
        "boosting methods were selected as the primary modeling approach for the ForeWatt platform. "
        "The CatBoost algorithm was adopted for consumption forecasting, achieving a sMAPE of "
        "1.95% on the test set, which represents a 5.7-fold improvement over the best deep learning "
        "result of 11.07% obtained by TFT. This dramatic performance enhancement demonstrated "
        "the superior ability of gradient boosting to extract predictive signals from limited "
        "training data while maintaining robust generalization to unseen periods. The ensemble "
        "nature of gradient boosting, which combines numerous weak learners through sequential "
        "optimization, provided an effective regularization mechanism that prevented the "
        "overfitting observed in deep learning approaches."
    )

    # Paragraph 6: Decision rationale for gradient boosting - part 2
    add_paragraph(
        doc,
        "For price forecasting, a hybrid ensemble combining CatBoost and LightGBM was implemented, "
        "achieving a sMAPE of 11.71% compared to the N-HITS result of 16.01%, representing a "
        "27% relative improvement. The hybrid architecture leveraged the complementary strengths "
        "of both gradient boosting frameworks, with CatBoost providing robust handling of "
        "categorical features and LightGBM contributing efficient leaf-wise tree growth. Beyond "
        "raw predictive accuracy, the gradient boosting approach offered several practical "
        "advantages critical for production deployment. Training time was reduced from hours "
        "to minutes, enabling rapid model iteration and retraining as new data became available. "
        "Inference latency decreased substantially, supporting the real-time forecasting "
        "requirements of the hourly prediction pipeline."
    )

    # Paragraph 7: Interpretability and conclusion
    add_paragraph(
        doc,
        "An additional consideration in the model selection decision was the interpretability "
        "of gradient boosting methods compared to deep learning architectures. The feature "
        "importance scores generated by CatBoost and LightGBM provided transparent insights "
        "into the factors driving price and consumption predictions, facilitating model validation "
        "by domain experts and enabling targeted feature engineering improvements. This "
        "interpretability was deemed essential for building trust with stakeholders in the "
        "Turkish electricity market and for diagnosing prediction errors in production. "
        "The combination of superior predictive performance, computational efficiency, and "
        "model interpretability established gradient boosting as the optimal approach for "
        "the ForeWatt forecasting platform, with deep learning architectures reserved for "
        "potential future investigation as training data volumes increase."
    )

    return doc, table_counter


if __name__ == "__main__":
    # Standalone test
    print("Generating Section 2.3: Model Selection...")

    # Create document and counter
    doc = create_document()
    table_counter = TableCounter(start=1)

    # Generate section
    doc, table_counter = generate_section(doc, table_counter)

    # Save document
    output_file = "section_2_3_model_selection.docx"
    save_section(doc, output_file)

    print(f"Section 2.3 generated successfully!")
    print(f"Tables used: {table_counter.current() - 1}")
