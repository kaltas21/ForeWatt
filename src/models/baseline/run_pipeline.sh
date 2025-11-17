#!/bin/bash
# Quick Start Script for Baseline Pipeline
# ========================================

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                     ForeWatt Baseline Pipeline                             ║"
echo "║            Demand & Price Forecasting with Feature Selection               ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "Project Root: $PROJECT_ROOT"
echo ""

# Parse arguments
RUN_MODE="${1:-all}"

case "$RUN_MODE" in
    "all")
        echo "Running FULL pipeline: Both targets, all models"
        python "$SCRIPT_DIR/pipeline_runner.py"
        ;;

    "demand")
        echo "Running pipeline for DEMAND forecasting only"
        python "$SCRIPT_DIR/pipeline_runner.py" --targets consumption
        ;;

    "price")
        echo "Running pipeline for PRICE forecasting only"
        python "$SCRIPT_DIR/pipeline_runner.py" --targets price_real
        ;;

    "boosting")
        echo "Running BOOSTING models only (both targets)"
        python "$SCRIPT_DIR/pipeline_runner.py" --models catboost xgboost lightgbm
        ;;

    "statistical")
        echo "Running STATISTICAL models only (both targets)"
        python "$SCRIPT_DIR/pipeline_runner.py" --models prophet sarimax
        ;;

    "quick")
        echo "Running QUICK test: Demand with CatBoost only"
        python "$SCRIPT_DIR/pipeline_runner.py" --targets consumption --models catboost
        ;;

    *)
        echo "Usage: $0 [all|demand|price|boosting|statistical|quick]"
        echo ""
        echo "Options:"
        echo "  all         - Both targets, all models (default)"
        echo "  demand      - Demand forecasting only"
        echo "  price       - Price forecasting only"
        echo "  boosting    - Boosting models only (CatBoost, XGBoost, LightGBM)"
        echo "  statistical - Statistical models only (Prophet, SARIMAX)"
        echo "  quick       - Quick test (Demand + CatBoost)"
        echo ""
        exit 1
        ;;
esac

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                           Pipeline Complete!                                ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Results saved to: $PROJECT_ROOT/reports/baseline/"
echo "📈 MLflow UI: mlflow ui --backend-store-uri $PROJECT_ROOT/mlruns"
echo ""
