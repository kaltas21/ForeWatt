#!/bin/bash
# ForeWatt Dashboard Launcher
# Quick script to start the Streamlit dashboard

echo "🚀 Starting ForeWatt Dashboard..."
echo ""
echo "Dashboard will be available at: http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""

# Navigate to dashboard directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Installing dependencies..."
    source venv/bin/activate
    pip install -r requirements.txt
else
    # Activate virtual environment
    echo "✅ Activating virtual environment..."
    source venv/bin/activate
fi

# Check if data exists
if [ ! -f "../data/gold/master/master_v2_fundamental.csv" ]; then
    echo "⚠️  Warning: Master data file not found"
    echo "   Expected: ../data/gold/master/master_v2_fundamental.csv"
    echo ""
else
    echo "✅ Master data file found (master_v2_fundamental.csv)"
fi

# Check if experiment results exist
if [ ! -f "../reports/new_experiment/baseline/results.csv" ]; then
    echo "⚠️  Warning: Baseline experiment results not found"
    echo "   Expected: ../reports/new_experiment/baseline/results.csv"
    echo ""
else
    echo "✅ Baseline experiment results found"
fi

if [ ! -f "../reports/new_experiment/deeplearning/results.csv" ]; then
    echo "⚠️  Warning: Deep learning experiment results not found"
    echo "   Expected: ../reports/new_experiment/deeplearning/results.csv"
    echo ""
else
    echo "✅ Deep learning experiment results found"
fi

# Launch Streamlit
echo "✅ Launching dashboard..."
streamlit run app.py
