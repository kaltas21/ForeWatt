#!/bin/bash
# Wrapper script to run validation with correct venv Python

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found at .venv"
    echo "Please create it first: python3 -m venv .venv"
    exit 1
fi

# Use venv Python directly
echo "Using Python: .venv/bin/python"
.venv/bin/python --version

# Run validation
.venv/bin/python src/data/validate_deflation_pipeline.py "$@"
