#!/bin/bash
################################################################################
# GPU Monitoring Script
# Continuously monitors GPU usage during training
################################################################################

echo "████████████████████████████████████████████████████████████████████████████████"
echo "GPU Monitor (Press Ctrl+C to stop)"
echo "████████████████████████████████████████████████████████████████████████████████"
echo ""

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found"
    exit 1
fi

# Monitor in a loop
while true; do
    clear
    echo "████████████████████████████████████████████████████████████████████████████████"
    echo "ForeWatt GPU Monitor - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "████████████████████████████████████████████████████████████████████████████████"
    echo ""

    # GPU info
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
        --format=csv,noheader,nounits | while IFS=',' read -r idx name temp gpu_util mem_util mem_used mem_total power; do
        echo "GPU $idx: $name"
        echo "  Temperature: ${temp}°C"
        echo "  GPU Utilization: ${gpu_util}%"
        echo "  Memory Utilization: ${mem_util}%"
        echo "  Memory: ${mem_used}MB / ${mem_total}MB"
        echo "  Power: ${power}W"
        echo ""
    done

    # Running processes
    echo "Running Processes:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits | while IFS=',' read -r pid name mem; do
        echo "  PID $pid: $name (${mem}MB)"
    done

    # Training status
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Training Status:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Check for running Python processes
    if pgrep -f "grid_search_runner.py" > /dev/null; then
        echo "✓ Training is running"

        # Show last log line from grid search
        LAST_LOG=$(find reports/*/logs/grid_search_run_*.log -type f 2>/dev/null | tail -n 1)
        if [ -n "$LAST_LOG" ]; then
            echo ""
            echo "Latest log entry:"
            tail -n 3 "$LAST_LOG"
        fi
    else
        echo "✗ No training process detected"
    fi

    echo ""
    echo "Updating every 10 seconds... (Press Ctrl+C to stop)"

    sleep 10
done
