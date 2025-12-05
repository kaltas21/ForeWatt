FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional scheduler dependencies
RUN pip install --no-cache-dir \
    apscheduler>=3.10.0 \
    influxdb-client>=1.38.0

# Copy source code
COPY src/ ./src/
COPY services/ ./services/

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python", "services/scheduler.py", "--interval", "60"]
