# ForeWatt API Dockerfile
# For Cloud Run deployment

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY main.py .
COPY src/ ./src/
COPY models/ ./models/

# Copy forecast parquet files (for validation/history view)
COPY data/forecasts/ ./data/forecasts/

# Create master data directory (master data loaded from GCS/EPIAS at runtime)
RUN mkdir -p data/gold/master

# Expose port
ENV PORT=8080
EXPOSE 8080

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
