# Production Multi-Stage Dockerfile for GeoTree Web & AI Engine
FROM python:3.10-slim

# System dependencies for PyTorch, GDAL, Rasterio & OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    gdal-bin \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency requirements
COPY requirements.txt .

# Install Python requirements
RUN pip install --no-cache-dir -r requirements.txt uvicorn gunicorn

# Copy source repository
COPY . .

# Environment setup
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Expose Render default port
EXPOSE 8000

# Command to run production FastAPI app using Uvicorn
CMD ["sh", "-c", "uvicorn web.backend.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
