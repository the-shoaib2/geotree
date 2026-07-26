FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for GDAL/Rasterio
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir rasterio geopandas pyyaml shapely

COPY . .

ENTRYPOINT ["python3", "main.py"]
CMD ["preprocess"]
