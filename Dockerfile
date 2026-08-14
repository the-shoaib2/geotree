# ── Stage 1: Build Frontend (Node.js & Vite) ──
FROM node:20-slim AS frontend-builder
WORKDIR /app/web/frontend

# Install pnpm and copy package definitions
RUN npm install -g pnpm
COPY web/frontend/package.json web/frontend/pnpm-lock.yaml* ./
RUN pnpm install --frozen-lockfile || pnpm install

# Copy frontend code and compile React Vite distribution bundle
COPY web/frontend ./
RUN pnpm run build

# ── Stage 2: Production Backend & Combined Runner (Python 3.10) ──
FROM python:3.10-slim AS runner

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

# Copy built static frontend assets from Stage 1 into web/frontend/dist
COPY --from=frontend-builder /app/web/frontend/dist /app/web/frontend/dist

# Environment setup
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV PYTHONPATH=/app

# Expose Render port
EXPOSE 8000

# Command to run production FastAPI app using Uvicorn
CMD ["sh", "-c", "uvicorn web.backend.server:app --host 0.0.0.0 --port ${PORT:-8000}"]

