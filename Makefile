# GeoTree Project Makefile
.PHONY: install dev-server dev-web dev build-web docker-build docker-run clean test

install:
	@echo "Installing Python requirements..."
	pip install -r requirements.txt
	@echo "Installing Web Frontend packages..."
	cd web/frontend && pnpm install

dev:
	@echo "Starting GeoTree Backend (port 8080) & React Vite Frontend (port 5173)..."
	@$(MAKE) -j2 dev-server dev-web

dev-server:
	@echo "Starting GeoTree Backend API Server on port 8080..."
	python3 -m uvicorn web.backend.server:app --host 0.0.0.0 --port 8080 --reload

dev-web:
	@echo "Starting GeoTree React Vite Dev Server..."
	cd web/frontend && pnpm dev

build-web:
	@echo "Building React Vite production bundle..."
	cd web/frontend && pnpm run build

docker-build:
	@echo "Building GeoTree Docker image..."
	docker build -t geotree:latest .

docker-run:
	@echo "Running GeoTree Docker container on port 8080..."
	docker run -p 8080:8000 geotree:latest

test:
	@echo "Running test suite..."
	python3 -m pytest tests/
