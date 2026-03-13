FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-install-project

# Copy project files
COPY src/ src/
COPY main.py .

# Create directories for mounting and runtime data
RUN mkdir -p data output models logs visualizations

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV UV_CACHE_DIR=/tmp/uv-cache

# Entrypoint
ENTRYPOINT ["uv", "run", "python", "main.py", "--data_dir", "data", "--output_dir", "output", "--model_size", "nano", "--log_dir", "logs", "--viz_dir", "visualizations"]
