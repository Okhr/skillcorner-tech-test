# SkillCorner Video Inference Technical Test

This project implements a video inference pipeline for football match analysis. It identifies players and the ball in broadcast footage at a target of 10 FPS, providing detailed metrics, visualizations, and automated model management.

## Core Features

- **Continuous Processing**: Automatically watches the `data/` directory for new videos and processes them in real-time. Skip already processed videos based on output existence.
- **Monitoring Stack**: Integrated Prometheus and Grafana for real-time performance and inference analytics.
- **File Hashing**: Uses file hashes as `video_id` for consistent tracking across runs.
- **Automated Model Management**: Automatically downloads RF-DETR models from Hugging Face based on the selected size.
- **Dynamic Configuration**: Infers input dimensions and normalization parameters directly from model metadata and `preprocessor_config.json`.
- **Efficient Decoding**: Uses `PyAV` for low-latency frame extraction and precise FPS sampling.
- **Inference Optimization**: Leverages `ONNX Runtime` for fast execution with multi-provider support (CPU, CUDA, TensorRT).
- **Structured Logging**: Uses `structlog` for machine-readable JSON logs with `video_id` context. Every run automatically generates a timestamped log file in the `logs/` directory.
- **Data Export**: Generates results in `Parquet` format for efficient data analysis, one file per video.
- **Visualization**: Produces annotated frames using the `supervision` library, organized in video-specific subdirectories.
- **Containerized**: Fully Dockerized for production-grade reproducibility.

## Project Structure

```
.
├── src/
│   ├── config.py      # Centralized configuration & defaults
│   ├── video.py       # Optimized video decoding (PyAV)
│   ├── inference.py   # ONNX inference engine & model downloader
│   ├── pipeline.py    # Orchestration & visualization logic
│   ├── metrics.py     # Prometheus metrics definitions
│   └── logger.py      # Structured JSON logger setup
├── grafana/           # Automated dashboard provisioning
├── models/            # Automatically managed model storage
├── main.py            # CLI Entry point
├── Dockerfile         # Production-optimized container
├── docker-compose.yml # Full stack orchestration (App + Prometheus + Grafana)
├── prometheus.yml     # Scrape configuration
├── pyproject.toml     # Dependency management (uv)
└── data/              # Input video storage
```

## Observability Stack

The project includes a fully automated monitoring stack.
- **Prometheus**: Aggregates real-time performance (latency per step) and inference metrics (detection counts, spatial distributions).
- **Grafana**: Pre-configured with a "SkillCorner Inference Dashboard" for immediate visualization.

Access locally via Docker Compose:
- **Grafana**: [http://localhost:3000](http://localhost:3000) (User: `admin`, Pass: `admin`)
- **Prometheus**: [http://localhost:9090](http://localhost:9090)

## Setup and Usage

### 1. Docker Compose (Preferred)
Use Docker Compose to run the inference service alongside the monitoring stack. This is the simplest way to get everything running with zero manual configuration.

```bash
docker compose up --build
```
Once started, access the following services:
- **Grafana (Dashboards)**: [http://localhost:3000](http://localhost:3000) (User: `admin`, Pass: `admin`)
- **Prometheus (Metrics)**: [http://localhost:9090](http://localhost:9090)

### 2. Local Execution
#### Prerequisites
- [uv](https://github.com/astral-sh/uv) installed.
- Python 3.12.

#### Steps
1. **Install dependencies**:
   ```bash
   uv sync
   ```
2. **Run the pipeline**:
   The model will be automatically downloaded on the first run.
   ```bash
   # Basic run
   uv run python main.py --model_size nano

   # Advanced run with custom logging frequency
   uv run python main.py --model_size small --fps 10 --log_freq 50
   ```

### Command Line Arguments
- `--data_dir`: Directory containing input videos (default: `data/`).
- `--model_size`: RF-DETR model size: `nano`, `small`, `base`, `medium`, `large` (default: `nano`).
- `--output_dir`: Directory for output parquet files (default: `output/`).
- `--log_dir`: Directory for log files (default: `logs/`).
- `--log_freq`: Logging frequency - log metrics every X frames (default: `10`).
- `--viz_dir`: Base directory for visualization frames (default: `visualizations/`).
- `--fps`: Target FPS for inference (default: `10`).
- `--conf`: Confidence threshold for detections (default: `0.5`).
- `--viz_freq`: Visualization frequency - save every X frames (default: `100`).

### Docker Execution (Standalone Container)
Mount your data and output folders to ensure results persist outside the container.

You can pull the prebuilt image from GitHub Container Registry:
```bash
docker pull ghcr.io/okhr/skillcorner-tech-test:latest
```

Or build the image locally:
```bash
docker build -t skillcorner-inference .
```

1. **Run with mounted volumes**:
   Using the prebuilt image:
   ```bash
   docker run -v $(pwd)/data:/data \
              -v $(pwd)/output:/output \
              -v $(pwd)/models:/models \
              -v $(pwd)/logs:/logs \
              -v $(pwd)/visualizations:/visualizations \
              ghcr.io/okhr/skillcorner-tech-test:latest --model_size nano
   ```

   Alternatively, using a locally built image:
   ```bash
   docker run -v $(pwd)/data:/data \
              -v $(pwd)/output:/output \
              -v $(pwd)/models:/models \
              -v $(pwd)/logs:/logs \
              -v $(pwd)/visualizations:/visualizations \
              skillcorner-inference --model_size nano
   ```

## Design Choices

1. **Modular Architecture**: Separated responsibilities into clean modules (`video`, `inference`, `pipeline`) for better testability and maintenance.
2. **Centralized Configuration**: All parameters are managed in `config.py`, making it easy to swap models or adjust performance targets.
3. **ONNX Runtime**: Chosen for its high performance and broad hardware compatibility without the weight of full ML frameworks.
4. **Parquet Format**: Used for output to support large-scale data handling and efficient storage.
5. **Supervision Library**: Employed to ensure professional and accurate visualizations.
6. **Observability**: Real-time tracking using Prometheus Histograms for latency (loading, inference, post-processing) and detection analytics. A pre-provisioned Grafana dashboard provides visual insights into pitch density and performance bottlenecks.
7. **Idempotency**: The pipeline skips videos that already have a corresponding parquet file in the output directory.
