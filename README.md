# SkillCorner Video Inference Technical Test

This project implements a video inference pipeline for football match analysis. It identifies players and the ball in broadcast footage at a target of 10 FPS, providing detailed metrics, visualizations, and automated model management.

## Core Features

- **Batch Processing**: Automatically processes all videos in the `data/` directory. Skip already processed videos based on output existence.
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
│   └── logger.py      # Structured JSON logger setup
├── models/            # Automatically managed model storage
├── main.py            # CLI Entry point
├── Dockerfile         # Production-optimized container
├── pyproject.toml     # Dependency management (uv)
└── data/              # Input video storage
```

## Setup and Usage

### Prerequisites
- [uv](https://github.com/astral-sh/uv) installed.
- Python 3.12.

### Local Execution
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

### Docker Execution
Mount your data and output folders to ensure results persist outside the container.

1. **Build the image**:
   ```bash
   docker build -t skillcorner-inference .
   ```
2. **Run with mounted volumes**:
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
6. **Observability**: Periodically (controlled by `--log_freq`), the pipeline logs performance metrics (FPS, detections count). A full telemetry summary is provided upon completion. Each log entry is enriched with `video_id`.
7. **Idempotency**: The pipeline skips videos that already have a corresponding parquet file in the output directory.
