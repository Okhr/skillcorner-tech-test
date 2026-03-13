import argparse
import os
import glob
import hashlib
import time
from datetime import datetime
from src.pipeline import InferencePipeline
from src.logger import setup_logger, logger
from src.metrics import start_metrics_server
from src.config import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_MODEL_SIZE,
    DEFAULT_LOG_DIR,
    LOG_FREQUENCY,
    TARGET_FPS,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_PROVIDER,
    VIZ_FREQUENCY,
    DEFAULT_VIZ_DIR
)


def calculate_file_hash(file_path, block_size=65536):
    """Calculate MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read(block_size)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(block_size)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="SkillCorner Video Inference Technical Test")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Directory containing input videos")
    parser.add_argument("--model_size", type=str, default=DEFAULT_MODEL_SIZE, help="RF-DETR model size (nano, small, base, medium, large)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory for output parquet files")
    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR, help="Directory for log files")
    parser.add_argument("--log_freq", type=int, default=LOG_FREQUENCY, help="Logging frequency (every X frames)")
    parser.add_argument("--fps", type=int, default=TARGET_FPS, help="Target FPS for inference")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--provider", type=str, default=DEFAULT_PROVIDER, help="ONNX Runtime provider")
    parser.add_argument("--viz_dir", type=str, default=DEFAULT_VIZ_DIR, help="Base directory for visualization frames")
    parser.add_argument("--viz_freq", type=int, default=VIZ_FREQUENCY, help="Visualization frequency (every X frames)")
    parser.add_argument("--metrics_port", type=int, default=8000, help="Exporter port for Prometheus")

    args = parser.parse_args()

    # Start Prometheus Metrics Server
    start_metrics_server(args.metrics_port)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(args.log_dir, f"{timestamp}.log")
    setup_logger(log_path)
    main_logger = logger.bind(model_variant=args.model_size)

    # Find all video files in data_dir
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_paths = []
    for ext in video_extensions:
        video_paths.extend(glob.glob(os.path.join(args.data_dir, f"*{ext}")))

    video_paths.sort()

    if not video_paths:
        main_logger.warning("no_videos_found", data_dir=args.data_dir)
        return

    main_logger.info("starting_continuous_video_watch", data_dir=args.data_dir)

    try:
        while True:
            # Find all video files in data_dir
            video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
            video_paths = []
            for ext in video_extensions:
                video_paths.extend(glob.glob(os.path.join(args.data_dir, f"*{ext}")))

            video_paths.sort()

            for video_path in video_paths:
                video_id = calculate_file_hash(video_path)
                output_path = os.path.join(args.output_dir, f"{video_id}.parquet")
                viz_output_dir = os.path.join(args.viz_dir, video_id)

                if os.path.exists(output_path):
                    continue

                main_logger.info("processing_video", video_id=video_id, video_path=video_path)

                pipeline = InferencePipeline(
                    video_path=video_path,
                    model_size=args.model_size,
                    output_path=output_path,
                    log_frequency=args.log_freq,
                    target_fps=args.fps,
                    confidence_threshold=args.conf,
                    provider=args.provider,
                    viz_output_dir=viz_output_dir,
                    viz_frequency=args.viz_freq,
                    video_id=video_id
                )

                try:
                    pipeline.run()
                except Exception as e:
                    main_logger.error("video_processing_failed", video_id=video_id, video_path=video_path, error=str(e))

            # Sleep before next scan
            time.sleep(5)
    except KeyboardInterrupt:
        main_logger.info("stopping_continuous_watch")


if __name__ == "__main__":
    main()
