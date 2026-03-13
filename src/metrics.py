from prometheus_client import Histogram, Counter, start_http_server
import time

# Performance Metrics
INFERENCE_STEP_LATENCY = Histogram(
    'inference_step_latency_seconds',
    'Time taken for each frame processing phase',
    ['step', 'video_id'],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, float("inf"))
)

# Inference Analytics
DETECTIONS_PER_FRAME = Histogram(
    'detections_per_frame',
    'Count of detections per class per frame',
    ['class_name', 'video_id'],
    buckets=(0, 1, 2, 5, 10, 15, 20, 25, 30, float("inf"))
)

DETECTION_BBOX_AREA = Histogram(
    'detection_bbox_area_pixels',
    'Distribution of object sizes (width * height)',
    ['class_name', 'video_id'],
    buckets=(100, 500, 1000, 2500, 5000, 10000, 25000, 50000, float("inf"))
)

DETECTION_SPATIAL_BIN = Counter(
    'detection_spatial_bin_total',
    '16x9 grid binning of detection coordinates',
    ['bin_x', 'bin_y', 'video_id', 'class_name']
)

FRAMES_PROCESSED = Counter(
    'frames_processed_total',
    'Cumulative frame count per video',
    ['video_id']
)


def start_metrics_server(port: int = 8000):
    """Start the Prometheus metrics server."""
    start_http_server(port)
