import time
import os
import cv2
import pandas as pd
import numpy as np
import supervision as sv
from tqdm import tqdm
from .logger import logger
from .video import VideoProcessor
from .inference import InferenceEngine
from .metrics import (
    INFERENCE_STEP_LATENCY,
    DETECTIONS_PER_FRAME,
    DETECTION_BBOX_AREA,
    DETECTION_SPATIAL_BIN,
    FRAMES_PROCESSED
)


from .config import (
    TARGET_FPS,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_PROVIDER,
    LOG_FREQUENCY,
    VIZ_FREQUENCY,
    DEFAULT_VIZ_DIR
)


class InferencePipeline:
    def __init__(self,
                 video_path: str,
                 model_size: str,
                 output_path: str,
                 target_fps: int = TARGET_FPS,
                 confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
                 provider: str = DEFAULT_PROVIDER,
                 log_frequency: int = LOG_FREQUENCY,
                 viz_frequency: int = VIZ_FREQUENCY,
                 viz_output_dir: str = DEFAULT_VIZ_DIR,
                 video_id: str = None):
        self.video_id = video_id
        self.model_size = model_size
        self.logger = logger.bind(video_id=video_id, model_variant=model_size) if video_id else logger.bind(model_variant=model_size)

        self.video_processor = VideoProcessor(video_path, target_fps=target_fps, video_id=video_id, model_variant=model_size)
        self.inference_engine = InferenceEngine(model_size, confidence_threshold=confidence_threshold, provider=provider)
        self.output_path = output_path
        self.log_frequency = log_frequency
        self.viz_frequency = viz_frequency
        self.viz_output_dir = viz_output_dir
        self.results = []

        # Initialize supervision annotators
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def run(self):
        start_time = time.time()
        periodic_start_time = time.time()
        frame_count = 0
        periodic_detections_count = 0

        self.logger.info("start_inference_pipeline")

        try:
            # We wrap the generator to measure loading time
            video_iter = self.video_processor.get_frames()

            while True:
                # 1. Load frame
                load_start = time.time()
                try:
                    frame_data = next(video_iter)
                except StopIteration:
                    break
                frame, timestamp, frame_idx = frame_data
                INFERENCE_STEP_LATENCY.labels(step='load', video_id=self.video_id).observe(time.time() - load_start)

                # 2. Inference
                infer_start = time.time()
                detections_list = self.inference_engine.run(frame)
                INFERENCE_STEP_LATENCY.labels(step='infer', video_id=self.video_id).observe(time.time() - infer_start)

                # 3. Post-processing (Supervision conversion, metrics, etc.)
                post_start = time.time()
                # Convert detections to supervision Detections object
                if detections_list:
                    xyxy = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in detections_list])
                    confidence = np.array([d["confidence"] for d in detections_list])
                    class_id = np.array([d["class"] for d in detections_list])
                    sv_detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
                else:
                    sv_detections = sv.Detections.empty()

                for det in detections_list:
                    det["timestamp"] = timestamp
                    det["frame_idx"] = frame_idx
                    self.results.append(det)

                # Update Prometheus Metrics
                FRAMES_PROCESSED.labels(video_id=self.video_id).inc()

                # Group detections by class for the histogram
                class_counts = {}

                for det in detections_list:
                    cls_label = det["label"]
                    class_counts[cls_label] = class_counts.get(cls_label, 0) + 1

                    # BBox Area
                    area = (det["x2"] - det["x1"]) * (det["y2"] - det["y1"])
                    DETECTION_BBOX_AREA.labels(class_name=cls_label, video_id=self.video_id).observe(area)

                    # Spatial Binning (16x9 grid)
                    # Coordinates are already scaled to frame size in detections_list
                    # Target size is (1280, 720) in config.py
                    bin_x = int((det["x1"] + det["x2"]) / 2 / 80)  # 1280 / 16 = 80
                    bin_y = int((det["y1"] + det["y2"]) / 2 / 80)  # 720 / 9 = 80
                    # Clip to grid bounds
                    bin_x = max(0, min(15, bin_x))
                    bin_y = max(0, min(8, bin_y))

                    DETECTION_SPATIAL_BIN.labels(
                        bin_x=bin_x,
                        bin_y=bin_y,
                        video_id=self.video_id,
                        class_name=cls_label
                    ).inc()

                for cls_label, count in class_counts.items():
                    DETECTIONS_PER_FRAME.labels(class_name=cls_label, video_id=self.video_id).observe(count)

                INFERENCE_STEP_LATENCY.labels(step='post_process', video_id=self.video_id).observe(time.time() - post_start)

                frame_count += 1
                periodic_detections_count += len(detections_list)

                # Visualization frequency
                if frame_count % self.viz_frequency == 0:
                    self.save_visualization(frame, sv_detections, frame_idx)

                # Periodic metrics
                if frame_count % self.log_frequency == 0:
                    elapsed = time.time() - periodic_start_time
                    self.logger.info("periodic_metrics",
                                     frames=self.log_frequency,
                                     elapsed_seconds=elapsed,
                                     fps=self.log_frequency / elapsed,
                                     detections_in_period=periodic_detections_count,
                                     mean_detections_per_frame=periodic_detections_count / self.log_frequency)
                    periodic_start_time = time.time()
                    periodic_detections_count = 0

        except Exception as e:
            self.logger.error("pipeline_error", error=str(e))
            raise e
        finally:
            total_elapsed = time.time() - start_time
            total_detections = len(self.results)
            self.logger.info("total_metrics",
                             total_frames=frame_count,
                             total_seconds=total_elapsed,
                             avg_fps=frame_count/total_elapsed if total_elapsed > 0 else 0,
                             total_detections=total_detections,
                             avg_detections_per_frame=total_detections/frame_count if frame_count > 0 else 0)

        # Save results
        self.save_results()

    def save_results(self):
        df = pd.DataFrame(self.results)
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        df.to_parquet(self.output_path)
        self.logger.info("results_saved", path=self.output_path)

    def save_visualization(self, frame, detections, frame_idx):
        # Annotate frame using supervision
        annotated_frame = frame.copy()

        if not detections.empty():
            id2label = getattr(self.inference_engine, "id2label", {})
            labels = [
                f"{id2label.get(class_id, str(class_id))} {conf:.2f}"
                for class_id, conf in zip(detections.class_id, detections.confidence)
            ]
            annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        os.makedirs(self.viz_output_dir, exist_ok=True)
        output_name = os.path.join(self.viz_output_dir, f"frame_{frame_idx:06d}.jpg")

        cv2.imwrite(output_name, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
        self.logger.info("visualization_saved", path=output_name)
