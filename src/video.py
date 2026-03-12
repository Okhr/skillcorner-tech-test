import av
import numpy as np
import cv2
from .logger import logger
from .config import TARGET_FPS, TARGET_SIZE


class VideoProcessor:
    def __init__(self, video_path: str, target_fps: int = TARGET_FPS, target_size: tuple = TARGET_SIZE, video_id: str = None, model_variant: str = None):
        self.video_path = video_path
        self.target_fps = target_fps
        self.target_size = target_size
        self.video_id = video_id
        self.model_variant = model_variant

        # Bind context to localized logger
        log_context = {}
        if video_id:
            log_context["video_id"] = video_id
        if model_variant:
            log_context["model_variant"] = model_variant

        self.logger = logger.bind(**log_context) if log_context else logger

        self.container = av.open(video_path)
        self.stream = self.container.streams.video[0]

        self.source_fps = float(self.stream.average_rate)
        self.total_frames = self.stream.frames
        self.duration = float(self.stream.duration * self.stream.time_base)

        # Calculate sampling interval to match target FPS
        self.sampling_interval = self.source_fps / self.target_fps

        self.logger.info("video_metadata",
                         fps=self.source_fps,
                         total_frames=self.total_frames,
                         duration=self.duration,
                         width=self.stream.width,
                         height=self.stream.height)

    def get_frames(self):
        """Generator to yield frames at the target FPS."""
        frame_idx = 0
        target_frame_count = 0

        # Decoding specific stream is more efficient than decoding the whole container
        for frame in self.container.decode(self.stream):
            # Check if this frame should be sampled
            if frame_idx >= target_frame_count * self.sampling_interval:
                # Convert to RGB image
                img = frame.to_image()
                img_rgb = np.array(img)
                # Resize to target size
                img_resized = cv2.resize(img_rgb, self.target_size)

                yield img_resized, frame.time, frame_idx
                target_frame_count += 1

            frame_idx += 1

    def __del__(self):
        if hasattr(self, 'container'):
            self.container.close()
