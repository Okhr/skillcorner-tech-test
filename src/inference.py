import onnxruntime as ort
import cv2
import numpy as np
import os
import urllib.request
import json
from .logger import logger
from .config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_PROVIDER,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PERSON_CLASS_ID,
    BALL_CLASS_ID,
    MODELS_DIR,
    AVAILABLE_MODEL_SIZES
)


class InferenceEngine:
    def __init__(self,
                 model_size: str,
                 confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
                 provider: str = DEFAULT_PROVIDER):

        if model_size not in AVAILABLE_MODEL_SIZES:
            raise ValueError(f"Model size '{model_size}' is not supported. Available sizes: {AVAILABLE_MODEL_SIZES}")

        self.model_size = model_size
        self.logger = logger.bind(model_variant=model_size)

        self.model_path, self.preprocessor_config = self._ensure_model_and_config(model_size)

        self.id2label = {
            PERSON_CLASS_ID: "person",
            BALL_CLASS_ID: "ball"
        }

        # Determine providers based on availability and preference
        available_providers = ort.get_available_providers()

        self.logger.info("onnx_available_providers", providers=available_providers)

        providers = []
        if provider and provider in available_providers:
            providers.append(provider)

        # Priority list for providers
        if "CUDAExecutionProvider" in available_providers and "CUDAExecutionProvider" not in providers:
            providers.append("CUDAExecutionProvider")
        if "TensorrtExecutionProvider" in available_providers and "TensorrtExecutionProvider" not in providers:
            providers.append("TensorrtExecutionProvider")
        if "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")

        self.logger.info("onnx_session_providers", requested_provider=provider, selected_providers=providers)

        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, H, W]

        # Handle dynamic input sizes (sometimes strings in ONNX)
        try:
            self.h = int(self.input_shape[2])
            self.w = int(self.input_shape[3])
        except (ValueError, TypeError):
            # Try to get from preprocessor_config
            if self.preprocessor_config and "size" in self.preprocessor_config:
                size_config = self.preprocessor_config["size"]
                self.h = size_config.get("height")
                self.w = size_config.get("width")
            else:
                self.h = None
                self.w = None

        if self.h is None or self.w is None:
            raise ValueError(f"Could not infer input size from ONNX model or preprocessor_config for model_size '{model_size}'")

        self.logger.info("model_input_info", input_size=(self.w, self.h), input_shape=self.input_shape)

        self.confidence_threshold = confidence_threshold

        # Normalization constants from config or fallback
        mean = self.preprocessor_config.get("image_mean", IMAGENET_MEAN) if self.preprocessor_config else IMAGENET_MEAN
        std = self.preprocessor_config.get("image_std", IMAGENET_STD) if self.preprocessor_config else IMAGENET_STD
        self.mean = np.array(mean).astype(np.float32)
        self.std = np.array(std).astype(np.float32)

    def _ensure_model_and_config(self, model_size: str):
        model_filename = f"rf_detr_{model_size}.onnx"
        config_filename = f"rf_detr_{model_size}_preprocessor_config.json"

        model_path = os.path.join(MODELS_DIR, model_filename)
        config_path = os.path.join(MODELS_DIR, config_filename)

        os.makedirs(MODELS_DIR, exist_ok=True)

        # Download model if missing
        if not os.path.exists(model_path):
            url = f"https://huggingface.co/onnx-community/rfdetr_{model_size}-ONNX/resolve/main/onnx/model.onnx"
            self.logger.info("downloading_model", size=model_size, url=url)
            try:
                urllib.request.urlretrieve(url, model_path)
                self.logger.info("model_downloaded", path=model_path)
            except Exception as e:
                self.logger.error("model_download_failed", error=str(e))
                raise e

        # Download config if missing
        if not os.path.exists(config_path):
            url = f"https://huggingface.co/onnx-community/rfdetr_{model_size}-ONNX/resolve/main/preprocessor_config.json"
            self.logger.info("downloading_config", size=model_size, url=url)
            try:
                urllib.request.urlretrieve(url, config_path)
                self.logger.info("config_downloaded", path=config_path)
            except Exception as e:
                self.logger.warning("config_download_failed", error=str(e))

        # Load config
        preprocessor_config = None
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    preprocessor_config = json.load(f)
            except Exception as e:
                self.logger.warning("config_load_failed", error=str(e))

        return model_path, preprocessor_config

    def preprocess(self, img):
        # Resize to model input size
        img_input = cv2.resize(img, (self.w, self.h))
        # Normalize to [0, 1]
        img_input = img_input.astype(np.float32) / 255.0
        # Apply ImageNet normalization (mean/std)
        img_input = (img_input - self.mean) / self.std
        # HWC to CHW
        img_input = img_input.transpose(2, 0, 1)
        # Add batch dimension
        img_input = np.expand_dims(img_input, axis=0)
        return img_input

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def run(self, img):
        img_input = self.preprocess(img)
        # RF-DETR outputs: ['pred_boxes', 'pred_logits']
        # pred_boxes: [1, 300, 4] -> [cx, cy, w, h] normalized
        # pred_logits: [1, 300, 80] -> classification logits
        outputs = self.session.run(None, {self.input_name: img_input})
        pred_boxes = outputs[0][0]  # [300, 4]
        pred_logits = outputs[1][0]  # [300, 80]

        # Convert logits to probabilities using sigmoid (multi-label)
        # Some DETR models use sigmoid, others might use softmax if mutual exclusive
        probs = self.sigmoid(pred_logits)

        # Get max score and class ID for each of the 300 predictions
        scores = np.max(probs, axis=-1)
        # Using probs > threshold to see all candidates
        class_ids = np.argmax(probs, axis=-1)

        results = []
        frame_h, frame_w = img.shape[:2]

        for box, score, cls in zip(pred_boxes, scores, class_ids):
            class_id = int(cls)
            if score > self.confidence_threshold and class_id in [PERSON_CLASS_ID, BALL_CLASS_ID]:
                cx, cy, w, h = box

                # Convert normalized [cx, cy, w, h] to absolute [x1, y1, x2, y2]
                x1 = (cx - w / 2) * frame_w
                y1 = (cy - h / 2) * frame_h
                x2 = (cx + w / 2) * frame_w
                y2 = (cy + h / 2) * frame_h

                label = self.id2label.get(class_id, str(class_id))

                results.append({
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "confidence": float(score),
                    "class": class_id,
                    "label": label
                })
        return results
