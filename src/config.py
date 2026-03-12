import os

# Video configuration
DEFAULT_DATA_DIR = "data"
TARGET_FPS = 10
TARGET_SIZE = (1280, 720)  # (width, height)

# Inference configuration
MODELS_DIR = "models"
AVAILABLE_MODEL_SIZES = ["nano", "small", "base", "medium", "large"]  # onnx-community/rfdetr_<size>-ONNX
DEFAULT_MODEL_SIZE = "nano"
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# Class mapping (Model dependent)
# For onnx-community/rfdetr_nano-ONNX
PERSON_CLASS_ID = 1
BALL_CLASS_ID = 37

# Output configuration
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_LOG_DIR = "logs"
LOG_FREQUENCY = 10
VIZ_FREQUENCY = 100
DEFAULT_VIZ_DIR = "visualizations"

# ONNX Runtime configuration
DEFAULT_PROVIDER = "CPUExecutionProvider"

# Normalization (ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
