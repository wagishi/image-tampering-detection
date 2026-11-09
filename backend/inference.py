"""Validated single-image inference for the VGG19 binary classifier.

Artifacts in backend/models/:
  model.keras
  class_indices.json  (class name -> output index)
  model_config.json   (image_size and preprocessing; optional for older exports)

The preprocessing value must describe the preprocessing used during training.
"""
from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(BASE_DIR / "models")))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(MODEL_DIR / "model.keras")))
CLASS_MAP_PATH = Path(os.getenv("CLASS_MAP_PATH", str(MODEL_DIR / "class_indices.json")))
CONFIG_PATH = Path(os.getenv("MODEL_CONFIG_PATH", str(MODEL_DIR / "model_config.json")))

if not MODEL_PATH.is_file():
    raise RuntimeError(
        f"Trained model not found: {MODEL_PATH}. Copy/export your existing "
        "trained model to this path; retraining is not required."
    )
if not CLASS_MAP_PATH.is_file():
    raise RuntimeError(
        f"Class mapping not found: {CLASS_MAP_PATH}. Generate it from the "
        "training data loader's original class_indices mapping."
    )

try:
    MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
except Exception as exc:
    raise RuntimeError(
        "Could not load the trained model. Check that its TensorFlow/Keras "
        "version is compatible with the deployment environment."
    ) from exc

with CLASS_MAP_PATH.open("r", encoding="utf-8") as file:
    CLASS_INDICES = json.load(file)

if not isinstance(CLASS_INDICES, dict) or len(CLASS_INDICES) != 2:
    raise RuntimeError("class_indices.json must contain exactly two class names and indices.")
try:
    INDEX_TO_CLASS = {int(index): str(name) for name, index in CLASS_INDICES.items()}
except (TypeError, ValueError) as exc:
    raise RuntimeError("Class indices must be integer values 0 and 1.") from exc

if set(INDEX_TO_CLASS) != {0, 1} or len(INDEX_TO_CLASS) != 2:
    raise RuntimeError(f"Expected class indices 0 and 1; received {CLASS_INDICES!r}.")

CONFIG = {"image_size": [224, 224], "preprocessing": "vgg19"}
if CONFIG_PATH.is_file():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        CONFIG.update(json.load(file))

IMAGE_SIZE = tuple(int(value) for value in CONFIG["image_size"])
if len(IMAGE_SIZE) != 2 or min(IMAGE_SIZE) <= 0:
    raise RuntimeError("model_config.json image_size must contain two positive integers.")

PREPROCESSING = CONFIG["preprocessing"]
if PREPROCESSING not in {"vgg19", "rescale_0_1", "none"}:
    raise RuntimeError("Unsupported preprocessing. Use vgg19, rescale_0_1, or none.")

input_shape = MODEL.input_shape
output_shape = MODEL.output_shape
if isinstance(input_shape, list) or isinstance(output_shape, list):
    raise RuntimeError("This inference pipeline expects a single-input, single-output model.")
if len(input_shape) != 4 or tuple(input_shape[1:]) != (*IMAGE_SIZE, 3):
    raise RuntimeError(
        f"Expected model input (None, {IMAGE_SIZE[0]}, {IMAGE_SIZE[1]}, 3); got {input_shape}."
    )
if len(output_shape) != 2 or output_shape[-1] != 1:
    raise RuntimeError(f"Expected one sigmoid output per image (shape (None, 1)); got {output_shape}.")


def predict_image(image_bytes: bytes) -> dict[str, Any]:
    """Run binary inference and return the predicted class and model score."""
    started = time.perf_counter()
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("The uploaded file is not a valid supported image.") from exc

    image = image.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.Resampling.BILINEAR)
    pixels = np.asarray(image, dtype=np.float32)

    if PREPROCESSING == "vgg19":
        pixels = tf.keras.applications.vgg19.preprocess_input(pixels)
    elif PREPROCESSING == "rescale_0_1":
        pixels = pixels / 255.0

    batch = np.expand_dims(pixels, axis=0)
    values = np.asarray(MODEL.predict(batch, verbose=0))
    if values.shape != (1, 1):
        raise RuntimeError(f"Expected prediction shape (1, 1); received {values.shape}.")

    score_class_1 = float(values[0, 0])
    if not np.isfinite(score_class_1) or not 0.0 <= score_class_1 <= 1.0:
        raise RuntimeError("Model returned an invalid sigmoid score in [0, 1].")

    predicted_index = 1 if score_class_1 >= 0.5 else 0
    predicted_class = INDEX_TO_CLASS[predicted_index]
    confidence = score_class_1 if predicted_index == 1 else 1.0 - score_class_1

    return {
        "prediction": predicted_class,
        "predicted_class_index": predicted_index,
        "confidence": round(float(confidence), 6),
        "class_1_probability": round(score_class_1, 6),
        "class_1_name": INDEX_TO_CLASS[1],
        "threshold": 0.5,
        "inference_time_ms": round((time.perf_counter() - started) * 1000, 2),
        "disclaimer": (
            "This is a model classification, not proof of authenticity "
            "and not a pixel-level localization of manipulation."
        ),
    }
