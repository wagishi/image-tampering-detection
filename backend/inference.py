"""Single-image inference for the CASIA VGG19 classifier.

Expected artifacts in backend/models/:
  - model.keras
  - class_indices.json (Keras flow_from_directory class mapping)

The model must be trained with the same preprocessing used below:
VGG19 preprocess_input on RGB pixels in the [0, 255] range.
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
CLASS_MAP_PATH = Path(
    os.getenv("CLASS_MAP_PATH", str(MODEL_DIR / "class_indices.json"))
)

if not MODEL_PATH.is_file():
    raise RuntimeError(
        f"Model not found at {MODEL_PATH}. Train/export the model and place "
        "model.keras in backend/models/ before starting the API."
    )
if not CLASS_MAP_PATH.is_file():
    raise RuntimeError(
        f"Class mapping not found at {CLASS_MAP_PATH}. Save "
        "train_generator.class_indices as class_indices.json."
    )

# Load once per API worker, not once per request.
MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
with CLASS_MAP_PATH.open("r", encoding="utf-8") as f:
    CLASS_INDICES: dict[str, int] = json.load(f)

INDEX_TO_CLASS = {int(index): name for name, index in CLASS_INDICES.items()}
if sorted(INDEX_TO_CLASS) != [0, 1]:
    raise RuntimeError(
        "This API expects exactly two classes with indices 0 and 1; "
        f"received {CLASS_INDICES!r}."
    )

IMAGE_SIZE = (224, 224)


def predict_image(image_bytes: bytes) -> dict[str, Any]:
    """Run the binary classifier and return class-aware score information."""
    started = time.perf_counter()

    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("The uploaded file is not a valid supported image.") from exc

    image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    pixels = np.asarray(image, dtype=np.float32)

    # Must match training preprocessing exactly.
    batch = tf.keras.applications.vgg19.preprocess_input(pixels)
    batch = np.expand_dims(batch, axis=0)

    raw = MODEL.predict(batch, verbose=0)
    score_class_1 = float(np.asarray(raw).reshape(-1)[0])
    if not np.isfinite(score_class_1) or not 0.0 <= score_class_1 <= 1.0:
        raise RuntimeError("Model returned an invalid sigmoid score.")

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
