"""Train and export a VGG19 binary classifier compatible with backend/inference.py.

Set DATASET_DIR to a directory with exactly two class subdirectories, e.g.:
  dataset/
    authentic/
    tampered/

Example:
  DATASET_DIR=/path/to/dataset python training/train_and_export.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = int(os.getenv("EPOCHS", "10"))
SEED = 42

dataset_dir = os.getenv("DATASET_DIR")
if not dataset_dir:
    raise SystemExit(
        "Set DATASET_DIR to the CASIA dataset directory containing exactly "
        "two class subfolders."
    )

dataset_path = Path(dataset_dir)
if not dataset_path.is_dir():
    raise SystemExit(f"Dataset directory does not exist: {dataset_path}")

# Apply augmentation only to the training subset. Both subsets use the same
# VGG19 preprocessing as the inference API.
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg19.preprocess_input,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    validation_split=0.2,
)

validation_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg19.preprocess_input,
    validation_split=0.2,
)

train_generator = train_datagen.flow_from_directory(
    str(dataset_path),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True,
    seed=SEED,
)

validation_generator = validation_datagen.flow_from_directory(
    str(dataset_path),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False,
    seed=SEED,
)

if len(train_generator.class_indices) != 2:
    raise ValueError(
        "Expected exactly two class folders for binary classification; "
        f"found {train_generator.class_indices}"
    )
if train_generator.class_indices != validation_generator.class_indices:
    raise ValueError("Training and validation class mappings do not match.")

base_model = VGG19(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
)
base_model.trainable = False

model = Sequential(
    [
        base_model,
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ],
    name="casia_vgg19_binary_classifier",
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

model.summary()
model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
)

output_dir = Path(__file__).resolve().parents[1] / "backend" / "models"
output_dir.mkdir(parents=True, exist_ok=True)

model.save(output_dir / "model.keras")
with (output_dir / "class_indices.json").open("w", encoding="utf-8") as f:
    json.dump(train_generator.class_indices, f, indent=2)

print(f"Saved model and class mapping to: {output_dir}")
print(f"Class mapping: {train_generator.class_indices}")
