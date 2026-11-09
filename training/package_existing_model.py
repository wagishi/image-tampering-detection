"""Package an already-trained Keras model for this repository.

Example:
python training/package_existing_model.py \
  --model /path/to/trained_model.h5 \
  --class-0-name authentic \
  --class-1-name tampered \
  --preprocessing vgg19

IMPORTANT: class-0/class-1 names must match the exact output-index mapping
used during training. Do not guess this mapping. The preprocessing option
must match the preprocessing used during training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "backend" / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to the existing .keras or .h5 model.")
    parser.add_argument("--class-0-name", required=True, help="Name mapped to sigmoid output index 0.")
    parser.add_argument("--class-1-name", required=True, help="Name mapped to sigmoid output index 1.")
    parser.add_argument(
        "--preprocessing",
        required=True,
        choices=("vgg19", "rescale_0_1", "none"),
        help="Exact input preprocessing used during training.",
    )
    args = parser.parse_args()

    if args.class_0_name == args.class_1_name:
        parser.error("The two class names must be different.")

    source = Path(args.model).expanduser().resolve()
    if not source.is_file():
        parser.error(f"Model file does not exist: {source}")

    model = tf.keras.models.load_model(source, compile=False)
    input_shape = model.input_shape
    output_shape = model.output_shape
    if isinstance(input_shape, list) or isinstance(output_shape, list):
        parser.error("Only single-input, single-output models are supported.")
    if len(input_shape) != 4 or tuple(input_shape[1:]) != (224, 224, 3):
        parser.error(f"Expected input shape (None, 224, 224, 3), got {input_shape}.")
    if len(output_shape) != 2 or output_shape[-1] != 1:
        parser.error(f"Expected binary sigmoid output shape (None, 1), got {output_shape}.")

    # A zero-image smoke test catches incompatible or non-finite model outputs.
    prediction = np.asarray(model.predict(np.zeros((1, 224, 224, 3), dtype=np.float32), verbose=0))
    if prediction.shape != (1, 1) or not np.isfinite(prediction).all():
        parser.error(f"Model smoke test failed; prediction shape/value was {prediction!r}.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(OUTPUT_DIR / "model.keras")
    (OUTPUT_DIR / "class_indices.json").write_text(
        json.dumps({args.class_0_name: 0, args.class_1_name: 1}, indent=2) + "\\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "model_config.json").write_text(
        json.dumps({"image_size": [224, 224], "preprocessing": args.preprocessing}, indent=2) + "\\n",
        encoding="utf-8",
    )

    print(f"Packaged model: {OUTPUT_DIR / 'model.keras'}")
    print(f"Class mapping: {args.class_0_name}=0, {args.class_1_name}=1")
    print(f"Preprocessing: {args.preprocessing}")
    print("Next: run the local inference smoke test and deploy these three files.")


if __name__ == "__main__":
    main()
