# Image Tampering Detection using VGG19

A deep-learning project that classifies images into two dataset classes (authentic/tampered) using a VGG19-based CNN and the CASIA 2.0 dataset.

## Repository contents

- `image_tampering_detection.py`: original Colab-generated training notebook export.
- `training/train_and_export.py`: reproducible transfer-learning training script that exports artifacts for the API.
- `backend/`: FastAPI inference service for single-image predictions.

## Model and dataset

The model uses VGG19 without its ImageNet classification head, followed by Flatten, Dense(256, ReLU), Dropout(0.5), and a sigmoid output. The export training script initializes VGG19 with ImageNet weights and freezes the convolutional backbone while training the classification head.

The dataset directory must contain exactly two class subdirectories. Keras assigns numeric labels alphabetically; the script saves this mapping to `class_indices.json`. Do not assume which label is authentic or tampered—check the saved mapping.

## Train and export

Install TensorFlow and the training dependencies in your Python environment. Set `DATASET_DIR` to the directory containing the two class folders.

PowerShell:

```powershell
$env:DATASET_DIR = "C:\path\to\CASIA_dataset"
python training/train_and_export.py
```

Linux/macOS:

```bash
export DATASET_DIR=/path/to/CASIA_dataset
python training/train_and_export.py
```

The script writes:

- `backend/models/model.keras`
- `backend/models/class_indices.json`

Model artifacts are excluded from Git by `backend/.gitignore`; keep large trained weights local or distribute them through an appropriate model-artifact host. The API cannot perform real inference until both files are present.

### Important note about the original training script

The original `image_tampering_detection.py` initializes VGG19 with `weights=None` and freezes it. That means its random feature-extractor weights are not updated. It also uses `rescale=1./255` and does not export a model. The new export script is a separate, corrected transfer-learning path and uses `tf.keras.applications.vgg19.preprocess_input`, matching the API inference pipeline. The original file is preserved.

## Run the API locally

Python 3.10 or 3.11 is recommended. From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
uvicorn main:app --app-dir backend --reload
```

On Linux/macOS, activate with `source .venv/bin/activate`.

Open the interactive API docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## API endpoints

### `GET /` and `GET /health`

Return service status.

### `POST /predict`

Upload one JPEG or PNG image as multipart form data using the `file` field. Maximum upload size defaults to 5 MiB.

Example response shape:

```json
{
  "filename": "sample.jpg",
  "content_type": "image/jpeg",
  "size_bytes": 123456,
  "prediction": "tampered",
  "predicted_class_index": 1,
  "confidence": 0.91,
  "class_1_probability": 0.91,
  "class_1_name": "tampered",
  "threshold": 0.5,
  "inference_time_ms": 120.5,
  "disclaimer": "This is a model classification, not proof of authenticity and not a pixel-level localization of manipulation."
}
```

The values above are illustrative, not measured model results. The returned class names come from the saved mapping.

## Limitations and responsible interpretation

- This is a binary image classifier, not a forensic guarantee.
- It does not generate a pixel-level tampering mask or locate manipulated regions.
- Performance on CASIA does not guarantee generalization to images from other sources or manipulation methods.
- The sigmoid score is a model score; it is not necessarily a calibrated probability of correctness.
- Evaluate with a held-out test set and report precision, recall, F1-score, confusion matrix, and relevant failure cases before making performance claims.

## Tech stack

Python, TensorFlow/Keras, VGG19, FastAPI, Pillow, NumPy, Uvicorn.
