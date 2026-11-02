# Image Tampering Detection using VGG19

A deep-learning project that classifies images into two dataset classes (authentic/tampered) using a VGG19-based CNN and the CASIA 2.0 dataset.

## Repository contents

- `image_tampering_detection.py`: original Colab-generated training notebook export.
- `training/train_and_export.py`: reproducible transfer-learning training script that exports model artifacts.
- `backend/`: FastAPI inference service and shared prediction pipeline.
- `streamlit_app.py`: public-facing Streamlit image-upload website.

## Model and dataset

The export model uses VGG19 with ImageNet weights and its ImageNet classification head removed, followed by Flatten, Dense(256, ReLU), Dropout(0.5), and a sigmoid output. The convolutional backbone is frozen while the classification head is trained.

The dataset directory must contain exactly two class subdirectories. Keras assigns numeric labels alphabetically; the script saves this mapping to `class_indices.json`. Do not assume which label is authentic or tampered—check the saved mapping.

## Train and export

The web app needs a trained model artifact. Install the training dependencies and set `DATASET_DIR` to the directory containing the two class folders.

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

Model artifacts are excluded from Git by `backend/.gitignore`. Upload these two artifacts to your deployment using Streamlit secrets / a private model-artifact host, or temporarily commit them only if their size and dataset licensing permit. The website displays a setup message until both files are available.

### Important note about the original training script

The original `image_tampering_detection.py` initializes VGG19 with `weights=None` and freezes it. That means its random feature-extractor weights are not updated. It also uses `rescale=1./255` and does not export a model. The new export script is a separate transfer-learning path and uses `tf.keras.applications.vgg19.preprocess_input`, matching the inference pipeline. The original file is preserved.

## Deploy the website (Streamlit Community Cloud)

1. Push or merge the `feature/image-forensics-web-app` branch.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and choose **Create app**.
3. Select this repository, the branch containing the website, and set the main file path to `streamlit_app.py`.
4. Ensure the trained files `backend/models/model.keras` and `backend/models/class_indices.json` are available at those paths in the app runtime.
5. Deploy. The first build can take several minutes because TensorFlow is a large dependency.

The website accepts JPEG and PNG images up to 5 MiB. Images are processed in memory for inference; the app does not intentionally save uploaded images.

## Run the website locally

Use Python 3.10 or 3.11:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Run the API locally

From the repository root:

```bash
pip install -r backend/requirements.txt
uvicorn main:app --app-dir backend --reload
```

Open the interactive API docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## API endpoints

- `GET /` and `GET /health`: service status.
- `POST /predict`: upload one JPEG or PNG image as multipart form data using the `file` field. Maximum upload size defaults to 5 MiB.

Example response shape (illustrative only):

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

## Limitations and responsible interpretation

- This is a binary image classifier, not a forensic guarantee.
- It does not generate a pixel-level tampering mask or locate manipulated regions.
- Performance on CASIA does not guarantee generalization to images from other sources or manipulation methods.
- The sigmoid score is a model score; it is not necessarily a calibrated probability of correctness.
- Evaluate with a held-out test set and report precision, recall, F1-score, confusion matrix, and relevant failure cases before making performance claims.

## Tech stack

Python, TensorFlow/Keras, VGG19, Streamlit, FastAPI, Pillow, NumPy, Uvicorn.
