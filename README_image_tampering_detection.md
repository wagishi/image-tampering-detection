# Image Tampering Detection using VGG19

A deep learning project for detecting whether an image is authentic or tampered. The model uses a VGG19-based convolutional neural network architecture with image augmentation to classify images into two categories: **authentic** and **tampered**.

This project is designed for image forensics and can be used as a starting point for detecting manipulated images such as copy-move, splicing, or other forms of image tampering.

## Repository Description

**Image Tampering Detection using Deep Learning** is a binary image classification project built with TensorFlow and Keras. It uses the CASIA 2.0 Image Tampering Detection Dataset and a VGG19-based CNN model to identify whether an input image has been manipulated. The notebook includes dataset loading, preprocessing, image augmentation, model building, training, validation, and visualization of accuracy and loss curves.

## Features

- Binary classification of images as authentic or tampered
- VGG19-based CNN architecture
- Image preprocessing and resizing to 224 × 224 pixels
- Data augmentation using rotation, shifting, shearing, zooming, and horizontal flipping
- Training and validation split using `ImageDataGenerator`
- Accuracy and loss visualization after training
- Simple TensorFlow/Keras implementation suitable for beginners and academic projects

## Tech Stack

- Python
- TensorFlow
- Keras
- VGG19
- Matplotlib
- Kaggle Notebook / Google Colab

## Dataset

The project uses the **CASIA 2.0 Image Tampering Detection Dataset**.

Expected dataset path in the notebook:

```text
/kaggle/input/casia-20-image-tampering-detection-dataset
```

The dataset should be arranged in class-wise folders so that Keras can load it using `flow_from_directory()`.

Example structure:

```text
casia-20-image-tampering-detection-dataset/
│
├── authentic/
│   ├── image1.jpg
│   ├── image2.jpg
│
└── tampered/
    ├── image1.jpg
    ├── image2.jpg
```

## Project Workflow

1. Import required libraries
2. Set image size, batch size, and number of epochs
3. Apply image preprocessing and augmentation
4. Load training and validation data from directory
5. Build a VGG19-based CNN model
6. Compile the model using Adam optimizer and binary cross-entropy loss
7. Train the model on the dataset
8. Plot training and validation accuracy/loss curves

## Model Architecture

The model uses VGG19 as the base feature extractor followed by fully connected layers:

```text
VGG19 Base Model
↓
Flatten Layer
↓
Dense Layer with 256 neurons and ReLU activation
↓
Dropout Layer with 0.5 dropout rate
↓
Dense Output Layer with Sigmoid activation
```

Since this is a binary classification problem, the final layer uses a sigmoid activation function.

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/image-tampering-detection.git
cd image-tampering-detection
```

Install the required dependencies:

```bash
pip install tensorflow matplotlib
```

If you are using Kaggle Notebook or Google Colab, TensorFlow and Matplotlib may already be installed.

## How to Run

1. Download or add the CASIA 2.0 dataset to your Kaggle/Colab environment.
2. Update the dataset path if required.
3. Open the notebook:

```text
image-tampering-detection.ipynb
```

4. Run all cells step by step.
5. After training, the notebook will display accuracy and loss graphs.

## Important Code Settings

```python
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 19
```

You can change these values depending on your GPU availability and experiment requirements.

## Results

The notebook plots:

- Training accuracy
- Validation accuracy
- Training loss
- Validation loss

These plots help analyze whether the model is learning properly or overfitting.

## Limitations

- The current notebook uses VGG19 with `weights=None`, meaning the model does not use ImageNet pre-trained weights.
- The base VGG19 model is set as non-trainable, so if `weights=None` is used, the frozen base contains random weights.
- For better performance, use `weights='imagenet'` or make selected VGG19 layers trainable during fine-tuning.
- The model performs binary classification and does not localize the tampered region inside an image.

## Future Improvements

- Use ImageNet pre-trained VGG19 weights
- Fine-tune deeper layers of VGG19
- Add confusion matrix, precision, recall, and F1-score
- Save the trained model as `.h5` or `.keras`
- Add prediction code for testing a single image
- Use advanced architectures like EfficientNet, ResNet, or Vision Transformers
- Add tampered-region localization using segmentation models
- Deploy the model using Streamlit or Flask

## Suggested Single Image Prediction Code

```python
import numpy as np
from tensorflow.keras.preprocessing import image

img_path = "path/to/test/image.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("Tampered Image")
else:
    print("Authentic Image")
```

## Project Applications

- Digital image forensics
- Fake image detection
- Media verification
- Cybersecurity and misinformation detection
- Academic deep learning projects

## Author

**Wagishi Jagat**  

