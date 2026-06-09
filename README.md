# Image Tampering Detection using VGG19

A deep learning project that detects whether an image is **authentic** or **tampered** using a VGG19-based Convolutional Neural Network.

## About the Project

This project focuses on digital image forensics by classifying images into two categories: authentic and tampered. It uses the CASIA 2.0 Image Tampering Detection Dataset and applies image preprocessing, data augmentation, model training, validation, and performance visualization.

## Features

- Detects authentic and tampered images
- Uses VGG19-based CNN model
- Image preprocessing and resizing
- Data augmentation for better training
- Accuracy and loss visualization
- Useful for digital image forensics and fake image detection

## Tech Stack

- Python
- TensorFlow
- Keras
- VGG19
- Matplotlib
- Kaggle / Google Colab

## Dataset

This project uses the **CASIA 2.0 Image Tampering Detection Dataset**.

## Model Workflow

1. Load image dataset
2. Preprocess and resize images
3. Apply image augmentation
4. Build VGG19-based CNN model
5. Train the model
6. Validate performance
7. Plot accuracy and loss graphs

## Model Architecture

```text
VGG19 Base Model
↓
Flatten Layer
↓
Dense Layer with ReLU Activation
↓
Dropout Layer
↓
Sigmoid Output Layer
```

## Applications

- Digital image forensics
- Fake image detection
- Cybersecurity
- Media verification
- Misinformation detection

## Future Improvements

- Add confusion matrix and classification report
- Save trained model as `.h5` or `.keras`
- Add single image prediction feature
- Deploy using Streamlit or Flask
- Improve accuracy using more advanced architectures

## Author

**Wagishi Jagat**
