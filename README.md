# Brain Tumor Detection Using Deep Learning

This project was developed as a Final Year Project for the **Bachelor of Science in Artificial Intelligence** at **The Islamia University of Bahawalpur**.

The goal of the project is to develop an AI-based system for detecting brain tumors from medical images using deep learning techniques. The system focuses on analyzing **MRI and CT scan images** and applying Convolutional Neural Networks (CNNs) to identify tumor-related patterns.

---

## Project Overview

Brain tumor detection is an important medical imaging problem where early and accurate identification can support timely diagnosis and treatment.

This project applies deep learning and computer vision techniques to automate the analysis of medical images. The system preprocesses input images, extracts relevant visual features, and uses a CNN-based model to classify images based on the presence of a brain tumor.

The project also includes a user-facing application where medical images can be uploaded and processed through the trained model.

---

## Objectives

- Develop a deep learning-based brain tumor detection system.
- Apply CNNs for medical image analysis.
- Preprocess MRI and CT scan images before model training.
- Train and validate the model using labeled medical images.
- Evaluate the model using standard classification metrics.
- Provide a simple interface for uploading and analyzing medical images.
- Demonstrate the practical use of Artificial Intelligence in healthcare and medical imaging.

---

## Technologies Used

- Python
- Jupyter Notebook
- TensorFlow
- NumPy
- Convolutional Neural Networks (CNN)
- Deep Learning
- Image Processing
- HTML/CSS
- Python-based application backend

---

## Methodology

The project follows the following workflow:

### 1. Image Collection and Preprocessing

Medical images are prepared before being passed to the deep learning model.

Preprocessing includes:

- Image resizing
- Pixel normalization
- Noise reduction
- Image standardization
- Data augmentation
- Rotation, flipping, scaling, and other transformations

These techniques help improve model robustness and reduce overfitting.

### 2. Model Development

A **Convolutional Neural Network (CNN)** is used to learn visual patterns from medical images.

The model includes:

- Convolutional layers
- Pooling layers
- Fully connected layers
- Feature extraction
- Classification layers

CNNs are particularly suitable for medical image classification because they can automatically learn spatial features such as shapes, textures, and image patterns.

### 3. Training and Validation

The dataset is divided into training, validation, and testing sets.

During training, the model learns to distinguish between normal and tumor-affected medical images.

Model performance is monitored during validation to improve generalization and reduce overfitting.

### 4. Model Evaluation

The model is evaluated using classification metrics such as:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

These metrics are used to assess how effectively the model identifies brain tumors in unseen images.

---

## Application Workflow

The application provides a simple workflow for users:

1. Select a medical image.
2. Upload the image.
3. Preprocess the image.
4. Pass the image through the trained model.
5. Generate a prediction.
6. Display the analysis result.

The goal of the interface is to make the AI model easier to use without requiring users to interact directly with the underlying code.

---

## Repository Structure

```text
Brain-Tumor-Detection/
│
├── model and data/
│   └── test images/
│
├── static/
│
├── templates/
│
├── Brain_Tumor_Detection.ipynb
│
├── braintumorApp.py
│
└── README.md
