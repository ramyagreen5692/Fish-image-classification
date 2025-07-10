# Fish-image-classification
# 🐟 Fish Species Classification Using Deep Learning

This project focuses on classifying fish images into multiple categories using Convolutional Neural Networks (CNNs) and Transfer Learning. The goal is to determine the best model architecture, save trained models, and deploy a Streamlit web application for real-time predictions.

---

## 🚀 Problem Statement

Develop a robust image classification pipeline to:
- Train a CNN from scratch and fine-tune five pre-trained models.
- Compare model performance using key metrics.
- Deploy the best model in a user-friendly Streamlit web app.
- Predict fish species from uploaded images in real-time.

---

## 📌 Business Use Cases

- 🎯 **Enhanced Accuracy**: Determine the most accurate model for fish classification.
- 💻 **Deployment Ready**: Provide an interactive prediction interface via a Streamlit app.
- 🔍 **Model Comparison**: Evaluate and select the most suitable model using classification metrics.

---

## 🧠 Approach

### 1. Data Preprocessing & Augmentation
- Images rescaled to [0, 1] range.
- Applied rotation, zoom, horizontal flip for robustness.

### 2. Model Training
- Train a custom CNN model.
- Experiment with pre-trained models:
  - VGG16
  - ResNet50
  - MobileNet
  - InceptionV3
  - EfficientNetB0
- Fine-tune all models and save the best based on validation accuracy.

### 3. Model Evaluation
- Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
- Visualizations: Training/validation loss and accuracy plots.

### 4. Deployment
- Developed a Streamlit app to:
  - Upload fish images.
  - Predict species and show confidence scores.

This app supports predictions using the following fine-tuned models:

- ✅ CNN from Scratch
- ✅ VGG16
- ✅ ResNet50
- ✅ MobileNet
- ✅ InceptionV3
- ✅ EfficientNetB0

All models were trained on a custom fish dataset with 10 species, including:
- `bass`, `trout`, `shrimp`, `gilt_head_bream`, etc.

## 📸 How to Use

1. Upload a fish image (`.jpg`, `.jpeg`, or `.png`)
2. Select a model from the dropdown
3. Click the **Submit** button
4. View:
   - Predicted species
   - Model confidence scores



