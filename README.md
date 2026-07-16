# BrainScan: Brain Tumor Detection System

BrainScan is a brain tumor classification project built with deep learning and transfer learning. The notebook uses MRI images to classify brain scans into four categories: glioma, meningioma, pituitary tumor, and no tumor.


<img width="1918" height="1078" alt="Screenshot 2026-07-16 223103" src="https://github.com/user-attachments/assets/db6297c7-8dd7-4fb5-8cc7-88c4edd78b3d" />
<img width="1918" height="1078" alt="Screenshot 2026-07-16 223139" src="https://github.com/user-attachments/assets/f4ad7212-7623-4c61-be2c-b23ea05990d4" />

## Project Goal

The goal of this project is to build an automated system that can analyze brain MRI images and predict the tumor type from the scan. The workflow covers data preparation, image preprocessing, model training, evaluation, and visual analysis of predictions.

## Dataset

The project uses the Brain Tumor MRI Dataset from Kaggle:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data

The dataset is organized into the following classes:

- Glioma
- Meningioma
- Pituitary
- No Tumor

## What Was Done

### 1. Environment Setup and Library Import

The notebook begins by importing the required Python libraries for image processing, data handling, visualization, and model development. These include NumPy, Pandas, Matplotlib, Pillow, TensorFlow, Keras, and scikit-learn tools for evaluation.

### 2. Image Preview

An example MRI image is loaded and resized to a consistent input size of 224 by 224 pixels. This helps verify the image format and confirms that the scans are being read correctly.

### 3. Dataset Split

The original training data is reorganized into separate folders for training and validation. An 80/20 split is applied for each class so the model can be trained and checked on unseen validation data during development.

### 4. Data Loading and Augmentation

Image data generators are used to load the scans in batches and normalize pixel values by rescaling them to the range 0 to 1.

To make the model more robust, brightness augmentation is applied to the training images. The validation and test images are only rescaled and are not augmented.

### 5. Transfer Learning with VGG16

The project uses VGG16 as the base convolutional network. A custom classifier is added on top of the pretrained base, and the model is trained for brain tumor classification.

### 6. Fine-Tuned Transfer Learning Model

The final model uses a fine-tuned VGG16 architecture. The early layers are frozen, while the last convolutional block is unfrozen so the model can adapt better to MRI images.

The classifier head includes:

- Flatten layer
- Dropout layers
- Dense hidden layer
- Softmax output layer with 4 classes

The model is compiled with the Adam optimizer and categorical cross-entropy loss.

### 7. Training Strategy

Training is performed with:

- Batch size: 64
- Image size: 224 by 224
- Epochs: 20
- Early stopping based on validation loss

Early stopping is used to reduce overfitting and restore the best-performing weights.

### 8. Model Evaluation

After training, the model is evaluated on the validation and test datasets. The notebook includes the following evaluation steps:

- Validation accuracy and validation loss
- Test loss and test accuracy
- Confusion matrix
- Classification report
- Per-class accuracy
- ROC curves for all 4 classes
- Macro AUC score

### 9. Prediction Review

A set of random test images is displayed with their true labels, predicted labels, and prediction confidence. Correct predictions are highlighted in green and incorrect predictions in red.

### 10. Model Saving

The trained fine-tuned model is saved as `model1.h5` for later use.

## Results

### Training Curves
<img width="556" height="435" alt="Training Accuracy vs Validation Accuracy" src="https://github.com/user-attachments/assets/1a22cfab-5b13-42c4-893e-f22cfd3b7b9e" />
<img width="547" height="435" alt="Training Loss vs Validation Loss" src="https://github.com/user-attachments/assets/50e5a8ad-979e-4ca0-974a-f5d603ee3605" />

### Confusion Matrix
<img width="649" height="547" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/081cb5eb-75ec-44ea-b8f1-1da8ac3369aa" />

### Classification Report
<img width="655" height="342" alt="Classification Report" src="https://github.com/user-attachments/assets/e42e92cf-3401-4dab-ab56-0feb1cbd7a50" />

### ROC Curves
<img width="691" height="547" alt="ROC Curve" src="https://github.com/user-attachments/assets/eef97162-ec52-4385-a440-e102c1ba08fb" />

### Sample Predictions
<img width="1732" height="1190" alt="Predictions" src="https://github.com/user-attachments/assets/004829fd-86f9-4503-bd6d-ea6458f7bbd4" />

## Project Output

This project produces a trained deep learning model that can classify brain MRI images into four tumor-related categories and provide evaluation metrics for model performance.

## Model Hosting

The trained model file is too large to host directly on GitHub, so it is available on Hugging Face:

https://huggingface.co/puspah/brain-tumor-detection-model/tree/main

## Run the App Locally

The project includes a Streamlit app for running local predictions with the saved model.

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

The app lets you upload an MRI image and returns the predicted class with a confidence score.

## Files

- `Brain_tumor_detection.ipynb` - Main notebook containing the full workflow
- `app.py` - Streamlit app for local inference
- `model1.h5` - Saved trained model

## Notes

- The notebook is written for a Kaggle-style environment, but it can also be adapted for local Jupyter or Colab use if the dataset paths are updated.
