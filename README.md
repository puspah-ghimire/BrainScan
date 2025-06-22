# BrainScan

An AI-powered application that uses deep learning to detect and classify brain tumors from MRI scans.

![Brain Scan Demo](https://github.com/user-attachments/assets/d2f47738-fd74-4a59-aed1-c68a64038b07)
![Brain Scan Demo](https://github.com/user-attachments/assets/963e8a69-b1b6-40c3-9fcc-636e3d7b553e)

## Overview

This project uses a trained deep learning model to detect and classify brain tumors from MRI scans. It includes a Flask web application that serves as an interface for users to upload MRI images and receive real-time predictions. The model classifies MRI scans into one of the following four categories:
- Pituitary Tumor
- Glioma
- Meningioma
- No Tumor (healthy brain)

## Features

- Upload and process brain MRI scans
- Real-time tumor classification
- Displays prediction confidence score
- User-friendly web interface
- Pre-trained model automatically downloaded from Hugging Face

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/username/BrainScan.git
   cd BrainScan
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python main.py
   ```

4. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

1. Access the web interface at http://localhost:5000
2. Upload an MRI scan image (JPG, PNG format)
3. Click the "Upload and Detect" button
4. View the results showing tumor classification and confidence score

## Sample Images

The repository includes sample MRI images in the `Sample MRI Images` folder that you can use for testing.

## Model Information

The model is automatically downloaded from Hugging Face on first run. It's a CNN-based architecture trained on a dataset of brain MRI scans with four classes. The model achieves high accuracy in classifying different types of brain tumors.

- The deep learning model is hosted on [Hugging Face](https://huggingface.co/puspah/brain-tumor-detection-model/tree/main)

