# Skin Disease Detection System

![Application Screenshot](screenshot.png) 

## Overview

This project is an advanced skin disease detection system that uses deep learning (ResNet152 architecture) to classify skin conditions from images. The application provides a user-friendly GUI for training models, making predictions, and evaluating performance.

## Features

- 🚀 **Model Training**: Train a ResNet152 model on your own dataset with customizable parameters
- 🔍 **Disease Prediction**: Upload skin images and get instant predictions with confidence scores
- 📊 **Performance Visualization**: Real-time training plots and evaluation metrics
- 🎯 **Confusion Matrix**: Visualize model performance across different classes
- ℹ️ **Model Information**: Detailed view of model architecture and training parameters
- ⚙️ **GPU Support**: Optimized for CUDA-enabled systems with dynamic memory allocation

## Installation

### Prerequisites

- Python 3.7+
- CUDA 11.2 (for GPU acceleration)
- cuDNN 8.0

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/skin-disease-detection.git
   cd skin-disease-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python finall.py
   ```

## Dataset Structure

Prepare your dataset in the following structure:
```
dataset_root/
├── 1. Eczema 1677/
├── 2. Melanoma 15.75k/
├── 3. Atopic Dermatitis - 1.25k/
├── 4. Basal Cell Carcinoma (BCC) 3323/
├── 5. Melanocytic Nevi (NV) - 7970/
├── 6. Benign Keratosis-like Lesions (BKL) 2624/
├── 7. Psoriasis pictures Lichen Planus and related diseases - 2k/
├── 8. Seborrheic Keratoses and other Benign Tumors - 1.8k/
├── 9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k/
└── 10. Warts Molluscum and other Viral Infections - 2103/
```

## Usage

1. **Training Tab**:
   - Select your dataset directory
   - Configure training parameters
   - Monitor real-time training progress

2. **Prediction Tab**:
   - Upload skin images
   - View predicted disease with confidence scores
   - See all possible diagnoses ranked by probability

3. **Evaluation Tab**:
   - Generate training history plots
   - View confusion matrix

4. **Model Info Tab**:
   - View model architecture details
   - Check system information

## Screenshots

Here are some example screenshots you might want to include (replace with your actual images):

1. **Main Interface**  
   ![Main Interface](images/main_interface.png)

2. **Training Progress**  
   ![Training](images/training.png)

3. **Prediction Results**  
   ![Prediction](images/prediction.png)

4. **Confusion Matrix**  
   ![Confusion Matrix](images/confusion_matrix.png)

## Technical Details

- **Model Architecture**: ResNet152 with custom top layers
- **Preprocessing**: ResNet-specific preprocessing with image augmentation
- **Training**: Uses class weighting for imbalanced datasets
- **Visualization**: Real-time matplotlib plots embedded in Tkinter

## Future Improvements

- Add support for more model architectures
- Implement patient history tracking
- Add multi-language support
- Develop mobile/cloud versions



