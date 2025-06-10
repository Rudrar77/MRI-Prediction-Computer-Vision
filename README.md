# Brain Tumor Classification using Deep Learning

A deep learning project for classifying brain MRI images into three categories: brain tumor, brain glioma, and brain meningioma using PyTorch and computer vision techniques.

## Overview

This project implements a convolutional neural network to classify brain MRI scans into three distinct categories:
- **Brain Tumor** (General tumor classification)
- **Brain Glioma** (Specific type of brain tumor)
- **Brain Meningioma** (Tumor of the meninges)

The model achieves **71.04% accuracy** on the test dataset with varying performance across different tumor types.

## Dataset

The project uses the Brain Cancer MRI Dataset from Kaggle, sourced via KaggleHub:
- **Source**: `orvile/brain-cancer-mri-dataset`
- **Classes**: 3 (brain_tumor, brain_glioma, brain_menin)
- **Split**: 80% training, 20% testing
- **Preprocessing**: Images resized to 256x256, normalized using ImageNet statistics

## Requirements

```python
torch
torchvision
kagglehub
numpy
matplotlib
sklearn
```

## Data Preprocessing

The images undergo the following transformations:
- Resize to 256x256 pixels
- Convert to tensor format
- Normalize with ImageNet mean and standard deviation:
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]

## Model Configuration

- **Batch Size**: 32
- **Epochs**: 10
- **Learning Rate**: 0.001
- **Architecture**: CNN (specific architecture not shown in code)

## Results

### Overall Performance
- **Test Accuracy**: 71.04%

### Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Brain Tumor (0) | 0.97 | 0.71 | 0.82 | 410 |
| Brain Glioma (1) | 0.53 | 0.96 | 0.69 | 391 |
| Brain Meningioma (2) | 0.95 | 0.47 | 0.63 | 411 |

### Analysis
- **Brain Tumor**: High precision (97%) but moderate recall (71%)
- **Brain Glioma**: High recall (96%) but lower precision (53%)
- **Brain Meningioma**: High precision (95%) but low recall (47%)

## Usage

### 1. Data Loading
```python
import kagglehub
orvile_brain_cancer_mri_dataset_path = kagglehub.dataset_download('orvile/brain-cancer-mri-dataset')
```

### 2. Data Preprocessing
```python
transformed_data = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
```

### 3. Dataset Split
```python
train_split = int(len(dataset) * 0.8)
test_split = len(dataset) - train_split
train_data, test_data = torch.utils.data.random_split(dataset, lengths=[train_split, test_split])
```

## Visualization

The project includes visualization capabilities:
- **Training Data Visualization**: Display sample images with their corresponding labels
- **Prediction Visualization**: Show test images with true vs predicted labels (color-coded: green for correct, red for incorrect predictions)

## Key Features

1. **Automated Data Download**: Uses KaggleHub for seamless dataset acquisition
2. **Image Preprocessing**: Standardized preprocessing pipeline for consistent input
3. **Train/Test Split**: Proper data splitting for unbiased evaluation
4. **Performance Metrics**: Comprehensive evaluation using precision, recall, and F1-score
5. **Visual Analysis**: Sample visualization and prediction comparison tools

## License

Please refer to the original dataset license from Kaggle for usage terms and conditions.

## Acknowledgments

- Dataset provided by Orvile on Kaggle
- Built using PyTorch and torchvision libraries
- Evaluation metrics from scikit-learn
