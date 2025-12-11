# Emotion Detection with Advanced Image Processing

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A state-of-the-art deep learning project for real-time emotion detection from facial expressions using advanced image processing techniques and convolutional neural networks (CNN).

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Image Processing Techniques](#image-processing-techniques)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

This project implements an advanced emotion detection system that recognizes seven different facial expressions: **Angry**, **Disgust**, **Fear**, **Happy**, **Sad**, **Surprise**, and **Neutral**. The system leverages deep learning with optimized CNN architecture and comprehensive image preprocessing techniques to achieve high accuracy in real-time emotion recognition.

### Key Highlights

- 🎭 **7 Emotion Classes**: Comprehensive emotion recognition across multiple categories
- 🚀 **High-Speed Training**: Optimized CNN architecture with attention mechanism
- 🔧 **Advanced Image Processing**: Multiple preprocessing techniques including CLAHE, unsharp masking, and bilateral filtering
- 💾 **Smart Caching**: Memory-efficient data loading with intelligent caching
- 🎯 **Attention Mechanism**: Enhanced feature extraction with spatial attention
- 📊 **Real-time Inference**: Optimized for real-time emotion detection

## ✨ Key Features

### Model Architecture

- **Custom CNN Architecture**: 4-layer deep convolutional network with residual connections
- **Attention Mechanism**: Spatial attention for enhanced feature extraction
- **Batch Normalization**: Improved training stability and convergence
- **Dropout Regularization**: Prevents overfitting with adaptive dropout rates
- **Global Average Pooling**: Reduces overfitting compared to fully connected layers

### Image Processing Pipeline

1. **Noise Reduction**: Bilateral filtering and Gaussian blur
2. **Contrast Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
3. **Edge Enhancement**: Unsharp masking and Laplacian filtering
4. **Gamma Correction**: Dynamic brightness adjustment
5. **Morphological Operations**: Noise cleaning and edge refinement

### Optimization Features

- **Mixed Precision Training**: Accelerated training with FP16
- **Dynamic Batch Sizing**: Automatic optimal batch size detection
- **Data Augmentation**: Controlled random transformations
- **Multi-worker Loading**: Parallel data loading with prefetching
- **GPU Memory Management**: Efficient CUDA memory utilization

## 🏗️ Architecture

### Model Structure

```
HighSpeedEmotionCNN
├── Conv Block 1: 1 → 64 channels
├── Conv Block 2: 64 → 128 channels
├── Conv Block 3: 128 → 256 channels
├── Conv Block 4: 256 → 512 channels
├── Attention Module
├── Global Average Pooling
└── Classifier
    ├── Dropout (0.3)
    ├── Linear: 512 → 256
    ├── BatchNorm + ReLU
    ├── Dropout (0.15)
    ├── Linear: 256 → 128
    ├── BatchNorm + ReLU
    ├── Dropout (0.09)
    └── Linear: 128 → 7 (output)
```

### Image Processing Techniques

The project implements multiple image preprocessing strategies:

1. **Original**: Baseline without preprocessing
2. **Combined Enhancement Fast**: Balanced preprocessing pipeline
3. **Enhancement Aggressive**: Heavy preprocessing for challenging images
4. **Enhancement Subtle**: Light preprocessing for high-quality images
5. **Unsharp Masking**: Edge enhancement technique
6. **Adaptive Techniques**: Dynamic preprocessing based on image characteristics

## 📊 Dataset

### FER2013 Dataset

- **Total Images**: ~35,000 grayscale images
- **Image Size**: 48×48 pixels
- **Classes**: 7 emotion categories
- **Split**: Training and testing sets
- **Format**: Grayscale, single channel

### Emotion Categories

| Emotion  | Description |
|----------|-------------|
| Angry    | Angry facial expression |
| Disgust  | Disgusted facial expression |
| Fear     | Fearful facial expression |
| Happy    | Happy/joyful facial expression |
| Sad      | Sad facial expression |
| Surprise | Surprised facial expression |
| Neutral  | Neutral/no emotion expression |

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM
- 5GB+ disk space

### Step 1: Clone the Repository

```bash
git clone https://github.com/ChanukaWelagedara/Emotion-Detection-with-Advanced-Image-Processing.git
cd Emotion-Detection-with-Advanced-Image-Processing
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n emotion-detection python=3.8
conda activate emotion-detection
```

### Step 3: Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python numpy pandas matplotlib seaborn scikit-learn
pip install pillow dlib jupyter
```

### Step 4: Download Dataset

```bash
# Install kaggle CLI
pip install kaggle

# Setup Kaggle API credentials (place kaggle.json in ~/.kaggle/)
# Download FER2013 dataset
kaggle datasets download -d msambare/fer2013 -p ./fer2013
unzip ./fer2013/fer2013.zip -d ./fer2013
```

## 💻 Usage

### Training the Model

#### Option 1: Using Jupyter Notebook

```bash
jupyter notebook code.ipynb
```

Run all cells sequentially to:
1. Load and visualize the dataset
2. Apply image processing techniques
3. Train the model with different preprocessing methods
4. Evaluate performance and generate metrics

#### Option 2: Python Script (Custom)

```python
import torch
from code import HighSpeedEmotionCNN, create_optimized_dataloaders

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = "./fer2013"
technique_name = "combined_enhancement_fast"

# Create dataloaders
train_loader, test_loader, num_classes = create_optimized_dataloaders(
    dataset_path, 
    batch_size=128, 
    technique_name=technique_name
)

# Initialize model
model = HighSpeedEmotionCNN(num_classes=num_classes).to(device)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(50):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

### Inference on External Images

```python
import cv2
from PIL import Image
from torchvision import transforms

# Load trained model
model = HighSpeedEmotionCNN(num_classes=7)
model.load_state_dict(torch.load('best_model_unsharp_masking.pth'))
model.eval()

# Load and preprocess image
image = cv2.imread('external/happy.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (48, 48))
image = Image.fromarray(image)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

input_tensor = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    emotion = emotions[predicted.item()]
    print(f"Predicted Emotion: {emotion}")
```

### Model Evaluation

```python
from sklearn.metrics import classification_report, confusion_matrix

# Evaluate on test set
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Generate metrics
print(classification_report(all_labels, all_predictions))
print(confusion_matrix(all_labels, all_predictions))
```

## 🔧 Image Processing Techniques

### Available Techniques

| Technique | Description | Use Case |
|-----------|-------------|----------|
| `original` | No preprocessing | Baseline comparison |
| `combined_enhancement_fast` | Balanced pipeline with CLAHE, gamma correction, and sharpening | General purpose |
| `enhancement_aggressive` | Heavy noise reduction and strong contrast enhancement | Low-quality images |
| `enhancement_subtle` | Light preprocessing | High-quality images |
| `unsharp_masking` | Edge enhancement with Gaussian blur | Detail preservation |

### Custom Technique Selection

```python
# In dataset creation
train_dataset = OptimizedFER2013Dataset(
    root_dir="./fer2013/train",
    transform=train_transform,
    technique_name="combined_enhancement_fast"  # Change here
)
```

## 📈 Model Performance

### Training Configuration

- **Optimizer**: AdamW with weight decay
- **Learning Rate**: 0.001 with ReduceLROnPlateau scheduler
- **Batch Size**: 128 (auto-optimized)
- **Epochs**: 50-100
- **Loss Function**: Cross-Entropy Loss
- **Mixed Precision**: Enabled (FP16)

### Expected Results

| Metric | Value |
|--------|-------|
| Training Accuracy | ~85-90% |
| Test Accuracy | ~70-75% |
| Training Time | ~30-45 min (GPU) |
| Inference Time | ~10-15ms per image |

## 📁 Project Structure

```
Emotion-Detection-with-Advanced-Image-Processing/
│
├── code.ipynb                          # Main Jupyter notebook
├── best_model_unsharp_masking.pth     # Pre-trained model weights
├── README.md                           # Project documentation
│
├── external/                           # External test images
│   ├── angry.png
│   ├── disgust.jpg
│   ├── fear.png
│   ├── sad1.jpg
│   ├── surprise.png
│   └── h1.jpg
│
└── fer2013/                            # Dataset directory (after download)
    ├── train/
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── sad/
    │   ├── surprise/
    │   └── neutral/
    └── test/
        ├── angry/
        ├── disgust/
        ├── fear/
        ├── happy/
        ├── sad/
        ├── surprise/
        └── neutral/
```

## 📦 Requirements

### Core Dependencies

```
python>=3.8
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
Pillow>=10.0.0
dlib>=19.24.0
```

### Optional Dependencies

```
jupyter>=1.0.0
kaggle>=1.5.0
tqdm>=4.65.0
```

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 🙏 Acknowledgments

- **FER2013 Dataset**: [Kaggle - FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
- **PyTorch Team**: For the excellent deep learning framework
- **OpenCV Community**: For comprehensive computer vision tools
- **Research Papers**: Various papers on emotion recognition and CNN architectures


---


