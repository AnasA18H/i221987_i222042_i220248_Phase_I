# Network Intrusion Detection using Deep Learning

This project implements and compares three deep learning models (CNN, LSTM, and CNN-LSTM hybrid) for network intrusion detection using the NSL-KDD dataset.

## Overview

The project performs binary classification to detect network attacks vs normal traffic using:
- **CNN (Convolutional Neural Network)**: 1D convolutional layers for feature extraction
- **LSTM (Long Short-Term Memory)**: Recurrent layers for sequence modeling
- **CNN-LSTM Hybrid**: Combined architecture leveraging both convolutional and recurrent layers

## Dataset

The project uses the **NSL-KDD** dataset, which is an improved version of the KDD Cup 99 dataset for network intrusion detection. The dataset contains network connection records with 41 features and binary labels (Normal = 0, Attack = 1).

### Dataset Structure
- Training set: KDDTrain+.txt
- Test set: KDDTest+.txt
- Features: 41 network connection features (duration, protocol_type, service, flag, etc.)
- Labels: Binary classification (Normal/Attack)

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the NSL-KDD dataset and place it in the appropriate directory:
   - Update the file paths in the notebook to point to your dataset location
   - Default paths: `/kaggle/input/nslkdd/KDDTrain+.txt` and `/kaggle/input/nslkdd/KDDTest+.txt`

## Usage

1. Open the Jupyter notebook: `i221987_i222042_i222048.ipynb`
2. Update the dataset file paths if necessary
3. Run all cells sequentially

The notebook will:
- Load and preprocess the NSL-KDD dataset
- Train three models (CNN, LSTM, CNN-LSTM)
- Evaluate each model and generate performance metrics
- Save results and visualizations

## Model Architectures

### CNN Model
- Conv1D layer (64 filters, kernel size 3)
- MaxPooling1D
- Conv1D layer (128 filters, kernel size 3)
- GlobalAveragePooling1D
- Dense layers with dropout

### LSTM Model
- LSTM layer (64 units)
- Dropout layer
- Dense layers

### CNN-LSTM Hybrid Model
- Conv1D layer (64 filters, kernel size 3)
- MaxPooling1D
- LSTM layer (64 units)
- Dropout and Dense layers

## Outputs

The project generates:
- **Confusion matrices** for each model (saved as PNG files)
- **Performance comparison chart** (model_comparison.png)
- **Results CSV file** (nsl_kdd_results.csv) containing:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Matthews Correlation Coefficient (MCC)

Results are saved in the `/content/results` directory (or as specified in the notebook).

## Evaluation Metrics

The models are evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **MCC**: Matthews Correlation Coefficient

## Notes

- The project uses TensorFlow/Keras for deep learning
- Data preprocessing includes standardization and one-hot encoding
- Models are trained for 10 epochs with a validation split of 0.2
- Random seed is set to 42 for reproducibility

## Authors

- i221987
- i222042
- i222048

