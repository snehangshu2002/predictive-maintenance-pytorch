# Predictive Maintenance for Manufacturing Equipment

A PyTorch-based binary classification project that predicts machine failure using the **AI4I 2020 Predictive Maintenance Dataset**. The model uses SMOTE for class balancing, a feedforward neural network for prediction, and comprehensive evaluation metrics.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Dependencies](#dependencies)
- [References](#references)

---

## Overview

Manufacturing equipment failures are costly in terms of downtime and repair expenses. This project builds a neural network model that predicts whether a machine will fail based on sensor readings and operational parameters. The key challenge is the **severe class imbalance** (~3% failure rate), addressed using **SMOTE** oversampling.

---

## Dataset

| Property | Value |
|----------|-------|
| **Source** | [AI4I 2020 Predictive Maintenance Dataset (Kaggle)](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020) |
| **Samples** | 10,000 |
| **Classes** | Binary (0 = No Failure, 1 = Failure) |
| **Imbalance** | 9,661 : 339 (~96.6% : 3.4%) |

### Features (11 total)

| Feature | Type | Description |
|---------|------|-------------|
| `Type` | Categorical | Machine type (L, M, H) — label encoded |
| `Air temperature [K]` | Continuous | Ambient air temperature in Kelvin |
| `Process temperature [K]` | Continuous | Process temperature in Kelvin |
| `Rotational speed [rpm]` | Integer | Machine rotational speed |
| `Torque [Nm]` | Continuous | Torque applied in Newton-meters |
| `Tool wear [min]` | Integer | Tool wear duration in minutes |
| `TWF` | Binary | Tool Wear Failure flag |
| `HDF` | Binary | Heat Dissipation Failure flag |
| `PWF` | Binary | Power Failure flag |
| `OSF` | Binary | Overstrain Failure flag |
| `RNF` | Binary | Random Failure flag |

---

## Features

- **SMOTE oversampling** to balance the training set (7,729 : 7,729 after resampling)
- **Stratified train/test split** (80/20) preserving the failure ratio
- **Standard Scaling** applied after SMOTE to prevent data leakage
- **Feedforward Neural Network** with Dropout regularization
- **Binary Cross-Entropy** loss with `BCEWithLogitsLoss`
- **Comprehensive evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

---

## Architecture

```
Input (11 features)
        │
        ▼
  Linear(11 → 128)
        │
        ▼
    ReLU + Dropout(0.3)
        │
        ▼
  Linear(128 → 64)
        │
        ▼
    ReLU + Dropout(0.3)
        │
        ▼
   Linear(64 → 1)     ← Single logit output
        │
        ▼
  torch.sigmoid()     ← During inference only
        │
        ▼
  Threshold (≥ 0.5)   ← Binary prediction
```

---

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
# Clone or navigate to the project directory
cd "Predictive Maintenance for Manufacturing Equipment.ipynb"

# Install all dependencies
pip install -e .

# Or install manually
pip install pandas numpy matplotlib seaborn torch scikit-learn imbalanced-learn optuna torchinfo kagglehub
```

---

## Usage

Open the Jupyter notebook and run all cells sequentially:

```bash
jupyter notebook Predictive.ipynb
```

The notebook executes the full pipeline:

1. Load & preprocess data
2. Apply SMOTE to training set
3. Scale features
4. Build & train the neural network
5. Evaluate on the held-out test set

---

## Project Structure

```
Predictive Maintenance for Manufacturing Equipment.ipynb/
├── Predictive.ipynb                        # Main notebook
├── ai4i2020_saved.csv                      # Dataset file
├── pyproject.toml                          # Project dependencies
├── README.md                               # This file
├── .gitignore                              # Ignored files
└── chat-PyTorch Binary Classification Guide.md  # AI Q&A guide (not tracked)
```

---

## Model Training

| Hyperparameter | Value |
|----------------|-------|
| **Optimizer** | AdamW |
| **Loss** | `BCEWithLogitsLoss` |
| **Batch size** | 32 |
| **Epochs** | 100 |
| **Learning rate** | Default (AdamW) |
| **Dropout rate** | 0.3 |
| **Activation** | ReLU |
| **Random seed** | 42 |

### Training Loop

```python
for epoch in range(epochs):
    total_epoch_loss = 0
    for batch_features, batch_labels in train_loader:
        outputs = model(batch_features).view(-1)
        loss = criterion(outputs, batch_labels.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_epoch_loss += loss.item()
    avg_loss = total_epoch_loss / len(train_loader)
```

---

## Evaluation

The model is evaluated on the **unseen test set** (not touched by SMOTE) using:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct prediction rate |
| **Precision** | Of predicted failures, how many were real? |
| **Recall** | Of actual failures, how many did we catch? |
| **F1-Score** | Harmonic mean of precision and recall |
| **Confusion Matrix** | TP / FP / TN / FN breakdown |

```python
with torch.no_grad():
    model.eval()
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch).view(-1)
        predicted = (torch.sigmoid(outputs) > 0.5).long()
        # Collect all_preds and all_labels
```

---

## Results

> **Note**: Results depend on training configuration and random seed. See the notebook for actual output.

### Common Challenge

Due to extreme class imbalance, the model may learn the "lazy" strategy of always predicting class 0 (no failure), yielding high accuracy but zero precision/recall for the failure class. Potential remedies:

- Increase training epochs
- Adjust learning rate
- Add `pos_weight` to the loss function
- Tune the classification threshold (e.g., 0.3 instead of 0.5)
- Use focal loss or cost-sensitive learning

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Data loading & manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting & visualization |
| `seaborn` | Statistical visualization (confusion matrix heatmap) |
| `torch` | Neural network framework |
| `scikit-learn` | Train/test split, scaling, evaluation metrics |
| `imbalanced-learn` | SMOTE oversampling |
| `optuna` | Hyperparameter tuning (available for future use) |
| `torchinfo` | Model architecture summary |
| `kagglehub` | Dataset download from Kaggle |

---

## References

- [AI4I 2020 Dataset Paper](https://www.mdpi.com/2504-4494/4/3/35)
- [SMOTE Paper — Chawla et al. (2002)](https://arxiv.org/abs/1106.1813)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

*Built with PyTorch · AI4I 2020 Dataset · SMOTE Balanced*
