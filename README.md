# Predictive Maintenance for Manufacturing Equipment

Notebook-first PyTorch project for binary machine-failure prediction on the AI4I 2020 dataset.

The notebook implements two stages:
- A baseline ANN with fixed hyperparameters
- An Optuna-tuned ANN, then checkpoint export and inference helpers

## Project Objective

Build a reliable binary classification system that predicts machine failure early from sensor and operational features, so maintenance can be scheduled proactively and unplanned downtime can be reduced.

Key goals:
- Handle severe class imbalance using SMOTE
- Improve failure detection quality beyond accuracy alone (precision/recall/F1)
- Use Optuna to tune ANN hyperparameters for better generalization
- Export a reusable PyTorch checkpoint for real-world inference

## What is in `Predictive.ipynb`

1. Data loading and inspection
2. Train/test split with stratification
3. Label encoding for `Type`
4. SMOTE on training data only
5. Feature scaling with `StandardScaler`
6. Baseline model training and evaluation
7. Optuna search (`n_trials=20`) for ANN hyperparameters
8. Best model retraining, checkpoint save, reload, and prediction examples

## Model and Tuning Details

PyTorch baseline model:
- Framework: `torch` / `torch.nn`
- Architecture: `11 -> 128 -> 64 -> 1`
- Activations: `ReLU`
- Regularization: `Dropout(0.3)`
- Loss: `BCEWithLogitsLoss`
- Optimizer: `AdamW`
- Baseline training setup: `epochs=100`, `batch_size=32`, `learning_rate=0.001`

Optuna hyperparameter tuning:
- Study direction: maximize validation accuracy
- Sampler: `TPESampler`
- Trials: `20`
- Early stopping + pruning used during tuning
- Tuned parameters include:
  - `num_hidden_layers`
  - `neurons_per_layer`
  - `epochs`
  - `learning_rate`
  - `dropout_rate`
  - `batch_size`
  - `optimizer` (`Adam`, `SGD`, `RMSprop`)
  - `weight_decay`

## Dataset

- Source: [AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)
- Samples: 10,000
- Target: `Machine failure` (0/1)
- Imbalance: about 3.4% failures

Feature columns used by the model:
- `Type`
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`
- `TWF`, `HDF`, `PWF`, `OSF`, `RNF`

## Current Notebook Metrics (latest saved outputs)

Baseline model:
- Accuracy: `0.9645`
- Precision: `0.4891`
- Recall: `0.9853`
- F1: `0.6537`

Optuna-selected model evaluation:
- Accuracy: `0.9815`
- Precision: `0.6535`
- Recall: `0.9706`
- F1: `0.7811`

Best Optuna trial reported:
- Trial: `10`
- Best validation accuracy during tuning: `0.9985`

## Installation

From this project folder:

```bash
pip install -e .
```

Optional (only if you want KaggleHub dataset download flow):

```bash
pip install -e ".[data]"
```

## Run

```bash
jupyter notebook Predictive.ipynb
```

## Path Note (important)

The notebook currently reads/saves files using Google Drive Colab-style paths such as:
- `/content/drive/MyDrive/.../ai4i2020_saved.csv`
- `/content/drive/MyDrive/.../best_predictive_model.pth`

If you run locally, update those paths to local files in this repository:
- `ai4i2020_saved.csv`
- `best_predictive_model.pth`

## Project Files

- `Predictive.ipynb`: main end-to-end workflow
- `ai4i2020_saved.csv`: dataset copy
- `best_predictive_model.pth`: trained checkpoint artifact
- `pyproject.toml`: package/dependency metadata
- `README.md`: project documentation
