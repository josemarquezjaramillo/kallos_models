# Kallos Models

`kallos_models` is a Python package for training and tuning deep learning forecasting models for cryptocurrency price prediction. It uses a walk-forward validation strategy for robust hyperparameter tuning with Optuna and saves final models and data scalers for inference.

## Features

- **Walk-Forward Validation:** Employs a robust time-series cross-validation methodology.
- **Hyperparameter Tuning:** Integrated with `Optuna` to find the best model parameters.
- **Modular Architecture:** Clean separation of data loading, preprocessing, model architecture, tuning, and training.
- **Extensible:** Easily add new models or preprocessing steps.
- **Supported Models:** GRU, LSTM, Transformer.

## Installation

To install the package, clone the repository and install it in editable mode with all dependencies:

```bash
git clone https://github.com/your_username/kallos_models.git
cd kallos_models
pip install -e .
```

## Usage

The package is run via the command-line interface.

### 1. Tune Hyperparameters

First, run the tuner to find the optimal hyperparameters for a given model and asset.

```bash
kallos-run tune \
    --model-name gru \
    --asset-id BTC \
    --end-date 2023-01-01 \
    --db-url "postgresql://user:password@host:port/dbname" \
    --n-trials 50
```

This will run 50 optimization trials and save the best parameters to a JSON file (e.g., `gru_BTC_best_params.json`).

### 2. Train Final Model

Next, use the best parameters to train a final model on the entire dataset.

```bash
kallos-run train \
    --model-name gru \
    --asset-id BTC \
    --end-date 2023-01-01 \
    --db-url "postgresql://user:password@host:port/dbname" \
    --params-file gru_BTC_best_params.json \
    --output-path ./trained_models
```

This will save the trained Darts model and the fitted scikit-learn scaler to the `./trained_models` directory.
