# Kallos Models

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

`kallos_models` is a structured Python package for training, tuning, and evaluating deep learning time-series models for cryptocurrency price prediction. It provides a command-line interface to manage an end-to-end MLOps workflow, from robust hyperparameter optimization to final model evaluation on hold-out data.

## Features

-   **End-to-End Workflow**: A CLI for a complete workflow: tuning, training, and evaluation.
-   **Robust Validation**: Employs walk-forward cross-validation for reliable hyperparameter tuning and a hold-out set for final evaluation.
-   **Hyperparameter Optimization**: Integrated with `Optuna` to systematically find the best model parameters.
-   **Modular Architecture**: Clean separation of data loading, preprocessing, model architecture, and evaluation logic.
-   **Custom Preprocessing**: Applies tailored transformation pipelines to different groups of financial features.
-   **Supported Models**: Easily extensible factory for `darts` models like GRU, LSTM, and Transformer.

## Architecture

The package is composed of several key modules that work in concert:

-   `preprocessing.py`: Creates `scikit-learn` pipelines for feature-specific normalization.
-   `datasets.py`: Handles data loading from a database and generates walk-forward validation splits.
-   `architectures.py`: A factory for instantiating `darts` forecasting models.
-   `tuner.py`: Implements the hyperparameter optimization logic using `Optuna`.
-   `trainer.py`: Trains a final model on all available data using optimal hyperparameters.
-   `evaluation.py`: Evaluates a trained model on a hold-out test set, generating metrics and plots.
-   `main.py`: Provides the command-line interface (CLI) to orchestrate the entire workflow.

## Installation

Clone the repository and use `pip` to install the package in editable mode. This will also install all required dependencies listed in `setup.py`.

```bash
git clone https://github.com/your_username/kallos_models.git
cd kallos_models
pip install -e .
```

## Workflow and Usage

The recommended workflow ensures that the model is trained and evaluated robustly, without data leakage. It consists of three distinct steps performed via the CLI.

### Step 1: Tune Hyperparameters

First, run the tuner to find the optimal hyperparameters for a given model and asset. This is done on a dataset that **excludes the final hold-out test set**.

```bash
kallos-run tune \
    --model-name gru \
    --asset-id BTC \
    --end-date 2023-09-30 \
    --db-url "postgresql://user:pass@host/db" \
    --n-trials 100
```

This command runs 100 optimization trials on data up to September 30, 2023. It saves the best parameters to a JSON file (e.g., `gru_BTC_best_params.json`).

### Step 2: Train Final Model

Next, use the best parameters to train a final model. This should be done on the **exact same data period** as the tuning step.

```bash
kallos-run train \
    --model-name gru \
    --asset-id BTC \
    --end-date 2023-09-30 \
    --db-url "postgresql://user:pass@host/db" \
    --params-file gru_BTC_best_params.json \
    --output-path ./trained_models
```

This saves the trained Darts model (`gru_BTC.pt`) and the fitted scikit-learn scaler (`gru_BTC_scaler.pkl`) to the `./trained_models` directory.

### Step 3: Evaluate on Hold-Out Data

Finally, evaluate the trained model's performance on a completely unseen hold-out test set (e.g., the period after the training `end-date`).

```bash
kallos-run evaluate \
    --model-name gru \
    --asset-id BTC \
    --model-path ./trained_models/gru_BTC.pt \
    --scaler-path ./trained_models/gru_BTC_scaler.pkl \
    --test-start-date 2023-10-01 \
    --test-end-date 2023-12-31 \
    --db-url "postgresql://user:pass@host/db" \
    --output-path ./evaluation_results
```

This command loads the saved model and scaler, tests performance on Q4 2023 data, and saves two artifacts to `./evaluation_results`:
-   `BTC_evaluation_plot.png`: A chart comparing the model's forecast to the actual values.
-   `BTC_evaluation_metrics.json`: A report with RMSE, MAE, and MAPE scores.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
