# Kallos Models: Project Overview

## 1. Introduction

`kallos_models` is a specialized Python package designed for training, tuning, and deploying deep learning time-series models for cryptocurrency price prediction. It provides a structured, modular, and robust framework that leverages industry-standard libraries like `darts`, `optuna`, and `scikit-learn`.

The core philosophy is to separate concerns into distinct modules, enabling a clean and maintainable workflow from data loading to model saving.

## 2. Modules

The package is composed of five key modules that work in concert:

-   **`preprocessing.py`**: Contains functions to create sophisticated data scaling and transformation pipelines tailored to different types of financial features.
-   **`datasets.py`**: Acts as the data bridge, responsible for fetching data from a database, generating robust walk-forward validation splits, and preparing `darts.TimeSeries` objects for modeling.
-   **`architectures.py`**: A simple factory module for instantiating `darts` forecasting models (like GRU, LSTM, Transformer) with specific hyperparameters.
-   **`tuner.py`**: Implements the hyperparameter optimization logic using `Optuna`. It systematically searches for the best model parameters by evaluating them across multiple walk-forward validation folds.
-   **`trainer.py`**: Handles the final stage of the process: training a model on the entire dataset using the optimal hyperparameters found by the tuner and saving the model and its associated data scaler to disk.

## 3. Module Interaction

The modules are designed to interact in a two-phase process: **Tuning** and **Training**.

### Tuning Phase

This phase is orchestrated by `tuner.py` to find the best hyperparameters.

1.  **`tuner.run_tuning`** initiates an `Optuna` study.
2.  For each trial, the `objective` function calls **`datasets.load_features_from_db`** to get the full historical data.
3.  **`datasets.get_walk_forward_splits`** generates a series of training and validation dataframes.
4.  For each split, **`datasets.prepare_darts_timeseries`** is called.
    -   Inside this function, **`preprocessing.create_feature_transformer`** creates a scaler.
    -   This scaler is **fitted only on the current split's training data**.
    -   Both training and validation data are transformed.
5.  Back in the `tuner`, **`architectures.create_model`** instantiates a model with the trial's hyperparameters.
6.  The model is trained on the split's training data and evaluated on the validation data.
7.  The average validation score across all splits is returned to Optuna, which uses it to guide the search for better parameters.

### Training Phase

This phase is orchestrated by `trainer.py` to produce the final, deployable artifacts.

1.  **`trainer.train_and_save_model`** is called with the best hyperparameters found during tuning.
2.  **`datasets.load_features_from_db`** loads the *entire* dataset.
3.  **`preprocessing.create_feature_transformer`** creates a scaler.
4.  This scaler is **fitted on the entire dataset**.
5.  The entire dataset is transformed using the fitted scaler.
6.  **`architectures.create_model`** instantiates a model with the optimal hyperparameters.
7.  The model is trained on the full, normalized dataset.
8.  The final `darts` model and the `scikit-learn` scaler object are saved to disk.

## 4. Example Workflow

A user interacts with the system via the `main.py` command-line interface.

**Step 1: Hyperparameter Optimization**

The user first runs the `tune` command to find the best set of hyperparameters for a GRU model on BTC data up to January 1st, 2023.

```bash
kallos-run tune \
    --model-name gru \
    --asset-id BTC \
    --end-date 2023-01-01 \
    --db-url "postgresql://user:pass@host/db" \
    --n-trials 100
```

This process performs 100 trials of walk-forward cross-validation. At the end, it creates a file named `gru_BTC_best_params.json` containing the winning hyperparameter combination.

**Step 2: Final Model Training**

With the optimal parameters identified, the user runs the `train` command.

```bash
kallos-run train \
    --model-name gru \
    --asset-id BTC \
    --end-date 2023-01-01 \
    --db-url "postgresql://user:pass@host/db" \
    --params-file gru_BTC_best_params.json \
    --output-path ./production_models
```

This loads the entire dataset, trains one final GRU model using the parameters from the JSON file, and saves two artifacts in the `production_models` directory:
-   `gru_BTC.pt`: The serialized, trained Darts model.
-   `gru_BTC_scaler.pkl`: The pickled scikit-learn scaler fitted on the full dataset.
