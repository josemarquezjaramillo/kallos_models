# Kallos Models: Project Overview

## 1. Introduction

`kallos_models` is a specialized Python package designed for training, tuning, and deploying deep learning time-series models for cryptocurrency price prediction. It provides a structured, modular, and robust framework that leverages industry-standard libraries like `darts`, `optuna`, and `scikit-learn`.

The core philosophy is to separate concerns into distinct modules, enabling a clean and maintainable workflow from data loading to model saving.

## 2. Modules

The package is composed of six key modules that work in concert:

-   **`preprocessing.py`**: Contains functions to create sophisticated data scaling and transformation pipelines tailored to different types of financial features.
-   **`datasets.py`**: Acts as the data bridge, responsible for fetching data from a database, generating robust walk-forward validation splits, and preparing `darts.TimeSeries` objects for modeling.
-   **`architectures.py`**: A simple factory module for instantiating `darts` forecasting models (like GRU, LSTM, Transformer) with specific hyperparameters.
-   **`tuner.py`**: Implements the hyperparameter optimization logic using `Optuna`. It systematically searches for the best model parameters by evaluating them across multiple walk-forward validation folds.
-   **`trainer.py`**: Handles the final stage of the process: training a model on the entire dataset using the optimal hyperparameters found by the tuner and saving the model and its associated data scaler to disk.
-   **`evaluation.py`**: Provides tools for evaluating trained models on hold-out test data, calculating performance metrics, and visualizing results.

## 3. Module Interaction

The modules are designed to interact in a three-phase process: **Tuning**, **Training**, and **Evaluation**.

### Tuning Phase

This phase is orchestrated by `tuner.py` to find the best hyperparameters.

1.  **`tuner.run_tuning`** initiates an `Optuna` study, creating or resuming a study in a PostgreSQL database.
2.  For each trial, the `objective` function calls **`datasets.load_features_from_db`** to get the full historical data.
3.  **`datasets.calculate_dynamic_wf_kwargs`** automatically determines appropriate parameters for walk-forward validation.
4.  **`datasets.get_walk_forward_splits`** generates a series of training and validation dataframes.
5.  For each split, **`datasets.prepare_darts_timeseries`** is called.
    -   Inside this function, **`preprocessing.create_feature_transformer`** creates a scaler.
    -   This scaler is **fitted only on the current split's training data**.
    -   Both training and validation data are transformed.
6.  Back in the `tuner`, **`architectures.create_model`** instantiates a model with the trial's hyperparameters.
7.  The model is trained on the split's training data and evaluated on the validation data.
8.  The average validation score across all splits is returned to Optuna, which uses it to guide the search for better parameters.

### Training Phase

This phase is orchestrated by `trainer.py` to produce the final, deployable artifacts.

1.  **`trainer.train_and_save_model`** is called with the best hyperparameters found during tuning.
2.  **`datasets.load_features_from_db`** loads the *entire* dataset up to a specified end date.
3.  **`preprocessing.create_feature_transformer`** creates a scaler.
4.  This scaler is **fitted on the entire dataset**.
5.  The entire dataset is transformed using the fitted scaler.
6.  **`architectures.create_model`** instantiates a model with the optimal hyperparameters.
7.  The model is trained on the full, normalized dataset.
8.  The final `darts` model and the `scikit-learn` scaler object are saved to disk.

### Evaluation Phase

This phase is orchestrated by `evaluation.py` to assess model performance on unseen data.

1.  **`evaluation.run_evaluation`** is called with paths to the saved model and scaler.
2.  **`datasets.load_features_from_db`** loads test data plus a lookback window for the specified period.
3.  The saved scaler is used to transform the test data **without refitting**.
4.  The model generates forecasts for the test period.
5.  **`evaluation.generate_evaluation_report`** calculates performance metrics like RMSE, MAE, and MAPE.
6.  **`evaluation.plot_forecast`** creates a visualization comparing actual vs. predicted values.
7.  The metrics and visualization are saved to disk for analysis.

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

**Step 3: Model Evaluation**

Finally, the user evaluates the trained model on a completely unseen test set (e.g., Q1 2023).

```bash
kallos-run evaluate \
    --model-name gru \
    --asset-id BTC \
    --model-path ./production_models/gru_BTC.pt \
    --scaler-path ./production_models/gru_BTC_scaler.pkl \
    --test-start-date 2023-01-01 \
    --test-end-date 2023-03-31 \
    --db-url "postgresql://user:pass@host/db" \
    --output-path ./evaluation_results
```

This command loads the saved model and scaler, evaluates performance on the specified test period, and saves two files to the `evaluation_results` directory:
-   `BTC_evaluation_metrics.json`: A JSON file with RMSE, MAE, and MAPE metrics.
-   `BTC_evaluation_plot.png`: A chart visualizing the model's forecasts against actual values.

## 5. Implementation Details

### Database Schema Expectations

The package expects a specific database schema with at least two tables:
- `DAILY_MARKET_DATA`: Contains price and volume data
- `DAILY_TECHNICAL_INDICATORS`: Contains technical indicators

### Default Feature Groups

The CLI uses the following default feature grouping:
```python
FEATURE_GROUPS = {
    'volume_features': ['volume', 'taker_buy_base_asset_volume'],
    'bounded_features': ['rsi', 'mfi'],
    'unbounded_features': ['macd_diff', 'bollinger_hband_indicator', 'bollinger_lband_indicator'],
    'roc_features': ['roc_1', 'roc_3', 'roc_7']
}
```

### Supported Models

Currently, the package supports three types of models from the Darts library:
1. GRU (Gated Recurrent Unit)
2. LSTM (Long Short-Term Memory)
3. Transformer (with attention mechanism)

Additional model types can be added by extending the `architectures.py` module.

## 6. For Researchers and Analysts

This package implements several best practices for time series forecasting in finance:

1. **Walk-forward validation** prevents look-ahead bias by simulating real-world forecasting conditions
2. **Feature-specific preprocessing** applies domain knowledge to handle different types of financial indicators
3. **Automated hyperparameter optimization** systematically finds the best model configuration
4. **Clean separation of concerns** improves reproducibility and maintainability
5. **Complete workflow** from data loading to model evaluation ensures consistent methodology

Researchers can extend this framework by:
- Adding new model architectures in `architectures.py`
- Implementing additional feature transformations in `preprocessing.py`
- Creating new evaluation metrics in `evaluation.py`
- Modifying the walk-forward validation strategy in `datasets.py`
