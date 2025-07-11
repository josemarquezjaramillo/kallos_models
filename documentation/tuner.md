# Module: `tuner.py`

## 1. Purpose

This module orchestrates the hyperparameter optimization process using the `Optuna` library. Its goal is to find the best set of model hyperparameters by performing a robust, time-series-aware cross-validation.

## 2. Core Components

-   **`objective(trial, ...)`**: The objective function that `Optuna` seeks to minimize. It defines a single evaluation of a set of hyperparameters.
-   **`run_tuning(...)`**: The main entry point for the tuning process. It sets up the `Optuna` study and calls the `objective` function for a specified number of trials.

## 3. Detailed Explanation

### The Objective Function
The `objective` function is the heart of the tuning process. For each `trial` run by Optuna, it performs the following steps:

1.  **Define PyTorch Lightning Training Parameters**: It sets up `EarlyStopping` and other PyTorch Lightning trainer configurations to ensure consistent training behavior across trials.
   
2.  **Define Search Space**: It uses the `trial.suggest_*` methods (e.g., `trial.suggest_int`, `trial.suggest_float`) to define the search space for each hyperparameter. Optuna will pick a value from this space for the current trial. The hyperparameters include:
    - `hidden_dim`: Size of the hidden layers (log-scale search between 32 and 256)
    - `n_rnn_layers`: Number of RNN layers (between 1 and 4)
    - `dropout`: Dropout rate for regularization (between 0.0 and 0.7)
    - `batch_size`: Mini-batch size for training (categorical choice among 32, 64, or 128)
    - `input_chunk_length`: Size of the lookback window (between 30 and 70 days)
    - `learning_rate`: Learning rate for the optimizer (log-scale search between 1e-5 and 1e-2)

3.  **Load Data**: It calls `datasets.load_features_from_db` to get the complete dataset for the tuning period.

4.  **Calculate Dynamic Walk-Forward Parameters**: It calls `datasets.calculate_dynamic_wf_kwargs` to automatically determine appropriate parameters for walk-forward validation based on the dataset size.

5.  **Get Walk-Forward Splits**: It calls `datasets.get_walk_forward_splits` to get a generator for the training/validation folds.

6.  **Iterate and Evaluate**: It loops through each `(train_df, val_df)` pair from the generator.
    a. It calls `datasets.prepare_darts_timeseries` to get normalized, preprocessed `TimeSeries` objects for the current fold.
    b. It calls `architectures.create_model` to instantiate a model with the current trial's hyperparameters and the PyTorch Lightning trainer configuration.
    c. It trains the model on the training data (`model.fit`), providing both validation data for early stopping.
    d. It generates predictions on the validation data's timeline (`model.predict`).
    e. It calculates the Root Mean Squared Error (RMSE) between the predictions and the actual validation data.
    f. The RMSE score for the fold is stored.

7.  **Return Average Score**: After the loop finishes, it calculates the mean of all the stored RMSE scores. This average score is the final result for the trial, which is returned to Optuna.

### The Tuning Runner
The `run_tuning` function sets up and executes the study.

1.  **Create Database Storage**: It first creates a persistent storage for the Optuna study in a PostgreSQL database. This allows:
    - Resuming interrupted tuning sessions
    - Parallel tuning across multiple processes or machines
    - Long-term storage of all trial results

2.  **Handle Optional Schema**: The function accepts an `optuna_schema` parameter that allows specifying a custom schema in the database for storing Optuna studies. This is useful for organizing multiple studies in the same database.

3.  **Name the Study**: It uses the `study_name` parameter to identify the study in the database, allowing multiple studies to coexist (e.g., separate studies for different models or assets).

4.  **Create or Resume Study**: It initializes an `Optuna` study with `optuna.create_study(direction='minimize')`, with `load_if_exists=True` to resume an existing study if one with the same name exists in the database.

5.  **Check Existing Trials**: If the study already has equal or more than the requested number of trials (`n_trials`), it returns the study without running additional trials.

6.  **Run Optimization**: If more trials are needed, it calls `study.optimize()`, passing the `objective` function and the remaining number of trials needed to reach `n_trials`. Optuna handles the search strategy (e.g., TPE) to intelligently explore the hyperparameter space.

7.  **Log Best Results**: Once all trials are complete, it logs the best parameters and their performance score.

8.  **Return Study**: It returns the complete study object, which contains all trial information and the best parameters.

## 4. Usage in Workflow

The `run_tuning` function is the primary function called by the `main.py` script when the user executes the `kallos-run tune` command. The full function signature is:

```python
def run_tuning(
    asset_id: str,
    end_date: str,
    db_kwargs: Dict[str, Union[int, str]],
    optuna_schema: Union[str, None],
    model_name: str,
    study_name: str,
    target_col: str,
    feature_groups: Dict[str, List[str]],
    n_trials: int
) -> optuna.Study:
```

Example usage:

```python
best_study = tuner.run_tuning(
    asset_id="BTC",
    end_date="2023-01-01",
    db_kwargs={
        "postgres_user": "user",
        "postgres_password": "pass",
        "postgres_host": "localhost",
        "postgres_port": 5432,
        "postgres_db": "crypto_db"
    },
    optuna_schema="optuna_studies",  # Optional schema for organizing studies
    model_name="gru",
    study_name="gru_btc_optimization",  # Unique study identifier
    target_col="close",
    feature_groups={
        "volume_features": ["volume", "taker_buy_base_asset_volume"],
        "bounded_features": ["rsi", "mfi"],
        "unbounded_features": ["macd_diff"]
    },
    n_trials=100
)

# Extract best parameters for saving or further use
best_params = best_study.best_params
```

The result of this process (a dictionary of the best parameters) is typically saved to a JSON file to be used in the final training phase.
