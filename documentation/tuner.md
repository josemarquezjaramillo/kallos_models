# Module: `tuner.py`

## 1. Purpose

This module orchestrates the hyperparameter optimization process using the `Optuna` library. Its goal is to find the best set of model hyperparameters by performing a robust, time-series-aware cross-validation.

## 2. Core Components

-   **`objective(trial, ...)`**: The objective function that `Optuna` seeks to minimize. It defines a single evaluation of a set of hyperparameters.
-   **`run_tuning(...)`**: The main entry point for the tuning process. It sets up the `Optuna` study and calls the `objective` function for a specified number of trials.

## 3. Detailed Explanation

### The Objective Function
The `objective` function is the heart of the tuning process. For each `trial` run by Optuna, it performs the following steps:

1.  **Define Search Space**: It uses the `trial.suggest_*` methods (e.g., `trial.suggest_int`, `trial.suggest_float`) to define the search space for each hyperparameter. Optuna will pick a value from this space for the current trial.
2.  **Load Data**: It calls `datasets.load_features_from_db` to get the complete dataset for the tuning period.
3.  **Get Walk-Forward Splits**: It calls `datasets.get_walk_forward_splits` to get a generator for the training/validation folds.
4.  **Iterate and Evaluate**: It loops through each `(train_df, val_df)` pair from the generator.
    a. It calls `datasets.prepare_darts_timeseries` to get normalized, preprocessed `TimeSeries` objects for the current fold.
    b. It calls `architectures.create_model` to instantiate a model with the current trial's hyperparameters.
    c. It trains the model on the training data (`model.fit`).
    d. It generates predictions on the validation data's timeline (`model.predict`).
    e. It calculates the Root Mean Squared Error (RMSE) between the predictions and the actual validation data.
    f. The RMSE score for the fold is stored.
5.  **Return Average Score**: After the loop finishes, it calculates the mean of all the stored RMSE scores. This average score is the final result for the trial, which is returned to Optuna.

### The Tuning Runner
The `run_tuning` function sets up and executes the study.

1.  **Create Study**: It initializes an `Optuna` study with `optuna.create_study(direction='minimize')`, telling Optuna that the goal is to find the parameters that result in the lowest possible score from the `objective` function.
2.  **Run Optimization**: It calls `study.optimize()`, passing the `objective` function and the desired `n_trials`. Optuna handles the search strategy (e.g., TPE) to intelligently explore the hyperparameter space.
3.  **Return Best Parameters**: Once all trials are complete, it extracts and returns the best set of hyperparameters found during the study (`study.best_params`).

## 4. Usage in Workflow

The `run_tuning` function is the primary function called by the `main.py` script when the user executes the `kallos-run tune` command. The result of this process (a dictionary of the best parameters) is typically saved to a JSON file to be used in the final training phase.
