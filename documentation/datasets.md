# Module: `datasets.py`

## 1. Purpose

This module serves as the data access layer, bridging the raw data in the database with the format required by the `darts` library for modeling.

## 2. Core Components

-   **`load_features_from_db(...)`**: Connects to a SQL database, executes a query to fetch all features for a specific asset up to a given date, and returns a pandas DataFrame.
-   **`get_walk_forward_splits(...)`**: An essential generator function for robust time-series cross-validation. It yields pairs of training and validation DataFrames that simulate a real-world forecasting scenario.
-   **`prepare_darts_timeseries(...)`**: Takes training and validation DataFrames for a single fold, orchestrates their preprocessing, and converts them into `darts.TimeSeries` objects.

## 3. Detailed Explanation

### Data Loading
`load_features_from_db` is a straightforward utility that uses `SQLAlchemy` and `pandas` to pull data. It expects the table to have an `asset_id` and a `date` column for filtering.

### Walk-Forward Splitting
`get_walk_forward_splits` implements a "Sliding Window with Static Start" (also known as an expanding window) methodology. It works as follows:
1.  It starts with an initial training period (e.g., 2 years).
2.  The period immediately following is designated as the validation set (e.g., 3 months).
3.  It `yields` this first `(train_df, validation_df)` pair.
4.  It then expands the training set by a `step_months` (e.g., 3 months) and defines a new validation set.
5.  It `yields` this new, larger training set and the next validation set.
6.  This process repeats until the validation period would extend beyond the available data.

This method ensures that the model is always validated on data "in the future" relative to its training data, preventing look-ahead bias.

### TimeSeries Preparation
`prepare_darts_timeseries` is the critical link between raw data and the model. For a given train/validation split, it:
1.  Separates the target column from the feature (covariate) columns.
2.  Calls `preprocessing.create_feature_transformer` to get an unfitted scaler.
3.  **Crucially, it fits the scaler *only* on the training covariates (`covariates_train_df`).**
4.  It uses this fitted scaler to transform both the training and validation covariates. This prevents information from the validation set from "leaking" into the training process.
5.  Finally, it converts the four pandas DataFrames (target_train, target_val, covariates_train_norm, covariates_val_norm) into `darts.TimeSeries` objects, which is the required input format for all Darts models.
6.  It returns the `TimeSeries` objects and the scaler that was fitted on the training data.

## 4. Usage in Workflow

-   The `tuner.objective` function uses all three functions in this module to get the data needed for each trial of the hyperparameter search.
-   The `trainer.train_and_save_model` function uses `load_features_from_db` to get the full dataset for final training.
