# Module: `datasets.py`

## 1. Purpose

This module serves as the data access layer, bridging the raw data in the database with the format required by the `darts` library for modeling.

## 2. Core Components

-   **`load_features_from_db(...)`**: Connects to a SQL database, executes a query to fetch all features for a specific asset up to a given date, and returns a pandas DataFrame.
-   **`calculate_dynamic_wf_kwargs(...)`**: Automatically calculates appropriate parameters for walk-forward validation based on the dataset's size and characteristics.
-   **`get_walk_forward_splits(...)`**: An essential generator function for robust time-series cross-validation. It yields pairs of training and validation DataFrames that simulate a real-world forecasting scenario.
-   **`prepare_darts_timeseries(...)`**: Takes training and validation DataFrames for a single fold, orchestrates their preprocessing, and converts them into `darts.TimeSeries` objects.

## 3. Detailed Explanation

### Data Loading
`load_features_from_db` is a utility that uses `SQLAlchemy` and `pandas` to pull data. It creates a database connection using the provided credentials and executes a sophisticated SQL query that:
1. Fetches price data, volume data, and numerous technical indicators
2. Calculates log returns and other derived features directly in SQL
3. Handles time alignment and potential missing values
4. Allows specifying both start and end dates for the data window

The function processes the query results to ensure proper datetime indexing and frequency (daily data), with timezone handling to make the DataFrame compatible with Darts' requirements.

### Dynamic Walk-Forward Parameter Calculation
`calculate_dynamic_wf_kwargs` is a key innovation that automatically determines appropriate walk-forward validation parameters based on the dataset's characteristics:

1. It analyzes the total time span of the available data
2. Based on the desired number of validation folds, it calculates:
   - The appropriate number of years for the initial training period (ensuring sufficient data for model learning)
   - The appropriate number of months for each validation window (ensuring statistical significance)
   - The step size for advancing between folds (typically equal to the validation window size)

This dynamic approach ensures that the walk-forward validation methodology adapts to different assets and time periods without requiring manual parameter tuning.

### Walk-Forward Splitting
`get_walk_forward_splits` implements a "Sliding Window with Static Start" (also known as an expanding window) methodology. It works as follows:
1.  It starts with an initial training period (e.g., 2 years).
2.  The period immediately following is designated as the validation set (e.g., 3 months).
3.  It `yields` this first `(train_df, validation_df)` pair.
4.  It then expands the training set by a `step_months` (e.g., 3 months) and defines a new validation set.
5.  It `yields` this new, larger training set and the next validation set.
6.  This process repeats until the validation period would extend beyond the available data.

This method ensures that the model is always validated on data "in the future" relative to its training data, preventing look-ahead bias while maximizing the use of historical data for training.

### TimeSeries Preparation
`prepare_darts_timeseries` is the critical link between raw data and the model. For a given train/validation split, it:
1.  Separates the target column from the feature (covariate) columns.
2.  Calls `preprocessing.create_feature_transformer` to get an unfitted scaler.
3.  **Crucially, it fits the scaler *only* on the training covariates (`covariates_train_df`).**
4.  It uses this fitted scaler to transform both the training and validation covariates. This prevents information from the validation set from "leaking" into the training process.
5.  Finally, it converts the four pandas DataFrames (target_train, target_val, covariates_train_norm, covariates_val_norm) into `darts.TimeSeries` objects, which is the required input format for all Darts models.
6.  It returns the `TimeSeries` objects and the scaler that was fitted on the training data.

## 4. Usage in Workflow

-   The `tuner.objective` function uses all functions in this module to:
    1. Load data with `load_features_from_db`
    2. Calculate appropriate walk-forward parameters with `calculate_dynamic_wf_kwargs`
    3. Generate multiple training/validation folds with `get_walk_forward_splits`
    4. Prepare normalized TimeSeries for each fold with `prepare_darts_timeseries`

-   The `trainer.train_and_save_model` function uses `load_features_from_db` to get the full dataset for final training.

-   The `evaluation.run_evaluation` function uses `load_features_from_db` to get the test data and a sufficient lookback period for model evaluation.

## 5. Example Usage

```python
# Load data and calculate dynamic walk-forward parameters
full_df = datasets.load_features_from_db(asset_id, end_date, db_kwargs)
wf_kwargs = datasets.calculate_dynamic_wf_kwargs(full_df, target_folds=5)

# Generate walk-forward splits
splits_generator = datasets.get_walk_forward_splits(
    full_df,
    train_years=wf_kwargs["train_years"],
    val_months=wf_kwargs["val_months"],
    step_months=wf_kwargs["step_months"]
)

# Process each fold
for train_df, val_df in splits_generator:
    target_train, target_val, cov_train, cov_val, scaler = datasets.prepare_darts_timeseries(
        train_df, val_df, target_col, feature_groups
    )
    
    # Train and evaluate model for this fold
    model.fit(target_train, past_covariates=cov_train)
    predictions = model.predict(n=len(target_val), past_covariates=cov_val)
```
