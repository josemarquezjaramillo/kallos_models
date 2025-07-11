# Module: `preprocessing.py`

## 1. Purpose

This module provides functions to create a `scikit-learn` `ColumnTransformer`, which applies specific, research-backed normalization and transformation strategies to different groups of financial features.

## 2. Core Components

-   **`log_modified_z_score(series)`**: A helper function that applies a custom transformation (`sign(x) * log(1 + |x|)`) suitable for features with heavy-tailed distributions, like rate-of-change indicators.
-   **`ffill_and_handle_missing(X)`**: A utility function that performs forward-filling of missing values and handles any remaining NaNs, compatible with scikit-learn's `FunctionTransformer`.
-   **`create_feature_transformer(feature_groups)`**: The main factory function. It takes a dictionary that maps feature types (e.g., `volume_features`) to lists of column names and builds a `ColumnTransformer` pipeline.

## 3. Detailed Explanation

Financial features have very different statistical properties. Applying a single scaling method (like `StandardScaler`) to all features is suboptimal. This module addresses that by creating tailored pipelines for distinct feature categories:

### Handling Missing Values

The `ffill_and_handle_missing` function is a critical component of each pipeline, addressing the common issue of missing values in financial time series data:

1. It first applies forward-filling (`ffill`), which propagates the last valid observation forward until another valid observation is found
2. If any missing values remain after forward-filling (e.g., at the beginning of the time series), they are replaced with zeros
3. The function is designed to work with both NumPy arrays and pandas DataFrames, making it compatible with scikit-learn's `FunctionTransformer`

This approach preserves the temporal nature of the data by not using future information to impute past values, which is crucial for preventing look-ahead bias in financial modeling.

### Feature-Specific Pipelines

-   **Volume Features**: These are often log-normally distributed with high skewness. The pipeline:
    1. Forward-fills missing values
    2. Applies a `log1p` transformation to make the distribution more Gaussian
    3. Applies `StandardScaler` to normalize the log-transformed values

-   **Bounded Features**: Indicators like RSI or MFI are bounded (typically between 0 and 100). The pipeline:
    1. Forward-fills missing values
    2. Applies `MinMaxScaler` to scale these features to a consistent range (e.g., 0 to 1) without distorting their distribution

-   **Unbounded Features**: Indicators like MACD can have significant outliers. The pipeline:
    1. Forward-fills missing values
    2. Applies `RobustScaler` with quantile ranges (5th and 95th percentiles), making it resilient to extreme values

-   **Rate-of-Change (ROC) Features**: These features can have extreme positive and negative values (heavy tails). The pipeline:
    1. Forward-fills missing values
    2. Applies the custom `log_modified_z_score` transformation to tame outliers while preserving sign
    3. Applies `StandardScaler` to normalize the transformed values

-   **Log Return Features**: When present, these features often benefit from dimension reduction due to potential collinearity. The pipeline:
    1. Forward-fills missing values
    2. Applies `StandardScaler` to normalize the features
    3. Applies `PCA` to reduce dimensionality while preserving 95% of the variance

The `ColumnTransformer` ensures that each feature group receives its specific pipeline. The `remainder='passthrough'` argument guarantees that any columns not specified in the `feature_groups` dictionary are kept in the dataset without being transformed.

## 4. Usage in Workflow

The `create_feature_transformer` function is called by `datasets.prepare_darts_timeseries` (during tuning) and `trainer.train_and_save_model` (during final training).

-   **In Tuning**: A new, unfitted transformer is created for each walk-forward split. It is then fitted **only on that split's training data** to prevent data leakage from the validation set.
    ```python
    # Inside datasets.prepare_darts_timeseries
    scaler = preprocessing.create_feature_transformer(feature_groups)
    scaler.fit(covariates_train_df)
    covariates_train_norm = scaler.transform(covariates_train_df)
    covariates_val_norm = scaler.transform(covariates_val_df)
    ```

-   **In Training**: A single transformer is created and fitted on the **entire dataset** before the final model is trained. This fitted scaler is then saved to disk alongside the model so that the exact same transformation can be applied to new data during inference.
    ```python
    # Inside trainer.train_and_save_model
    scaler = preprocessing.create_feature_transformer(feature_groups)
    scaler.fit(features_df)  # Fit on all available data
    features_norm = scaler.transform(features_df)
    
    # Later, save the fitted scaler
    with open(scaler_filepath, 'wb') as f:
        pickle.dump(scaler, f)
    ```

## 5. Research Foundations

The transformation approaches in this module are based on financial econometrics research, which has shown that:

1. Volume data typically follows a log-normal distribution
2. Technical indicators with bounded ranges are best preserved with min-max scaling
3. Financial time series often contain outliers that can distort models, requiring robust scaling methods
4. Rate-of-change features often exhibit "fat tails" that benefit from a log transformation to reduce their impact

These domain-specific transformations significantly improve model performance compared to applying a one-size-fits-all approach to feature preprocessing.
