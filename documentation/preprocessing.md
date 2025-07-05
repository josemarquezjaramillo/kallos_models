# Module: `preprocessing.py`

## 1. Purpose

This module provides functions to create a `scikit-learn` `ColumnTransformer`, which applies specific, research-backed normalization and transformation strategies to different groups of financial features.

## 2. Core Components

-   **`log_modified_z_score(series)`**: A helper function that applies a custom transformation (`sign(x) * log(1 + |x|)`) suitable for features with heavy-tailed distributions, like rate-of-change indicators.
-   **`create_feature_transformer(feature_groups)`**: The main factory function. It takes a dictionary that maps feature types (e.g., `volume_features`) to lists of column names and builds a `ColumnTransformer` pipeline.

## 3. Detailed Explanation

Financial features have very different statistical properties. Applying a single scaling method (like `StandardScaler`) to all features is suboptimal. This module addresses that by creating tailored pipelines for distinct feature categories:

-   **Volume Features**: These are often log-normally distributed. The pipeline first applies a `log1p` transformation to make the distribution more Gaussian, followed by a `StandardScaler`.
-   **Bounded Features**: Indicators like RSI or MFI are bounded (typically between 0 and 100). A `MinMaxScaler` is ideal for scaling these features to a consistent range (e.g., 0 to 1) without distorting their distribution.
-   **Unbounded Features**: Indicators like MACD can have significant outliers. A `RobustScaler` is used, which scales data based on quantile ranges (e.g., 5th and 95th percentiles), making it resilient to extreme values.
-   **Rate-of-Change (ROC) Features**: These features can have extreme positive and negative values (heavy tails). The custom `log_modified_z_score` is applied to tame these outliers, followed by a `StandardScaler`.

The `ColumnTransformer` ensures that each feature group receives its specific pipeline. The `remainder='passthrough'` argument guarantees that any columns not specified in the `feature_groups` dictionary are kept in the dataset without being transformed.

## 4. Usage in Workflow

The `create_feature_transformer` function is called by `datasets.prepare_darts_timeseries` (during tuning) and `trainer.train_and_save_model` (during final training).

-   **In Tuning**: A new, unfitted transformer is created for each walk-forward split. It is then fitted **only on that split's training data** to prevent data leakage from the validation set.
-   **In Training**: A single transformer is created and fitted on the **entire dataset** before the final model is trained. This fitted scaler is then saved to disk alongside the model so that the exact same transformation can be applied to new data during inference.
