import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, RobustScaler, StandardScaler

# Set up a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def log_modified_z_score(series: pd.Series) -> pd.Series:
    """Applies a log-modified Z-score transformation suitable for heavy-tailed distributions.

    This function calculates sign(x) * log(1 + |x|).

    Args:
        series (pd.Series): The input data series.

    Returns:
        pd.Series: The transformed data series.
    """
    return np.sign(series) * np.log1p(np.abs(series))


def create_feature_transformer(feature_groups: Dict[str, List[str]]) -> ColumnTransformer:
    """Creates a scikit-learn ColumnTransformer to apply group-specific normalizations.

    Args:
        feature_groups (Dict[str, List[str]]): A dictionary where keys are group names
            (e.g., 'volume_features', 'bounded_features') and values are lists of
            column names belonging to that group.

    Returns:
        ColumnTransformer: An unfitted scikit-learn ColumnTransformer object.
    """
    transformer_pipelines = []

    # Pipeline for 'volume_features'
    if 'volume_features' in feature_groups:
        volume_pipeline = Pipeline([
            ('log_transform', FunctionTransformer(np.log1p)),
            ('scaler', StandardScaler())
        ])
        transformer_pipelines.append(('volume_pipeline', volume_pipeline, feature_groups['volume_features']))

    # Pipeline for 'bounded_features' (e.g., RSI, MFI)
    if 'bounded_features' in feature_groups:
        bounded_pipeline = Pipeline([
            ('scaler', MinMaxScaler(feature_range=(0, 1)))
        ])
        transformer_pipelines.append(('bounded_pipeline', bounded_pipeline, feature_groups['bounded_features']))

    # Pipeline for 'unbounded_features' (e.g., MACD diff)
    if 'unbounded_features' in feature_groups:
        unbounded_pipeline = Pipeline([
            ('scaler', RobustScaler(quantile_range=(5.0, 95.0)))
        ])
        transformer_pipelines.append(('unbounded_pipeline', unbounded_pipeline, feature_groups['unbounded_features']))

    # Pipeline for 'roc_features' (Rate of Change)
    if 'roc_features' in feature_groups:
        roc_pipeline = Pipeline([
            ('log_z_score', FunctionTransformer(log_modified_z_score)),
            ('scaler', StandardScaler())
        ])
        transformer_pipelines.append(('roc_pipeline', roc_pipeline, feature_groups['roc_features']))

    # Instantiate ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=transformer_pipelines,
        remainder='passthrough'
    )

    return preprocessor
