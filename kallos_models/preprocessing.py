"""
Feature Preprocessing Module
===========================

This module provides specialized preprocessing transformations for different types of financial features.
It creates scikit-learn pipelines tailored for various categories of cryptocurrency indicators
and technical analysis features.

The module defines:
1. Transformation functions for specific financial data distributions
2. A factory function to create composite preprocessing pipelines for different feature groups

Example:
    from kallos_models.preprocessing import create_feature_transformer
    import pandas as pd
    
    # Define feature groups
    feature_groups = {
        'volume_features': ['volume', 'taker_volume'],
        'bounded_features': ['rsi_14', 'mfi_14'],
        'unbounded_features': ['macd_diff', 'ema_diff']
    }
    
    # Create transformer
    transformer = create_feature_transformer(feature_groups)
    
    # Use transformer in a preprocessing workflow
    features_df = pd.DataFrame(...)  # Your feature dataframe
    features_normalized = transformer.fit_transform(features_df)
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, RobustScaler, StandardScaler

# Set up a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_modified_z_score(series: pd.Series) -> pd.Series:
    """
    Apply a log-modified Z-score transformation for heavy-tailed distributions.
    
    This transformation calculates sign(x) * log(1 + |x|), which is effective for
    financial data with extreme values while preserving the sign of the original value.
    The transformation dampens the effect of outliers while maintaining directionality.
    
    Parameters:
        series (pd.Series): The input data series to transform
    
    Returns:
        pd.Series: The transformed data series
    
    Notes:
        - Useful for rate-of-change (ROC) indicators and other unbounded metrics
        - Preserves the sign of the original values
        - Compresses extreme values to reduce their impact
    
    Examples:
        >>> import pandas as pd
        >>> data = pd.Series([-100, -10, -1, 0, 1, 10, 100])
        >>> log_modified_z_score(data)
        0   -4.615121
        1   -2.397895
        2   -0.693147
        3    0.000000
        4    0.693147
        5    2.397895
        6    4.615121
        dtype: float64
    """
    return np.sign(series) * np.log1p(np.abs(series))

# Create a forward-fill function that can work with sklearn's FunctionTransformer
def ffill_and_handle_missing(X):
    """Forward fills missing values and handles any remaining NaNs"""
    if isinstance(X, np.ndarray):
        # Convert numpy array to pandas for ffill then back
        df = pd.DataFrame(X)
        df = df.ffill().fillna(0)  # Forward fill then fill remaining with zeros
        return df.values
    else:
        # Already pandas
        return X.ffill().fillna(0)


def create_feature_transformer(feature_groups: Dict[str, List[str]]) -> ColumnTransformer:
    """
    Create a scikit-learn ColumnTransformer with specialized pipelines for different financial feature types.
    
    This factory function builds a composite transformer that applies different preprocessing
    strategies to different types of financial features based on their statistical properties.
    
    Parameters:
        feature_groups (Dict[str, List[str]]): Dictionary mapping feature group names to lists of column names.
            Supported group names include:
            - 'volume_features': Features like trading volume that follow a log-normal distribution
            - 'bounded_features': Oscillator indicators with fixed bounds (e.g., RSI, MFI)
            - 'unbounded_features': Features without natural bounds that may have outliers
            - 'roc_features': Rate-of-change indicators that can have extreme values
            - 'log_return_features': Log return features that benefit from dimension reduction
    
    Returns:
        ColumnTransformer: A scikit-learn ColumnTransformer that can be used to preprocess features
    
    Notes:
        - Volume features: Log-transformed then standardized
        - Bounded features: Min-max scaled to [0, 1]
        - Unbounded features: Robust-scaled using 5th and 95th percentiles
        - ROC features: Log-modified Z-score transformed then standardized
        - Log return features: Standardized then reduced with PCA
        - All pipelines include handling for missing values via forward-fill
    
    Example:
        >>> feature_groups = {
        ...     'volume_features': ['volume', 'taker_buy_volume'],
        ...     'bounded_features': ['rsi_14', 'mfi_14'],
        ...     'roc_features': ['price_roc_5', 'volume_roc_7']
        ... }
        >>> transformer = create_feature_transformer(feature_groups)
        >>> normalized_features = transformer.fit_transform(feature_df)
    """
    
    transformer_pipelines = []

    # Modified pipeline for 'log_return_features' with PCA
    if 'log_return_features' in feature_groups:
        log_return_pipeline = Pipeline([
            ('handle_missing', FunctionTransformer(ffill_and_handle_missing)),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=2))  # Keep 2 components for dimension reduction
        ])
        transformer_pipelines.append(('log_return_features', log_return_pipeline, 
                                     feature_groups['log_return_features']))

    # Pipeline for 'volume_features'
    if 'volume_features' in feature_groups:
        volume_pipeline = Pipeline([
            ('handle_missing', FunctionTransformer(ffill_and_handle_missing)),
            ('log_transform', FunctionTransformer(np.log1p)),
            ('scaler', StandardScaler())
        ])
        transformer_pipelines.append(('volume_pipeline', volume_pipeline, feature_groups['volume_features']))

    # Pipeline for 'bounded_features' (e.g., RSI, MFI)
    if 'bounded_features' in feature_groups:
        bounded_pipeline = Pipeline([
            ('handle_missing', FunctionTransformer(ffill_and_handle_missing)),
            ('scaler', MinMaxScaler(feature_range=(0, 1)))
        ])
        transformer_pipelines.append(('bounded_pipeline', bounded_pipeline, feature_groups['bounded_features']))

    # Pipeline for 'unbounded_features' (e.g., MACD diff)
    if 'unbounded_features' in feature_groups:
        unbounded_pipeline = Pipeline([
            ('handle_missing', FunctionTransformer(ffill_and_handle_missing)),
            ('scaler', RobustScaler(quantile_range=(5.0, 95.0)))
        ])
        transformer_pipelines.append(('unbounded_pipeline', unbounded_pipeline, feature_groups['unbounded_features']))

    # Pipeline for 'roc_features' (Rate of Change)
    if 'roc_features' in feature_groups:
        roc_pipeline = Pipeline([
            ('handle_missing', FunctionTransformer(ffill_and_handle_missing)),
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

def transform_features_to_dataframe(
    transformer: ColumnTransformer,
    features_df: pd.DataFrame,
    feature_groups: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Transform features using the fitted transformer and convert the result to a DataFrame
    with appropriate column names, preserving original names where possible.
    
    Assumes that PCA is only applied to 'log_return_features' and produces exactly 2 components.
    
    Parameters:
        transformer (ColumnTransformer): A fitted scikit-learn ColumnTransformer
        features_df (pd.DataFrame): DataFrame of features to transform
        feature_groups (Dict[str, List[str]]): Dictionary mapping feature group names to lists of column names
    
    Returns:
        pd.DataFrame: DataFrame with transformed features and appropriate column names
    """
    # Apply transformation
    transformed_features = transformer.transform(features_df)
    
    # First, try to get feature names from transformer directly
    try:
        output_feature_names = transformer.get_feature_names_out()
        return pd.DataFrame(
            transformed_features,
            index=features_df.index,
            columns=output_feature_names
        )
    except (AttributeError, NotImplementedError) as e:
        logging.debug(f"Could not get feature names from transformer: {e}")
        
        # Get original feature names
        all_feature_cols = [col for group in feature_groups.values() for col in group]
        
        # If dimensions match, use original column names
        if transformed_features.shape[1] == len(all_feature_cols):
            return pd.DataFrame(
                transformed_features,
                index=features_df.index,
                columns=all_feature_cols
            )
        else:
            # Assume PCA is only applied to log_return_features and creates exactly 2 components
            if 'log_return_features' in feature_groups:
                # Count features before log_return_features
                features_before_log_return = 0
                for group_name, cols in feature_groups.items():
                    if group_name == 'log_return_features':
                        break
                    features_before_log_return += len(cols)
                
                # Count features after log_return_features
                log_return_original_count = len(feature_groups.get('log_return_features', []))
                features_after_log_return = len(all_feature_cols) - features_before_log_return - log_return_original_count
                
                # Create column names with the original columns preserved and two PCA components for log_return_features
                new_cols = []
                
                # Add columns before log_return_features
                for group_name, cols in feature_groups.items():
                    if group_name == 'log_return_features':
                        break
                    new_cols.extend(cols)
                
                # Add the two PCA components
                new_cols.extend(['log_return_pc1', 'log_return_pc2'])
                
                # Add columns after log_return_features
                for group_name, cols in list(feature_groups.items())[list(feature_groups.keys()).index('log_return_features') + 1:]:
                    new_cols.extend(cols)
                
                logging.info(f"PCA reduced log_return_features from {log_return_original_count} to 2 components. "
                            f"Total features: {len(new_cols)}")
                
                return pd.DataFrame(
                    transformed_features,
                    index=features_df.index,
                    columns=new_cols
                )
            else:
                # Default case - use generic names
                new_cols = [f"feature_{i}" for i in range(transformed_features.shape[1])]
                logging.info(f"Feature dimensions changed from {len(all_feature_cols)} to "
                            f"{transformed_features.shape[1]}. Using generic column names.")
                
                return pd.DataFrame(
                    transformed_features,
                    index=features_df.index,
                    columns=new_cols
                )
