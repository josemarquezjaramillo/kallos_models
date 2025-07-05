import logging
from typing import Dict, Generator, List, Tuple

import pandas as pd
from darts import TimeSeries
from sklearn.compose import ColumnTransformer
from sqlalchemy import create_engine

from .preprocessing import create_feature_transformer

# Set up a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_features_from_db(asset_id: str, end_date: str, db_url: str) -> pd.DataFrame:
    """Loads all features for a given asset from the database up to a specific end date.

    Args:
        asset_id (str): The identifier for the cryptocurrency (e.g., 'BTC').
        end_date (str): The final date for the data ('YYYY-MM-DD').
        db_url (str): The SQLAlchemy database connection URL.

    Returns:
        pd.DataFrame: A DataFrame with a datetime index and feature columns.
    """
    logging.info(f"Loading features for asset '{asset_id}' up to {end_date}...")
    query = f"""
    SELECT * FROM features
    WHERE asset_id = '{asset_id}' AND date <= '{end_date}'
    ORDER BY date;
    """
    engine = create_engine(db_url)
    with engine.connect() as connection:
        df = pd.read_sql(query, connection, index_col='date', parse_dates=['date'])
    logging.info(f"Loaded {len(df)} records.")
    return df


def get_walk_forward_splits(
    full_df: pd.DataFrame,
    train_years: int,
    val_months: int,
    step_months: int
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """Generates walk-forward training and validation splits from a DataFrame.

    This implements a "Sliding Window with Static Start" approach.

    Args:
        full_df (pd.DataFrame): The entire historical DataFrame for an asset.
        train_years (int): The initial number of years for the first training set.
        val_months (int): The number of months for each validation set.
        step_months (int): The number of months to expand the training set by in each step.

    Yields:
        Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]: A generator that yields
            tuples of (train_dataframe, validation_dataframe).
    """
    start_date = full_df.index.min()
    end_of_data = full_df.index.max()
    
    current_train_end = start_date + pd.DateOffset(years=train_years)

    while True:
        val_end = current_train_end + pd.DateOffset(months=val_months)
        if val_end > end_of_data:
            break

        train_df = full_df.loc[start_date:current_train_end]
        val_df = full_df.loc[current_train_end:val_end]

        yield train_df, val_df

        current_train_end += pd.DateOffset(months=step_months)


def prepare_darts_timeseries(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str,
    feature_groups: Dict[str, List[str]]
) -> Tuple[TimeSeries, TimeSeries, TimeSeries, TimeSeries, ColumnTransformer]:
    """Prepares normalized Darts TimeSeries objects for a single walk-forward fold.

    Args:
        train_df (pd.DataFrame): The training data for the current fold.
        val_df (pd.DataFrame): The validation data for the current fold.
        target_col (str): The name of the target variable column.
        feature_groups (Dict[str, List[str]]): The feature group dictionary for the preprocessor.

    Returns:
        Tuple[TimeSeries, TimeSeries, TimeSeries, TimeSeries, ColumnTransformer]: A tuple containing
            (target_train, target_val, covariates_train, covariates_val, fitted_scaler).
    """
    all_feature_cols = [col for group in feature_groups.values() for col in group]

    target_train_df = train_df[[target_col]]
    covariates_train_df = train_df[all_feature_cols]
    target_val_df = val_df[[target_col]]
    covariates_val_df = val_df[all_feature_cols]

    scaler = create_feature_transformer(feature_groups)
    scaler.fit(covariates_train_df)

    covariates_train_norm = scaler.transform(covariates_train_df)
    covariates_val_norm = scaler.transform(covariates_val_df)

    covariates_train_norm_df = pd.DataFrame(covariates_train_norm, index=covariates_train_df.index, columns=all_feature_cols)
    covariates_val_norm_df = pd.DataFrame(covariates_val_norm, index=covariates_val_df.index, columns=all_feature_cols)

    target_train = TimeSeries.from_dataframe(target_train_df, freq=train_df.index.freq)
    target_val = TimeSeries.from_dataframe(target_val_df, freq=val_df.index.freq)
    covariates_train = TimeSeries.from_dataframe(covariates_train_norm_df, freq=train_df.index.freq)
    covariates_val = TimeSeries.from_dataframe(covariates_val_norm_df, freq=val_df.index.freq)

    return target_train, target_val, covariates_train, covariates_val, scaler
