"""
Data Access and Preparation Module
=================================

This module handles data loading from databases and preparation for time-series modeling.
It provides functions for loading cryptocurrency data, generating robust walk-forward
validation splits, and preprocessing data into Darts TimeSeries objects.

The module implements:
1. Database querying functionality to load historical cryptocurrency data
2. Walk-forward cross-validation generation for time-series model evaluation
3. Feature transformation and normalization for Darts models
4. Conversion of pandas DataFrames to Darts TimeSeries objects

Example:
    from kallos_models import datasets
    
    # Load data from database
    db_config = {
        'postgres_user': 'user',
        'postgres_password': 'pass',
        'postgres_host': 'localhost',
        'postgres_port': 5432,
        'postgres_db': 'crypto_db'
    }
    
    # Load historical data for Bitcoin
    btc_data = datasets.load_features_from_db('BTC', '2023-01-01', db_config)
    
    # Generate walk-forward splits for cross-validation
    for train_df, val_df in datasets.get_walk_forward_splits(btc_data, 
                                                            train_years=2,
                                                            val_months=3,
                                                            step_months=3):
        print(f"Train: {train_df.index.min()} to {train_df.index.max()}")
        print(f"Val: {val_df.index.min()} to {val_df.index.max()}")
"""

import logging
from typing import Dict, Generator, List, Tuple
import pandas as pd
from dateutil.relativedelta import relativedelta
from darts import TimeSeries
from sklearn.compose import ColumnTransformer
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

from .preprocessing import create_feature_transformer

# Set up a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_features_from_db(asset_id: str, end_date: str, db_kwargs: dict, start_date='2018-06-01') -> pd.DataFrame:
    """
    Load cryptocurrency features from database for a specified asset and time period.
    
    This function connects to a SQL database and executes a query to retrieve all relevant
    technical indicators and price data for the specified cryptocurrency. It processes
    the returned data to ensure proper datetime indexing and frequency.
    
    Parameters:
        asset_id (str): The identifier for the cryptocurrency (e.g., 'BTC', 'ETH')
        end_date (str): The final date for the data in ISO format ('YYYY-MM-DD')
        db_kwargs (dict): Database connection parameters containing:
            - postgres_user (str): Database username
            - postgres_password (str): Database password
            - postgres_host (str): Database host address
            - postgres_port (int): Database port number
            - postgres_db (str): Database name
            - drivername (str, optional): SQLAlchemy driver name, defaults to 'postgresql+psycopg2'
        start_date (str, optional): The start date for the data in ISO format ('YYYY-MM-DD').
            Defaults to '2018-06-01'.
    
    Returns:
        pd.DataFrame: A DataFrame with datetime index and feature columns
    
    Raises:
        ValueError: If required database configuration keys are missing
        SQLAlchemyError: If a database connection error occurs
    
    Notes:
        - The returned DataFrame has a timezone-naive DatetimeIndex with daily frequency
        - Missing values in the time series are forward-filled
        - The query fetches a wide range of features including:
          * Price data (open, high, low, close)
          * Volume indicators
          * Technical indicators (RSI, MACD, Bollinger Bands, etc.)
          * Rate-of-change metrics
        
    Example:
        >>> db_config = {
        ...     'postgres_user': 'crypto_user',
        ...     'postgres_password': 'password123',
        ...     'postgres_host': 'localhost',
        ...     'postgres_port': 5432,
        ...     'postgres_db': 'crypto_database'
        ... }
        >>> btc_df = load_features_from_db('BTC', '2023-01-01', db_config)
        >>> print(f"Loaded {len(btc_df)} rows of data")
        >>> print(f"Date range: {btc_df.index.min()} to {btc_df.index.max()}")
    """
    logging.info(f"Loading features for asset '{asset_id}' up to {end_date}...")
    query = f"""
                SELECT
                    *
                FROM
                    (
                        SELECT
                            DM.TIMESTAMP AS timestamp,
                            -- Corrected PARTITION BY and added default value to LAG to prevent NULLs
                            LOG(DM.PRICE / LAG(DM.PRICE, 1, DM.PRICE) OVER (PARTITION BY DM.ID ORDER BY DM.TIMESTAMP ASC)) AS price,
                            LOG(DM.OPEN / LAG(DM.OPEN, 1, DM.OPEN) OVER (PARTITION BY DM.ID ORDER BY DM.TIMESTAMP ASC)) AS open,
                            LOG(DM.HIGH / LAG(DM.HIGH, 1, DM.HIGH) OVER (PARTITION BY DM.ID ORDER BY DM.TIMESTAMP ASC)) AS high,
                            LOG(DM.LOW / LAG(DM.LOW, 1, DM.LOW) OVER (PARTITION BY DM.ID ORDER BY DM.TIMESTAMP ASC)) AS low,
                            -- Corrected table alias from TI to DTI
                            LOG(DTI.EMA_10 / LAG(DTI.EMA_10, 1, DTI.EMA_10) OVER (PARTITION BY DM.ID ORDER BY DM.TIMESTAMP ASC)) AS ema_10,
                            LOG(DTI.EMA_50 / LAG(DTI.EMA_50, 1, DTI.EMA_50) OVER (PARTITION BY DM.ID ORDER BY DM.TIMESTAMP ASC)) AS ema_50,
                            LOG(DTI.BBM_20_2 / LAG(DTI.BBM_20_2, 1, DTI.BBM_20_2) OVER (PARTITION BY DM.ID ORDER BY DM.TIMESTAMP ASC)) AS bbm_20_2,
                            -- volume features (Corrected typo in MARKET_CAP)
                            DM.VOLUME as volume,
                            DM.MARKET_CAP as market_cap,
                            DTI.ATR_14 as atr_14,
                            -- bounded features
                            DTI.RSI_14 as rsi_14,
                            DTI.STOCHK_14_3_3 as stochk_14_3_3,
                            DTI.STOCHD_14_3_3 as stochd_14_3_3,
                            DTI.MFI_14 as mfi_14,
                            DTI.CHOP_14 as chop_14,
                            DTI.UI_14 as ui_14,
                            DTI.ADX_14 as adx_14,
                            DTI.DMP_14 as dmp_14,
                            DTI.DMN_14 as dmn_14,
                            DTI.CMF_20 as cmf_20,
                            -- unbounded features
                            DTI.EMA_DIFF as ema_diff,
                            DTI.MACD_DIFF as macd_diff,
                            DTI.STOCH_DIFF as stoch_diff,
                            DTI.DI_DIFF as di_diff,
                            DTI.price_bb_m_diff as price_bb_m_diff,
                            DTI.RSI_DEV as rsi_dev,
                            DTI.KSTS_9 as ksts_9,
                            DTI.MACD_12_26_9 as macd_12_26_9,
                            DTI.MACDS_12_26_9 as macds_12_26_9,
                            DTI.MACDH_12_26_9 as macdh_12_26_9,
                            -- rate of change features
                            DTI.PRICE_ROC_30 as price_roc_30,
                            DTI.VOLUME_ROC_7 as volume_roc_7,
                            DTI.VOLATILITY_RATIO as volatility_ratio,
                            DTI.TREND_QUALITY as trend_quality,
                            DTI.ROC_5 as roc_5,
                            -- Target variable calculation
                            (LEAD(DM.PRICE, 30) OVER (PARTITION BY DM.ID ORDER BY DM.TIMESTAMP ASC) / DM.PRICE - 1) AS pct_return_30d
                        FROM
                            DAILY_MARKET_DATA DM
                            LEFT JOIN DAILY_TECHNICAL_INDICATORS DTI USING (ID, TIMESTAMP)
                        WHERE
                            DM.ID = '{asset_id}'
                            AND DM.TIMESTAMP::DATE BETWEEN '{start_date}' AND ('{end_date}'::DATE + INTERVAL '30 days')
                        ORDER BY
                            DM.TIMESTAMP
                    ) AS subquery
                WHERE
                    TIMESTAMP <= '{end_date}'::DATE
    """
    required_keys = ['postgres_user', 'postgres_password', 'postgres_host', 'postgres_port', 'postgres_db']
    for key in required_keys:
        if key not in db_kwargs:
            raise ValueError(f"Missing required database configuration key: {key}")
        
    db_url = URL.create(
        drivername= db_kwargs.get('drivername', 'postgresql+psycopg2'),
        username=db_kwargs.get('postgres_user', 'your_username'),
        password=db_kwargs.get('postgres_password', 'your_password'),
        host=db_kwargs.get('postgres_host', 'localhost'),
        port=db_kwargs.get('postgres_port', 5432),
        database=db_kwargs.get('postgres_db', 'your_db_name')
    )
    engine = create_engine(db_url)
    with engine.connect() as connection:
        df = pd.read_sql(query, connection)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['timestamp'] = df['timestamp'].dt.normalize()

        # Convert to timezone-naive index AFTER normalization.
        # This makes the entire pipeline consistent with Darts' behavior.
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)

        df = df.set_index('timestamp', drop=True)
        df = df.asfreq('D', method='ffill')
    logging.info(f"Loaded {len(df)} records.")
    return df

def calculate_dynamic_wf_kwargs(full_df: pd.DataFrame, target_folds: int = 5, initial_train_ratio: float = 0.8) -> Dict[str, int]:
    """
    Calculate dynamic walk-forward parameters based on the dataset length.
    
    This function automatically determines appropriate time windows for walk-forward
    validation based on the size of the dataset, ensuring a reasonable number of
    validation folds while maintaining adequate training data.
    
    Parameters:
        full_df (pd.DataFrame): The entire historical DataFrame with a DatetimeIndex
        target_folds (int, optional): The desired number of walk-forward validation folds.
            Defaults to 5.
        initial_train_ratio (float, optional): The proportion of data to use for the
            initial training set. Defaults to 0.8 (80%).
    
    Returns:
        Dict[str, int]: A dictionary with calculated parameters:
            - 'train_years': Number of years for the initial training period
            - 'val_months': Number of months for each validation period
            - 'step_months': Number of months to advance between folds
    
    Notes:
        - Ensures at least 1 year of training data regardless of dataset size
        - Makes validation windows at least 3 months long for statistical significance
        - The step size equals the validation window size, creating non-overlapping validation sets
    
    Example:
        >>> import pandas as pd
        >>> dates = pd.date_range('2018-01-01', '2023-01-01', freq='D')
        >>> df = pd.DataFrame(index=dates)
        >>> params = calculate_dynamic_wf_kwargs(df, target_folds=4)
        >>> print(params)
        {'train_years': 3, 'val_months': 3, 'step_months': 3}
    """
    start_date = full_df.index.min()
    end_date = full_df.index.max()
    
    total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    
    # Calculate initial training period
    initial_train_months = int(total_months * initial_train_ratio)
    train_years = max(1, initial_train_months // 12) # Ensure at least 1 year of training
    
    # Calculate remaining months for validation and stepping
    remaining_months = total_months - initial_train_months
    
    # Distribute remaining months across the target number of folds
    # This will be the size of each validation/step window
    step_val_months = max(3, remaining_months // target_folds)
    
    wf_kwargs = {
        'train_years': train_years,
        'val_months': step_val_months,
        'step_months': step_val_months
    }
    logging.info(f"Calculated dynamic walk-forward params: {wf_kwargs} for a total of {total_months} months.")
    return wf_kwargs


def get_walk_forward_splits(
    full_df: pd.DataFrame,
    train_years: int,
    val_months: int,
    step_months: int
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """
    Generate walk-forward training and validation splits from a time series DataFrame.
    
    This function implements a "Sliding Window with Static Start" (expanding window) approach
    for time series cross-validation. It creates multiple train/validation splits where:
    1. The training window always starts at the beginning of the data
    2. The training window expands over time to include more recent data
    3. The validation window follows immediately after the training window
    
    Parameters:
        full_df (pd.DataFrame): The entire historical DataFrame with a DatetimeIndex
        train_years (int): Number of years for the initial training period
        val_months (int): Number of months for each validation period
        step_months (int): Number of months to advance the end of the training period in each step
    
    Yields:
        Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]: 
            A generator providing (train_df, val_df) tuples for each fold
    
    Notes:
        - Uses integer-based slicing for precise, non-overlapping splits
        - Assumes a timezone-naive DatetimeIndex on the input DataFrame
        - The training window grows with each fold, while the validation window slides forward
        - This prevents look-ahead bias by always validating on future data
    
    Example:
        >>> import pandas as pd
        >>> # Create sample DataFrame with 3 years of daily data
        >>> dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')
        >>> df = pd.DataFrame(index=dates, data={'value': range(len(dates))})
        >>> 
        >>> # Generate 3 folds with 1-year initial training, 3-month validation
        >>> splits = get_walk_forward_splits(df, train_years=1, val_months=3, step_months=3)
        >>> 
        >>> # Process each fold
        >>> for i, (train, val) in enumerate(splits):
        ...     print(f"Fold {i+1}:")
        ...     print(f"  Train: {train.index.min()} to {train.index.max()} ({len(train)} days)")
        ...     print(f"  Val: {val.index.min()} to {val.index.max()} ({len(val)} days)")
    """
    start_date = full_df.index.min()
    end_date = full_df.index.max()
    
    current_train_end = start_date + relativedelta(years=train_years)

    while True:
        current_val_end = current_train_end + relativedelta(months=val_months)
        
        # Ensure the validation period does not exceed the data
        if current_val_end > end_date:
            break

        # Find integer locations for slicing using timezone-naive dates
        train_end_loc = full_df.index.get_loc(current_train_end)
        val_end_loc = full_df.index.get_loc(current_val_end)

        # Use .iloc for precise, non-overlapping splits
        train_df = full_df.iloc[:train_end_loc + 1]
        val_df = full_df.iloc[train_end_loc + 1:val_end_loc + 1]

        # Ensure validation set is not empty
        if not val_df.empty:
            yield train_df, val_df

        # Move to the next split
        current_train_end += relativedelta(months=step_months)


def prepare_darts_timeseries(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str,
    feature_groups: Dict[str, List[str]]
) -> Tuple[TimeSeries, TimeSeries, TimeSeries, TimeSeries, ColumnTransformer]:
    """
    Prepare normalized Darts TimeSeries objects for model training and validation.
    
    This function processes raw DataFrames into the specialized TimeSeries format
    required by Darts models. It:
    1. Separates target and feature columns
    2. Creates and fits a feature transformer on training data only
    3. Transforms both training and validation features consistently
    4. Converts all processed data to Darts TimeSeries objects
    
    Parameters:
        train_df (pd.DataFrame): Training data DataFrame with DatetimeIndex
        val_df (pd.DataFrame): Validation data DataFrame with DatetimeIndex
        target_col (str): Name of the target column to predict
        feature_groups (Dict[str, List[str]]): Dictionary mapping feature group names
            to lists of column names
    
    Returns:
        Tuple[TimeSeries, TimeSeries, TimeSeries, TimeSeries, ColumnTransformer]: 
            A tuple containing:
            - target_train: TimeSeries object with training target values
            - target_val: TimeSeries object with validation target values
            - covariates_train: TimeSeries object with normalized training features
            - covariates_val: TimeSeries object with normalized validation features
            - scaler: The fitted ColumnTransformer used for feature normalization
    
    Notes:
        - The scaler is fitted ONLY on training data to prevent data leakage
        - The same fitted scaler is applied to both training and validation features
        - All TimeSeries objects are created with daily frequency
    
    Example:
        >>> train_data = pd.DataFrame(
        ...     index=pd.date_range('2021-01-01', '2021-12-31', freq='D'),
        ...     data={'close': range(365), 'volume': range(10000, 10365), 'rsi': [50]*365}
        ... )
        >>> val_data = pd.DataFrame(
        ...     index=pd.date_range('2022-01-01', '2022-01-31', freq='D'),
        ...     data={'close': range(31), 'volume': range(20000, 20031), 'rsi': [60]*31}
        ... )
        >>> feature_groups = {
        ...     'volume_features': ['volume'],
        ...     'bounded_features': ['rsi']
        ... }
        >>> 
        >>> target_train, target_val, cov_train, cov_val, scaler = prepare_darts_timeseries(
        ...     train_data, val_data, 'close', feature_groups
        ... )
        >>> 
        >>> print(f"Target train shape: {target_train.shape}")
        >>> print(f"Target val shape: {target_val.shape}")
        >>> print(f"Covariates train shape: {cov_train.shape}")
        >>> print(f"Covariates val shape: {cov_val.shape}")
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

    target_train = TimeSeries.from_dataframe(target_train_df, freq='D')
    target_val = TimeSeries.from_dataframe(target_val_df, freq='D')
    covariates_train = TimeSeries.from_dataframe(covariates_train_norm_df, freq='D')
    covariates_val = TimeSeries.from_dataframe(covariates_val_norm_df, freq='D')

    return target_train, target_val, covariates_train, covariates_val, scaler
