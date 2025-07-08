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
    """Loads all features for a given asset from the database up to a specific end date.

    Args:
        asset_id (str): The identifier for the cryptocurrency (e.g., 'BTC').
        end_date (str): The final date for the data ('YYYY-MM-DD').
        db_kwargs (dict): Keyword arguments for creating the database URL.
            Requires keys: 'postgres_user', 'postgres_password', 'postgres_host', 'postgres_port', 'postgres_db'.

    Returns:
        pd.DataFrame: A DataFrame with a datetime index and feature columns.
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
    Calculates dynamic walk-forward parameters based on the dataset's length.

    Args:
        full_df (pd.DataFrame): The entire historical DataFrame.
        target_folds (int): The desired number of walk-forward folds.
        initial_train_ratio (float): The proportion of the data to use for the initial training set.

    Returns:
        Dict[str, int]: A dictionary with calculated 'train_years', 'val_months', and 'step_months'.
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
    """Generates walk-forward training and validation splits from a DataFrame.

    This implements a "Sliding Window with Static Start" approach using robust
    integer-based slicing to prevent gaps or overlaps. Assumes a timezone-naive index.

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

    target_train = TimeSeries.from_dataframe(target_train_df, freq='D')
    target_val = TimeSeries.from_dataframe(target_val_df, freq='D')
    covariates_train = TimeSeries.from_dataframe(covariates_train_norm_df, freq='D')
    covariates_val = TimeSeries.from_dataframe(covariates_val_norm_df, freq='D')

    return target_train, target_val, covariates_train, covariates_val, scaler
