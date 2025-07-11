"""
Hyperparameter Tuning Module
============================

This module provides functionality for optimizing the hyperparameters of time-series forecasting models
using the Optuna framework with a walk-forward cross-validation approach suitable for financial time-series data.

The module implements:
1. An objective function for Optuna that evaluates model performance across multiple walk-forward validation folds
2. A main tuning function that creates and runs an Optuna study to find optimal hyperparameters

Example:
    from kallos_models import tuner

    # Run hyperparameter optimization for a GRU model on Bitcoin data
    best_params = tuner.run_tuning(
        asset_id="BTC",
        end_date="2023-01-01",
        db_kwargs={
            "postgres_user": "user",
            "postgres_password": "pass",
            "postgres_host": "localhost",
            "postgres_port": 5432,
            "postgres_db": "crypto_db"
        },
        model_name="gru",
        study_name="gru_btc_study",
        target_col="close",
        feature_groups={
            "volume_features": ["volume", "taker_buy_base_asset_volume"],
            "bounded_features": ["rsi", "mfi"]
        },
        n_trials=100
    )
"""

import logging
from datetime import datetime
from typing import Dict, List, Union

import numpy as np
import optuna
from darts.metrics import rmse
from sqlalchemy.engine import URL
from pytorch_lightning.callbacks import EarlyStopping

from . import architectures, datasets


# Set up a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
optuna.logging.set_verbosity(optuna.logging.DEBUG)


def objective(
    trial: optuna.Trial,
    asset_id: str,
    end_date: str,
    db_kwargs: Dict[str, Union[int, str]],
    model_name: str,
    target_col: str,
    feature_groups: Dict[str, List[str]]
) -> float:
    """
    Objective function for Optuna to minimize during hyperparameter optimization.
    
    This function evaluates a set of hyperparameters by:
    1. Creating a model with the trial's suggested hyperparameters
    2. Training and evaluating the model across multiple walk-forward validation folds
    3. Returning the average validation RMSE across all folds
    
    Parameters:
        trial (optuna.Trial): The current Optuna trial object that suggests hyperparameter values
        asset_id (str): The asset identifier (e.g., "BTC", "ETH")
        end_date (str): The end date for the data in ISO format (YYYY-MM-DD)
        db_kwargs (Dict[str, Union[int, str]]): Database connection parameters containing:
            - postgres_user (str): Database username
            - postgres_password (str): Database password
            - postgres_host (str): Database host address
            - postgres_port (int): Database port number
            - postgres_db (str): Database name
        model_name (str): The type of model to create (e.g., "gru", "lstm", "transformer")
        target_col (str): The name of the target column in the dataset
        feature_groups (Dict[str, List[str]]): Dictionary mapping feature group names to lists of column names
    
    Returns:
        float: The average validation RMSE across all walk-forward folds (lower is better)
    
    Notes:
        - Uses early stopping during training to prevent overfitting
        - Logs the hyperparameters and performance of each trial
        - The objective function is designed to be passed to optuna.study.optimize()
    
    Example:
        # This function is not typically called directly but through run_tuning
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(
            trial, 
            asset_id="BTC", 
            end_date="2023-01-01",
            db_kwargs=db_config,
            model_name="gru", 
            target_col="close", 
            feature_groups=features
        ), n_trials=100)
    """
    
    # Define the PyTorch Lightning Trainer arguments, including EarlyStopping
    early_stopper = EarlyStopping(
        monitor="val_loss",   # The metric to monitor
        patience=50,          # Number of epochs to wait for improvement
        min_delta=0.0005,     # Minimum change to qualify as an improvement
        mode='min'            # Stop when the monitored metric stops decreasing
    )
    
    pl_trainer_kwargs = {
        "max_epochs": 500,
        "callbacks": [early_stopper]
    }
   
    # 1. Define hyperparameters for the model's structure and data loading
    hparams = {
        'hidden_dim': trial.suggest_int('hidden_dim', 32, 256, log=True),
        'n_rnn_layers': trial.suggest_int('n_rnn_layers', 1, 4),
        'dropout': trial.suggest_float('dropout', 0.0, 0.7),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'input_chunk_length': trial.suggest_int('input_chunk_length', 30, 70),
    }
    
    # 2. Define the learning rate for the optimizer separately
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    
    # 3. Add the learning rate to the optimizer_kwargs dictionary within hparams
    # This is the correct way to pass it to the Darts model.
    hparams['optimizer_kwargs'] = {'lr': learning_rate}
    
    output_chunk_length = 1 # Assuming we predict one step ahead

    full_df = datasets.load_features_from_db(asset_id, end_date, db_kwargs)
    wf_kwargs = datasets.calculate_dynamic_wf_kwargs(full_df, target_folds=5)

    splits_generator = datasets.get_walk_forward_splits(full_df,
                                                        train_years=wf_kwargs["train_years"],
                                                        val_months=wf_kwargs["val_months"],
                                                        step_months=wf_kwargs["step_months"])
    validation_scores = []
    for i, (train_df, val_df) in enumerate(splits_generator):
        logging.info(f"Trial {trial.number}, Fold {i+1}: Training on {len(train_df)} samples, validating on {len(val_df)}.")
        target_train, target_val, cov_train, cov_val, _ = datasets.prepare_darts_timeseries(
            train_df, val_df, target_col, feature_groups
        )
        
        model = architectures.create_model(
            model_name, hparams, pl_trainer_kwargs
        )

        # Darts models need a validation set during fit() for early stopping to work
        model.fit(
            target_train,
            past_covariates=cov_train,
            val_series=target_val,
            val_past_covariates=cov_val,
            verbose=False
        )
        # Create a new covariate series for prediction that includes historical data
        # This ensures the model has the required lookback period (input_chunk_length)
        prediction_covariates = cov_train.append(cov_val)
        preds = model.predict(n=len(target_val), past_covariates=prediction_covariates)

        # Calculate RMSE for the current fold       
        score = rmse(target_val, preds)
        validation_scores.append(score)

    mean_score = float(np.mean(validation_scores))
    logging.info(f"Trial {trial.number} finished. Avg RMSE: {mean_score:.4f}, Params: {trial.params}")
    return mean_score


def run_tuning(
    asset_id: str,
    end_date: str,
    db_kwargs: Dict[str, Union[int, str]],
    optuna_schema: Union[str, None],
    model_name: str,
    study_name: str,
    target_col: str,
    feature_groups: Dict[str, List[str]],
    n_trials: int
) -> Dict:
    """
    Run an Optuna hyperparameter optimization study for a specified model and dataset.
    
    This function creates an Optuna study (or loads an existing one) and runs the optimization
    process to find the best hyperparameters for a time-series forecasting model.
    
    Parameters:
        asset_id (str): The asset identifier (e.g., "BTC", "ETH")
        end_date (str): The end date for the data in ISO format (YYYY-MM-DD)
        db_kwargs (Dict[str, Union[int, str]]): Database connection parameters containing:
            - postgres_user (str): Database username
            - postgres_password (str): Database password
            - postgres_host (str): Database host address
            - postgres_port (int): Database port number
            - postgres_db (str): Database name
        optuna_schema (Union[str, None]): Optional schema name for Optuna's storage in the database.
            If None, the default schema is used.
        model_name (str): The type of model to create (e.g., "gru", "lstm", "transformer")
        study_name (str): The name to give the Optuna study (used when storing results in the database)
        target_col (str): The name of the target column in the dataset
        feature_groups (Dict[str, List[str]]): Dictionary mapping feature group names to lists of column names
        n_trials (int): The maximum number of optimization trials to run
    
    Returns:
        optuna.Study: The completed Optuna study object containing optimization results
    
    Notes:
        - Uses a persistent database storage for the Optuna study
        - If a study with the same name exists, it will be loaded and resumed
        - The function will skip running more trials if the study already has n_trials or more trials
    
    Example:
        best_study = run_tuning(
            asset_id="BTC",
            end_date="2023-01-01",
            db_kwargs={
                "postgres_user": "user", 
                "postgres_password": "pass",
                "postgres_host": "localhost", 
                "postgres_port": 5432,
                "postgres_db": "crypto_db"
            },
            optuna_schema="optuna_studies",
            model_name="gru",
            study_name="gru_btc_study",
            target_col="close",
            feature_groups={
                "volume_features": ["volume"],
                "bounded_features": ["rsi", "mfi"]
            },
            n_trials=100
        )
        
        print(f"Best parameters: {best_study.best_params}")
        print(f"Best value: {best_study.best_value}")
    """
    storage_url = (
        f"postgresql://{db_kwargs['postgres_user']}:{db_kwargs['postgres_password']}@"
        f"{db_kwargs['postgres_host']}:{db_kwargs['postgres_port']}/{db_kwargs['postgres_db']}"
    )

    if optuna_schema:
        storage_url += f"?options=-csearch_path%3D{optuna_schema}"
    
    storage = optuna.storages.RDBStorage(
        url=storage_url
    )

    study = optuna.create_study(direction='minimize',
                                storage=storage,
                                study_name=study_name,
                                load_if_exists=True)
    
    current_study_trials = len(study.trials)

    if current_study_trials >= n_trials:
        logging.info(f"Study '{study_name}' already has {current_study_trials} trials, which is >= requested {n_trials}.")
        return study

    objective_fn = lambda trial: objective(
        trial, asset_id, end_date, db_kwargs, model_name, target_col, feature_groups
    )

    study.optimize(objective_fn, n_trials=n_trials-current_study_trials)

    logging.info(f"Tuning finished. Best trial value: {study.best_value:.4f}")
    logging.info(f"Best params: {study.best_params}")

    return study
