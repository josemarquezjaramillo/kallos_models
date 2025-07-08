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
    """The objective function for Optuna to minimize.

    Args:
        trial (optuna.Trial): The current Optuna trial.
        asset_id (str): The asset identifier.
        end_date (str): The final date for the data.
        db_kwargs (dict): Keyword arguments for creating the database URL.
            Requires keys: 'postgres_user', 'postgres_password', 'postgres_host', 'postgres_port', 'postgres_db'.
        model_name (str): The name of the model to tune.
        target_col (str): The name of the target column.
        feature_groups (Dict[str, List[str]]): The feature group dictionary.

    Returns:
        float: The average validation RMSE across all walk-forward folds.
    """
    
    # Define the PyTorch Lightning Trainer arguments, including EarlyStopping
    early_stopper = EarlyStopping(
        monitor="val_loss",   # The metric to monitor
        patience=40,          # Number of epochs to wait for improvement
        min_delta=0.0005,     # Minimum change to qualify as an improvement
        mode='min'            # Stop when the monitored metric stops decreasing
    )
    
    pl_trainer_kwargs = {
        "max_epochs": 500,
        "callbacks": [early_stopper]
    }

    input_chunk_length = trial.suggest_categorical('input_chunk_length', [30, 40, 50, 60])
    
    # 1. Define hyperparameters for the model's structure and data loading
    hparams = {
        'hidden_dim': trial.suggest_int('hidden_dim', 32, 256, log=True),
        'n_rnn_layers': trial.suggest_int('n_rnn_layers', 1, 4),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'training_length': trial.suggest_int('training_length', input_chunk_length, input_chunk_length + 30)
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
            model_name, input_chunk_length, output_chunk_length, hparams, pl_trainer_kwargs
        )

        # Darts models need a validation set during fit() for early stopping to work
        model.fit(
            target_train,
            future_covariates=cov_train,
            val_series=target_val,
            val_future_covariates=cov_val,
            verbose=False
        )
        # Create a new covariate series for prediction that includes historical data
        # This ensures the model has the required lookback period (input_chunk_length)
        prediction_covariates = cov_train.append(cov_val)
        preds = model.predict(n=len(target_val), future_covariates=prediction_covariates)
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
    """Runs the Optuna hyperparameter tuning study.

    Args:
        asset_id (str): The asset identifier.
        end_date (str): The final date for the data.
        db_kwargs (dict): Keyword arguments for creating the database URL.
            Requires keys: 'postgres_user', 'postgres_password', 'postgres_host', 'postgres_port', 'postgres_db'.
        model_name (str): The name of the model to tune.
        target_col (str): The name of the target column.
        feature_groups (Dict[str, List[str]]): The feature group dictionary.
        n_trials (int): The number of optimization trials to run.

    Returns:
        Dict: A dictionary containing the best hyperparameters found.
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

    objective_fn = lambda trial: objective(
        trial, asset_id, end_date, db_kwargs, model_name, target_col, feature_groups
    )

    study.optimize(objective_fn, n_trials=n_trials)

    logging.info(f"Tuning finished. Best trial value: {study.best_value:.4f}")
    logging.info(f"Best params: {study.best_params}")

    return study
