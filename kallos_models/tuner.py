import logging
from typing import Dict, List

import numpy as np
import optuna
from darts.metrics import rmse

from . import architectures, datasets

# Set up a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(
    trial: optuna.Trial,
    asset_id: str,
    end_date: str,
    db_url: str,
    model_name: str,
    target_col: str,
    feature_groups: Dict[str, List[str]]
) -> float:
    """The objective function for Optuna to minimize.

    Args:
        trial (optuna.Trial): The current Optuna trial.
        asset_id (str): The asset identifier.
        end_date (str): The final date for the data.
        db_url (str): The database connection URL.
        model_name (str): The name of the model to tune.
        target_col (str): The name of the target column.
        feature_groups (Dict[str, List[str]]): The feature group dictionary.

    Returns:
        float: The average validation RMSE across all walk-forward folds.
    """
    hparams = {
        'hidden_dim': trial.suggest_int('hidden_dim', 32, 256, log=True),
        'n_rnn_layers': trial.suggest_int('n_rnn_layers', 1, 4),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
    }
    input_chunk_length = trial.suggest_categorical('input_chunk_length', [30, 60, 90])
    output_chunk_length = 1 # Assuming we predict one step ahead

    full_df = datasets.load_features_from_db(asset_id, end_date, db_url)
    splits_generator = datasets.get_walk_forward_splits(full_df, train_years=2, val_months=3, step_months=3)
    
    validation_scores = []
    for i, (train_df, val_df) in enumerate(splits_generator):
        logging.info(f"Trial {trial.number}, Fold {i+1}: Training on {len(train_df)} samples, validating on {len(val_df)}.")
        target_train, target_val, cov_train, cov_val, _ = datasets.prepare_darts_timeseries(
            train_df, val_df, target_col, feature_groups
        )
        
        model = architectures.create_model(model_name, input_chunk_length, output_chunk_length, hparams)
        
        model.fit(target_train, past_covariates=cov_train, verbose=False)
        preds = model.predict(n=len(target_val), past_covariates=cov_train)
        score = rmse(target_val, preds)
        validation_scores.append(score)

    mean_score = float(np.mean(validation_scores))
    logging.info(f"Trial {trial.number} finished. Avg RMSE: {mean_score:.4f}, Params: {trial.params}")
    return mean_score


def run_tuning(
    asset_id: str,
    end_date: str,
    db_url: str,
    model_name: str,
    target_col: str,
    feature_groups: Dict[str, List[str]],
    n_trials: int
) -> Dict:
    """Runs the Optuna hyperparameter tuning study.

    Args:
        asset_id (str): The asset identifier.
        end_date (str): The final date for the data.
        db_url (str): The database connection URL.
        model_name (str): The name of the model to tune.
        target_col (str): The name of the target column.
        feature_groups (Dict[str, List[str]]): The feature group dictionary.
        n_trials (int): The number of optimization trials to run.

    Returns:
        Dict: A dictionary containing the best hyperparameters found.
    """
    study = optuna.create_study(direction='minimize')
    
    objective_fn = lambda trial: objective(
        trial, asset_id, end_date, db_url, model_name, target_col, feature_groups
    )
    
    study.optimize(objective_fn, n_trials=n_trials)
    
    logging.info(f"Tuning finished. Best trial value: {study.best_value:.4f}")
    logging.info(f"Best params: {study.best_params}")
    
    return study.best_params
