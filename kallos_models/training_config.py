"""
Training Configuration Module
============================

This module centralizes training configurations for time series models, providing
consistent parameters for model training across different modules.

It defines standard configurations for callbacks, training parameters, and
evaluation settings to ensure consistency throughout the model lifecycle.

Example:
    from kallos_models import training_config
    
    # Get standard training configuration
    pl_trainer_kwargs = training_config.get_default_trainer_config()
    
    # Create a model with standard training settings
    model = architectures.create_model(
        model_name="gru",
        hparams=optimal_hparams,
        pl_trainer_kwargs=pl_trainer_kwargs
    )
"""

MAX_EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 50
EARLY_STOPPING_DELTA = 0.0005


from typing import Dict, List, Optional, Union

from pytorch_lightning.callbacks import Callback, EarlyStopping


def create_early_stopping_callback(
    patience: int = EARLY_STOPPING_PATIENCE,
    min_delta: float = EARLY_STOPPING_DELTA,
    monitor: str = "val_loss",
    mode: str = "min"
) -> EarlyStopping:
    """
    Create a PyTorch Lightning EarlyStopping callback with standard parameters.
    """
    return EarlyStopping(
        monitor=monitor,
        patience=patience,
        min_delta=min_delta,
        mode=mode
    )


def get_default_trainer_config(
    max_epochs: int = MAX_EPOCHS,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    early_stopping_delta: float = EARLY_STOPPING_DELTA,
    monitor: str = "val_loss",  # Add this parameter
    additional_callbacks: Optional[List[Callback]] = None,
    **kwargs
) -> Dict[str, Union[int, List[Callback]]]:
    """
    Get the default PyTorch Lightning trainer configuration with standard callbacks.
    """
    early_stopper = create_early_stopping_callback(
        patience=early_stopping_patience,
        min_delta=early_stopping_delta,
        monitor=monitor  # Pass the monitor parameter here
    )
    
    callbacks = [early_stopper]
    if additional_callbacks:
        callbacks.extend(additional_callbacks)
    
    config = {
        "max_epochs": max_epochs,
        "callbacks": callbacks,
    }
    
    # Add any additional keyword arguments
    config.update(kwargs)
    
    return config


def get_tuning_trainer_config() -> Dict[str, Union[int, List[Callback]]]:
    """
    Get trainer configuration specifically optimized for hyperparameter tuning.
    """
    return get_default_trainer_config(
        max_epochs=MAX_EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_delta=EARLY_STOPPING_DELTA,
        monitor="val_loss"  # During tuning, we have validation data
    )


def get_production_trainer_config() -> Dict[str, Union[int, List[Callback]]]:
    """
    Get trainer configuration optimized for final model training.
    """
    return get_default_trainer_config(
        max_epochs=MAX_EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_delta=EARLY_STOPPING_DELTA,
        monitor="train_loss"  # For production training, use train_loss instead of val_loss
    )
