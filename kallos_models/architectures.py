"""
Model Architecture Factory Module
================================

This module provides a factory function to create different types of Darts forecasting models
with consistent interfaces. It abstracts away model-specific implementation details and
parameter naming differences.

Example:
    from kallos_models.architectures import create_model
    
    # Create a GRU model with specific hyperparameters
    gru_model = create_model(
        model_name="gru",
        hparams={
            "hidden_dim": 128,
            "n_rnn_layers": 2,
            "dropout": 0.2,
            "batch_size": 64,
            "input_chunk_length": 60,
            "optimizer_kwargs": {"lr": 0.001}
        },
        pl_trainer_kwargs={
            "max_epochs": 100,
            "callbacks": [early_stopper]
        }
    )
    
    # Train the model
    gru_model.fit(target_series, past_covariates=covariate_series)
"""

import logging
from typing import Dict, Union

from darts.models import BlockRNNModel, TransformerModel
from darts.models.forecasting.forecasting_model import ForecastingModel

# Set up a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_model(
    model_name: str,
    hparams: Dict,
    pl_trainer_kwargs: Dict
) -> ForecastingModel:
    """
    Create a Darts forecasting model based on the specified type and hyperparameters.
    
    This factory function instantiates different types of time series forecasting models
    from the Darts library with a unified interface, handling model-specific parameter
    mappings internally.
    
    Parameters:
        model_name (str): The type of model to create. Supported values:
            - 'gru': Gated Recurrent Unit network
            - 'lstm': Long Short-Term Memory network
            - 'transformer': Transformer network with attention mechanism
        hparams (Dict): Hyperparameters for the model. Common parameters include:
            - hidden_dim (int): Size of hidden layers
            - n_rnn_layers (int): Number of recurrent layers (for RNN-based models)
            - dropout (float): Dropout rate for regularization
            - batch_size (int): Mini-batch size for training
            - input_chunk_length (int): Size of the lookback window
            - optimizer_kwargs (Dict): Parameters for the optimizer (e.g., {'lr': 0.001})
            
            For Transformer models, additional required parameters:
            - nhead (int): Number of attention heads
            - num_encoder_layers (int): Number of encoder layers
            - num_decoder_layers (int): Number of decoder layers
            - dim_feedforward (int): Dimension of feedforward network
        
        pl_trainer_kwargs (Dict): Keyword arguments for the PyTorch Lightning Trainer.
            Common parameters include:
            - max_epochs (int): Maximum number of training epochs
            - callbacks (List): List of PyTorch Lightning callbacks
            - gpus (int): Number of GPUs to use
    
    Returns:
        ForecastingModel: An unfitted Darts model instance
    
    Raises:
        ValueError: If an unsupported model_name is provided
        KeyError: If required hyperparameters for a specific model are missing
    
    Notes:
        - All models are initialized with output_chunk_length=1 (single-step forecasting)
        - A fixed random seed (42) is used for reproducibility
        - The hyperparameter dictionary structure varies by model type:
            * RNN models (GRU, LSTM): Use the same parameter names directly
            * Transformer models: Require different parameter names that are mapped internally
    
    Examples:
        >>> # Create a GRU model
        >>> from pytorch_lightning.callbacks import EarlyStopping
        >>> early_stopper = EarlyStopping(monitor="val_loss", patience=5)
        >>> 
        >>> gru_params = {
        ...     "hidden_dim": 64,
        ...     "n_rnn_layers": 2,
        ...     "dropout": 0.3,
        ...     "batch_size": 32,
        ...     "input_chunk_length": 60,
        ...     "optimizer_kwargs": {"lr": 0.001}
        ... }
        >>> 
        >>> gru_model = create_model(
        ...     model_name="gru", 
        ...     hparams=gru_params,
        ...     pl_trainer_kwargs={"max_epochs": 100, "callbacks": [early_stopper]}
        ... )
        
        >>> # Create a Transformer model
        >>> transformer_params = {
        ...     "hidden_dim": 64,  # Will be mapped to d_model
        ...     "nhead": 4,
        ...     "num_encoder_layers": 3,
        ...     "num_decoder_layers": 3,
        ...     "dim_feedforward": 256,
        ...     "dropout": 0.1,
        ...     "batch_size": 32,
        ...     "input_chunk_length": 60,
        ...     "optimizer_kwargs": {"lr": 0.0001}
        ... }
        >>> 
        >>> transformer_model = create_model(
        ...     model_name="transformer", 
        ...     hparams=transformer_params,
        ...     pl_trainer_kwargs={"max_epochs": 100, "callbacks": [early_stopper]}
        ... )
    """
    model_kwargs = {
        'output_chunk_length': 1,  # Default output chunk length
        'random_state': 42,
        'pl_trainer_kwargs': pl_trainer_kwargs,
        **hparams
    }

    model_name_lower = model_name.lower()
    logging.info(f"Creating model: {model_name_lower} with params: {model_kwargs}")

    if model_name_lower in ['gru', 'lstm']:
        # For RNN models, we use the generic RNNModel class and specify the model type.
        model = BlockRNNModel(model=model_name.upper(), **model_kwargs)
    elif model_name_lower == 'transformer':
        # TransformerModel has different arg names, so we adjust
        transformer_kwargs = model_kwargs.copy()
        # The following will raise KeyError if the keys are not in hparams.
        # This assumes the hyperparameter dictionary is correctly formatted for Transformer.
        transformer_kwargs['d_model'] = transformer_kwargs.pop('hidden_dim')
        transformer_kwargs['nhead'] = transformer_kwargs.pop('nhead')
        transformer_kwargs['num_encoder_layers'] = transformer_kwargs.pop('num_encoder_layers')
        transformer_kwargs['num_decoder_layers'] = transformer_kwargs.pop('num_decoder_layers')
        transformer_kwargs['dim_feedforward'] = transformer_kwargs.pop('dim_feedforward')
        model = TransformerModel(**transformer_kwargs)
    else:
        raise ValueError(f"Unsupported model name: '{model_name}'. Supported: 'lstm', 'gru', 'transformer'.")

    return model
