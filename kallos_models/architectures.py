import logging
from typing import Dict, Union

from darts.models import RNNModel, TransformerModel
from darts.models.forecasting.forecasting_model import ForecastingModel

# Set up a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_model(
    model_name: str,
    input_chunk_length: int,
    output_chunk_length: int,
    hparams: Dict,
    pl_trainer_kwargs: Dict
) -> ForecastingModel:
    """Factory function to create a Darts forecasting model.

    Args:
        model_name (str): The name of the model to create ('lstm', 'gru', 'transformer').
        input_chunk_length (int): The length of the lookback window.
        output_chunk_length (int): The length of the forecast horizon.
        hparams (Dict): A dictionary of hyperparameters for the model constructor.
        pl_trainer_kwargs (Dict): A dictionary of keyword arguments for the PyTorch Lightning Trainer.

    Returns:
        ForecastingModel: An unfitted Darts model instance.
    """
    model_kwargs = {
        'input_chunk_length': input_chunk_length,
        'output_chunk_length': output_chunk_length,
        'random_state': 42,
        'pl_trainer_kwargs': pl_trainer_kwargs,
        **hparams
    }

    model_name_lower = model_name.lower()
    logging.info(f"Creating model: {model_name_lower} with params: {model_kwargs}")

    if model_name_lower in ['gru', 'lstm']:
        # For RNN models, we use the generic RNNModel class and specify the model type.
        model = RNNModel(model=model_name.upper(), **model_kwargs)
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
