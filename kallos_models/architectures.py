import logging
from typing import Dict, Union

from darts.models import GRU, LSTM, TransformerModel
from darts.models.forecasting.forecasting_model import ForecastingModel

# Set up a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_model(
    model_name: str,
    input_chunk_length: int,
    output_chunk_length: int,
    hparams: Dict
) -> ForecastingModel:
    """Factory function to create a Darts forecasting model.

    Args:
        model_name (str): The name of the model to create ('lstm', 'gru', 'transformer').
        input_chunk_length (int): The length of the lookback window.
        output_chunk_length (int): The length of the forecast horizon.
        hparams (Dict): A dictionary of hyperparameters for the model constructor.

    Returns:
        ForecastingModel: An unfitted Darts model instance.
    """
    model_kwargs = {
        'input_chunk_length': input_chunk_length,
        'output_chunk_length': output_chunk_length,
        'random_state': 42,
        **hparams
    }

    model_name_lower = model_name.lower()
    logging.info(f"Creating model: {model_name_lower} with params: {model_kwargs}")

    if model_name_lower == 'gru':
        model = GRU(**model_kwargs)
    elif model_name_lower == 'lstm':
        model = LSTM(**model_kwargs)
    elif model_name_lower == 'transformer':
        # TransformerModel has different arg names, so we adjust
        transformer_kwargs = model_kwargs.copy()
        transformer_kwargs['d_model'] = transformer_kwargs.pop('hidden_dim')
        transformer_kwargs['nhead'] = transformer_kwargs.pop('nhead')
        transformer_kwargs['num_encoder_layers'] = transformer_kwargs.pop('num_encoder_layers')
        transformer_kwargs['num_decoder_layers'] = transformer_kwargs.pop('num_decoder_layers')
        transformer_kwargs['dim_feedforward'] = transformer_kwargs.pop('dim_feedforward')
        model = TransformerModel(**transformer_kwargs)
    else:
        raise ValueError(f"Unsupported model name: '{model_name}'. Supported: 'lstm', 'gru', 'transformer'.")

    return model
