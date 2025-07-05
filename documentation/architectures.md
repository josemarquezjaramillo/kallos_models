# Module: `architectures.py`

## 1. Purpose

This module acts as a simple factory for creating `darts` forecasting model instances. It decouples the model instantiation logic from the `tuner` and `trainer` modules, making it easy to manage model-specific arguments and add new models in the future.

## 2. Core Components

-   **`create_model(model_name, input_chunk_length, output_chunk_length, hparams)`**: The single factory function in this module. It takes the model's name and its configuration parameters and returns an unfitted `darts` model object.

## 3. Detailed Explanation

The `create_model` function provides a single, consistent interface for creating different types of Darts models.

1.  **Argument Consolidation**: It accepts a `hparams` dictionary containing model-specific hyperparameters (like `hidden_dim`, `n_rnn_layers`, `dropout`). It combines these with common arguments required by all Darts models (`input_chunk_length`, `output_chunk_length`, `random_state`) into a single `model_kwargs` dictionary. This simplifies the calling signature from the `tuner` and `trainer`.

2.  **Model Selection**: It uses a simple `if/elif/else` block to check the `model_name` string.
    -   If `model_name` is 'gru', it instantiates `darts.models.GRU`.
    -   If `model_name` is 'lstm', it instantiates `darts.models.LSTM`.
    -   If `model_name` is 'transformer', it instantiates `darts.models.TransformerModel`. It also handles the mapping of common hyperparameter names (e.g., `hidden_dim`) to the specific argument names required by the `TransformerModel` (e.g., `d_model`).

3.  **Extensibility**: To add a new model (e.g., `NBEATSModel`), one would simply add another `elif` block to this factory function. The `tuner` and `trainer` code would not need to be changed, as long as the new model's hyperparameters are defined in the `tuner`'s search space.

4.  **Error Handling**: If an unsupported `model_name` is provided, it raises a `ValueError`, preventing silent failures.

## 4. Usage in Workflow

-   **In Tuning**: The `tuner.objective` function calls `create_model` inside its loop for every trial, passing the hyperparameters suggested by Optuna for that trial.
-   **In Training**: The `trainer.train_and_save_model` function calls `create_model` once, passing the set of optimal hyperparameters that were found during the tuning phase.
