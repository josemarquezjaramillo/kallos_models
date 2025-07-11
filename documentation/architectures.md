# Module: `architectures.py`

## 1. Purpose

This module acts as a simple factory for creating `darts` forecasting model instances. It decouples the model instantiation logic from the `tuner` and `trainer` modules, making it easy to manage model-specific arguments and add new models in the future.

## 2. Core Components

-   **`create_model(model_name, hparams, pl_trainer_kwargs)`**: The single factory function in this module. It takes the model's name, hyperparameters, and PyTorch Lightning trainer configuration parameters and returns an unfitted `darts` model object.

## 3. Detailed Explanation

The `create_model` function provides a single, consistent interface for creating different types of Darts models.

1.  **Argument Consolidation**: It accepts a `hparams` dictionary containing model-specific hyperparameters (like `hidden_dim`, `n_rnn_layers`, `dropout`) and a `pl_trainer_kwargs` dictionary for training configuration (like `max_epochs`, `callbacks` for early stopping). It combines these with common arguments required by all Darts models (like `output_chunk_length` and `random_state`) into a single `model_kwargs` dictionary. This simplifies the calling signature from the `tuner` and `trainer`.

2.  **Model Selection**: It uses a simple `if/elif/else` block to check the `model_name` string.
    -   If `model_name` is 'gru' or 'lstm', it instantiates the generic `darts.models.BlockRNNModel` and passes the specific type (e.g., 'GRU') to the `model` parameter. This is the correct approach for modern versions of the Darts library.
    -   If `model_name` is 'transformer', it instantiates `darts.models.TransformerModel`. It also handles the mapping of common hyperparameter names (e.g., `hidden_dim`) to the specific argument names required by the `TransformerModel` (e.g., `d_model`).

3.  **PyTorch Lightning Integration**: The function accepts `pl_trainer_kwargs` which allows passing configuration for the PyTorch Lightning trainer that Darts uses internally. This is crucial for setting up early stopping, learning rate schedulers, and other training behaviors without modifying the model creation code.

4.  **Extensibility**: To add a new RNN-based model supported by `BlockRNNModel` (e.g., a vanilla 'RNN'), one would simply add its name to the list in the `if` condition. To add a completely different model (e.g., `NBEATSModel`), one would add another `elif` block.

5.  **Error Handling**: If an unsupported `model_name` is provided, it raises a `ValueError`, preventing silent failures.

## 4. Usage in Workflow

-   **In Tuning**: The `tuner.objective` function calls `create_model` inside its loop for every trial, passing:
    - The model name (e.g., 'gru')
    - The hyperparameters suggested by Optuna for that trial
    - The PyTorch Lightning trainer configuration with early stopping

    ```python
    # Example from tuner.py
    model = architectures.create_model(
        model_name, 
        hparams,
        pl_trainer_kwargs
    )
    ```

-   **In Training**: The `trainer.train_and_save_model` function calls `create_model` once, passing:
    - The model name
    - The set of optimal hyperparameters that were found during the tuning phase
    - Any training configuration parameters

    ```python
    # Example from trainer.py
    model = architectures.create_model(
        model_name, 
        **optimal_hparams  # Unpacked to provide all parameters
    )
    ```
