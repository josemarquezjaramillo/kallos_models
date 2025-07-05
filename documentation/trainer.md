# Module: `trainer.py`

## 1. Purpose

This module is responsible for the final step of the modeling pipeline: training a model on the entire available dataset using a given set of optimal hyperparameters and saving the resulting model and data scaler to disk for future use in inference.

## 2. Core Components

-   **`train_and_save_model(...)`**: The sole function in this module. It orchestrates the loading of data, fitting of the final scaler, training of the final model, and serialization of the artifacts.

## 3. Detailed Explanation

The `train_and_save_model` function executes a straightforward, linear process to produce the final, production-ready model.

1.  **Load Full Dataset**: It begins by calling `datasets.load_features_from_db` to load all historical data up to the specified `end_date`. Unlike the tuning phase, no splitting occurs.
2.  **Create and Fit Scaler**: It calls `datasets.create_feature_transformer` to get a `ColumnTransformer`. This scaler is then fitted on the **entire** feature DataFrame. This is a key difference from the tuning phase and is critical for ensuring that the model is trained on a consistently scaled version of all available information.
3.  **Transform Data**: The fitted scaler is used to transform the feature DataFrame.
4.  **Prepare TimeSeries**: The target and normalized feature DataFrames are converted into `darts.TimeSeries` objects.
5.  **Instantiate Model**: It calls `architectures.create_model`, passing it the `optimal_hparams` dictionary that was found during the tuning phase.
6.  **Train Final Model**: It calls the `model.fit()` method, passing the full target and covariate `TimeSeries`. This trains the model on all available data, which generally leads to the most robust and accurate final model.
7.  **Save Artifacts**:
    -   It creates the specified `output_path` directory if it doesn't exist.
    -   It saves the trained `darts` model using the model's built-in `.save()` method. This correctly serializes the PyTorch model architecture and weights.
    -   It saves the fitted `scikit-learn` scaler object using `pickle`. This is **extremely important**, as the exact same scaler (with the same internal statistics) must be used to transform new, unseen data before feeding it to the model for inference.

## 4. Usage in Workflow

This module is used in the final phase of the workflow. The `main.py` script calls `train_and_save_model` when the user executes the `kallos-run train` command, providing the path to the JSON file containing the optimal hyperparameters.
