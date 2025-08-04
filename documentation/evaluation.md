# Module: `evaluation.py`

## 1. Purpose

This module provides the tools to perform a final, objective evaluation of a trained model on a completely unseen hold-out test set. It separates evaluation from training, which is a critical practice for assessing a model's true generalization performance.

The module's functions handle loading saved model artifacts, fetching test data, generating forecasts, calculating standard regression metrics, and plotting the results for visual inspection.

## 2. Core Components

-   **`generate_evaluation_report(...)`**: Calculates a dictionary of key performance indicators including Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and specialized directional metrics.
-   **`plot_forecast(...)`**: Uses `matplotlib` to generate and save a plot comparing the model's forecast against the actual values from the test set.
-   **`run_evaluation(...)`**: The main orchestrator function that ties everything together, managing the entire evaluation workflow from loading artifacts to saving the final report and chart.
-   **`calculate_directional_weighted_mse(...)`**: Calculates a custom MSE metric that penalizes direction errors more heavily than magnitude errors.

## 3. Detailed Explanation

The evaluation workflow is designed to simulate how the model would perform in a real-world scenario on new data.

1.  **Load Model with Custom Loss**: `run_evaluation` begins by loading the serialized model using our custom loader function, which ensures that models are loaded with the appropriate DirectionSelectiveMSELoss function. This is especially important for BlockRNN models (GRU/LSTM).

2.  **Fetch Test Data**: It fetches data for the specified test period plus a lookback window equal to the model's `input_chunk_length`. This historical data is required by the model to generate predictions.

3.  **Preprocess Data**: It applies the **already-fitted scaler** to the test set's features, ensuring no data leakage occurs.

4.  **Generate Fixed-Window Forecast**: The model generates a standardized 90-day prediction window using only past covariates without requiring historical target values, matching the prediction approach used during training.

5.  **Evaluate and Report**:
    -   The generated forecast is compared against the true values from the test set.
    -   Performance metrics include standard regression metrics as well as specialized financial metrics for direction prediction accuracy.
    -   Both metrics and visualizations are saved with a consistent naming scheme based on the study name when available.

## 4. Usage in Workflow

The evaluation process is the final step after tuning and training. It is executed via the `kallos_model_evaluator.py` script, which loads evaluation tasks from a database view.

**Example Direct Usage:**

```python
from kallos_models.evaluation import run_evaluation

run_evaluation(
    model_path='trained_models/bitcoin/gru_bitcoin_2021_Q4_30D_customloss.pt',
    scaler_path='trained_models/bitcoin/gru_bitcoin_2021_Q4_30D_customloss_scaler.pkl',
    asset_id='bitcoin',
    test_start_date='2022-01-01',
    test_end_date='2022-03-31',
    db_kwargs={
        "postgres_user": "user", 
        "postgres_password": "pass",
        "postgres_host": "localhost", 
        "postgres_port": 5432,
        "postgres_db": "crypto_db"
    },
    target_col='pct_return_30d',
    feature_groups=feature_groups,
    output_path='evaluation_results/bitcoin',
    study_name='gru_bitcoin_2021_Q4_30D_customloss'  # Optional: Used for output file naming
)
```

This will evaluate the model on Q1 2022 data and save evaluation metrics and plots to the specified directory, with filenames based on the study_name if provided.
