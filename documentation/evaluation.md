# Module: `evaluation.py`

## 1. Purpose

This module provides the tools to perform a final, objective evaluation of a trained model on a completely unseen hold-out test set. It separates evaluation from training, which is a critical practice for assessing a model's true generalization performance.

The module's functions handle loading saved model artifacts, fetching test data, generating forecasts, calculating standard regression metrics, and plotting the results for visual inspection.

## 2. Core Components

-   **`generate_evaluation_report(...)`**: Calculates a dictionary of key performance indicators (KPIs), including Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE).
-   **`plot_forecast(...)`**: Uses `matplotlib` to generate and save a plot comparing the model's forecast against the actual values from the test set.
-   **`run_evaluation(...)`**: The main orchestrator function that ties everything together. It manages the entire evaluation workflow from loading artifacts to saving the final report and chart.

## 3. Detailed Explanation

The evaluation workflow is designed to simulate how the model would perform in a real-world scenario on new data.

1.  **Load Artifacts**: `run_evaluation` begins by loading the serialized Darts model (`.pt` file) and the pickled scikit-learn scaler (`.pkl` file) that were created by the `trainer` module.

2.  **Fetch Test Data**: It fetches data for the specified test period. Crucially, it also fetches data from before the test period starts, equal to the model's `input_chunk_length` (lookback window). This historical data is required by the model to generate the very first prediction in the test set.

3.  **Preprocess Data**: It applies the **already-fitted scaler** to the test set's features. It is vital not to re-fit the scaler on the test data, as this would cause data leakage and lead to an overly optimistic performance estimate.

4.  **Generate Forecast**: It calls the model's `.predict()` method to generate a forecast over the duration of the test period.

5.  **Evaluate and Report**:
    -   The generated forecast is compared against the true values from the test set.
    -   `generate_evaluation_report` is called to compute the performance metrics, which are then saved to a JSON file.
    -   `plot_forecast` is called to create a visual comparison chart, which is saved as a PNG image.

## 4. Usage in Workflow

The evaluation process is the final step after tuning and training. It is executed via the `main.py` CLI.

**Example Command:**

```bash
kallos-run evaluate \
    --model-name gru \
    --asset-id BTC \
    --model-path ./trained_models/gru_BTC.pt \
    --scaler-path ./trained_models/gru_BTC_scaler.pkl \
    --test-start-date 2023-01-01 \
    --test-end-date 2023-03-31 \
    --db-url "postgresql://user:pass@host/db" \
    --output-path ./evaluation_results
```

This command will test the specified model on Q1 2023 data and save `BTC_evaluation_metrics.json` and `BTC_evaluation_plot.png` into the `./evaluation_results` directory.
