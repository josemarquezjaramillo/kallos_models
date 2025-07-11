"""
Kallos Models Package
====================

A comprehensive package for training, tuning, and evaluating deep learning time-series 
models for cryptocurrency price prediction.

This package provides an end-to-end workflow for developing forecasting models,
from robust hyperparameter optimization to final model evaluation on hold-out data.

Modules
-------
preprocessing : Feature-specific normalization and transformation pipelines
datasets : Data loading, walk-forward validation splits, and TimeSeries preparation
architectures : Model factory for instantiating different forecasting models
tuner : Hyperparameter optimization using Optuna and walk-forward validation
trainer : Final model training and saving
evaluation : Model performance assessment on hold-out test data

Workflow
--------
The typical workflow consists of three steps:

1. Tune hyperparameters using walk-forward validation:
   ```
   kallos-run tune --model-name gru --asset-id BTC --end-date 2023-01-01 --db-url "..." --n-trials 100
   ```

2. Train a final model using the optimal hyperparameters:
   ```
   kallos-run train --model-name gru --asset-id BTC --end-date 2023-01-01 --db-url "..." 
                   --params-file gru_BTC_best_params.json --output-path ./models
   ```

3. Evaluate the trained model on a hold-out test set:
   ```
   kallos-run evaluate --model-name gru --asset-id BTC --model-path ./models/gru_BTC.pt 
                      --scaler-path ./models/gru_BTC_scaler.pkl --test-start-date 2023-02-01 
                      --test-end-date 2023-04-30 --db-url "..." --output-path ./results
   ```

Notes
-----
- All models use the Darts library (https://unit8co.github.io/darts/) for time-series forecasting
- Feature preprocessing is tailored for different types of financial indicators
- Walk-forward validation ensures robust evaluation of time-series models
"""

# This file makes the kallos_models directory a Python package.
