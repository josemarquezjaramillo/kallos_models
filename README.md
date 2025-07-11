# Kallos Models

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Darts](https://img.shields.io/badge/darts-0.21.0+-orange.svg)
![Optuna](https://img.shields.io/badge/optuna-3.0.0+-purple.svg)

A structured Python package for training, tuning, and evaluating deep learning time-series models for cryptocurrency price prediction. It provides a command-line interface to manage an end-to-end MLOps workflow, from robust hyperparameter optimization to final model evaluation on hold-out data.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Workflow and Usage](#workflow-and-usage)
  - [Step 1: Tune Hyperparameters](#step-1-tune-hyperparameters)
  - [Step 2: Train Final Model](#step-2-train-final-model)
  - [Step 3: Evaluate on Hold-Out Data](#step-3-evaluate-on-hold-out-data)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Development Guidelines](#development-guidelines)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **End-to-End Workflow**: A CLI for a complete workflow: tuning, training, and evaluation
- **Robust Validation**: Employs walk-forward cross-validation for reliable hyperparameter tuning and a hold-out set for final evaluation
- **Hyperparameter Optimization**: Integrated with `Optuna` to systematically find the best model parameters
- **Modular Architecture**: Clean separation of data loading, preprocessing, model architecture, and evaluation logic
- **Custom Preprocessing**: Applies tailored transformation pipelines to different groups of financial features
- **Supported Models**: Easily extensible factory for `darts` models like GRU, LSTM, and Transformer
- **Database Integration**: Loads financial data directly from SQL databases
- **Persistent Studies**: Stores optimization results in a database for resumable tuning sessions

## Project Structure

```
kallos_models/
├── README.md                 # Project documentation
├── setup.py                  # Package setup and dependencies
├── main.py                   # CLI entry point
├── .env                      # Database configuration (not versioned)
├── config.json               # Project configuration
├── kallos_models/            # Core package modules
│   ├── __init__.py           # Package initialization
│   ├── architectures.py      # Model factory implementations
│   ├── datasets.py           # Data loading and preparation
│   ├── evaluation.py         # Model evaluation logic
│   ├── preprocessing.py      # Feature transformation pipelines
│   ├── trainer.py            # Final model training logic
│   └── tuner.py              # Hyperparameter optimization
└── documentation/            # Detailed module documentation
    ├── architectures.md      # Model factory documentation
    ├── datasets.md           # Data handling documentation
    ├── evaluation.md         # Evaluation process documentation
    ├── kallos_models.md      # Project overview
    ├── preprocessing.md      # Feature preprocessing documentation
    ├── trainer.md            # Training workflow documentation
    └── tuner.md              # Tuning process documentation
```

## Installation

Clone the repository and use `pip` to install the package in editable mode. This will also install all required dependencies listed in `setup.py`.

```bash
# Clone the repository
git clone https://github.com/your_username/kallos_models.git
cd kallos_models

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # For Linux/Mac
# or
venv\Scripts\activate     # For Windows

# Install in editable mode with dependencies
pip install -e .
```

### Database Configuration

The project requires a PostgreSQL database with cryptocurrency data. Create a `.env` file in the project root with your database credentials:

```properties
{
    "postgres_user": "your_username",
    "postgres_password": "your_password",
    "postgres_host": "your_host",
    "postgres_port": 5432,
    "postgres_db": "your_database"
}
```

## Workflow and Usage

The recommended workflow ensures that the model is trained and evaluated robustly, without data leakage. It consists of three distinct steps performed via the CLI.

### Step 1: Tune Hyperparameters

First, run the tuner to find the optimal hyperparameters for a given model and asset. This is done on a dataset that **excludes the final hold-out test set**.

```bash
kallos-run tune \
    --model-name gru \
    --asset-id BTC \
    --end-date 2023-09-30 \
    --db-url "postgresql://user:pass@host/db" \
    --n-trials 100
```

This command runs 100 optimization trials on data up to September 30, 2023. It saves the best parameters to a JSON file (e.g., `gru_BTC_best_params.json`).

### Step 2: Train Final Model

Next, use the best parameters to train a final model. This should be done on the **exact same data period** as the tuning step.

```bash
kallos-run train \
    --model-name gru \
    --asset-id BTC \
    --end-date 2023-09-30 \
    --db-url "postgresql://user:pass@host/db" \
    --params-file gru_BTC_best_params.json \
    --output-path ./trained_models
```

This saves the trained Darts model (`gru_BTC.pt`) and the fitted scikit-learn scaler (`gru_BTC_scaler.pkl`) to the `./trained_models` directory.

### Step 3: Evaluate on Hold-Out Data

Finally, evaluate the trained model's performance on a completely unseen hold-out test set (e.g., the period after the training `end-date`).

```bash
kallos-run evaluate \
    --model-name gru \
    --asset-id BTC \
    --model-path ./trained_models/gru_BTC.pt \
    --scaler-path ./trained_models/gru_BTC_scaler.pkl \
    --test-start-date 2023-10-01 \
    --test-end-date 2023-12-31 \
    --db-url "postgresql://user:pass@host/db" \
    --output-path ./evaluation_results
```

This command loads the saved model and scaler, tests performance on Q4 2023 data, and saves two artifacts to `./evaluation_results`:
- `BTC_evaluation_plot.png`: A chart comparing the model's forecast to the actual values
- `BTC_evaluation_metrics.json`: A report with RMSE, MAE, and MAPE scores

## Configuration

### Command-line Arguments

The `kallos-run` command supports three subcommands: `tune`, `train`, and `evaluate`. Each has specific arguments:

**Common arguments for all commands:**
- `--model-name`: Type of model to use (e.g., 'gru', 'lstm', 'transformer')
- `--asset-id`: Cryptocurrency identifier (e.g., 'BTC')
- `--db-url`: Database connection URL

**Tune-specific arguments:**
- `--end-date`: End date for training data
- `--n-trials`: Number of Optuna trials to run

**Train-specific arguments:**
- `--end-date`: End date for training data
- `--params-file`: Path to JSON file with optimal parameters
- `--output-path`: Directory to save the trained model

**Evaluate-specific arguments:**
- `--model-path`: Path to the saved model file
- `--scaler-path`: Path to the saved scaler file
- `--test-start-date`: Start date for the test period
- `--test-end-date`: End date for the test period
- `--output-path`: Directory to save evaluation results

### Project Configuration File

You can also use a `config.json` file for default settings:

```json
{
    "asset_id": "bitcoin",
    "model_name": "gru",
    "tune_end_date": "2023-01-01",
    "train_end_date": "2023-01-01",
    "test_start_date": "2023-01-01",
    "test_end_date": "2023-07-01",
    "n_trials": 10,
    "output_path": "./kallos_run_output"
}
```

## Architecture

The package is composed of several key modules that work in concert:

- `preprocessing.py`: Creates `scikit-learn` pipelines for feature-specific normalization
- `datasets.py`: Handles data loading from a database and generates walk-forward validation splits
- `architectures.py`: A factory for instantiating `darts` forecasting models
- `tuner.py`: Implements the hyperparameter optimization logic using `Optuna`
- `trainer.py`: Trains a final model on all available data using optimal hyperparameters
- `evaluation.py`: Evaluates a trained model on a hold-out test set, generating metrics and plots
- `main.py`: Provides the command-line interface (CLI) to orchestrate the entire workflow

For more detailed information on each module, see the [Documentation](#documentation) section.

## Development Guidelines

### Code Style

The project follows [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines. We recommend using tools like `flake8` or `black` to ensure consistent code formatting.

```bash
# Install development tools
pip install flake8 black

# Check code style
flake8 kallos_models/

# Format code
black kallos_models/
```

### Branching Model

We follow the [GitFlow](https://nvie.com/posts/a-successful-git-branching-model/) branching model:

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: New feature development
- `bugfix/*`: Bug fixes
- `release/*`: Release preparation

### Testing

Add tests to the `tests/` directory and run them with `pytest`:

```bash
# Install pytest
pip install pytest

# Run tests
pytest tests/
```

## Documentation

Detailed documentation for each module can be found in the `documentation/` directory:

- [Project Overview](documentation/kallos_models.md)
- [Data Loading and Preparation](documentation/datasets.md)
- [Feature Preprocessing](documentation/preprocessing.md)
- [Model Architectures](documentation/architectures.md)
- [Hyperparameter Tuning](documentation/tuner.md)
- [Model Training](documentation/trainer.md)
- [Model Evaluation](documentation/evaluation.md)

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows our style guidelines and includes appropriate tests and documentation.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgements

- [Darts](https://unit8co.github.io/darts/) - The time series forecasting library
- [Optuna](https://optuna.org/) - The hyperparameter optimization framework
- [PyTorch](https://pytorch.org/) - The deep learning framework
- [scikit-learn](https://scikit-learn.org/) - For preprocessing pipelines
- [pandas](https://pandas.pydata.org/) - For data manipulation
- All contributors who have helped improve this project
