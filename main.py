import argparse
import json
import logging
import os

from kallos_models import tuner, trainer, evaluation

# Set up a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Hardcoded Feature Groups and Target ---
# In a real application, this might come from a config file.
TARGET_COL = 'close'
FEATURE_GROUPS = {
    'volume_features': ['volume', 'taker_buy_base_asset_volume'],
    'bounded_features': ['rsi', 'mfi'],
    'unbounded_features': ['macd_diff', 'bollinger_hband_indicator', 'bollinger_lband_indicator'],
    'roc_features': ['roc_1', 'roc_3', 'roc_7']
}

def run_tune(args):
    """Runs the hyperparameter tuning process."""
    logging.info(f"Starting tuning for model '{args.model_name}' on asset '{args.asset_id}'...")
    best_params = tuner.run_tuning(
        asset_id=args.asset_id,
        end_date=args.end_date,
        db_url=args.db_url,
        model_name=args.model_name,
        target_col=TARGET_COL,
        feature_groups=FEATURE_GROUPS,
        n_trials=args.n_trials
    )

    # Save best params to a file
    params_filename = f"{args.model_name}_{args.asset_id}_best_params.json"
    with open(params_filename, 'w') as f:
        json.dump(best_params, f, indent=4)
    logging.info(f"Best parameters saved to {params_filename}")

def run_train(args):
    """Runs the final model training process."""
    logging.info(f"Starting training for model '{args.model_name}' on asset '{args.asset_id}'...")
    
    # Load optimal hyperparameters from file
    try:
        with open(args.params_file, 'r') as f:
            optimal_hparams = json.load(f)
    except FileNotFoundError:
        logging.error(f"Error: Parameter file not found at '{args.params_file}'")
        return

    trainer.train_and_save_model(
        asset_id=args.asset_id,
        end_date=args.end_date,
        db_url=args.db_url,
        model_name=args.model_name,
        target_col=TARGET_COL,
        feature_groups=FEATURE_GROUPS,
        optimal_hparams=optimal_hparams,
        output_path=args.output_path
    )
    logging.info("Training complete.")

def run_evaluate(args):
    """Runs the model evaluation process on a hold-out test set."""
    logging.info(f"Starting evaluation for asset '{args.asset_id}'...")
    evaluation.run_evaluation(
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        asset_id=args.asset_id,
        test_start_date=args.test_start_date,
        test_end_date=args.test_end_date,
        db_url=args.db_url,
        target_col=TARGET_COL,
        feature_groups=FEATURE_GROUPS,
        output_path=args.output_path
    )

def main():
    """Main function to parse arguments and run selected mode."""
    parser = argparse.ArgumentParser(description="Kallos Models CLI for tuning and training.")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Select mode: 'tune', 'train', or 'evaluate'")

    # --- Common arguments for both subparsers ---
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--model-name", type=str, required=True, help="Name of the model (e.g., 'gru', 'lstm').")
    parent_parser.add_argument("--asset-id", type=str, required=True, help="Asset identifier (e.g., 'BTC').")
    parent_parser.add_argument("--end-date", type=str, required=True, help="End date for data loading (YYYY-MM-DD).")
    parent_parser.add_argument("--db-url", type=str, required=True, help="SQLAlchemy database connection URL.")

    # --- Tune subparser ---
    parser_tune = subparsers.add_parser("tune", parents=[parent_parser], help="Run hyperparameter tuning.")
    parser_tune.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials to run.")
    parser_tune.set_defaults(func=run_tune)

    # --- Train subparser ---
    parser_train = subparsers.add_parser("train", parents=[parent_parser], help="Run final model training.")
    parser_train.add_argument("--params-file", type=str, required=True, help="Path to the JSON file with optimal hyperparameters.")
    parser_train.add_argument("--output-path", type=str, default="./trained_models", help="Directory to save trained model and scaler.")
    parser_train.set_defaults(func=run_train)

    # --- Evaluate subparser ---
    parser_eval = subparsers.add_parser("evaluate", parents=[parent_parser], help="Evaluate a trained model on a test set.")
    parser_eval.add_argument("--model-path", type=str, required=True, help="Path to the saved .pt model file.")
    parser_eval.add_argument("--scaler-path", type=str, required=True, help="Path to the saved .pkl scaler file.")
    parser_eval.add_argument("--test-start-date", type=str, required=True, help="Start date of the hold-out test set (YYYY-MM-DD).")
    parser_eval.add_argument("--test-end-date", type=str, required=True, help="End date of the hold-out test set (YYYY-MM-DD).")
    parser_eval.add_argument("--output-path", type=str, default="./evaluation_results", help="Directory to save evaluation charts and metrics.")
    parser_eval.set_defaults(func=run_evaluate)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
