# %%
import os
import sys
import fnmatch
import argparse
from glob import glob
import pandas as pd

import torch
import ultralytics
from ultralytics import YOLO
from loguru import logger


# Import the custom YOLO class which is pre-configured with our custom trainer.
from src.yolo.custom_ultralytics import CustomYOLO


torch.cuda.empty_cache()

# Configure Loguru for colored output
logger.remove()
logger.add(sys.stderr, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")


def train_yolo(model_path, dataset_yaml_path, epochs, img_size, batch_size, project_name, experiment_name, patience, device, workers):
    """
    Trains a YOLO model with the given parameters.
    """
    # Check Ultralytics version
    ultralytics.checks()

    logger.info(f"Loading model from {model_path}")
    model = CustomYOLO(model_path) # Load the model from the specified path

    # The 'data' argument should be the path to your 'dataset.yaml'
    logger.info(f"Starting training for {experiment_name}...")
    logger.info(f"  - Dataset: {dataset_yaml_path}")
    logger.info(f"  - Epochs: {epochs}, Image Size: {img_size}, Batch Size: {batch_size}, Patience: {patience}")
    
    results = model.train(
        data=dataset_yaml_path,  # Use the absolute path
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=project_name,
        name=experiment_name,
        patience=patience, # Epochs to wait for no observable improvement before stopping training early
        device=device, # Specify GPU device
        workers=workers # Number of worker threads for data loading (adjust based on your system)
    )

    logger.success("Training completed.")
    logger.info(f"Results saved to: {model.trainer.save_dir}")
# %%
def total_results(project_name, best_metric):
    """
    Collects all training results, selects the best epoch for each run based on a metric,
    cleans the data, and saves it to an Excel file with separate sheets for each train_type.

    Args:
        project_name (str): The path to the project directory containing the results.
        best_metric (str): The column name of the metric to use for selecting the best model.

    Returns:
        pd.DataFrame: A DataFrame containing the combined and cleaned results.
    """
    all_best_models = []
    columns_to_drop = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss', 'lr/pg0', 'lr/pg1', 'lr/pg2']

    # 1. Collect results from all CSV files in the project directory
    results_csv_paths = glob(os.path.join(project_name, '**', 'results.csv'), recursive=True)
    logger.info(f"Found {len(results_csv_paths)} results.csv files in {project_name}")
    
    for path in results_csv_paths:
        try:
            results_df = pd.read_csv(path)
            results_df.columns = results_df.columns.str.strip()

            if best_metric not in results_df.columns:
                logger.warning(f"Warning: {best_metric} missing from metric column. Using 'mAP50-95' as default metric.")
                best_metric='metrics/mAP50-95(B)'
                if results_df.empty or best_metric not in results_df.columns:
                    logger.warning(f"Warning: Skipping '{path}' due to missing data or metric column.")
                    continue

            # Find the best performing epoch and create a copy
            best_model_series = results_df.loc[results_df[best_metric].idxmax()].copy()
            
            # Extract train_type from path if it contains 'augmented' or 'cleaned', otherwise use 'unknown'
            path_parts = path.split(os.sep)
            train_type = 'unknown'
            for part in path_parts:
                if 'augmented' in part.lower():
                    train_type = 'augmented'
                    break
                elif 'cleaned' in part.lower():
                    train_type = 'cleaned'
                    break
            
            best_model_series['train_type'] = train_type
            best_model_series['model'] = os.path.basename(os.path.dirname(path))
            all_best_models.append(best_model_series)
            logger.info(f"Added results for {best_model_series['model']} (train_type: {train_type})")
        except Exception as e:
            logger.warning(f"Error processing file {path}: {e}")

    if not all_best_models:
        logger.warning("No valid results found to process.")
        return pd.DataFrame()

    # 2. Create a single DataFrame and process it
    results_df = pd.DataFrame(all_best_models)

    # Clean column names
    # Only drop columns that actually exist
    existing_columns_to_drop = [col for col in columns_to_drop if col in results_df.columns]
    if existing_columns_to_drop:
        results_df.drop(columns=existing_columns_to_drop, inplace=True)
    
    results_df.columns = results_df.columns.str.replace(r'^metrics/', '', regex=True).str.replace(r'\(B\)$', '', regex=True)

    # Reorder columns
    first_cols = ['train_type', 'model']
    other_cols = [col for col in results_df.columns if col not in first_cols]
    results_df = results_df[first_cols + other_cols]

    results_df.reset_index(drop=True, inplace=True)

    # 3. Save the processed data to an Excel file with separate sheets
    output_excel_path = os.path.join(project_name, 'training_results.xlsx')
    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        for train_type, group_df in results_df.groupby('train_type'):
            logger.info(f"Saving sheet: {train_type}")
            group_df.to_excel(writer, sheet_name=train_type, index=False)
            logger.info(f"\n {group_df}")

    logger.success(f"Successfully created Excel file at: {output_excel_path}")
    return results_df
# %%
def main(models_dir, model_selection, project_name, dataset_yaml_path, epochs, img_size, batch_size, patience, device, workers, best_metric):
    # Check Ultralytics version
    ultralytics.checks()

    logger.info(f"Starting training process for models: {model_selection}")

    for model in model_selection:
        torch.cuda.empty_cache()
        logger.info(f"--- Processing model: {model} ---")
        model_path = os.path.join(models_dir, model + '.pt')
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found at {model_path}, attempting to load directly.")
            model_path = model + '.pt'


        # Define training parameters
        experiment_name = model  # Name of the experiment

        # Train the model
        model_batch_size = batch_size
        # if fnmatch.fnmatch(model, 'yolo*x'):
        #     logger.info(f"'{model_path}' is an 'x' model. Halving batch size from {batch_size} to {batch_size // 2}.")
        #     model_batch_size = batch_size // 2

        train_yolo(
            model_path=model_path,
            dataset_yaml_path=dataset_yaml_path,
            epochs=epochs,
            img_size=img_size,
            batch_size=batch_size,
            project_name=project_name,
            experiment_name=experiment_name,
            patience=patience,
            device=device,
            workers=workers
        )
    
    logger.success("All training sessions finished.")

    total_results(project_name, best_metric=best_metric)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLO models.')
    
    parser.add_argument('--models_dir', type=str, default='/mnt/d/Rowan/AeroDefence/models', help='Directory where base models are stored.')
    parser.add_argument('--model_selection', nargs='+', default=['yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x', 'yolo12n', 'yolo12s', 'yolo12m', 'yolo12l', 'yolo12x'], help='List of YOLO models to train.')
    parser.add_argument('--project_name', type=str, default='/mnt/d/Rowan/AeroDefence/exp/yolo/detect/Roboflow-2025-10-27_augmented', help='Project directory to save training experiments.')
    parser.add_argument('--dataset_yaml_path', type=str, default='/mnt/d/Rowan/AeroDefence/dataset/Roboflow-2025-10-27/augmented/data.yaml', help='Path to the dataset YAML file.')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--img_size', type=int, default=640, help='Input image size.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--patience', type=int, default=10, help='Epochs to wait for no observable improvement before early stopping.')
    parser.add_argument('--device', type=str, default='0', help='Device to run on, e.g., "0" or "0,1"')
    parser.add_argument('--workers', type=int, default=16, help='Number of worker threads for data loading.')
    parser.add_argument('--best_metric', type=str, default='metrics/mAP50-95(B)', help='Metric to use for selecting the best model.')

    args = parser.parse_args()

    main(
        models_dir=args.models_dir,
        model_selection=args.model_selection,
        project_name=args.project_name,
        dataset_yaml_path=args.dataset_yaml_path,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        patience=args.patience,
        device=args.device,
        workers=args.workers,
        best_metric=args.best_metric
    )


# %%
