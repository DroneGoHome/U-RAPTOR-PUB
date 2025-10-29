#!/usr/bin/env python3
"""
Model Evaluator for SNR-Based Performance Analysis

This module evaluates all trained YOLO models across different SNR test datasets
and generates comprehensive performance metrics and visualizations.

Key Features:
- Automatic model discovery from experiment directories
- Evaluation on all SNR test datasets
- Comprehensive metrics: mAP, Precision, Recall, F1, FDR, Misclassification Rate
- Per-class and aggregate performance analysis
- Excel/CSV output and visualization plots
"""

import os
import sys
import json
import yaml
import argparse
from glob import glob
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from loguru import logger

from ultralytics import YOLO

# Configure Loguru
logger.remove()
logger.add(sys.stderr, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")


def discover_trained_models(models_base_dir: str) -> list[dict]:
    """
    Discovers all trained YOLO models in the experiment directory.
    
    Args:
        models_base_dir: Base directory containing training experiments
        
    Returns:
        List of dictionaries with model information:
        [{'path': '/path/to/best.pt', 'name': 'yolo11n', 'train_type': 'augmented', 'experiment': 'path'}, ...]
    """
    models = []
    
    # Find all best.pt files
    best_model_paths = glob(os.path.join(models_base_dir, '**', 'best.pt'), recursive=True)
    logger.info(f"Found {len(best_model_paths)} trained models")
    
    for model_path in best_model_paths:
        # Extract information from path
        path_parts = Path(model_path).parts
        
        # Determine train_type (augmented, balanced, cleaned, etc.)
        train_type = 'unknown'
        for part in path_parts:
            if 'augmented' in part.lower():
                train_type = 'augmented'
                break
            elif 'balanced' in part.lower():
                train_type = 'balanced'
                break
            elif 'cleaned' in part.lower():
                train_type = 'cleaned'
                break
        
        # Extract model name (directory name containing best.pt)
        model_name = path_parts[-2]  # Parent directory of best.pt
        
        # Experiment path (relative to models_base_dir)
        experiment_path = os.path.dirname(model_path)
        
        models.append({
            'path': model_path,
            'name': model_name,
            'train_type': train_type,
            'experiment': experiment_path
        })
        
        logger.info(f"  - {model_name} ({train_type}): {model_path}")
    
    return models


def discover_test_datasets(test_datasets_dir: str) -> list[dict]:
    """
    Discovers all test datasets organized by SNR.
    
    Args:
        test_datasets_dir: Base directory containing SNR test datasets
        
    Returns:
        List of dictionaries with test dataset information:
        [{'path': '/path/to/snr_-15', 'snr': -15, 'yaml': '/path/to/data.yaml'}, ...]
    """
    test_datasets = []
    
    # Find all snr_* directories
    snr_dirs = glob(os.path.join(test_datasets_dir, 'snr_*'))
    logger.info(f"Found {len(snr_dirs)} test datasets")
    
    for snr_dir in sorted(snr_dirs):
        # Extract SNR value from directory name
        dir_name = os.path.basename(snr_dir)
        try:
            snr_value = float(dir_name.replace('snr_', ''))
        except ValueError:
            logger.warning(f"Could not parse SNR from directory name: {dir_name}")
            continue
        
        # Find data.yaml
        yaml_path = os.path.join(snr_dir, 'data.yaml')
        if not os.path.exists(yaml_path):
            logger.warning(f"No data.yaml found in {snr_dir}")
            continue
        
        test_datasets.append({
            'path': snr_dir,
            'snr': snr_value,
            'yaml': yaml_path
        })
        
        logger.info(f"  - SNR = {snr_value} dB: {snr_dir}")
    
    return test_datasets


def evaluate_model_on_dataset(model_path: str, dataset_yaml: str, **kwargs) -> dict:
    """
    Evaluates a YOLO model on a specific dataset.
    
    Args:
        model_path: Path to the trained model (.pt file)
        dataset_yaml: Path to the dataset YAML file
        **kwargs: Additional arguments for model.val()
        
    Returns:
        Dictionary containing evaluation results
    """
    try:
        # Load model
        model = YOLO(model_path)
        
        # Run validation
        results = model.val(data=dataset_yaml, **kwargs)
        
        # Extract metrics
        metrics = {
            'map50': results.box.map50 if hasattr(results.box, 'map50') else None,
            'map50_95': results.box.map if hasattr(results.box, 'map') else None,
            'precision': results.box.mp if hasattr(results.box, 'mp') else None,
            'recall': results.box.mr if hasattr(results.box, 'mr') else None,
            'f1': None,  # Will calculate
        }
        
        # Calculate F1 score
        if metrics['precision'] is not None and metrics['recall'] is not None:
            p, r = metrics['precision'], metrics['recall']
            metrics['f1'] = 2 * (p * r) / (p + r + 1e-12)
        
        # Per-class metrics
        if hasattr(results.box, 'ap50'):
            metrics['per_class_ap50'] = results.box.ap50
        if hasattr(results.box, 'ap'):
            metrics['per_class_ap'] = results.box.ap
        
        # Confusion matrix
        if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
            metrics['confusion_matrix'] = results.confusion_matrix.matrix
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating model {model_path} on {dataset_yaml}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def calculate_additional_metrics(confusion_matrix: np.ndarray) -> dict:
    """
    Calculates additional metrics from confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix (classes x classes)
        
    Returns:
        Dictionary with additional metrics
    """
    if confusion_matrix is None:
        return {}
    
    metrics = {}
    
    # Ensure matrix is 2D
    if len(confusion_matrix.shape) != 2:
        return metrics
    
    n_classes = confusion_matrix.shape[0]
    
    # Calculate per-class metrics
    tp = np.diag(confusion_matrix)
    fp = confusion_matrix.sum(axis=0) - tp
    fn = confusion_matrix.sum(axis=1) - tp
    
    # False Detection Rate (FDR) = FP / (TP + FP)
    fdr = fp / (tp + fp + 1e-12)
    metrics['false_detection_rate'] = fdr.mean()
    metrics['per_class_fdr'] = fdr
    
    # Miss Rate = FN / (TP + FN)
    miss_rate = fn / (tp + fn + 1e-12)
    metrics['miss_rate'] = miss_rate.mean()
    metrics['per_class_miss_rate'] = miss_rate
    
    # Misclassification Rate
    total_predictions = confusion_matrix.sum()
    correct_predictions = tp.sum()
    metrics['misclassification_rate'] = (total_predictions - correct_predictions) / (total_predictions + 1e-12)
    
    # Per-class accuracy
    per_class_total = confusion_matrix.sum(axis=0)
    per_class_accuracy = tp / (per_class_total + 1e-12)
    metrics['per_class_accuracy'] = per_class_accuracy
    metrics['mean_class_accuracy'] = per_class_accuracy.mean()
    
    return metrics


def evaluate_all_models(
    models: list[dict],
    test_datasets: list[dict],
    output_dir: str,
    batch_size: int = 16,
    conf_thresholds: list[float] = None,
    iou_thresholds: list[float] = None,
    device: str = '0',
    workers: int = 8
):
    """
    Evaluates all models on all test datasets at multiple threshold combinations.
    
    Args:
        models: List of model dictionaries from discover_trained_models()
        test_datasets: List of test dataset dictionaries from discover_test_datasets()
        output_dir: Directory to save evaluation results
        batch_size: Batch size for inference
        conf_thresholds: List of confidence thresholds to evaluate
        iou_thresholds: List of IoU thresholds to evaluate
        device: Device for inference
        workers: Number of data loading workers
    """
    if conf_thresholds is None:
        conf_thresholds = [0.25]
    if iou_thresholds is None:
        iou_thresholds = [0.45]
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'raw_results'), exist_ok=True)
    
    all_results = []
    
    total_evaluations = len(models) * len(test_datasets) * len(conf_thresholds) * len(iou_thresholds)
    logger.info(f"Starting {total_evaluations} evaluations ({len(models)} models × {len(test_datasets)} datasets × {len(conf_thresholds)} conf × {len(iou_thresholds)} iou)")
    
    with tqdm(total=total_evaluations, desc="Evaluating models") as pbar:
        for model_info in models:
            for dataset_info in test_datasets:
                for conf_thresh in conf_thresholds:
                    for iou_thresh in iou_thresholds:
                        # Evaluate
                        metrics = evaluate_model_on_dataset(
                            model_path=model_info['path'],
                            dataset_yaml=dataset_info['yaml'],
                            batch=batch_size,
                            conf=conf_thresh,
                            iou=iou_thresh,
                            device=device,
                            workers=workers,
                            plots=False,  # Don't generate plots during validation
                            save=False    # Don't save predictions
                        )
                        
                        if metrics is None:
                            pbar.update(1)
                            continue
                        
                        # Calculate additional metrics
                        if 'confusion_matrix' in metrics:
                            additional_metrics = calculate_additional_metrics(metrics['confusion_matrix'])
                            metrics.update(additional_metrics)
                        
                        # Build result record
                        result = {
                            'model_name': model_info['name'],
                            'train_type': model_info['train_type'],
                            'snr_db': dataset_info['snr'],
                            'conf_threshold': conf_thresh,
                            'iou_threshold': iou_thresh,
                            'map50': metrics.get('map50'),
                            'map50_95': metrics.get('map50_95'),
                            'precision': metrics.get('precision'),
                            'recall': metrics.get('recall'),
                            'f1': metrics.get('f1'),
                            'false_detection_rate': metrics.get('false_detection_rate'),
                            'miss_rate': metrics.get('miss_rate'),
                            'misclassification_rate': metrics.get('misclassification_rate'),
                            'mean_class_accuracy': metrics.get('mean_class_accuracy'),
                        }
                        
                        all_results.append(result)
                        
                        # Save raw results
                        raw_result_filename = f"{model_info['name']}_{model_info['train_type']}_snr_{int(dataset_info['snr'])}_conf{conf_thresh}_iou{iou_thresh}.json"
                        raw_result_path = os.path.join(output_dir, 'raw_results', raw_result_filename)
                        with open(raw_result_path, 'w') as f:
                            json.dump(metrics, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
                        
                        pbar.set_postfix({
                            'model': model_info['name'],
                            'snr': dataset_info['snr'],
                            'conf': conf_thresh,
                            'iou': iou_thresh,
                            'mAP50': f"{metrics.get('map50', 0):.3f}" if metrics.get('map50') is not None else 'N/A'
                        })
                        pbar.update(1)
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save summary tables
    save_summary_tables(results_df, output_dir)
    
    # Generate visualizations
    generate_visualizations(results_df, output_dir)
    
    logger.success(f"Evaluation complete! Results saved to {output_dir}")


def save_summary_tables(results_df: pd.DataFrame, output_dir: str):
    """
    Saves summary tables in various formats.
    
    Args:
        results_df: DataFrame with all evaluation results
        output_dir: Output directory
    """
    summary_dir = os.path.join(output_dir, 'summary_tables')
    os.makedirs(summary_dir, exist_ok=True)
    
    # Overall summary
    overall_path = os.path.join(summary_dir, 'overall_results.csv')
    results_df.to_csv(overall_path, index=False)
    logger.info(f"Saved overall results to {overall_path}")
    
    # Pivot by SNR (for default thresholds)
    metrics_cols = ['map50', 'map50_95', 'precision', 'recall', 'f1', 'false_detection_rate', 'misclassification_rate']
    
    # If we have multiple thresholds, create separate summaries for each combination
    if 'conf_threshold' in results_df.columns and 'iou_threshold' in results_df.columns:
        threshold_combinations = results_df[['conf_threshold', 'iou_threshold']].drop_duplicates()
        
        for _, row in threshold_combinations.iterrows():
            conf = row['conf_threshold']
            iou = row['iou_threshold']
            
            # Filter for this threshold combination
            df_filtered = results_df[
                (results_df['conf_threshold'] == conf) & 
                (results_df['iou_threshold'] == iou)
            ]
            
            # Save filtered results
            filtered_path = os.path.join(summary_dir, f'results_conf{conf}_iou{iou}.csv')
            df_filtered.to_csv(filtered_path, index=False)
            logger.info(f"Saved results for conf={conf}, iou={iou} to {filtered_path}")
            
            # Create pivots for each metric
            for metric in metrics_cols:
                if metric in df_filtered.columns:
                    pivot = df_filtered.pivot_table(
                        values=metric,
                        index=['model_name', 'train_type'],
                        columns='snr_db',
                        aggfunc='mean'
                    )
                    pivot_path = os.path.join(summary_dir, f'by_snr_{metric}_conf{conf}_iou{iou}.csv')
                    pivot.to_csv(pivot_path)
    else:
        # Legacy: single threshold case
        for metric in metrics_cols:
            if metric in results_df.columns:
                pivot = results_df.pivot_table(
                    values=metric,
                    index=['model_name', 'train_type'],
                    columns='snr_db',
                    aggfunc='mean'
                )
                pivot_path = os.path.join(summary_dir, f'by_snr_{metric}.csv')
                pivot.to_csv(pivot_path)
                logger.info(f"Saved {metric} by SNR to {pivot_path}")
    
    # Excel file with multiple sheets
    excel_path = os.path.join(summary_dir, 'evaluation_results.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='All Results', index=False)
        
        # Per train type
        for train_type in results_df['train_type'].unique():
            df_filtered = results_df[results_df['train_type'] == train_type]
            df_filtered.to_excel(writer, sheet_name=train_type, index=False)
        
        # Per threshold combination (if multiple)
        if 'conf_threshold' in results_df.columns and 'iou_threshold' in results_df.columns:
            threshold_combinations = results_df[['conf_threshold', 'iou_threshold']].drop_duplicates()
            for idx, row in threshold_combinations.iterrows():
                conf = row['conf_threshold']
                iou = row['iou_threshold']
                sheet_name = f'conf{conf}_iou{iou}'[:31]  # Excel sheet name limit
                df_filtered = results_df[
                    (results_df['conf_threshold'] == conf) & 
                    (results_df['iou_threshold'] == iou)
                ]
                df_filtered.to_excel(writer, sheet_name=sheet_name, index=False)
        
        logger.info(f"Saved Excel file to {excel_path}")


def generate_visualizations(results_df: pd.DataFrame, output_dir: str):
    """
    Generates visualization plots.
    
    Args:
        results_df: DataFrame with all evaluation results
        output_dir: Output directory
    """
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # Check if we have multiple thresholds
    has_multiple_thresholds = 'conf_threshold' in results_df.columns and 'iou_threshold' in results_df.columns
    
    if has_multiple_thresholds:
        # Get unique threshold combinations
        threshold_combinations = results_df[['conf_threshold', 'iou_threshold']].drop_duplicates()
        
        # Create plots for each threshold combination
        for _, thresh_row in threshold_combinations.iterrows():
            conf = thresh_row['conf_threshold']
            iou = thresh_row['iou_threshold']
            
            df_thresh = results_df[
                (results_df['conf_threshold'] == conf) & 
                (results_df['iou_threshold'] == iou)
            ]
            
            suffix = f"_conf{conf}_iou{iou}"
            _generate_plots(df_thresh, viz_dir, suffix)
        
        # Also create comparison plots across thresholds
        _generate_threshold_comparison_plots(results_df, viz_dir)
    else:
        # Single threshold case
        _generate_plots(results_df, viz_dir, "")
    
    logger.success("All visualizations generated")


def _generate_plots(df: pd.DataFrame, viz_dir: str, suffix: str = ""):
    """Helper function to generate standard plots for a filtered dataframe."""
    
    # 1. SNR vs mAP50 for all models
    plt.figure()
    for train_type in df['train_type'].unique():
        df_filtered = df[df['train_type'] == train_type]
        for model_name in df_filtered['model_name'].unique():
            df_model = df_filtered[df_filtered['model_name'] == model_name]
            df_model = df_model.sort_values('snr_db')
            plt.plot(df_model['snr_db'], df_model['map50'], marker='o', label=f"{model_name} ({train_type})")
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('mAP@50')
    plt.title(f'Model Performance vs SNR{suffix}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'snr_vs_map50{suffix}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Generated SNR vs mAP50 plot{suffix}")
    
    # 2. SNR vs FDR
    if 'false_detection_rate' in df.columns:
        plt.figure()
        for train_type in df['train_type'].unique():
            df_filtered = df[df['train_type'] == train_type]
            for model_name in df_filtered['model_name'].unique():
                df_model = df_filtered[df_filtered['model_name'] == model_name]
                df_model = df_model.sort_values('snr_db')
                plt.plot(df_model['snr_db'], df_model['false_detection_rate'], marker='s', label=f"{model_name} ({train_type})")
        
        plt.xlabel('SNR (dB)')
        plt.ylabel('False Detection Rate')
        plt.title(f'False Detection Rate vs SNR{suffix}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'snr_vs_fdr{suffix}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Generated SNR vs FDR plot{suffix}")
    
    # 3. Model comparison at specific SNR values
    snr_values = sorted(df['snr_db'].unique())
    if len(snr_values) > 0:
        fig, axes = plt.subplots(1, len(snr_values), figsize=(4*len(snr_values), 6), sharey=True)
        if len(snr_values) == 1:
            axes = [axes]
        
        for ax, snr in zip(axes, snr_values):
            df_snr = df[df['snr_db'] == snr]
            df_snr = df_snr.sort_values('map50', ascending=False)
            
            x_labels = [f"{row['model_name']}\n({row['train_type']})" for _, row in df_snr.iterrows()]
            ax.bar(range(len(df_snr)), df_snr['map50'])
            ax.set_xticks(range(len(df_snr)))
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
            ax.set_title(f'SNR = {int(snr)} dB')
            ax.set_ylabel('mAP@50')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'model_comparison_by_snr{suffix}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Generated model comparison plot{suffix}")


def _generate_threshold_comparison_plots(df: pd.DataFrame, viz_dir: str):
    """Generate plots comparing performance across different thresholds."""
    
    # For each model and SNR, plot performance vs confidence threshold
    for train_type in df['train_type'].unique():
        df_train = df[df['train_type'] == train_type]
        
        for model_name in df_train['model_name'].unique():
            df_model = df_train[df_train['model_name'] == model_name]
            
            # Plot mAP50 vs conf_threshold for each SNR
            plt.figure(figsize=(12, 6))
            for snr in sorted(df_model['snr_db'].unique()):
                df_snr = df_model[df_model['snr_db'] == snr]
                # Average over IoU thresholds
                df_grouped = df_snr.groupby('conf_threshold')['map50'].mean().reset_index()
                plt.plot(df_grouped['conf_threshold'], df_grouped['map50'], marker='o', label=f'SNR={int(snr)}dB')
            
            plt.xlabel('Confidence Threshold')
            plt.ylabel('mAP@50')
            plt.title(f'{model_name} ({train_type}) - mAP50 vs Confidence Threshold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f'conf_threshold_comparison_{model_name}_{train_type}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    logger.info("Generated threshold comparison plots")


def main(args):
    """Main evaluation function."""
    
    logger.info("="*80)
    logger.info("Starting Model Evaluation")
    logger.info("="*80)
    
    # Discover models
    logger.info(f"Discovering trained models in: {args.models_base_dir}")
    models = discover_trained_models(args.models_base_dir)
    
    if not models:
        logger.error("No trained models found!")
        return
    
    # Discover test datasets
    logger.info(f"Discovering test datasets in: {args.test_datasets_dir}")
    test_datasets = discover_test_datasets(args.test_datasets_dir)
    
    if not test_datasets:
        logger.error("No test datasets found!")
        return
    
    # Evaluate all models
    evaluate_all_models(
        models=models,
        test_datasets=test_datasets,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        conf_thresholds=args.conf_thresholds,
        iou_thresholds=args.iou_thresholds,
        device=args.device,
        workers=args.workers
    )
    
    logger.success("="*80)
    logger.success("Model Evaluation Complete!")
    logger.success("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained YOLO models on SNR test datasets.")
    
    # Input paths
    parser.add_argument('--models_base_dir', type=str, required=True,
                        help='Base directory containing trained models (experiment directory)')
    parser.add_argument('--test_datasets_dir', type=str, required=True,
                        help='Base directory containing SNR test datasets')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save evaluation results')
    
    # Inference parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference')
    parser.add_argument('--conf_thresholds', type=float, nargs='+', default=[0.25],
                        help='Confidence thresholds for detections (e.g., 0.1 0.25 0.5 0.75)')
    parser.add_argument('--iou_thresholds', type=float, nargs='+', default=[0.45],
                        help='IoU thresholds for NMS (e.g., 0.3 0.45 0.6 0.75)')
    parser.add_argument('--device', type=str, default='0',
                        help='Device for inference (e.g., "0" or "0,1")')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    main(args)
