# SNR-Based Model Evaluation System

This document describes the new testing and evaluation framework for systematic evaluation of YOLO models across different SNR (Signal-to-Noise Ratio) conditions.

## Overview

The system consists of three main components:

1. **`dataset_test_creator.py`** - Creates SNR-stratified test datasets
2. **`model_evaluator.py`** - Evaluates all trained models on all test datasets
3. **`download_and_train_yolo.sh`** (updated) - Orchestrates the entire pipeline

## Dataset Structure

```
dataset/
├── RoboFlow-2025-10-12/
│   ├── original/         # Downloaded from Roboflow
│   ├── cleaned/          # After filtering classes
│   ├── balanced/         # After balancing splits
│   └── augmented/        # With augmentations
└── test/                 # SNR test datasets
    ├── snr_-15/          # SNR = -15 dB
    ├── snr_-5/           # SNR = -5 dB
    ├── snr_5/            # SNR = 5 dB
    ├── snr_15/           # SNR = 15 dB
    ├── snr_25/           # SNR = 25 dB
    └── snr_35/           # SNR = 35 dB
```

Each SNR test dataset contains:
- **Mixed channel effects**: ~33% AWGN, ~33% Rayleigh fading, ~33% Rician fading
- **Balanced classes**: Background class = 3× other classes, other classes balanced equally
- **No frequency shifts**: Preserves ground truth bounding box locations
- **No signal mixing**: Single drone per sample (no multi-signal overlays)
- **Fixed color map**: 'viridis' for consistency

## Module 1: dataset_test_creator.py

### Purpose
Creates systematic test datasets at different SNR levels with controlled channel effects.

### Usage

```bash
python dataset_test_creator.py \
    --source_dir "../../dataset/RoboFlow-2025-10-12/balanced/" \
    --output_base_dir "../../dataset/test/" \
    --base_signals_dir "/mnt/d/Rowan/AeroDefence/dataset/" \
    --old_meta_file "/mnt/d/Rowan/AeroDefence/dataset/metadata/original_abspath_meta_data.json" \
    --new_meta_file "/mnt/d/Rowan/AeroDefence/dataset/unannotated_cage_dataset/images/meta_data.json" \
    --temp_meta_file "temp_test_meta.json" \
    --snr_values -15 -5 5 15 25 35 \
    --background_class_name "background" \
    --background_multiplier 3 \
    --color_map "viridis" \
    --center_dc true \
    --seed 42
```

### Key Parameters

- `--snr_values`: List of SNR values in dB (e.g., -15 -5 5 15 25 35)
- `--background_multiplier`: Background class count = multiplier × other class count (default: 3)
- `--color_map`: Fixed colormap for all test images (default: 'viridis')
- `--seed`: Random seed for reproducibility

### Channel Effects

For each SNR level:
- **AWGN**: Pure additive white Gaussian noise at specified SNR
- **Rayleigh Fading**: Rayleigh fading + AWGN at specified SNR
- **Rician Fading**: Rician fading + AWGN at specified SNR
  - K-factor randomized in range: [SNR - 5 dB, SNR + 5 dB]
  - Example: For SNR = 15 dB, K-factor ∈ [10, 20] dB

### Output

Each SNR folder (e.g., `snr_15/`) contains:
- Standard YOLO dataset structure: `train/`, `val/`, `test/`
- `data.yaml` with paths configured
- Images with filename format: `{original_name}_snr{SNR}_{effect_type}[_K{K_factor}].png`
- Corresponding annotation files (unchanged from source)

## Module 2: model_evaluator.py

### Purpose
Evaluates all trained YOLO models on all SNR test datasets and generates comprehensive metrics and visualizations.

### Usage

```bash
python model_evaluator.py \
    --models_base_dir "/mnt/d/Rowan/AeroDefence/exp/yolo/Roboflow-2025-10-12" \
    --test_datasets_dir "../../dataset/test" \
    --output_dir "/mnt/d/Rowan/AeroDefence/exp/yolo/Roboflow-2025-10-12/evaluation_results" \
    --batch_size 16 \
    --conf_thresholds 0.1 0.25 0.5 0.75 \
    --iou_thresholds 0.3 0.45 0.6 0.75 \
    --device "0" \
    --workers 8
```

### Key Parameters

- `--models_base_dir`: Directory containing trained model experiments
- `--test_datasets_dir`: Base directory with SNR test datasets
- `--conf_thresholds`: List of confidence thresholds to evaluate (default: [0.25])
- `--iou_thresholds`: List of IoU thresholds for NMS (default: [0.45])
- Multiple thresholds enable ROC-like analysis and operating point selection

### Metrics Computed

#### Detection Performance
- **mAP@50**: Mean Average Precision at IoU=0.50
- **mAP@50-95**: Mean Average Precision averaged over IoU thresholds [0.50:0.95]
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: Harmonic mean of precision and recall

#### Classification Performance
- **Per-class Accuracy**: Accuracy for each drone class
- **Mean Class Accuracy**: Average accuracy across all classes
- **Confusion Matrix**: Full confusion matrix saved in raw results

#### Reliability Metrics
- **False Detection Rate (FDR)**: FP / (TP + FP)
- **Miss Rate**: FN / (TP + FN)
- **Misclassification Rate**: (Total - Correct) / Total

### Output Structure

```
evaluation_results/
├── raw_results/                          # JSON files with detailed metrics
│   ├── yolo11n_augmented_snr_-15_conf0.25_iou0.45.json
│   ├── yolo11n_augmented_snr_-15_conf0.5_iou0.45.json
│   ├── yolo11n_augmented_snr_-5_conf0.25_iou0.45.json
│   └── ...
├── summary_tables/                       # CSV/Excel summaries
│   ├── overall_results.csv              # All results in one table
│   ├── results_conf0.25_iou0.45.csv     # Results for specific threshold
│   ├── by_snr_map50_conf0.25_iou0.45.csv
│   ├── by_snr_false_detection_rate_conf0.25_iou0.45.csv
│   └── evaluation_results.xlsx          # Multi-sheet Excel file
└── visualizations/                       # Plots
    ├── snr_vs_map50_conf0.25_iou0.45.png         # Performance vs SNR per threshold
    ├── snr_vs_fdr_conf0.25_iou0.45.png           # FDR vs SNR per threshold
    ├── model_comparison_by_snr_conf0.25_iou0.45.png
    ├── conf_threshold_comparison_yolo11n_augmented.png  # Threshold analysis
    └── ...
```

### Visualizations

1. **SNR vs mAP@50**: Line plot showing model performance degradation with decreasing SNR (per threshold)
2. **SNR vs FDR**: Line plot showing false detection rate trends (per threshold)
3. **Model Comparison**: Bar charts comparing all models at each SNR level (per threshold)
4. **Threshold Comparison** (NEW): Shows how confidence threshold affects mAP@50 at different SNR levels

## Module 3: download_and_train_yolo.sh (Updated)

### New Configuration Variables

```bash
# Test Dataset Parameters
SNR_VALUES=(-15 -5 5 15 25 35)
BACKGROUND_CLASS_NAME="background"
BACKGROUND_MULTIPLIER=3
TEST_COLOR_MAP="viridis"

```bash
# Evaluation Configuration
EVAL_BATCH_SIZE=16
EVAL_CONF_THRESHOLDS=(0.1 0.25 0.5 0.75)  # Multiple confidence thresholds
EVAL_IOU_THRESHOLDS=(0.3 0.45 0.6 0.75)    # Multiple IoU thresholds
EVAL_DEVICE="0"
EVAL_WORKERS=8
```
```

### Updated Pipeline

The full pipeline now includes:

1. **Download** - Download dataset from Roboflow
2. **Clean** - Filter unwanted classes
3. **Balance** - Balance train/val/test splits
4. **Augment** - Create augmented training dataset
5. **Create Test Sets** (NEW) - Generate SNR test datasets
6. **Train (Balanced)** - Train models on balanced dataset
7. **Train (Augmented)** - Train models on augmented dataset
8. **Evaluate** (NEW) - Evaluate all models on all test datasets

### Running the Full Pipeline

```bash
cd /mnt/d/Rowan/AeroDefence/src/yolo
bash download_and_train_yolo.sh
```

### Running Individual Steps

Uncomment the desired sections in the script. For example, to only create test datasets:

```bash
# Uncomment step 5 in the script, then run:
bash download_and_train_yolo.sh
```

## Workflow Example

### Step 1: Create Test Datasets

```bash
python dataset_test_creator.py \
    --source_dir "../../dataset/RoboFlow-2025-10-12/balanced/" \
    --output_base_dir "../../dataset/test/" \
    --base_signals_dir "/mnt/d/Rowan/AeroDefence/dataset/" \
    --old_meta_file "/mnt/d/Rowan/AeroDefence/dataset/metadata/original_abspath_meta_data.json" \
    --new_meta_file "/mnt/d/Rowan/AeroDefence/dataset/unannotated_cage_dataset/images/meta_data.json" \
    --snr_values -15 -5 5 15 25 35 \
    --background_multiplier 3 \
    --seed 42
```

**Output**: 6 test datasets at `dataset/test/snr_{-15,-5,5,15,25,35}/`

### Step 2: Train Models

(Assuming models are already trained via `train_yolo.py`)

### Step 3: Evaluate Models

```bash
python model_evaluator.py \
    --models_base_dir "/mnt/d/Rowan/AeroDefence/exp/yolo/Roboflow-2025-10-12" \
    --test_datasets_dir "../../dataset/test" \
    --output_dir "/mnt/d/Rowan/AeroDefence/exp/yolo/Roboflow-2025-10-12/evaluation_results"
```

**Output**: 
- Evaluation results in `evaluation_results/`
- Summary tables in Excel/CSV
- Visualization plots

### Step 4: Analyze Results

1. Open `evaluation_results/summary_tables/evaluation_results.xlsx`
2. Review plots in `evaluation_results/visualizations/`
3. Identify best performing models at different SNR levels
4. Check detailed per-model results in `evaluation_results/raw_results/`

## Key Design Decisions

### Why No Frequency Shifts in Test Data?
- Preserves ground truth bounding box locations
- Enables accurate evaluation of detection performance
- Frequency shifts are only used in training augmentation to improve model robustness

### Why No Signal Mixing in Test Data?
- Simplifies interpretation of results
- Each test sample has exactly one drone (or background)
- Makes per-class metrics more meaningful

### Why Background = 3× Other Classes?
- Background/negative samples are crucial for reducing false positives
- Mimics real-world scenarios where most of the spectrum is background noise
- Prevents models from being biased toward always predicting a drone class

### Why Fixed Color Map?
- Eliminates color variation as a confounding variable
- Ensures models are evaluated on signal characteristics, not visualization artifacts
- 'viridis' is chosen for perceptual uniformity

### Rician K-Factor Range
- K-factor centered around the SNR value (SNR ± 5 dB)
- Physically realistic: stronger LoS component correlates with better SNR
- Provides variation within each SNR test set

## Expected Outcomes

### Operating Point Analysis (Multiple Thresholds)
When evaluating with multiple confidence and IoU thresholds:
- **Low Conf (0.1)**: High recall, potentially higher FDR - good for "don't miss anything" scenarios
- **Medium Conf (0.25, 0.5)**: Balanced precision/recall tradeoff
- **High Conf (0.75)**: High precision, lower recall - good for "only high confidence" scenarios
- **Threshold comparison plots** show optimal operating points for different deployment requirements

### Model Performance Curves
- **High SNR (25-35 dB)**: All models should perform near their best
- **Medium SNR (5-15 dB)**: Separation between model architectures expected
- **Low SNR (-15 to -5 dB)**: Significant performance degradation, augmented models should outperform

### Augmentation Validation
- Models trained on augmented data should show:
  - Better performance at low SNR
  - More graceful degradation across SNR range
  - Lower false detection rates

### Model Selection Insights
- Identify which model architectures (11n, 11s, 11m, etc.) are most robust
- Determine SNR thresholds for acceptable performance
- Guide deployment decisions based on expected operating conditions

## Troubleshooting

### Issue: "No trained models found"
**Solution**: Ensure models are trained and `best.pt` files exist in the experiment directory

### Issue: "No test datasets found"
**Solution**: Run `dataset_test_creator.py` first to create test datasets

### Issue: Out of memory during evaluation
**Solution**: Reduce `--batch_size` or `--eval_batch_size`

### Issue: Evaluation takes too long
**Solution**: 
- Use fewer models (comment out large models in MODEL_SELECTION)
- Use fewer SNR values
- Increase `--workers` for faster data loading

## Future Enhancements

Potential additions to the framework:

1. **Confidence Score Analysis**: Distribution of confidence scores vs SNR
2. **Per-Class SNR Curves**: Individual performance curves for each drone class
3. **Statistical Significance Testing**: ANOVA/t-tests between model pairs
4. **Deployment Recommendations**: Automated report generation with model selection guidance
5. **Real-Time Dashboard**: Web-based visualization of evaluation results

## References

- YOLO Documentation: https://docs.ultralytics.com/
- Confusion Matrix Metrics: https://en.wikipedia.org/wiki/Confusion_matrix
- Channel Modeling: Rayleigh and Rician fading models
