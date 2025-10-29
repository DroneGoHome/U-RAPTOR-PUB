#!/bin/bash

# --- Configuration ---
VERSION=29
# CURRENT_DATE=$(date +%F)
CURRENT_DATE=2025-10-27

# --- Dataset Paths ---
BASE_DATASET_PATH="../../dataset/RoboFlow-$CURRENT_DATE"
BASE_SIGNALS_PATH="/mnt/d/Rowan/AeroDefence/dataset/"
ORIGINAL_DIR="$BASE_DATASET_PATH/original/"
CLEANED_DIR="$BASE_DATASET_PATH/cleaned/"
BALANCED_DIR="$BASE_DATASET_PATH/balanced/"
AUGMENTED_DIR="$BASE_DATASET_PATH/augmented/"
TEST_DIR="../../dataset/RoboFlow-$CURRENT_DATE/test/"
USE_CLEANED_FOR_SAMPLE2=true  # Set to true to use cleaned dataset as sample_2 pool

# --- Dataset Splits ---
TRAIN_RATIO=0.7
VAL_RATIO=0.3
TEST_RATIO=0.0
SEED=42


# --- Metadata Paths ---
META_FILES=("$BASE_SIGNALS_PATH/metadata/original_abspath_meta_data.json" "$BASE_SIGNALS_PATH/unannotated_cage_dataset/images/meta_data.json" "$BASE_SIGNALS_PATH/metadata/temp_meta.json" "$BASE_SIGNALS_PATH/SiteSurveyRecord/spectrogram - 2025-09-08 - 2025-09-15/meta_data.json")
TEMP_META_FILE="temp_meta.json"

# --- Augmentation Schedule ---
BASE_AUGMENTATIONS=2
MAX_AUGMENTATIONS=5

# --- Augmentation Parameters ---
NUM_AUGMENTATIONS_PER_IMAGE=2
MAX_ABS_SHIFT_HZ=20e6
CENTER_DC=true
MIN_BOX_HEIGHT_RATIO=0.50
MIN_SIGNAL_WEIGHT=0.3
MAX_SIGNAL_WEIGHT=0.9
ADD_SECOND_SAMPLE_PROB=0.8
BACKGROUND_SAMPLE_PROB=0.3
FREQUENCY_SHIFT_PROB=0.9
CHANNEL_EFFECTS_PROB=0.7
MAX_CHANNEL_EFFECTS=1
SNR_DB_MIN=-15
SNR_DB_MAX=35
RICIAN_PROB=0.6
RICIAN_K_DB_MIN=-15
RICIAN_K_DB_MAX=35
OTHER_NORMALIZATION_PROB=0.2
SEED=42
VISUALIZE=false # Set to true to show a visualization at the end

# --- Test Dataset Parameters ---
SNR_VALUES=(-15 -5 5 15 25 35)
BACKGROUND_CLASS_NAME="background"
BACKGROUND_MULTIPLIER=3
TEST_COLOR_MAP="viridis"

# --- Training Configuration ---
MODELS_DIR="/mnt/d/Rowan/AeroDefence/models"
MODEL_SELECTION=('yolo11n' 'yolo11s' 'yolo11m' 'yolo11l' 'yolo11x' 'yolo12n' 'yolo12s' 'yolo12m' 'yolo12l' 'yolo12x'
) # Array for model selection
PROJECT_NAME="/mnt/d/Rowan/AeroDefence/exp/yolo/Roboflow-$CURRENT_DATE"
EPOCHS=100
IMG_SIZE=640
BATCH_SIZE=16
PATIENCE=10
DEVICE="0,1"
WORKERS=2  # Reduced from 4 to prevent DataLoader segfaults
BEST_METRIC='metrics/mAP90-95(B)'

# --- Evaluation Configuration ---
EVAL_BATCH_SIZE=16
EVAL_CONF_THRESHOLDS=(0.1 0.25 0.5 0.75)  # Multiple confidence thresholds
EVAL_IOU_THRESHOLDS=(0.3 0.45 0.6 0.75)    # Multiple IoU thresholds
EVAL_DEVICE="0"
EVAL_WORKERS=4  # Reduced from 8 to prevent issues

# --- Logging Configuration ---
LOG_DIR="$PROJECT_NAME/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"

# --- Script Execution ---

# Log initial information
{
    echo "--- Script Start: $(date) ---"
    echo "Current date variable: $CURRENT_DATE"
    echo "Roboflow version: $VERSION"
    echo "Project Name: $PROJECT_NAME"
    echo "Log file: $LOG_FILE"
    echo "---------------------------------"
} | tee -a "$LOG_FILE"

# # 1. Download dataset from Roboflow
# echo "Downloading dataset from Roboflow..." | tee -a "$LOG_FILE"
# roboflow download -f yolo11 -l "$ORIGINAL_DIR" "aerodefense/u-raptor/$VERSION" | tee -a "$LOG_FILE"

# # 2. Clean the downloaded dataset
# echo "Cleaning the dataset..." | tee -a "$LOG_FILE"
# FILTER_CLASSES=$(python3 -c "import yaml, re; data=yaml.safe_load(open('$ORIGINAL_DIR/data.yaml')); print(' '.join([name for name in data['names'] if not re.search(r'_RC(-[a-z])?$', name)]))")
# echo "Filter classes: $FILTER_CLASSES" | tee -a "$LOG_FILE"
# python dataset_cleaner.py \
#     --dataset_dir "$ORIGINAL_DIR" \
#     --destination_dir  "$CLEANED_DIR" \
#     --filter_classes $FILTER_CLASSES \
#     2>&1 | tee -a "$LOG_FILE"

# # 3. Balance the cleaned dataset
# echo "Balancing the dataset..." | tee -a "$LOG_FILE"
# python dataset_balancer.py \
#     --source_dir "$CLEANED_DIR" \
#     --output_dir "$BALANCED_DIR" \
#     --train_ratio $TRAIN_RATIO \
#     --val_ratio $VAL_RATIO \
#     --test_ratio $TEST_RATIO \
#     --seed $SEED \
#     2>&1 | tee -a "$LOG_FILE"

# # 4. Augment the balanced dataset
# echo "Augmenting the dataset..." | tee -a "$LOG_FILE"
# python augment_annotations.py \
#     --base_dir "$BALANCED_DIR" \
#     --base_signals_dir "$BASE_SIGNALS_PATH" \
#     --yolo_dataset_dir "$BALANCED_DIR" \
#     --augmented_dataset_dir "$AUGMENTED_DIR" \
#     $(if [ "$USE_CLEANED_FOR_SAMPLE2" = true ]; then echo "--cleaned_dataset_dir $CLEANED_DIR"; fi) \
#     --meta_files "${META_FILES[@]}" \
#     --temp_meta_file "$TEMP_META_FILE" \
#     --base_augmentations $BASE_AUGMENTATIONS \
#     --max_augmentations $MAX_AUGMENTATIONS \
#     --num_augmentations_per_image $NUM_AUGMENTATIONS_PER_IMAGE \
#     --max_abs_shift_hz $MAX_ABS_SHIFT_HZ \
#     --center_dc $CENTER_DC \
#     --min_box_height_ratio $MIN_BOX_HEIGHT_RATIO \
#     --min_signal_weight $MIN_SIGNAL_WEIGHT \
#     --max_signal_weight $MAX_SIGNAL_WEIGHT \
#     --add_second_sample_prob $ADD_SECOND_SAMPLE_PROB \
#     --background_sample_prob $BACKGROUND_SAMPLE_PROB \
#     --frequency_shift_prob $FREQUENCY_SHIFT_PROB \
#     --channel_effects_prob $CHANNEL_EFFECTS_PROB \
#     --max_channel_effects $MAX_CHANNEL_EFFECTS \
#     --snr_db_min $SNR_DB_MIN \
#     --snr_db_max $SNR_DB_MAX \
#     --rician_prob $RICIAN_PROB \
#     --rician_K_db_min $RICIAN_K_DB_MIN \
#     --rician_K_db_max $RICIAN_K_DB_MAX \
#     --other_normalization_prob $OTHER_NORMALIZATION_PROB \
#     --seed $SEED \
#     $(if [ "$VISUALIZE" = true ]; then echo "--visualize"; fi) \
#     2>&1 | tee -a "$LOG_FILE"

# 5. Create SNR test datasets
echo "Creating SNR test datasets..." | tee -a "$LOG_FILE"
python dataset_test_creator.py \
    --source_dir "$ORIGINAL_DIR" \
    --output_base_dir "$TEST_DIR" \
    --snr_values "${SNR_VALUES[@]}" \
    --background_class_name "$BACKGROUND_CLASS_NAME" \
    --background_multiplier $BACKGROUND_MULTIPLIER \
    --seed $SEED \
    2>&1 | tee -a "$LOG_FILE"

# # 6. Train on the balanced dataset
# echo "Training on the balanced dataset..." | tee -a "$LOG_FILE"
# python train_yolo.py \
#     --models_dir "$MODELS_DIR" \
#     --model_selection "${MODEL_SELECTION[@]}" \
#     --project_name "$PROJECT_NAME/balanced" \
#     --dataset_yaml_path "$BALANCED_DIR/data.yaml" \
#     --epochs $EPOCHS \
#     --img_size $IMG_SIZE \
#     --batch_size $BATCH_SIZE \
#     --patience $PATIENCE \
#     --device $DEVICE \
#     --workers $WORKERS \
#     --best_metric "$BEST_METRIC" \
#     2>&1 | tee -a "$LOG_FILE"

# 7. Train on the augmented dataset
echo "Training on the augmented dataset..." | tee -a "$LOG_FILE"
python train_yolo.py \
    --models_dir "$MODELS_DIR" \
    --model_selection "${MODEL_SELECTION[@]}" \
    --project_name "$PROJECT_NAME/augmented" \
    --dataset_yaml_path "$AUGMENTED_DIR/data.yaml" \
    --epochs $EPOCHS \
    --img_size $IMG_SIZE \
    --batch_size $BATCH_SIZE \
    --patience $PATIENCE \
    --device $DEVICE \
    --workers $WORKERS \
    --best_metric "$BEST_METRIC" \
    2>&1 | tee -a "$LOG_FILE"

# 8. Evaluate all models on test datasets
echo "Evaluating all models on test datasets..." | tee -a "$LOG_FILE"
python model_evaluator.py \
    --models_base_dir "$PROJECT_NAME" \
    --test_datasets_dir "$TEST_DIR" \
    --output_dir "$PROJECT_NAME/evaluation_results" \
    --batch_size $EVAL_BATCH_SIZE \
    --conf_thresholds "${EVAL_CONF_THRESHOLDS[@]}" \
    --iou_thresholds "${EVAL_IOU_THRESHOLDS[@]}" \
    --device $EVAL_DEVICE \
    --workers $EVAL_WORKERS \
    2>&1 | tee -a "$LOG_FILE"

echo "--- Script finished: $(date) ---" | tee -a "$LOG_FILE"


 
