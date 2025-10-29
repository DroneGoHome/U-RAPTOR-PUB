#!/bin/bash

# Default arguments (edit as needed)
BASE_PATH="/mnt/d/Rowan/AeroDefence"
DATASET_PATH="$BASE_PATH/data/Related work from Rowan/new_dataset/10db/"
CLASS0_SUBPATH="no_drone"
CLASS1_SUBPATH="drones"
NUM_SAMPLES=650
BATCH_SIZE=64
VAL_RATIO=0.2
EPOCHS=20
LR=0.001
SAVE_PATH="$BASE_PATH/models/best_resnet18.pth"
CM_DIR="$BASE_PATH/exp/cm"

python3 binary_classifier.py \
  --main_path "$DATASET_PATH" \
  --class0_subpath "$CLASS0_SUBPATH" \
  --class1_subpath "$CLASS1_SUBPATH" \
  --num_samples $NUM_SAMPLES \
  --batch_size $BATCH_SIZE \
  --val_ratio $VAL_RATIO \
  --epochs $EPOCHS \
  --lr $LR \
  --save_path "$SAVE_PATH" \
  --cm_dir "$CM_DIR"
