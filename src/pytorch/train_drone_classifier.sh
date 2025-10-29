#!/bin/bash

# Default arguments (edit as needed)
master_meta_path='../../dataset/master_meta_data.json'
BASE_PATH="/mnt/d/Rowan/AeroDefence"
DATASET_PATH="$BASE_PATH/data/Related work from Rowan/new_dataset/10db/"
NUM_CLASSES=23
BATCH_SIZE=32
VAL_RATIO=0.2
EPOCHS=20
LR=0.001
SAVE_PATH="$BASE_PATH/models/best_resnet18_drones_moe_2.pth"
CM_DIR="$BASE_PATH/exp/cm_drones_moe_2"

python3 drone_classification.py \
  --main_path "$master_meta_path" \
  --num_classes $NUM_CLASSES \
  --batch_size $BATCH_SIZE \
  --val_ratio $VAL_RATIO \
  --epochs $EPOCHS \
  --lr $LR \
  --save_path "$SAVE_PATH" \
  --cm_dir "$CM_DIR"
