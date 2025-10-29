# Quick Reference: SNR Test & Evaluation

## Create Test Datasets

```bash
cd /mnt/d/Rowan/AeroDefence/src/yolo

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

## Evaluate Models

```bash
cd /mnt/d/Rowan/AeroDefence/src/yolo

python model_evaluator.py \
    --models_base_dir "/mnt/d/Rowan/AeroDefence/exp/yolo/Roboflow-2025-10-12" \
    --test_datasets_dir "../../dataset/test" \
    --output_dir "/mnt/d/Rowan/AeroDefence/exp/yolo/Roboflow-2025-10-12/evaluation_results" \
    --batch_size 16 \
    --conf_thresholds 0.1 0.25 0.5 0.75 \
    --iou_thresholds 0.3 0.45 0.6 0.75 \
    --device "0"
```

## Run Full Pipeline

```bash
cd /mnt/d/Rowan/AeroDefence/src/yolo
bash download_and_train_yolo.sh
```

## Key Files

- **Create Tests**: `src/yolo/dataset_test_creator.py`
- **Evaluate**: `src/yolo/model_evaluator.py`
- **Pipeline**: `src/yolo/download_and_train_yolo.sh`
- **Full Docs**: `docs/SNR_EVALUATION_SYSTEM.md`

## Output Locations

- **Test Datasets**: `dataset/test/snr_{-15,-5,5,15,25,35}/`
- **Results**: `exp/yolo/Roboflow-2025-10-12/evaluation_results/`
- **Excel Summary**: `evaluation_results/summary_tables/evaluation_results.xlsx`
- **Plots**: `evaluation_results/visualizations/*.png`
