# Hierarchical Drone RF Classification

## Overview

This repository provides a comprehensive pipeline for hierarchical classification of drones using radio frequency (RF) spectrogram data. It includes data processing, model training, evaluation, and inference scripts, supporting multi-level classification (modulation, protocol, and model).

## Structure

- `hierarchy_updated.py`: Main training script for hierarchical classification using PyTorch and ResNet18.
- `hierarchy_inference.py`: Inference script to evaluate trained models and save results.
- `cleaned_meta_data.json`: Metadata file containing hierarchical labels for each sample.
- `Moe/src/constants.py`: Contains the hierarchy dictionary used for label mapping.
- `Moe/generator_data_loader.py`: Defines the `SpectogramDataset` class for loading spectrogram data.

## Workflow

1. **Data Preparation**: Ensure `cleaned_meta_data.json` is generated and contains hierarchical labels (`modulation_label`, `protocol_label`, `model_label`).
2. **Training**: Run `hierarchy_updated.py` to train the hierarchical classification model. This script:
   - Loads data and hierarchical mappings.
   - Trains a ResNet18-based model with three classification heads.
   - Saves the trained model and evaluation metrics.
3. **Inference**: Run `hierarchy_inference.py` to evaluate the trained model on the validation set and save predictions to `hierarchical_inference_results.csv`.
4. **Evaluation**: Confusion matrices and classification reports are saved in the `cm/` directory for each hierarchical level.

## Usage

### Training

```sh
python hierarchy_updated.py
```

### Inference

```sh
python hierarchy_inference.py
```

## Pretrained Model

The trained model weights (`hierarchical_model.pth`) are available for download on Google Drive:

[Download hierarchical_model.pth](https://drive.google.com/file/d/1CxAxt6SWFRZ5mlaSb9vKksEF2UkV9wyW/view?usp=drive_link)

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- tqdm

Install dependencies with:

```sh
pip install torch torchvision pandas numpy scikit-learn matplotlib seaborn tqdm
```
