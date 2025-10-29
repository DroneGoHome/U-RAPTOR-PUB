# Binary and Drone Classification with ResNet18

This repository contains PyTorch implementations for two classification tasks:
1. **Binary Classification**: Distinguishing between "no_drone" and "drones" images
2. **Drone Classification**: Multi-class classification of different drone types

## Project Structure

```
binary_classification/
├── binary_classifier.py      # Main binary classification training script
├── drone_classification.py   # Main drone classification training script
├── pickle_classifier.py      # Training script for pickle data format
├── inference.py              # Inference script for trained models
├── datagen.py                # Data loading utilities for regular images
├── datagen_pickle.py         # Data loading utilities for pickle files
├── train_binary_classifier.sh # Bash script to run binary classification
├── train_drone_classifier.sh # Bash script to run drone classification
├── models/                   # Directory for saved models
├── cm/                       # Confusion matrices for binary classification
├── cm_drones/               # Confusion matrices for drone classification
└── cm_pickles/              # Confusion matrices for pickle data
```

## Pretrained Weights

Pretrained model weights for binary and drone classification are available for download:

- [Google Drive Folder (Pretrained Weights)](https://drive.google.com/drive/folders/1Nq-3iHiry5nc_f7PAfnF-uXKTSHHijL8)

Download the desired `.pth` files and place them in the `models/` directory.

## Environment Setup

### Prerequisites
- Python 3.8+ 
- CUDA-capable GPU (recommended)

### Setup
```bash
# Create environment
conda env create -f environment.yml
conda activate raptor

# Install additional packages
pip install scikit-learn matplotlib
```

### Verify
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

### Data Structure

#### For Binary Classification
```
/mnt/d/Raptor/binary/
├── no_drone/          # Class 0: Images without drones
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── drones/            # Class 1: Images with drones
    ├── subfolder1/    # Can have subfolders
    │   ├── drone1.jpg
    │   └── drone2.jpg
    └── subfolder2/
        └── drone3.jpg
```

#### For Drone Classification
```
/mnt/d/Raptor/binary/drones/
├── drone_type_1/      # Each subfolder represents a drone class
│   ├── image1.jpg
│   └── image2.jpg
├── drone_type_2/
│   ├── image3.jpg
│   └── image4.jpg
└── ... (up to 10 classes)
```

#### For Pickle Data
```
/mnt/d/Raptor/binary/drones/
├── class_0/
│   ├── data1.pkl
│   └── data2.pkl
└── class_1/
    ├── data3.pkl
    └── data4.pkl
```

## Running the Classification Tasks

### 1. Binary Classification

#### Quick Start with Default Settings
```bash
# Make script executable
chmod +x train_binary_classifier.sh

# Run training
./train_binary_classifier.sh
```

#### Manual Execution
```bash
python3 binary_classifier.py \
  --main_path "/mnt/d/Raptor/binary" \
  --class0_subpath "no_drone" \
  --class1_subpath "drones" \
  --num_samples 650 \
  --batch_size 64 \
  --val_ratio 0.2 \
  --epochs 20 \
  --lr 0.001 \
  --save_path "models/best_resnet18_binary.pth" \
  --cm_dir "cm"
```

#### Key Parameters
- `--main_path`: Root directory containing class folders
- `--class0_subpath`: Folder name for negative class (no_drone)
- `--class1_subpath`: Folder name for positive class (drones)
- `--num_samples`: Number of samples per class to use
- `--batch_size`: Training batch size
- `--val_ratio`: Validation split ratio (0.2 = 20% for validation)
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--save_path`: Where to save the best model
- `--cm_dir`: Directory to save confusion matrices

### 2. Drone Classification

#### Quick Start with Default Settings
```bash
# Make script executable
chmod +x train_drone_classification.sh

# Run training
./train_drone_classification.sh
```

#### Manual Execution
```bash
python3 drone_classification.py \
  --main_path "/mnt/d/Raptor/binary/drones" \
  --num_classes 10 \
  --batch_size 64 \
  --val_ratio 0.2 \
  --epochs 20 \
  --lr 0.001 \
  --save_path "models/best_resnet18_drones.pth" \
  --cm_dir "cm_drones"
```

#### Key Parameters
- `--main_path`: Root directory with class subfolders
- `--num_classes`: Number of drone classes (automatically detected from folders)
- Other parameters same as binary classification

### 3. Pickle Data Classification

For data stored in pickle format:

```bash
python3 pickle_classifier.py \
  --main_path "/mnt/d/Raptor/binary/drones" \
  --num_classes 10 \
  --batch_size 64 \
  --val_ratio 0.2 \
  --epochs 20 \
  --lr 0.001 \
  --save_path "models/best_resnet18_pickles.pth" \
  --cm_dir "cm_pickles" \
  --file_ext ".pkl"
```

## Model Inference

### Running Inference on Test Data
```bash
python3 inference.py \
  --model_path "models/best_resnet18_binary.pth" \
  --main_path "/mnt/d/Raptor/binary" \
  --class0_subpath "no_drone" \
  --class1_subpath "drones" \
  --output_dir "eval_results" \
  --use_gpu
```

### Single Image Prediction
```bash
python3 inference.py \
  --model_path "models/best_resnet18_binary.pth" \
  --image_path "/path/to/single/image.jpg" \
  --use_gpu
```

## Output Files

### Model Files
- `models/best_resnet18_binary.pth`: Best binary classification model
- `models/best_resnet18_drones.pth`: Best drone classification model

### Confusion Matrices
- `cm/confusion_matrix_best.png`: Binary classification confusion matrix
- `cm_drones/confusion_matrix_best.png`: Drone classification confusion matrix
- `cm_pickles/confusion_matrix_epoch_X.png`: Pickle data confusion matrices

### Evaluation Results
- `eval_results/confusion_matrix.png`: Test set confusion matrix
- `eval_results/metrics.txt`: Detailed evaluation metrics

## GPU Support

The scripts automatically detect and use:
- Multiple GPUs with DataParallel if available
- Single GPU if available
- CPU as fallback

To check GPU availability:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
```

## Customization

### Modifying Training Parameters

Edit the bash scripts or pass parameters directly:

```bash
# Custom training with different parameters
python3 binary_classifier.py \
  --epochs 50 \
  --batch_size 32 \
  --lr 0.0005 \
  --num_samples 1000
```

### Adding Data Augmentation

Modify the transforms in the Python scripts:

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
])
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size
2. **File not found**: Check data paths and structure
3. **No samples found**: Verify directory structure and file extensions

### Data Validation

Test your data loading:
```bash
# For regular images
python3 datagen.py

# For pickle files
python3 datagen_pickle.py --data_path "/your/data/path" --file_ext ".pkl"
```

## Performance Tips

1. **Use appropriate batch size**: Start with 64, adjust based on GPU memory
2. **Enable mixed precision**: Add `--fp16` flag if implemented
3. **Use multiple workers**: Increase `num_workers` in DataLoader for faster data loading
4. **Monitor training**: Watch for overfitting and adjust learning rate

## Results Interpretation

- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-score**: Harmonic mean of precision and recall

The confusion matrix shows:
- Rows: True labels
- Columns: Predicted labels
- Diagonal: Correct predictions
- Off-diagonal: Misclassifications
