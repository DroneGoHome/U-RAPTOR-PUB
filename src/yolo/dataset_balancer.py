# %%
import os
import sys
import yaml
import json
import shutil
import random
from glob import glob
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse
import re

import matplotlib.pyplot as plt
from loguru import logger

from utils import parallel_copy_files

# Configure Loguru for colored output
logger.remove()
logger.add(sys.stderr, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")


def load_yaml_file(yaml_path: str) -> dict:
    """Load YAML configuration file."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def save_yaml_file(yaml_path: str, data: dict) -> None:
    """Save YAML configuration file."""
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)


def clean_class_name(class_name: str) -> str:
    """
    Clean class names with special rules:
    1. Normalize '2' to 'Skydio_2'
    2. Normalize 'NoDrone' to 'background'
    3. Remove _RC* suffixes (e.g., _RC, _RC-m, _RC-s)
    4. Preserve -armed suffix for DJI_Mini3 and DJI_MavicMini4 (these are separate classes from base)
    5. Remove other detail suffixes like _hovering, _ground, _flying, _landed
    
    Examples:
        'NoDrone' -> 'background' (merged)
        'DJI_Mini3-armed' -> 'DJI_Mini3-armed' (preserved)
        'DJI_Mini3_hovering' -> 'DJI_Mini3' (suffix removed)
        'DJI_Mavic3_RC-m' -> 'DJI_Mavic3' (RC removed)
        'DJI_Phantom4_flying' -> 'DJI_Phantom4' (suffix removed)
    """
    import re
    
    # Special case: normalize '2' to 'Skydio_2'
    if class_name == '2':
        return 'Skydio_2'
    
    # Special case: normalize 'NoDrone' to 'background'
    if class_name == 'NoDrone':
        return 'background'
    
    # Remove _RC* suffixes (e.g., _RC, _RC-m, _RC-s) - always remove these
    if re.search(r'_RC(-[ms])?$', class_name):
        class_name = re.sub(r'_RC(-[ms])?$', '', class_name)
        return class_name
    
    # Define drones where -armed is a SEPARATE class (not just a suffix to remove)
    # For these drones, DJI_Mini3 and DJI_Mini3-armed are DIFFERENT classes
    armed_as_separate_class = ['DJI_Mini3', 'DJI_MavicMini4']
    
    # Check if this is an -armed variant of a drone where -armed means separate class
    if class_name.endswith('-armed'):
        base_class = class_name[:-6]  # Remove '-armed'
        if base_class in armed_as_separate_class:
            # Keep -armed as it's a separate class
            return class_name
        # For other drones, -armed is just a detail suffix, so fall through to remove it
    
    # Remove detail suffixes (including -armed for non-special drones)
    suffixes_to_remove = ['-armed', '_hovering', '_ground', '_flying', '_landed']
    for suffix in suffixes_to_remove:
        if class_name.endswith(suffix):
            class_name = class_name[:-len(suffix)]
            break
    
    return class_name


def analyze_dataset_distribution(dataset_dir: str) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """
    Analyze the distribution of classes in the dataset.
    
    Args:
        dataset_dir: Directory containing the cleaned dataset
        
    Returns:
        Tuple of (class_to_files dict, class_counts dict)
    """
    logger.info(f"Analyzing dataset distribution in '{dataset_dir}'...")
    
    # Load class names from yaml
    yaml_path = os.path.join(dataset_dir, 'data.yaml')
    if not os.path.exists(yaml_path):
        logger.error(f"data.yaml not found in {dataset_dir}")
        return {}, {}
    
    yaml_data = load_yaml_file(yaml_path)
    class_names = yaml_data.get('names', {})
    
    # Convert list to dict if needed
    if isinstance(class_names, list):
        class_names = {i: name for i, name in enumerate(class_names)}
    
    # Dictionary to store files by class
    class_to_files = defaultdict(list)
    class_counts = defaultdict(int)
    
    # Find all image files
    img_files = glob(f"{dataset_dir}/**/images/**/*.png", recursive=True)
    img_files += glob(f"{dataset_dir}/**/images/**/*.jpg", recursive=True)
    
    logger.info(f"Found {len(img_files)} image files to analyze...")
    
    for img_file in img_files:
        # Get corresponding annotation file
        label_file = img_file.replace('/images/', '/labels/').replace('.png', '.txt').replace('.jpg', '.txt')
        
        if os.path.exists(label_file):
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                if len(lines) > 0 and lines[0].strip():
                    # Get class ID from first line
                    class_id = int(lines[0].split()[0])
                    
                    # Get and clean class name
                    full_class_name = class_names.get(class_id, f'class_{class_id}')
                    cleaned_class_name = full_class_name #clean_class_name(full_class_name)
                    
                    # Store the file path for this class
                    class_to_files[cleaned_class_name].append(img_file)
                    class_counts[cleaned_class_name] += 1
                else:
                    # Empty annotation - background
                    class_to_files['background'].append(img_file)
                    class_counts['background'] += 1
                    
            except Exception as e:
                logger.warning(f"Error processing {label_file}: {e}")
        else:
            logger.warning(f"No label file found for {img_file}")
    
    logger.info(f"Found {len(class_counts)} classes in dataset")
    return dict(class_to_files), dict(class_counts)


def plot_class_distribution(class_counts: Dict[str, int], output_path: str):
    """
    Create and save a bar plot showing class distribution.
    
    Args:
        class_counts: Dictionary mapping class names to counts
        output_path: Path to save the plot
    """
    logger.info(f"Creating distribution plot...")
    
    # Sort classes by count
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    classes = [c[0] for c in sorted_classes]
    counts = [c[1] for c in sorted_classes]
    
    # Create figure
    plt.figure(figsize=(max(12, len(classes) * 0.5), 8))
    bars = plt.bar(range(len(classes)), counts, color='steelblue', alpha=0.8)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01, 
                str(count), ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Class Name', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Samples', fontsize=12, fontweight='bold')
    plt.title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Save plot
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.success(f"Distribution plot saved to: {output_path}")
    plt.close()
    
    # Print statistics
    logger.info("\n" + "="*60)
    logger.info("CLASS DISTRIBUTION STATISTICS")
    logger.info("="*60)
    for class_name, count in sorted_classes:
        logger.info(f"{class_name:30s}: {count:5d} samples")
    logger.info("="*60)
    logger.info(f"Total classes: {len(class_counts)}")
    logger.info(f"Total samples: {sum(counts)}")
    logger.info(f"Min samples: {min(counts)} ({classes[counts.index(min(counts))]})")
    logger.info(f"Max samples: {max(counts)} ({classes[0]})")
    logger.info(f"Mean samples: {sum(counts)/len(counts):.1f}")
    logger.info("="*60 + "\n")


def split_dataset(class_to_files: Dict[str, List[str]], 
                 train_ratio: float = 0.7, 
                 val_ratio: float = 0.2, 
                 test_ratio: float = 0.1,
                 seed: int = 42) -> Dict[str, Dict[str, List[str]]]:
    """
    Split dataset into train/val/test while maintaining class balance.
    
    Args:
        class_to_files: Dictionary mapping class names to file paths
        train_ratio: Percentage of data for training (default: 0.7)
        val_ratio: Percentage of data for validation (default: 0.2)
        test_ratio: Percentage of data for testing (default: 0.1)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys, each containing class_to_files dict
    """
    random.seed(seed)
    
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        logger.warning(f"Ratios sum to {total_ratio}, normalizing to 1.0")
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
    
    splits = {'train': {}, 'val': {}, 'test': {}}
    
    # Determine which splits are active (non-zero)
    active_splits = []
    if train_ratio > 0:
        active_splits.append('train')
    if val_ratio > 0:
        active_splits.append('val')
    if test_ratio > 0:
        active_splits.append('test')
    
    if not active_splits:
        logger.error("At least one split ratio must be greater than 0")
        return splits
    
    logger.info(f"Splitting dataset: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}")
    logger.info(f"Active splits: {', '.join(active_splits)}")
    
    for class_name, files in class_to_files.items():
        # Shuffle files for this class
        shuffled_files = files.copy()
        random.shuffle(shuffled_files)
        
        num_files = len(shuffled_files)
        train_count = int(num_files * train_ratio) if train_ratio > 0 else 0
        val_count = int(num_files * val_ratio) if val_ratio > 0 else 0
        # test_count gets the remainder
        
        # Split the files only for active splits
        if train_ratio > 0:
            splits['train'][class_name] = shuffled_files[:train_count]
        else:
            splits['train'][class_name] = []
            
        if val_ratio > 0:
            splits['val'][class_name] = shuffled_files[train_count:train_count + val_count]
        else:
            splits['val'][class_name] = []
            
        if test_ratio > 0:
            splits['test'][class_name] = shuffled_files[train_count + val_count:]
        else:
            splits['test'][class_name] = []
        
        logger.debug(f"{class_name}: {len(splits['train'][class_name])} train, "
                    f"{len(splits['val'][class_name])} val, "
                    f"{len(splits['test'][class_name])} test")
    
    return splits


def balance_dataset(class_to_files: Dict[str, List[str]], 
                   output_dir: str, 
                   source_dir: str,
                   min_samples: int = None,
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.2,
                   test_ratio: float = 0.1,
                   seed: int = 42):
    """
    Balance the dataset by keeping only min_samples from each class and split into train/val/test.
    
    Args:
        class_to_files: Dictionary mapping class names to file paths
        output_dir: Directory to save balanced dataset
        source_dir: Source directory containing original dataset
        min_samples: Number of samples to keep per class (if None, uses minimum class count)
        train_ratio: Percentage of data for training (default: 0.7)
        val_ratio: Percentage of data for validation (default: 0.2)
        test_ratio: Percentage of data for testing (default: 0.1)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Determine number of samples to keep per class
    if min_samples is None:
        min_samples = min(len(files) for files in class_to_files.values())
        # Ensure min_samples is at least 105
        min_samples = 105 if min_samples < 105 else min_samples
    
    logger.info(f"Balancing dataset to {min_samples} samples per class...")
    
    # First, balance the dataset by selecting min_samples from each class
    balanced_class_to_files = {}
    for class_name, files in class_to_files.items():
        if len(files) > min_samples:
            selected_files = random.sample(files, min_samples)
        else:
            selected_files = files
        balanced_class_to_files[class_name] = selected_files
    
    # Now split the balanced dataset into train/val/test
    splits = split_dataset(balanced_class_to_files, train_ratio, val_ratio, test_ratio, seed)
    
    # Copy yaml and other metadata files
    yaml_src = os.path.join(source_dir, 'data.yaml')
    yaml_data = None
    if os.path.exists(yaml_src):
        yaml_data = load_yaml_file(yaml_src)
    
    readme_src = os.path.join(source_dir, 'README.roboflow.txt')
    if os.path.exists(readme_src):
        shutil.copy(readme_src, os.path.join(output_dir, 'README.roboflow.txt'))
    
    # Copy original_path_mapping.json if it exists
    mapping_src = os.path.join(source_dir, 'original_path_mapping.json')
    if os.path.exists(mapping_src):
        mapping_dst = os.path.join(output_dir, 'original_path_mapping.json')
        shutil.copy(mapping_src, mapping_dst)
        logger.info(f"Copied original path mapping to balanced dataset")
    
    total_copied = 0
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    
    # Determine which splits are active (non-zero)
    active_splits = []
    if train_ratio > 0:
        active_splits.append('train')
    if val_ratio > 0:
        active_splits.append('val')
    if test_ratio > 0:
        active_splits.append('test')
    
    # Process each split (only active ones) using parallel copying
    for split_name in active_splits:
        split_class_files = splits[split_name]
        logger.info(f"\nProcessing {split_name} split...")
        
        # Collect all copy tasks
        img_files = []
        dest_imgs = []
        label_files = []
        dest_labels = []
        
        for class_name, files in split_class_files.items():
            if not files:  # Skip if no files for this split
                continue
            logger.debug(f"  {class_name}: {len(files)} samples")
            
            for img_file in files:
                # Create clean path structure: split_name/images/class_name/filename
                filename = os.path.basename(img_file)
                dest_img = os.path.join(output_dir, split_name, 'images', class_name, filename)
                
                img_files.append(img_file)
                dest_imgs.append(dest_img)
                
                # Add corresponding label
                label_file = img_file.replace('/images/', '/labels/').replace('.png', '.txt').replace('.jpg', '.txt')
                label_filename = os.path.basename(label_file)
                dest_label = os.path.join(output_dir, split_name, 'labels', class_name, label_filename)
                
                label_files.append(label_file)
                dest_labels.append(dest_label)
        
        # Execute copies in parallel
        file_pairs = list(zip(img_files, dest_imgs))
        label_pairs = list(zip(label_files, dest_labels))
        
        errors = parallel_copy_files(
            file_pairs, 
            label_pairs=label_pairs, 
            max_workers=8,
            desc=f"Copying {split_name} split"
        )
        
        copied_count = len(file_pairs) - len(errors)
        total_copied += copied_count
        split_counts[split_name] = copied_count
    
    # Update and save yaml with correct paths
    if yaml_data:
        yaml_data['path'] = output_dir
        # Only set paths for active splits (non-zero ratios)
        if train_ratio > 0:
            yaml_data['train'] = 'train/images'
        else:
            yaml_data.pop('train', None)
            
        if val_ratio > 0:
            yaml_data['val'] = 'val/images'
        else:
            yaml_data.pop('val', None)
            
        if test_ratio > 0:
            yaml_data['test'] = 'test/images'
        else:
            yaml_data.pop('test', None)
        
        save_yaml_file(os.path.join(output_dir, 'data.yaml'), yaml_data)
    
    logger.success(f"\nBalanced dataset created:")
    if train_ratio > 0:
        logger.success(f"  Train: {split_counts['train']} samples")
    if val_ratio > 0:
        logger.success(f"  Val: {split_counts['val']} samples")
    if test_ratio > 0:
        logger.success(f"  Test: {split_counts['test']} samples")
    logger.success(f"  Total: {total_copied} samples ({len(class_to_files)} classes × ~{min_samples} samples)")
    logger.success(f"Balanced dataset saved to: {output_dir}")


def main(source_dir: str, output_dir: str, plot_path: str, min_samples: int = None, 
         train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1, seed: int = 42):
    """
    Main function to analyze and balance dataset.
    
    Args:
        source_dir: Source directory containing cleaned dataset
        output_dir: Output directory for balanced dataset
        plot_path: Path to save distribution plot. If None, defaults to `output_dir/class_distribution_original.png`.
        min_samples: Number of samples per class (None = use minimum)
        train_ratio: Percentage for training split (default: 0.7)
        val_ratio: Percentage for validation split (default: 0.2)
        test_ratio: Percentage for test split (default: 0.1)
        seed: Random seed for reproducibility
    """
    logger.info("="*60)
    logger.info("DATASET BALANCER")
    logger.info("="*60)

    # If plot_path is not provided, create a default path inside the output directory
    if plot_path is None:
        plot_path = os.path.join(output_dir, 'class_distribution_original.png')
    
    # Step 1: Analyze dataset
    class_to_files, class_counts = analyze_dataset_distribution(source_dir)
    
    if not class_counts:
        logger.error("No classes found in dataset. Aborting.")
        return
    
    # Step 2: Plot distribution
    plot_class_distribution(class_counts, plot_path)
    
    # Step 3: Balance dataset with train/val/test split
    balance_dataset(class_to_files, output_dir, source_dir, min_samples, 
                   train_ratio, val_ratio, test_ratio, seed)
    
    # Step 4: Analyze and plot balanced dataset
    logger.info("\nAnalyzing balanced dataset...")
    balanced_class_to_files, balanced_class_counts = analyze_dataset_distribution(output_dir)
    balanced_plot_path = plot_path.replace('_original.png', '_balanced.png')
    plot_class_distribution(balanced_class_counts, balanced_plot_path)
    
    logger.success("Dataset balancing complete!")


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Balance YOLO dataset by equalizing class distribution and split into train/val/test.')
    parser.add_argument('--source_dir', type=str, 
                       default='../../dataset/RoboFlow-2025-09-15/cleaned',
                       help='Directory of the cleaned dataset to balance.')
    parser.add_argument('--output_dir', type=str, 
                       default='../../dataset/RoboFlow-2025-09-15/balanced',
                       help='Directory to save the balanced dataset.')
    parser.add_argument('--plot_path', type=str, default=None,
                       help='Path to save the class distribution plot. Defaults to a file in the output directory.')
    parser.add_argument('--min_samples', type=int, default=None,
                       help='Minimum number of samples per class (default: use smallest class count).')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Percentage of data for training (default: 0.7).')
    parser.add_argument('--val_ratio', type=float, default=0.3,
                       help='Percentage of data for validation (default: 0.3).')
    parser.add_argument('--test_ratio', type=float, default=0.0,
                       help='Percentage of data for testing (default: 0.0).')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility.')
    
    args = parser.parse_args()
    
    main(args.source_dir, args.output_dir, args.plot_path, args.min_samples,
         args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
# %%
