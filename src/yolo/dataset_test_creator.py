#!/usr/bin/env python3
"""
Dataset Test Creator for SNR-Based Model Evaluation

This module creates systematic test datasets with controlled SNR levels and channel effects.
It generates test datasets at different SNR values, each containing a mix of:
- AWGN (Additive White Gaussian Noise)
- Rayleigh fading + AWGN
- Rician fading + AWGN

Key Features:
- Pure image-based augmentation (no signal processing needed!)
- Reads class information directly from YOLO annotations
- Creates clean organized test dataset by class
- Generates SNR-augmented versions for robustness testing
- Balanced test sets (background class = 3x other classes)
- No metadata matching required - works directly with images

Workflow:
1. Read images and annotations from RoboFlow test split
2. Extract class from annotations
3. Balance dataset (background = 3x other classes)
4. Save clean organized dataset: test/images/class_name/
5. Create SNR-augmented datasets by applying noise to images
"""

import os
import sys
import json
import yaml
import shutil
import random
import argparse
import re
from tqdm import tqdm
from glob import glob
from collections import defaultdict

import numpy as np
from PIL import Image
from loguru import logger

# Configure Loguru for colored output
logger.remove()
logger.add(sys.stderr, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")


def should_filter_class(class_name: str) -> bool:
    """
    Check if a class should be filtered out.
    
    Filters out:
    - Remote control classes (ending with _RC, _RC-s, _RC-m)
    
    Args:
        class_name: The class name to check
        
    Returns:
        True if the class should be filtered out, False otherwise
    """
    if class_name == 'background':
        return False
    
    # Filter out remote control classes
    if re.search(r'_RC(-[a-z])?$', class_name):
        return True
    
    return False


def normalize_class_name(class_name: str) -> str:
    """
    Normalize class name by removing variant suffixes.
    
    Examples:
    - DJI_Mini3-armed -> DJI_Mini3
    - DJI_Mavic3_RC-s -> DJI_Mavic3 (but this should be filtered first)
    
    Args:
        class_name: The original class name
        
    Returns:
        Normalized class name
    """
    if class_name == 'background':
        return 'background'
    
    # Remove -armed suffix
    class_name = re.sub(r'-armed$', '', class_name)
    
    return class_name


def get_test_data_from_original(source_dir: str) -> tuple[list[dict], int]:
    """
    Get test data from original RoboFlow dataset format.
    
    The original dataset has .jpg files in test/images/ folder with format:
    - test/images/DroneModel_params_sample_X_png.rf.hash.jpg
    - test/labels/DroneModel_params_sample_X_png.rf.hash.txt
    
    This function:
    1. Finds all .jpg files in test/images/ folder
    2. Reads data.yaml to get class names
    3. Reads annotations to determine actual class of each image
    4. Returns list of samples with image path, label path, class name, and annotations
    
    No metadata matching needed - we work directly with images!
    """
    logger.info(f"Loading test data from original RoboFlow format: {source_dir}")
    
    # Load class names from data.yaml
    yaml_path = os.path.join(source_dir, 'data.yaml')
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    class_names = yaml_data.get('names', {})
    if isinstance(class_names, list):
        class_names = {i: name for i, name in enumerate(class_names)}
    
    logger.info(f"Loaded {len(class_names)} classes from data.yaml")
    
    # Find all .jpg files in test/images folder
    test_images_dir = os.path.join(source_dir, 'test', 'images')
    test_labels_dir = os.path.join(source_dir, 'test', 'labels')
    img_files = glob(os.path.join(test_images_dir, '*.jpg'))
    logger.info(f"Found {len(img_files)} image files in test/images folder")
    
    # Process each image file
    test_samples = []
    background_count = 0
    class_distribution = defaultdict(int)
    filtered_count = 0
    normalized_count = 0
    
    for img_file in tqdm(img_files, desc="Processing test images"):
        # Read annotation file to determine the actual class
        img_basename = os.path.basename(img_file)
        label_basename = os.path.splitext(img_basename)[0] + '.txt'
        label_file = os.path.join(test_labels_dir, label_basename)
        
        # Determine class from annotation
        original_class = 'background'  # default
        annotations = []
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                annotations = f.readlines()
            
            if annotations:
                # Get first annotation's class (in case of multiple objects, use first)
                first_class_id = int(annotations[0].strip().split()[0])
                original_class = class_names.get(first_class_id, 'background')
        
        # Apply filtering
        if should_filter_class(original_class):
            filtered_count += 1
            continue
        
        # Apply normalization (e.g., DJI_Mini3-armed -> DJI_Mini3)
        normalized_class = normalize_class_name(original_class)
        if normalized_class != original_class:
            normalized_count += 1
        
        # Create sample dict
        sample = {
            'img_file_path': img_file,
            'label_file_path': label_file,
            'annotations': annotations,
            'image_class': normalized_class,  # Use normalized class
            'original_class': original_class,  # Keep original for reference
            'img_short_path': f"{normalized_class}/{img_basename}"
        }
        
        # Skip background samples for now (we'll handle them separately)
        if normalized_class == 'background':
            background_count += 1
            continue
        
        class_distribution[normalized_class] += 1
        test_samples.append(sample)
    
    logger.info(f"Filtered out {filtered_count} samples (remote controls)")
    logger.info(f"Normalized {normalized_count} class names (removed -armed suffix)")
    logger.info(f"Skipped {background_count} background samples (will add them back after balancing)")
    logger.info(f"Class distribution (non-background, after filtering & normalization): {dict(class_distribution)}")
    logger.success(f"Loaded {len(test_samples)} non-background test samples")
    
    return test_samples, background_count

def balance_test_samples(
    test_samples: list[dict],
    source_dir: str,
    background_class_name: str = 'background',
    background_multiplier: int = 3
) -> dict:
    """
    Balance test samples so that background class has background_multiplier times the count of other classes.
    Other classes are balanced equally among themselves.
    
    Args:
        test_samples: List of non-background test samples with metadata
        source_dir: Source dataset directory to find background images
        background_class_name: Name of the background class
        background_multiplier: How many times more background samples than other classes
        
    Returns:
        Dictionary with balanced samples grouped by class: {class_name: [samples]}
    """
    logger.info("Balancing test samples by class...")
    
    # Group samples by class
    samples_by_class = defaultdict(list)
    for sample in test_samples:
        class_name = sample['image_class']
        samples_by_class[class_name].append(sample)
    
    # Log initial distribution
    logger.info("Initial class distribution:")
    for class_name, samples in sorted(samples_by_class.items()):
        logger.info(f"  {class_name}: {len(samples)} samples")
    
    # Find minimum count among non-background classes
    if not samples_by_class:
        logger.error("No non-background classes found!")
        return {}
    
    min_count = min(len(v) for v in samples_by_class.values())
    logger.info(f"Minimum class count = {min_count}")
    
    # Balance non-background classes
    balanced_samples = {}
    for class_name, class_samples in samples_by_class.items():
        if len(class_samples) >= min_count:
            sampled = random.sample(class_samples, min_count)
        else:
            sampled = class_samples
            logger.warning(f"{class_name}: Only {len(class_samples)} samples available (target: {min_count})")
        
        balanced_samples[class_name] = sampled
        logger.info(f"Balanced {class_name}: {len(sampled)} samples")
    
    # Now add background samples
    target_background_count = min_count * background_multiplier
    logger.info(f"Target background count: {target_background_count}")
    
    # Load class names
    yaml_path = os.path.join(source_dir, 'data.yaml')
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    class_names = yaml_data.get('names', {})
    if isinstance(class_names, list):
        class_names = {i: name for i, name in enumerate(class_names)}
    
    # Find background images
    test_images_dir = os.path.join(source_dir, 'test', 'images')
    test_labels_dir = os.path.join(source_dir, 'test', 'labels')
    img_files = glob(os.path.join(test_images_dir, '*.jpg'))
    
    background_samples = []
    for img_file in img_files:
        img_basename = os.path.basename(img_file)
        label_basename = os.path.splitext(img_basename)[0] + '.txt'
        label_file = os.path.join(test_labels_dir, label_basename)
        
        # Check if this is a background image (no annotations or class is background)
        is_background = True
        annotations = []
        original_class = 'background'
        
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                annotations = f.readlines()
            
            if annotations:
                first_class_id = int(annotations[0].strip().split()[0])
                original_class = class_names.get(first_class_id, 'background')
                
                # Filter out unwanted classes (e.g., remote controls)
                if should_filter_class(original_class):
                    continue
                
                # Normalize class name
                normalized_class = normalize_class_name(original_class)
                is_background = (normalized_class == 'background')
        
        if is_background:
            background_samples.append({
                'img_file_path': img_file,
                'label_file_path': label_file,
                'annotations': annotations,
                'image_class': 'background',
                'img_short_path': f"background/{img_basename}"
            })
    
    logger.info(f"Found {len(background_samples)} background images")
    
    # Sample background images
    if len(background_samples) >= target_background_count:
        balanced_samples[background_class_name] = random.sample(background_samples, target_background_count)
    else:
        balanced_samples[background_class_name] = background_samples
        logger.warning(f"Only {len(background_samples)} background samples available (target: {target_background_count})")
    
    logger.info(f"Balanced background: {len(balanced_samples[background_class_name])} samples")
    
    # Log final distribution
    total = sum(len(v) for v in balanced_samples.values())
    logger.success(f"Final balanced dataset: {total} samples across {len(balanced_samples)} classes")
    for class_name, samples in sorted(balanced_samples.items()):
        logger.info(f"  {class_name}: {len(samples)} samples")
    
    return balanced_samples


def save_clean_test_dataset(
    balanced_samples: dict,
    source_dir: str,
    output_dir: str
):
    """
    Save the clean, organized test dataset with structure:
    test/images/class_name/image.jpg
    test/labels/class_name/annotation.txt
    
    This creates a balanced, class-organized dataset without any augmentation.
    
    Args:
        balanced_samples: Dictionary with {class_name: [samples]}
        source_dir: Source dataset directory (for data.yaml)
        output_dir: Output directory for clean test dataset
    """
    logger.info(f"Saving clean test dataset to: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy and update data.yaml
    source_yaml = os.path.join(source_dir, 'data.yaml')
    dest_yaml = os.path.join(output_dir, 'data.yaml')
    if os.path.exists(source_yaml):
        shutil.copy2(source_yaml, dest_yaml)
        with open(dest_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        data_config['path'] = output_dir
        data_config['test'] = 'test/images'
        # YOLO validation requires train/val keys even for test-only datasets
        # Point them to test images as dummy values
        data_config['train'] = 'test/images'
        data_config['val'] = 'test/images'
        with open(dest_yaml, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
    
    # Save samples by class
    total_saved = 0
    for class_name, samples in sorted(balanced_samples.items()):
        logger.info(f"Saving {len(samples)} samples for class: {class_name}")
        
        for sample in tqdm(samples, desc=f"Saving {class_name}"):
            # Create class directories
            images_class_dir = os.path.join(output_dir, 'test', 'images', class_name)
            labels_class_dir = os.path.join(output_dir, 'test', 'labels', class_name)
            os.makedirs(images_class_dir, exist_ok=True)
            os.makedirs(labels_class_dir, exist_ok=True)
            
            # Copy image
            img_basename = os.path.basename(sample['img_file_path'])
            dest_img = os.path.join(images_class_dir, img_basename)
            shutil.copy2(sample['img_file_path'], dest_img)
            
            # Copy/save annotation
            label_basename = os.path.splitext(img_basename)[0] + '.txt'
            dest_label = os.path.join(labels_class_dir, label_basename)
            if 'annotations' in sample and sample['annotations']:
                with open(dest_label, 'w') as f:
                    f.writelines(sample['annotations'])
            elif 'label_file_path' in sample and os.path.exists(sample['label_file_path']):
                shutil.copy2(sample['label_file_path'], dest_label)
            else:
                # Create empty annotation for background
                open(dest_label, 'w').close()
            
            total_saved += 1
    
    logger.success(f"Clean test dataset saved: {total_saved} samples in {len(balanced_samples)} classes")
    logger.info(f"Dataset location: {output_dir}")


def apply_channel_effects_to_image(
    img_path: str,
    snr_db: float,
    channel_effect_type: str,
    rician_K_db: float = None
) -> Image.Image:
    """
    Apply channel effects directly to a spectrogram image.
    
    This is much simpler and faster than regenerating from raw signals.
    We simulate channel effects by adding noise to the image pixel values.
    
    Args:
        img_path: Path to input image
        snr_db: Signal-to-noise ratio in dB
        channel_effect_type: One of ['awgn', 'rayleigh', 'rician']
        rician_K_db: Rician K-factor in dB (only used if channel_effect_type='rician')
        
    Returns:
        PIL Image with channel effects applied
    """
    # Load image and convert to float array [0, 1]
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Calculate noise power from SNR
    # SNR_dB = 10 * log10(signal_power / noise_power)
    signal_power = np.mean(img_array ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power)
    
    # Apply channel effects
    if channel_effect_type == 'awgn':
        # Just add Gaussian noise
        noise = np.random.normal(0, noise_std, img_array.shape)
        noisy_img = img_array + noise
        
    elif channel_effect_type == 'rayleigh':
        # Rayleigh fading: multiply by Rayleigh-distributed gain, then add noise
        h = np.random.rayleigh(1.0 / np.sqrt(2))
        noisy_img = img_array * h
        noise = np.random.normal(0, noise_std, img_array.shape)
        noisy_img = noisy_img + noise
        
    elif channel_effect_type == 'rician':
        # Rician fading: LOS + scattered components, then add noise
        if rician_K_db is None:
            raise ValueError("rician_K_db must be provided for Rician fading")
        
        K_linear = 10 ** (rician_K_db / 10.0)
        los_component = np.sqrt(K_linear / (K_linear + 1))
        scatter_component = np.sqrt(1 / (K_linear + 1)) * np.random.rayleigh(1.0 / np.sqrt(2))
        h = los_component + scatter_component
        
        noisy_img = img_array * h
        noise = np.random.normal(0, noise_std, img_array.shape)
        noisy_img = noisy_img + noise
    else:
        raise ValueError(f"Unknown channel effect type: {channel_effect_type}")
    
    # Clip to valid range and convert back to uint8
    noisy_img = np.clip(noisy_img, 0, 1)
    noisy_img_uint8 = (noisy_img * 255).astype(np.uint8)
    
    return Image.fromarray(noisy_img_uint8)




def create_snr_test_dataset(
    source_dir: str,
    output_dir: str,
    snr_db: float,
    balanced_samples: dict,
    channel_effects_distribution: dict = None,
    seed: int = 42
):
    """
    Creates a test dataset for a specific SNR value with mixed channel effects.
    Takes the clean balanced dataset and applies channel effects at the specified SNR.
    
    Args:
        source_dir: Source dataset directory (for data.yaml)
        output_dir: Output directory for this SNR test dataset
        snr_db: SNR value in dB for this test dataset
        balanced_samples: Dictionary with {class_name: [samples]}
        channel_effects_distribution: Dict with probabilities for each effect type
        seed: Random seed
    """
    if channel_effects_distribution is None:
        channel_effects_distribution = {
            'awgn': 0.33,
            'rayleigh': 0.33,
            'rician': 0.34
        }
    
    random.seed(seed)
    np.random.seed(seed)
    
    logger.info(f"Creating test dataset for SNR = {snr_db} dB at {output_dir}")
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy data.yaml from source
    source_yaml = os.path.join(source_dir, 'data.yaml')
    dest_yaml = os.path.join(output_dir, 'data.yaml')
    if os.path.exists(source_yaml):
        shutil.copy2(source_yaml, dest_yaml)
        # Update paths in data.yaml
        with open(dest_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        data_config['path'] = output_dir
        data_config['test'] = 'test/images'
        # YOLO validation requires train/val keys even for test-only datasets
        # Point them to test images as dummy values
        data_config['train'] = 'test/images'
        data_config['val'] = 'test/images'
        with open(dest_yaml, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
    
    # Calculate Rician K-factor range for this SNR
    rician_K_min = snr_db - 5
    rician_K_max = snr_db + 5
    
    # Process samples for each class
    total_processed = 0
    for class_name, samples in sorted(balanced_samples.items()):
        logger.info(f"Processing {len(samples)} samples for class: {class_name}")
        
        for sample in tqdm(samples, desc=f"Creating SNR={snr_db}dB samples for {class_name}"):
            try:
                # Get image info
                img_file = sample['img_file_path']
                img_basename = os.path.basename(img_file)
                sample_name = os.path.splitext(img_basename)[0]
                
                # Randomly select channel effect type
                rand_val = random.random()
                cumulative = 0
                channel_effect_type = 'awgn'  # default
                for effect_type, prob in channel_effects_distribution.items():
                    cumulative += prob
                    if rand_val <= cumulative:
                        channel_effect_type = effect_type
                        break
                
                # Generate Rician K-factor if needed
                rician_K_db = None
                if channel_effect_type == 'rician':
                    rician_K_db = round(random.uniform(rician_K_min, rician_K_max), 2)
                
                # Apply channel effects directly to existing image
                test_image = apply_channel_effects_to_image(
                    img_path=img_file,
                    snr_db=snr_db,
                    channel_effect_type=channel_effect_type,
                    rician_K_db=rician_K_db
                )
                
                # Add suffix to indicate SNR and effect
                suffix_parts = [f"snr{int(snr_db)}", channel_effect_type]
                if rician_K_db is not None:
                    suffix_parts.append(f"K{rician_K_db}")
                filename_stem = f"{sample_name}_{'_'.join(suffix_parts)}"
                
                # Save image with class structure: test/images/class_name/filename.png
                image_save_dir = os.path.join(output_dir, 'test', 'images', class_name)
                os.makedirs(image_save_dir, exist_ok=True)
                test_image.save(os.path.join(image_save_dir, f"{filename_stem}.png"))
                
                # Save annotation with class structure: test/labels/class_name/filename.txt
                label_save_dir = os.path.join(output_dir, 'test', 'labels', class_name)
                os.makedirs(label_save_dir, exist_ok=True)
                with open(os.path.join(label_save_dir, f"{filename_stem}.txt"), 'w') as f:
                    if 'annotations' in sample and sample['annotations']:
                        f.writelines(sample['annotations'])
                
                total_processed += 1
                
            except Exception as e:
                logger.error(f"Failed to process sample {sample.get('img_file_path', 'unknown')}: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    logger.success(f"Test dataset for SNR = {snr_db} dB created: {total_processed} samples at {output_dir}")


def main(args):
    """Main function to create all SNR test datasets."""
    
    logger.info("="*80)
    logger.info("Starting Test Dataset Creation")
    logger.info("="*80)
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Get test samples from original RoboFlow format (with class info from annotations)
    # No metadata matching needed - we work directly with images!
    test_samples, background_count = get_test_data_from_original(
        source_dir=args.source_dir
    )
    
    if not test_samples:
        logger.error("No test samples found. Cannot create test datasets.")
        return
    
    logger.info(f"Found {len(test_samples)} non-background test samples")
    logger.info(f"Found {background_count} background samples")
    
    # Balance the test samples (adds background samples automatically)
    logger.info("Balancing test samples...")
    balanced_samples = balance_test_samples(
        test_samples=test_samples,
        source_dir=args.source_dir,
        background_class_name=args.background_class_name,
        background_multiplier=args.background_multiplier
    )
    
    if not balanced_samples:
        logger.error("Balancing failed. Cannot create test datasets.")
        return
    
    total_samples = sum(len(v) for v in balanced_samples.values())
    logger.info(f"Total balanced samples: {total_samples}")
    
    # Step 1: Save clean organized test dataset (no augmentation)
    clean_test_dir = os.path.join(args.output_base_dir, "clean")
    logger.info("="*80)
    logger.info("Step 1: Creating clean organized test dataset")
    logger.info("="*80)
    save_clean_test_dataset(
        balanced_samples=balanced_samples,
        source_dir=args.source_dir,
        output_dir=clean_test_dir
    )
    
    # Step 2: Create test datasets for each SNR value
    logger.info("="*80)
    logger.info("Step 2: Creating SNR-augmented test datasets")
    logger.info("="*80)
    for snr_db in args.snr_values:
        output_dir = os.path.join(args.output_base_dir, f"snr_{int(snr_db)}")
        
        create_snr_test_dataset(
            source_dir=args.source_dir,
            output_dir=output_dir,
            snr_db=snr_db,
            balanced_samples=balanced_samples,
            seed=args.seed + int(snr_db)  # Different seed per SNR
        )
    
    logger.success("="*80)
    logger.success("All test datasets created successfully!")
    logger.success(f"  - Clean test dataset: {clean_test_dir}")
    logger.success(f"  - SNR test datasets: {args.output_base_dir}/snr_*")
    logger.success("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create SNR-stratified test datasets for model evaluation.")
    
    # Input/Output paths
    parser.add_argument('--source_dir', type=str, required=True,
                        help='Source dataset directory (original RoboFlow dataset with test split)')
    parser.add_argument('--output_base_dir', type=str, required=True,
                        help='Base output directory for test datasets (e.g., ../../dataset/test)')
    
    # SNR configuration
    parser.add_argument('--snr_values', type=float, nargs='+', required=True,
                        help='List of SNR values in dB (e.g., -15 -5 5 15 25 35)')
    
    # Test dataset parameters
    parser.add_argument('--background_class_name', type=str, default='background',
                        help='Name of the background class')
    parser.add_argument('--background_multiplier', type=int, default=3,
                        help='Background class will have this many times the count of other classes')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    main(args)
