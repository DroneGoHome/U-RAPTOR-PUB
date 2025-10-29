# %%
import os
import sys
import json
import yaml
import shutil
import random
import re
from tqdm import tqdm
from glob import glob
from collections.abc import Callable
from collections import defaultdict


import numpy as np
from PIL import Image
from loguru import logger
from scipy.signal import stft
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

from src.yolo.metadata import find_first_float_index, match_images_to_metadata
from src.yolo.utils import parallel_copy_directory

# Configure Loguru for colored output
logger.remove()
logger.add(sys.stderr, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

def load_and_truncate_signal(meta_data: dict) -> np.ndarray:
    """
    Loads a signal from a file and truncates it to the specified length.

    Args:
        meta_data (dict): Dictionary containing signal metadata.
    
    Returns:
        np.ndarray: The truncated signal sample.
    """
    FFT_SIZE = meta_data['fft_size']
    SAMPLING_RATE = meta_data['sampling_rate']
    
    # Load the signal from the file
    signal = np.memmap(meta_data['dat_file_path'], dtype=np.complex64, mode='r')
    
    # Calculate start and end indices based on time in seconds
    start_index = int(meta_data['time_start_sec'] * SAMPLING_RATE)
    end_index = int(meta_data['time_end_sec'] * SAMPLING_RATE)
    
    # Extract the signal sample
    signal_sample = signal[start_index:end_index]
    
    # Truncate the sample to the required length
    truncated_sample = signal_sample[:meta_data['num_fft_spec'] * FFT_SIZE]
    
    return truncated_sample

def get_centered_stft_matrix(signal_sample: np.ndarray, sampling_rate: float, fft_size: int, center_dc: bool = True) -> np.ndarray:
    """
    Computes the Short-Time Fourier Transform (STFT) of a signal sample and centers the frequency bins.

    Args:
        signal_sample (np.ndarray): The input signal sample.
        sampling_rate (float): The sampling rate of the signal.
        fft_size (int): The FFT size to use for the STFT.

    Returns:
        np.ndarray: The centered STFT matrix.
    """
    # Compute the STFT
    _, _, stft_matrix = stft(signal_sample, fs=sampling_rate, nperseg=fft_size, noverlap=128, window='hann', return_onesided=False)

    # Center the DC component (0 Hz)
    if center_dc:
        stft_matrix = np.fft.fftshift(stft_matrix, axes=0)

    return stft_matrix

def normalize_complex_signal(signal):
    """Normalize complex signal while preserving phase relationships."""
    # Find the maximum magnitude
    max_magnitude = np.max(np.abs(signal))
    # Normalize by the maximum magnitude
    return signal / (max_magnitude + 1e-12)

def frequency_normalize_spectrogram(spectrum_db):
    # Normalize each frequency bin independently
    mean_freq = np.mean(spectrum_db, axis=1, keepdims=True)
    std_freq = np.std(spectrum_db, axis=1, keepdims=True)
    # Apply normalization (z-score along frequency dimension)
    norm_spectrum = (spectrum_db - mean_freq) / (std_freq + 1e-12)
    return norm_spectrum

def time_normalize_spectrogram(spectrum_db):
    # Normalize each time slice independently  
    mean_time = np.mean(spectrum_db, axis=0, keepdims=True)
    std_time = np.std(spectrum_db, axis=0, keepdims=True)
    # Apply normalization (z-score along time dimension)
    norm_spectrum = (spectrum_db - mean_time) / (std_time + 1e-12)
    return norm_spectrum

def minmax_normalize_spectrogram(spectrum_db):
    # Scale entire spectrogram to [0,1] range
    min_val = np.min(spectrum_db)
    max_val = np.max(spectrum_db)
    norm_spectrum = (spectrum_db - min_val) / (max_val - min_val + 1e-12)
    return norm_spectrum

def power_normalize(signal):
    """Normalize signal to have unit power."""
    signal_power = np.mean(np.abs(signal)**2)
    return signal / np.sqrt(signal_power + 1e-12)

def z_normalize_complex(signal):
    """Z-normalize a complex signal (separately for real and imaginary parts)."""
    real = np.real(signal)
    imag = np.imag(signal)
    
    # Normalize real part
    real_mean, real_std = np.mean(real), np.std(real)
    real_norm = (real - real_mean) / (real_std + 1e-12)
    
    # Normalize imaginary part
    imag_mean, imag_std = np.mean(imag), np.std(imag)
    imag_norm = (imag - imag_mean) / (imag_std + 1e-12)
    
    return real_norm + 1j * imag_norm

def frequency_shift_signal(signal: np.ndarray, shift_hz: float, sampling_rate: float) -> np.ndarray:
    """
    Shifts the frequency of a complex signal.

    Args:
        signal (np.ndarray): The input complex signal.
        shift_hz (float): The frequency shift in Hz. Can be positive or negative.
        sampling_rate (float): The sampling rate of the signal in Hz.

    Returns:
        np.ndarray: The frequency-shifted complex signal.
    """
    # Create the time array for the signal's duration
    t = np.arange(len(signal)) / sampling_rate
    
    # Create the complex exponential for the shift
    shift_exponential = np.exp(1j * 2 * np.pi * shift_hz * t)
    
    # Apply the shift by element-wise multiplication
    shifted_signal = signal * shift_exponential
    
    return shifted_signal

def get_annotated_data(base_dir: str = '../../data/Related work from Rowan/new_dataset/original/') -> tuple[list[str], list[str], list[str]]:
    txt_files: list[str] = glob(f"{base_dir}/**/*.txt", recursive=True)
    # remove classes.txt
    txt_files = [f for f in txt_files if 'classes.txt' not in f and "README.roboflow.txt" not in f]
    logger.info(f"Found {len(txt_files)} '.txt' annotation files.")
    img_files: list[str] = glob(f"{base_dir}/**/*.png", recursive=True)
    classes: list[str] = list(set([os.path.basename(os.path.dirname(f)) for f in txt_files]))
    classes.sort()
    logger.info(f"\nDiscovered classes:\n{classes}")
    logger.debug(f"Total images: {len(img_files)}, Total annotations: {len(txt_files)}")
    logger.debug(f"\nSample images:\n{img_files[:3]}\nSample labels:\n{txt_files[:3]}")
    return img_files, txt_files, classes

def load_json(json_file: str) -> list[dict]:
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def save_json(data: list[dict], json_file: str) -> None:
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)

def get_dat_source_files(img_files, meta_file: str = '../data/Related work from Rowan/new_dataset/original/all_meta_data.json', base_dir: str='', base_signals_dir: str='') -> list[dict]:
    meta_data: list[dict] = load_json(meta_file)
    structure_template = find_dataset_structure(base_dir)

    # Use the new matching utility to handle all the path normalization
    img_short_names, img_spect_path, matched_img_paths = match_images_to_metadata(
        img_files=img_files,
        meta_data=meta_data,
        verbose=True
    )
    
    # Build annotation_samples dictionary
    annotation_samples = {'train':[], 'val':[], 'test':[]}
    
    for sample in meta_data:
        if sample['spect_path'] in matched_img_paths:
            # Find image_spect_index
            img_spect_index = img_spect_path.index(sample['spect_path'])
            split_name = find_split_name(img_short_names[img_spect_index], base_dir, structure_template)
            copy_sample = sample.copy()
            copy_sample['img_short_path'] = img_short_names[img_spect_index]
            if not os.path.isabs(copy_sample['dat_file_path']):
                common_folder = base_signals_dir.strip(os.path.sep).split(os.path.sep)[-1]
                copy_sample['dat_file_path'] = os.path.join(base_signals_dir, *copy_sample['dat_file_path'].split(common_folder)[-1].strip(os.path.sep).split(os.path.sep))
            annotation_samples[split_name].append(copy_sample)

    logger.success(f"Found {sum(len(v) for v in annotation_samples.values())} samples in the metadata that match the image files.")
    return annotation_samples

def get_spectrogram(sample, sampling_rate, fft_size, spectrogram_normalize: Callable | None, center_dc: bool = True) -> None:
    # --- STFT Calculation ---
    stft_matrix = get_centered_stft_matrix(signal_sample=sample, sampling_rate=sampling_rate, fft_size=fft_size, center_dc=center_dc)

    # --- Image Generation (from the sliced matrix) ---
    spectrum = np.abs(stft_matrix) ** 2
    spectrum_db = 10 * np.log10(spectrum + 1e-12)

    # # Normalize the sliced spectrum to create the image
    final_spectrum: np.ndarray = spectrogram_normalize(spectrum_db) if spectrogram_normalize is not None else spectrum_db

    return final_spectrum

def spectrogram_to_image(spectrum: np.ndarray, color_map_name='viridis') -> Image:
    """
    Converts a normalized spectrogram to a PIL Image.

    Args:
        final_spectrum (np.ndarray): The normalized spectrogram data.
        color_map_name (str): The name of the colormap to use.

    Returns:
        Image: A PIL Image object representing the spectrogram.
    """
    rgba_img: np.ndarray = plt.get_cmap(color_map_name)(spectrum)
    alpha = rgba_img[..., 3:]       # shape (H, W, 1)
    rgb_channels = rgba_img[..., :3]     # shape (H, W, 3), floats in [0,1]
    bg_color = np.ones_like(rgb_channels)    # white background (1,1,1)

    # composite
    color_img = rgb_channels * alpha + bg_color * (1 - alpha)

    rgb_image_uint8 = (color_img * 255).astype(np.uint8)
    pil_image = Image.fromarray(rgb_image_uint8, 'RGB')
    
    return pil_image

def calculate_yolo_shifted_bboxes(annotation_line: str, shift_hz: float, sampling_rate: float) -> list:
    """
    Calculates new YOLO-formatted bounding boxes after a frequency shift, handling wrap-around.

    Args:
        annotation_line (str): A string containing one YOLO annotation "class_id x y w h".
        shift_hz (float): The frequency shift in Hz.
        sampling_rate (float): The sampling rate of the signal.

    Returns:
        list: A list of new annotation strings. Returns two strings if the box wraps around.
    """
    class_id, x_center, y_center, width, height = map(float, annotation_line.strip().split())
    
    # Calculate the shift in normalized coordinates
    # A positive shift moves to a higher frequency, which is DOWN in the image (increasing y).
    y_shift_norm = shift_hz / sampling_rate
    
    # Calculate the original top edge of the bounding box.
    y_min_orig = y_center - height / 2
    
    # Calculate the new top edge, using modulo to handle shifts > sampling_rate.
    # The result of the modulo will always be in the range [0, 1).
    y_min_new = (y_min_orig + y_shift_norm) % 1.0
    y_max_new = y_min_new + height
    
    new_annotations = []

    # Case 1: The shifted box is fully within the image (no wrap-around)
    if y_min_new >= 0 and y_max_new <= 1.0:
        new_y_center = y_min_new + height / 2
        new_annotations.append(f"{int(class_id)} {x_center} {new_y_center} {width} {height}")
        return new_annotations

    # Case 2: The box wraps around the bottom edge (appears at the top)
    if y_max_new > 1.0:
        # Part 1: The portion at the bottom of the image
        h1 = 1.0 - y_min_new
        if h1 > 1e-6: # Check for non-zero height
            y1_center = y_min_new + h1 / 2
            new_annotations.append(f"{int(class_id)} {x_center} {y1_center} {width} {h1}")
        
        # Part 2: The portion that wraps to the top
        h2 = y_max_new - 1.0
        if h2 > 1e-6:
            y2_center = h2 / 2
            new_annotations.append(f"{int(class_id)} {x_center} {y2_center} {width} {h2}")

    # Case 3: The box wraps around the top edge (appears at the bottom)
    elif y_min_new < 0:
        # Part 1: The portion at the top of the image
        h1 = y_max_new
        if h1 > 1e-6:
            y1_center = h1 / 2
            new_annotations.append(f"{int(class_id)} {x_center} {y1_center} {width} {h1}")
            
        # Part 2: The portion that wraps to the bottom
        h2 = -y_min_new
        if h2 > 1e-6:
            y2_center = 1.0 - h2 / 2
            new_annotations.append(f"{int(class_id)} {x_center} {y2_center} {width} {h2}")
            
    return new_annotations

def find_dataset_structure(base_dir: str):
    images = glob(os.path.join(base_dir, '**', 'images', '**', '*.png'), recursive=True)
    struct = images[0].replace(base_dir, "").split(os.path.sep)[:2]
    assert 'images' in struct, "Could not find 'images' in the dataset structure."
    assert len(struct) <= 2, "Expected at most two levels in the dataset structure after base_dir."
    images_index = struct.index('images')
    if images_index != -1:
        struct[images_index] = '<**>'
    struct[(images_index+1)%2] = '<*>'

    return os.path.sep.join(struct)

def find_split_name(sample: dict, base_dir: str, structure_template: str) -> str:
    # figure out if the sample is in the train or val or test
    sample_index, _ = os.path.splitext(sample)
    for split in ['train', 'val', 'test', '']:
        annotate_path = os.path.join(base_dir,structure_template.replace('<*>', split).replace('<**>', 'labels'), f"{sample_index}.txt")
        split_name = split
        if os.path.exists(annotate_path):
            break

    return split_name

def get_info(sample_info, base_dir: str = '../../dataset/yolo_detection_dataset/', structure_template: str = '', split: str = '') -> dict:
    model_name, sample_index = sample_info['img_short_path'].split(os.path.sep)
    sample_index, _ = os.path.splitext(sample_index)
    # split = find_split_name(sample_info, base_dir, structure_template)
    annotate_path = os.path.join(base_dir,structure_template.replace('<*>', split).replace('<**>', 'labels'), f"{model_name}{os.sep}{sample_index}.txt")
    with open(annotate_path, 'r') as f:
        annotations = f.readlines()

    return {"model": model_name, "sample": sample_index, "annotation_path": annotate_path, "annotations": annotations}

def augment_spectrogram(sample_1, sample_2: dict,
    spectrogram_normalize, signal_normalize, shift_amount_hz_1=None, shift_amount_hz_2=None, signal_weight=0.5,
    color_map_name='viridis', center_dc: bool = True, snr_db: float | None = None,
    fading_type: str | None = None, # 'rayleigh' or 'rician'
    rician_K_db: float | None = None) -> Image:
    """
    Generates a spectrogram image from signal data, allowing for slicing
    a specific frequency band.

    Args:
        meta_data (dict): Dictionary containing signal metadata.
    """

    corrected_sample_1 = load_and_truncate_signal(sample_1)

    # --- FREQUENCY SHIFT ---
    if shift_amount_hz_1 is not None:
        # --- APPLY FREQUENCY SHIFT ---
        # Example: Shift the first signal up by 1 MHz
        corrected_sample_1 = frequency_shift_signal(
            corrected_sample_1,
            shift_hz=shift_amount_hz_1,
            sampling_rate=sample_1['sampling_rate']
        )
    # --- END SHIFT ---

    corrected_sample_1 = signal_normalize(corrected_sample_1)

    if sample_2 is not None:
        corrected_sample_2 = load_and_truncate_signal(sample_2)
        corrected_sample_2 = signal_normalize(corrected_sample_2)
        # --- FREQUENCY SHIFT ---
        if shift_amount_hz_2 is not None:
            # --- APPLY FREQUENCY SHIFT ---
            # Example: Shift the first signal up by 1 MHz
            corrected_sample_2 = frequency_shift_signal(
                corrected_sample_2,
                shift_hz=shift_amount_hz_2,
                sampling_rate=sample_2['sampling_rate']
            )
        # --- END SHIFT ---

    merge_sample =  corrected_sample_1 + signal_weight * corrected_sample_2 if sample_2 is not None else corrected_sample_1
    
    # --- ADD FADING AND NOISE ---
    if fading_type:
        if fading_type == 'rayleigh':
            merge_sample = apply_rayleigh_fading(merge_sample)
        elif fading_type == 'rician' and rician_K_db is not None:
            merge_sample = apply_rician_fading(merge_sample, rician_K_db)
            
    if snr_db is not None:
        merge_sample = add_awgn(merge_sample, snr_db)
    # --- END NOISE ---
    
    merge_sample = signal_normalize(merge_sample)

    final_spectrum: np.ndarray = get_spectrogram(sample= merge_sample, sampling_rate=sample_1['sampling_rate'], fft_size=sample_1['fft_size'], spectrogram_normalize=spectrogram_normalize, center_dc=center_dc)

    pil_image = spectrogram_to_image(final_spectrum, color_map_name=color_map_name)

    return pil_image

def add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Adds Additive White Gaussian Noise (AWGN) to a complex signal.

    Args:
        signal (np.ndarray): The input complex signal.
        snr_db (float): The desired Signal-to-Noise Ratio in decibels (dB).

    Returns:
        np.ndarray: The signal with added AWGN.
    """
    # Calculate signal power
    signal_power = np.mean(np.abs(signal)**2)
    
    # Convert SNR from dB to linear
    snr_linear = 10**(snr_db / 10.0)
    
    # Calculate noise power
    noise_power = signal_power / snr_linear
    
    # Generate complex Gaussian noise
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    
    return signal + noise

def apply_rayleigh_fading(signal: np.ndarray) -> np.ndarray:
    """
    Applies flat Rayleigh fading to a signal.

    Args:
        signal (np.ndarray): The input complex signal.

    Returns:
        np.ndarray: The faded signal.
    """
    # Generate a single complex fading coefficient for flat fading
    h = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
    return signal * h

def apply_rician_fading(signal: np.ndarray, K_db: float) -> np.ndarray:
    """
    Applies flat Rician fading to a signal.

    Args:
        signal (np.ndarray): The input complex signal.
        K_db (float): The Rician K-factor in decibels (dB).

    Returns:
        np.ndarray: The faded signal.
    """
    K_linear = 10**(K_db / 10.0)
    
    # Line-of-sight (LOS) component (deterministic, assumed phase 0)
    los_component = np.sqrt(K_linear / (K_linear + 1))
    
    # Scattered component (Rayleigh)
    scatter_component = np.sqrt(1 / (K_linear + 1)) * (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
    
    h = los_component + scatter_component
    return signal * h

def plot_annotations_on_ax(ax, annotations: list[str], img_width: int, img_height: int, class_names: dict, color: str = 'lime'):
    """
    Draws bounding boxes for a given set of annotations on a matplotlib axes object.

    Args:
        ax: The matplotlib axes to draw on.
        annotations (list[str]): A list of YOLO-formatted annotation strings.
        img_width (int): The width of the image.
        img_height (int): The height of the image.
        class_names (dict): A dictionary mapping class IDs to class names.
        color (str): The color for the bounding boxes and text.
    """
    for line in annotations:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        class_name = class_names.get(int(class_id), "Unknown")

        box_width_px = width * img_width
        box_height_px = height * img_height
        x_min_px = (x_center * img_width) - (box_width_px / 2)
        y_min_px = (y_center * img_height) - (box_height_px / 2)

        rect = patches.Rectangle(
            (x_min_px, y_min_px), box_width_px, box_height_px,
            linewidth=6, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(x_min_px, y_min_px - 5, class_name, color=color, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))
# %%
def shift_annotations(sample, info, shift, min_box_height_ratio: float = 0.10) -> list[str]:
    final_annotations = []
    for line in info['annotations']:
        clean_line = line.strip()
        if not clean_line:
            continue
        _, _, _, _, original_height = map(float, clean_line.split())
        if shift is None or shift == 0:
            final_annotations.append(f"{clean_line}\n")
        else:
            new_boxes = calculate_yolo_shifted_bboxes(clean_line, shift, sample['sampling_rate'])
            for box in new_boxes:
                *_, width, height = map(float, box.split())
                if height >= (original_height * min_box_height_ratio) and width > 0:
                    final_annotations.append(f"{box}\n")
    return final_annotations
# %%
def calculate_class_distribution(txt_files: list[str]) -> tuple[dict[int, int], dict[str, set[int]]]:
    """
    Analyzes annotation files to determine class distribution and image-to-class mapping.

    Args:
        txt_files (list[str]): A list of paths to YOLO annotation .txt files.

    Returns:
        tuple[dict[int, int], dict[str, set[int]]]:
            - A dictionary mapping class IDs to their instance counts.
            - A dictionary mapping annotation file paths to a set of class IDs they contain.
    """
    class_counts = defaultdict(int)
    image_to_classes = {}

    for txt_file in txt_files:
        classes_in_file = set()
        try:
            with open(txt_file, 'r') as f:
                for line in f:
                    try:
                        class_id = int(line.strip().split()[0])
                        class_counts[class_id] += 1
                        classes_in_file.add(class_id)
                    except (ValueError, IndexError):
                        logger.warning(f"Could not parse line in {txt_file}: {line.strip()}")
            image_to_classes[txt_file] = classes_in_file
        except FileNotFoundError:
            logger.warning(f"Annotation file not found: {txt_file}")
            
    return dict(class_counts), image_to_classes


def calculate_augmentation_schedule(
    txt_files: list[str],
    base_augmentations: int = 1,
    max_augmentations: int = 20
) -> dict[str, int]:
    """
    Calculates the number of augmentations for each image to balance the class distribution.
    Images containing rarer classes will be scheduled for more augmentations.

    Args:
        txt_files (list[str]): A list of paths to YOLO annotation .txt files.
        base_augmentations (int): The minimum number of augmentations for any image.
        max_augmentations (int): The maximum number of augmentations for an image containing the rarest class.

    Returns:
        dict[str, int]: A dictionary mapping annotation file path to the recommended number of augmentations.
    """
    class_counts, image_to_classes = calculate_class_distribution(txt_files)

    if not class_counts:
        logger.warning("No classes found in annotation files. Returning empty schedule.")
        return {}

    max_freq = max(class_counts.values())
    # Add 1 to avoid division by zero for classes not in counts, though defaultdict handles this.
    class_weights = {cid: max_freq / (count + 1e-6) for cid, count in class_counts.items()}

    image_scores = {}
    for img_path, classes in image_to_classes.items():
        if not classes:
            image_scores[img_path] = 0
            continue
        # Score is based on the rarest class in the image
        max_weight = max(class_weights.get(cid, 0) for cid in classes)
        image_scores[img_path] = max_weight

    min_score = min(image_scores.values())
    max_score = max(image_scores.values())

    augmentation_schedule = {}
    # Handle case where all scores are the same (e.g., single class dataset)
    if max_score - min_score < 1e-6:
        for img_path in image_to_classes:
            augmentation_schedule[img_path] = base_augmentations
        return augmentation_schedule

    for img_path, score in image_scores.items():
        # Normalize score to [0, 1]
        normalized_score = (score - min_score) / (max_score - min_score)
        num_augs = base_augmentations + round(normalized_score * (max_augmentations - base_augmentations))
        augmentation_schedule[img_path] = int(num_augs)

    return augmentation_schedule

def create_augment_dataset(
    original_dataset_dir: str,
    output_dir: str,
    annotation_samples: list[dict],
    cleaned_annotation_samples: list[dict] | None = None,
    num_augmentations_per_image: int = 1,
    augmentation_schedule: dict[str, int] | None = None,
    max_abs_shift_hz: float = 8e6,
    center_dc: bool = True,
    min_box_height_ratio: float = 0.30,
    min_signal_weight: float = 0.1,
    max_signal_weight: float = 0.9,
    add_second_sample_prob: float = 0.6,
    background_sample_prob: float = 0.3,
    frequency_shift_prob: float = 0.9,
    unshiftable_classes: list[str] = ['Autel_EXOII'],
    channel_effects_prob: float = 0.6,
    max_channel_effects: int = 2,
    snr_db_range: tuple[float, float] = (5, 20),
    rician_prob: float = 0.5,
    rician_K_db_range: tuple[float, float] = (1, 10),
    other_normalization_prob: float = 0.2,
    seed: int | None = 42,
):
    """
    Creates a new YOLO dataset by copying an original dataset and adding augmented samples.

    Args:
        original_dataset_dir (str): Path to the source YOLO dataset.
        output_dir (str): Path to create the new augmented dataset.
        annotation_samples (list[dict]): List of metadata for samples that can be augmented.
        cleaned_annotation_samples (list[dict] | None): List of metadata from cleaned (unbalanced) dataset to use for sample_2 selection. If None, uses annotation_samples.
        num_augmentations_per_image (int): How many augmented versions to create for each original image. This is ignored if `augmentation_schedule` is provided.
        augmentation_schedule (dict[str, int] | None): A dictionary mapping annotation file paths to the number of augmentations to create. Overrides `num_augmentations_per_image`.
        max_abs_shift_hz (float): The maximum absolute frequency shift to apply.
        center_dc (bool): Whether to center the DC component in the spectrogram.
        min_box_height_ratio (float): The minimum height for a split bounding box, as a ratio of the original box's height, to be kept after augmentation.
        min_signal_weight (float): The minimum weight for the second signal when mixing.
        max_signal_weight (float): The maximum weight for the second signal when mixing.
        add_second_sample_prob (float): Probability of adding a second signal to an augmentation.
        background_sample_prob (float): Probability that sample_2 will be from the background class (when adding a second sample). Ensures controlled background representation.
        frequency_shift_prob (float): Probability of applying a frequency shift.
        unshiftable_classes (list[str]): List of class names that should not be frequency shifted.
        channel_effects_prob (float): Probability of applying any channel effects (noise/fading).
        max_channel_effects (int): Maximum number of channel effects to apply at once (e.g., 2 means AWGN and fading can be combined).
        snr_db_range (tuple[float, float]): Range of SNR values (in dB) for AWGN.
        rician_prob (float): Probability of choosing Rician fading over Rayleigh when fading is applied.
        rician_K_db_range (tuple[float, float]): Range of Rician K-factor values (in dB).
        seed (int | None): Optional random seed for reproducibility.
    """
    if seed is not None:
        logger.info(f"Using random seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)

    logger.info(f"Creating augmented dataset at: {output_dir}")
    
    # 1. Clean and create directory structure, then copy original dataset
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    logger.info("Copying original dataset...")
    errors = parallel_copy_directory(original_dataset_dir, output_dir, max_workers=8)
    if errors:
        logger.error(f"Failed to copy {len(errors)} files")
        return
    logger.success("Copy complete.")
    
    # Fix data.yaml to point to the augmented dataset path
    yaml_path = os.path.join(output_dir, 'data.yaml')
    if os.path.exists(yaml_path):
        logger.info("Updating data.yaml path to point to augmented dataset...")
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Update path to point to augmented directory
        yaml_data['path'] = output_dir
        
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False)
        
        logger.success(f"Updated data.yaml path to: {output_dir}")

    structure_template: str = find_dataset_structure(output_dir)
    
    # 2. Organize samples by class for balanced sample_2 selection, SEPARATELY FOR EACH SPLIT
    # Use cleaned_annotation_samples if provided, otherwise use annotation_samples
    if cleaned_annotation_samples:
        logger.info(f"Using cleaned_annotation_samples for sample_2 selection ({len(cleaned_annotation_samples)} total samples)")
        # Convert flat list to split-organized dict (simple approach: all go to same splits as annotation_samples)
        # This is a simplified approach - just organize by the splits we already have
        samples_for_selection = {'train': [], 'val': [], 'test': []}
        for sample in cleaned_annotation_samples:
            # Use the img_short_path to figure out which split
            split_name = find_split_name(sample['img_short_path'], output_dir, structure_template)
            if split_name in samples_for_selection:
                samples_for_selection[split_name].append(sample)
    else:
        logger.info("Using annotation_samples (balanced dataset) for sample_2 selection")
        samples_for_selection = annotation_samples
    
    # Organize samples by class for EACH split separately
    logger.info("Organizing samples by class for balanced selection...")
    samples_by_class_by_split = {}
    background_samples_by_split = {}
    non_background_classes_by_split = {}
    all_samples_by_split = {}
    
    for split in ['train', 'val', 'test']:
        samples_by_class = defaultdict(list)
        for sample in samples_for_selection.get(split, []):
            # Get class from the sample's img_short_path (format: "class_name/sample_index.ext")
            class_name = sample['img_short_path'].split(os.path.sep)[0]
            samples_by_class[class_name].append(sample)
        
        # Separate background samples
        background_samples = samples_by_class.get('background', [])
        non_background_classes = [cls for cls in samples_by_class.keys() if cls != 'background']
        
        samples_by_class_by_split[split] = samples_by_class
        background_samples_by_split[split] = background_samples
        non_background_classes_by_split[split] = non_background_classes
        all_samples_by_split[split] = samples_for_selection.get(split, [])
        
        logger.info(f"  {split.capitalize()} split: {len(non_background_classes)} non-background classes, {len(background_samples)} background samples")
        for class_name in sorted(samples_by_class.keys()):
            samples = samples_by_class[class_name]
            logger.debug(f"    {class_name}: {len(samples)} samples")

    # Define a list of visually distinct color maps for augmentation
    color_maps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'gray', 'hot']

    shift_range_hz: list[float | int] = np.linspace(-max_abs_shift_hz, max_abs_shift_hz, int(1e6)).tolist()

    # 2. Generate and save augmented samples
    if augmentation_schedule:
        total_augs = sum(augmentation_schedule.values())
        logger.info(f"Generating {total_augs} augmented samples based on the provided schedule...")
    else:
        total_augs = num_augmentations_per_image * sum(len(v) for v in annotation_samples.values())
        logger.info(f"Generating {total_augs} augmented samples...")

    # --- DEBUG: Print some keys from the schedule to see their format
    schedule_keys = list(augmentation_schedule.keys())
    logger.debug(f"DEBUG: Sample keys from augmentation_schedule: \n{schedule_keys[:3]}")
    
    for split in annotation_samples.keys():
        # Get the split-specific sample pools for balanced selection
        samples_by_class = samples_by_class_by_split[split]
        background_samples = background_samples_by_split[split]
        non_background_classes = non_background_classes_by_split[split]
        all_samples = all_samples_by_split[split]
        
        for sample_1 in tqdm(annotation_samples[split], desc=f"Augmenting {split.capitalize()} Samples"):
            
            sample_1_info = get_info(sample_1, base_dir=output_dir, structure_template=structure_template, split=split)
            
            if augmentation_schedule:
                # The schedule is based on the original dataset dir, but the info path is in the output dir.
                # We need to create the correct key for the lookup by replacing the output path with the original.
                lookup_path = sample_1_info['annotation_path'].replace(output_dir, original_dataset_dir)
                num_augmentations = augmentation_schedule.get(lookup_path, 0)
            else:
                num_augmentations = num_augmentations_per_image

            for i in range(num_augmentations):
                # --- Get random augmentation parameters ---
                # Balanced sampling for sample_2 FROM THE SAME SPLIT
                sample_2 = None
                if random.random() < add_second_sample_prob:
                    # Decide if we want background or non-background
                    if random.random() < background_sample_prob and background_samples:
                        # Select from background (same split)
                        sample_2 = random.choice([s for s in background_samples if s != sample_1])
                    elif non_background_classes:
                        # Select a random class, then a random sample from that class (same split)
                        # This ensures each class has equal probability
                        selected_class = random.choice(non_background_classes)
                        class_samples = samples_by_class[selected_class]
                        sample_2 = random.choice([s for s in class_samples if s != sample_1])

                sample_2_info = None
                if sample_2 is not None:
                    sample_2_info = get_info(sample_2, base_dir=output_dir, structure_template=structure_template, split=split)
                    
                shift_amount_hz_1 = random.choice(shift_range_hz) if random.random() < frequency_shift_prob else None
                shift_amount_hz_2 = random.choice(shift_range_hz) if random.random() < frequency_shift_prob else None
                # Don't shift unshiftable classes
                if sample_1_info['model'] in unshiftable_classes:
                    shift_amount_hz_1 = None
                if (sample_2_info is not None) and sample_2_info['model'] in unshiftable_classes:
                    shift_amount_hz_2 = None

                signal_weight = round(random.uniform(min_signal_weight, max_signal_weight),2)
                color_map_name = random.choice(color_maps)

                # --- Noise parameters ---
                snr_db = None
                fading_type = None
                rician_K_db = None
                effects_to_apply = []
                
                if random.random() < channel_effects_prob:
                    available_effects = ['awgn', 'fading']
                    # Ensure max_channel_effects is not greater than the number of available effects
                    max_effects = min(max_channel_effects, len(available_effects))
                    if max_effects > 0:
                        num_effects = random.randint(1, max_effects)
                        effects_to_apply = random.sample(available_effects, num_effects)

                        # Unify SNR for this augmentation if any relevant effect is chosen
                        if 'awgn' in effects_to_apply or 'fading' in effects_to_apply:
                            snr_db = round(random.uniform(*snr_db_range), 2)
                        
                        if 'fading' in effects_to_apply:
                            if random.random() < rician_prob:
                                fading_type = 'rician'
                                rician_K_db = round(random.uniform(*rician_K_db_range), 2)
                            else:
                                fading_type = 'rayleigh'
                
                spectrogram_normalize=minmax_normalize_spectrogram
                signal_normalize=power_normalize

                if random.random() < other_normalization_prob:
                    spectrogram_normalize = random.choice([frequency_normalize_spectrogram, time_normalize_spectrogram])
                    
                if random.random() < other_normalization_prob:
                    signal_normalize = random.choice([z_normalize_complex, normalize_complex_signal])


                # --- Generate augmented image ---
                augmented_image = augment_spectrogram(
                    sample_1, sample_2,
                    spectrogram_normalize=spectrogram_normalize,
                    signal_normalize=signal_normalize,
                    shift_amount_hz_1=shift_amount_hz_1,
                    shift_amount_hz_2=shift_amount_hz_2,
                    signal_weight=signal_weight,
                    color_map_name=color_map_name,
                    center_dc=center_dc,
                    snr_db=snr_db,
                    fading_type=fading_type,
                    rician_K_db=rician_K_db
                )

                # --- Calculate new annotations ---
                final_annotations = []

                # --- Get annotations for both source samples ---
                
                # Calculate and add (potentially shifted) annotations from sample_1
                final_annotations.extend(shift_annotations(sample_1, sample_1_info, shift_amount_hz_1, min_box_height_ratio))
                
                # Calculate and add (potentially shifted) annotations from sample_2
                if sample_2 is not None:
                    final_annotations.extend(shift_annotations(sample_2, sample_2_info, shift_amount_hz_2, min_box_height_ratio))

                # --- Save the new image and label file ---
                try:
                    model_name = sample_1_info['model']

                    # Build the filename stem safely
                    name_parts = [f"{sample_1_info['sample']}_aug_{i}"]
                    if sample_2_info:
                        name_parts.append(sample_2_info['sample'])
                    if effects_to_apply:
                        name_parts.append('_'.join(effects_to_apply))
                    if snr_db is not None:
                        name_parts.append(f"snr{snr_db}")
                    if rician_K_db is not None:
                        name_parts.append(f"K{rician_K_db}")
                    new_filename_stem = '_'.join(name_parts)

                    save_dir = os.path.join(output_dir, structure_template.replace('<*>', split), model_name)
                    
                    # Save image
                    image_save_dir = save_dir.replace('<**>', 'images')
                    os.makedirs(image_save_dir, exist_ok=True)
                    augmented_image.save(os.path.join(image_save_dir, f"{new_filename_stem}.png"))

                    # Save label
                    label_save_dir = save_dir.replace('<**>', 'labels')
                    os.makedirs(label_save_dir, exist_ok=True)
                    with open(os.path.join(label_save_dir, f"{new_filename_stem}.txt"), 'w') as f:
                        f.writelines(final_annotations)
                except Exception:
                    logger.error(f"!!! FAILED to save augmented sample for {sample_1['spect_path']} !!!")
                    import traceback
                    logger.error(traceback.format_exc())

    logger.success("\nAugmentation complete.")
    logger.info(f"New dataset saved to: {output_dir}")


# %%
if __name__ == "__main__":
    # # base_dir = '../../data/Related work from Rowan/new_dataset/original/'
    # # yolo_dataset_dir: str = '../../dataset/yolo_detection_dataset/'
    # # augmented_dataset_dir: str = '../../dataset/yolo_augmented_dataset/'
    # old_meta_file = '/mnt/d/Rowan/AeroDefence/dataset/original_abspath_meta_data.json'

    # # class_names = {
    # # 0: '2', 1: 'Anafi', 2: 'EXOII', 3: 'F11GIM', 4: 'FPV', 5: 'HS110G',
    # # 6: 'HS720E', 7: 'Inspire1', 8: 'KY601S', 9: 'Mavic2Pro', 10: 'Mavic3',
    # # 11: 'MavicAir', 12: 'MavicMini4', 13: 'MavicPro', 14: 'Mini3', 15: 'Phantom4',
    # # 16: 'PhantomAdv3', 17: 'Tello', 18: 'X4', 19: 'Xstar'
    # # }

    # base_dir = '/mnt/d/Rowan/AeroDefence/dataset/Roboflow-2025-08-21/cleaned/'
    # yolo_dataset_dir: str = '/mnt/d/Rowan/AeroDefence/dataset/Roboflow-2025-08-21/cleaned/'
    # augmented_dataset_dir: str = '/mnt/d/Rowan/AeroDefence/dataset/Roboflow-2025-08-21/augmented/'
    # new_meta_file = '/mnt/d/Rowan/AeroDefence/dataset/unannotated_cage_dataset/images/meta_data.json'

    parser = argparse.ArgumentParser(description="Create an augmented dataset for YOLO training.")
    
    # --- File Paths ---
    parser.add_argument('--base_dir', type=str, default='/mnt/d/Rowan/AeroDefence/dataset/Roboflow-2025-10-12/balanced/', help='Base directory of the balanced, original dataset.')
    parser.add_argument('--base_signals_dir', type=str, default='/mnt/d/Rowan/AeroDefence/dataset/', help='Base directory of the balanced, original dataset.')
    parser.add_argument('--yolo_dataset_dir', type=str, default='/mnt/d/Rowan/AeroDefence/dataset/Roboflow-2025-10-12/balanced/', help='Source YOLO dataset directory to be augmented.')
    parser.add_argument('--augmented_dataset_dir', type=str, default='/mnt/d/Rowan/AeroDefence/dataset/Roboflow-2025-10-12/augmented/', help='Output directory for the augmented dataset.')
    parser.add_argument('--cleaned_dataset_dir', type=str, default=None, help='Path to cleaned (unbalanced) dataset for sample_2 selection. If not provided, uses yolo_dataset_dir.')
    parser.add_argument('--meta_files', type=str, nargs='+', 
                        default=['/mnt/d/Rowan/AeroDefence/dataset/metadata/original_abspath_meta_data.json',
                                 '/mnt/d/Rowan/AeroDefence/dataset/unannotated_cage_dataset/images/meta_data.json'], 
                        help='Paths to metadata JSON files to merge (supports multiple files).')
    parser.add_argument('--temp_meta_file', type=str, default='temp_meta.json', help='Path for the temporary merged metadata file.')

    # --- Augmentation Schedule ---
    parser.add_argument('--base_augmentations', type=int, default=10, help='Base number of augmentations for the most common classes.')
    parser.add_argument('--max_augmentations', type=int, default=50, help='Max number of augmentations for the rarest classes.')

    # --- Augmentation Parameters ---
    parser.add_argument('--num_augmentations_per_image', type=int, default=10, help='Number of augmentations per image (ignored if schedule is used).')
    parser.add_argument('--max_abs_shift_hz', type=float, default=20e6, help='Maximum absolute frequency shift in Hz.')
    parser.add_argument('--center_dc', type=bool, default=True, help='Whether to center the DC component.')
    parser.add_argument('--min_box_height_ratio', type=float, default=0.50, help='Minimum height ratio for a split bounding box to be kept.')
    parser.add_argument('--min_signal_weight', type=float, default=0.3, help='Minimum weight for the second signal when mixing.')
    parser.add_argument('--max_signal_weight', type=float, default=0.9, help='Maximum weight for the second signal when mixing.')
    parser.add_argument('--add_second_sample_prob', type=float, default=0.7, help='Probability of adding a second signal.')
    parser.add_argument('--background_sample_prob', type=float, default=0.3, help='Probability that sample_2 is from background class (when adding second sample).')
    parser.add_argument('--frequency_shift_prob', type=float, default=0.9, help='Probability of applying a frequency shift.')
    parser.add_argument('--channel_effects_prob', type=float, default=0.7, help='Probability of applying channel effects (noise/fading).')
    parser.add_argument('--max_channel_effects', type=int, default=1, help='Maximum number of channel effects to apply at once.')
    parser.add_argument('--snr_db_min', type=float, default=-15, help='Minimum SNR in dB for AWGN.')
    parser.add_argument('--snr_db_max', type=float, default=35, help='Maximum SNR in dB for AWGN.')
    parser.add_argument('--rician_prob', type=float, default=0.6, help='Probability of choosing Rician fading over Rayleigh.')
    parser.add_argument('--rician_K_db_min', type=float, default=-15, help='Minimum Rician K-factor in dB.')
    parser.add_argument('--rician_K_db_max', type=float, default=35, help='Maximum Rician K-factor in dB.')
    parser.add_argument('--other_normalization_prob', type=float, default=0.2, help='Probability of applying other normalization techniques.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--visualize', action='store_true', help='If set, shows a visualization of a random augmented sample at the end.')

    args = parser.parse_args()

    # Load and merge multiple metadata files
    meta_data = []
    for meta_file in args.meta_files:
        logger.info(f"Loading metadata from: {meta_file}")
        meta_data.extend(load_json(meta_file))
    
    logger.info(f"Merged {len(meta_data)} total metadata entries from {len(args.meta_files)} file(s)")
    save_json(meta_data, args.temp_meta_file)

    meta_file = args.temp_meta_file

    # getting class names from data.yaml
    with open(os.path.join(args.yolo_dataset_dir, 'data.yaml'), 'r') as f:
        data_yaml = yaml.safe_load(f)
        class_names = data_yaml.get('names', {})
        # If class_names is a list, convert it to a dict with indices as keys
        if isinstance(class_names, list):
            class_names = {i: name for i, name in enumerate(class_names)}

    logger.info(f"Loaded {len(class_names)} classes from data.yaml")
    logger.info(f"\nLoaded class names:\n{class_names}")


    img_files, txt_files, classes = get_annotated_data(base_dir=args.base_dir)
    logger.info(f"Found {len(img_files)} images, {len(txt_files)} labels, for {len(classes)} classes.")
    annotation_samples = get_dat_source_files(img_files=img_files, meta_file=meta_file, base_dir=args.base_dir, base_signals_dir=args.base_signals_dir)

    # Load cleaned dataset samples if provided
    cleaned_annotation_samples_list = None
    if args.cleaned_dataset_dir:
        logger.info(f"Loading cleaned dataset from: {args.cleaned_dataset_dir}")
        cleaned_img_files, cleaned_txt_files, cleaned_classes = get_annotated_data(base_dir=args.cleaned_dataset_dir)
        logger.info(f"Found {len(cleaned_img_files)} cleaned images, {len(cleaned_txt_files)} labels, for {len(cleaned_classes)} classes.")
        cleaned_annotation_samples = get_dat_source_files(
            img_files=cleaned_img_files, 
            meta_file=meta_file, 
            base_dir=args.cleaned_dataset_dir, 
            base_signals_dir=args.base_signals_dir
        )
        # Flatten the dict of splits into a single list
        cleaned_annotation_samples_list = []
        for split_samples in cleaned_annotation_samples.values():
            cleaned_annotation_samples_list.extend(split_samples)
        logger.info(f"Total cleaned samples available for sample_2 selection: {len(cleaned_annotation_samples_list)}")

    # --- Run the dataset augmentation process ---
    augmentation_schedule = calculate_augmentation_schedule(
        txt_files,
        base_augmentations=args.base_augmentations,
        max_augmentations=args.max_augmentations
    )
    
    create_augment_dataset(
        original_dataset_dir=args.yolo_dataset_dir,
        output_dir=args.augmented_dataset_dir,
        annotation_samples=annotation_samples,
        cleaned_annotation_samples=cleaned_annotation_samples_list,
        num_augmentations_per_image=args.num_augmentations_per_image, # is ignored when schedule is provided
        augmentation_schedule=augmentation_schedule,
        max_abs_shift_hz=args.max_abs_shift_hz,
        center_dc=args.center_dc,
        min_box_height_ratio=args.min_box_height_ratio,
        min_signal_weight=args.min_signal_weight,
        max_signal_weight=args.max_signal_weight,
        add_second_sample_prob=args.add_second_sample_prob,
        background_sample_prob=args.background_sample_prob,
        frequency_shift_prob=args.frequency_shift_prob,
        channel_effects_prob=args.channel_effects_prob,
        max_channel_effects=args.max_channel_effects,
        snr_db_range=(args.snr_db_min, args.snr_db_max),
        rician_prob=args.rician_prob,
        rician_K_db_range=(args.rician_K_db_min, args.rician_K_db_max),
        seed=args.seed,
    )

    # --- Example visualization of one augmented sample (optional) ---
    if args.visualize:
        print("\n--- Visualizing a random augmented sample ---")
        
        # Flatten the dictionary of samples into a single list for visualization
        all_samples = [sample for split_samples in annotation_samples.values() for sample in split_samples]
        
        if all_samples:
            sample_1 = random.choice(all_samples)
            
            # Find which split the chosen sample belongs to
            sample_1_split = None
            for split, samples in annotation_samples.items():
                if sample_1 in samples:
                    sample_1_split = split
                    break

            # Ensure sample_2 is different and exists
            possible_sample_2 = [s for s in all_samples if s != sample_1]
            sample_2 = random.choice(possible_sample_2) if possible_sample_2 else None
            
            sample_2_split = None
            if sample_2:
                for split, samples in annotation_samples.items():
                    if sample_2 in samples:
                        sample_2_split = split
                        break
            
            shift_amount_hz = random.uniform(-8e6, 8e6)

            spectrogram_image = augment_spectrogram(
                sample_1, sample_2,
                spectrogram_normalize=minmax_normalize_spectrogram,
                signal_normalize=power_normalize,
                shift_amount_hz_1=shift_amount_hz,
                shift_amount_hz_2=None,  # No shift for sample_2
                signal_weight=0.6,
                color_map_name='viridis',
                center_dc=True,
                snr_db=random.uniform(-15, 35),
                fading_type='rayleigh'
            )
            
            structure_template=find_dataset_structure(args.base_dir)
            sample_1_info = get_info(sample_1, base_dir=args.base_dir, structure_template=structure_template, split=sample_1_split)
            
            all_shifted_annotations = []
            for annotation_line in sample_1_info['annotations']:
                all_shifted_annotations.extend(
                    calculate_yolo_shifted_bboxes(annotation_line, shift_amount_hz, sample_1['sampling_rate'])
                )
            
            fig, ax = plt.subplots(1, 1, figsize=(18, 12))
            ax.imshow(spectrogram_image)
            ax.set_title(f"Example Augmented Spectrogram (Shift: {shift_amount_hz:.2f} Hz)")
            ax.axis('off')
            img_width, img_height = spectrogram_image.size
            
            if sample_2 and sample_2_split:
                sample_2_info = get_info(sample_2, base_dir=args.base_dir, structure_template=structure_template, split=sample_2_split)
                plot_annotations_on_ax(ax, sample_2_info['annotations'], img_width, img_height, class_names, color='cyan')
                
            plot_annotations_on_ax(ax, all_shifted_annotations, img_width, img_height, class_names, color='lime')
            plt.show()
        else:
            print("No samples available to visualize.")




# %%
