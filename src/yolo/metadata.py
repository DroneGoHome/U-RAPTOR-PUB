import os
import re
import csv
import json
import traceback
from glob import glob
from typing import Any, List

import numpy as np
from tqdm import tqdm
from loguru import logger

from src.data.dataset import save_to_json


def find_first_float_index(components: list[str]) -> int:
    """
    Finds the index of the first component in a list that can be converted to a float.

    Args:
        components (list[str]): List of string components.

    Returns:
        int: Index of the first float-convertible component, or -1 if none found.
    """
    for i, comp in enumerate(components):
        try:
            float(comp)
            return i
        except ValueError:
            continue
    return -1

def load_csv_as_json(csv_file_path: str) -> list[dict]:
    json_like_data = []
    
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        # DictReader automatically uses the first row as keys
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            json_like_data.append(row)
    
    return json_like_data

def return_as_if_possible(variable: Any, as_type: Any, multiply: float = 1.0) -> Any:
    try:
        return as_type(as_type(variable) * multiply)
    except (ValueError, TypeError):
        return variable

def get_info(file_path: str, from_path: bool=True) -> dict:
    """
    Extracts metadata from a .dat file.

    Args:
        file_path (str): Path to the .dat file.

    Returns:
        dict: A dictionary containing metadata information.
    """

    base_dir = os.path.dirname(file_path)
    
    csv_file_path = os.path.join(base_dir, 'parameter_summary.csv')
    csv_data = load_csv_as_json(csv_file_path) if os.path.exists(csv_file_path) else [{},]
    drone_2_info = csv_data[1] if len(csv_data) > 1 else {}

    # Extract components from the file name
    components = os.path.splitext(os.path.basename(file_path))[0].split('_')
    first_float_index = find_first_float_index(components)
    splitter_index = find_first_float_index(components[first_float_index + 5:]) + first_float_index + 5

    if from_path:
        basename_components = os.path.basename(base_dir).split('_')
        basename_first_float_index = find_first_float_index(basename_components)
        
        return {
                'device': "_".join(components[0:first_float_index]),
                'status': "_".join(components[first_float_index + 5: splitter_index]),
                'env': "_".join(basename_components[0:basename_first_float_index]),
                'sdr_gain': return_as_if_possible(basename_components[basename_first_float_index], int),
                'splitter': basename_components[basename_first_float_index + 3],
                'recording_duration': return_as_if_possible(basename_components[basename_first_float_index + 2], int),
                'distance': return_as_if_possible(components[first_float_index + 4], int),
                'altitude': return_as_if_possible(components[first_float_index + 3], int),
                'center_freq': return_as_if_possible(components[first_float_index], float),
                'drone_center_freq': return_as_if_possible(components[first_float_index+2], float),
                'bandwidth': return_as_if_possible(components[first_float_index + 1], int, multiply=1e6),
                'SNR': return_as_if_possible(components[splitter_index], float),
                'sampling_rate': return_as_if_possible(basename_components[basename_first_float_index + 1], int, multiply=1e6),
                'timestamp': "_".join(basename_components[basename_first_float_index + 4:]).replace(' ',':'),
                'original_file_path': file_path,
                # Additional fields for drone 2
                'device_2': drone_2_info.get('device', None),
                'center_freq_2':  return_as_if_possible(drone_2_info.get('center_freq', None), float),
                'drone2_center_freq': return_as_if_possible(drone_2_info.get('drone_c_freq', None), float),
                'bandwidth_2': return_as_if_possible(drone_2_info.get('bw', None), int, multiply=1e6),
                'SNR_2': return_as_if_possible(drone_2_info.get('snr', None), float),
                'sampling_rate_2': return_as_if_possible(drone_2_info.get('s', None), int, multiply=1e6),
            }
        
    else:
        return {
                'device': csv_data[0].get('device', None),
                'status': csv_data[0].get('status', None),
                'env': csv_data[0].get('env', None),
                'sdr_gain': return_as_if_possible(csv_data[0].get('sdr_gain', None), int),
                'splitter': csv_data[0].get('splitter', None),
                'recording_duration': return_as_if_possible(csv_data[0].get('duration_recording', None), int),
                'distance': return_as_if_possible(csv_data[0].get('distance', None), int),
                'altitude': return_as_if_possible(csv_data[0].get('altitude', None), int),
                'center_freq': return_as_if_possible(csv_data[0].get('center_freq', None), float),
                'drone_center_freq': return_as_if_possible(components[first_float_index+2], float),
                'bandwidth': return_as_if_possible(csv_data[0].get('bw', None), int, multiply=1e6),
                'SNR': return_as_if_possible(csv_data[0].get('snr', None), float),
                'sampling_rate': return_as_if_possible(csv_data[0].get('s', None), int, multiply=1e6),
                'timestamp': csv_data[0].get('timestamp', '').replace(' ',':'),
                'original_file_path': file_path,
                'center_freq_2':  return_as_if_possible(drone_2_info.get('center_freq', None), float),
                'drone2_center_freq': return_as_if_possible(drone_2_info.get('drone_c_freq', None), float),
                'bandwidth_2': return_as_if_possible(drone_2_info.get('bw', None), int, multiply=1e6),
                'SNR_2': return_as_if_possible(drone_2_info.get('snr', None), float),
                'sampling_rate_2': return_as_if_possible(drone_2_info.get('s', None), int, multiply=1e6),
            }

def create_master_metadata(
    dat_files_list: list[str],
    output_json_path: str,
    sample_time: float,
    step_time: float,
    fft_size: int, # Needed for length_corrector and truncation logic
    num_fft_spec: int # Needed for truncation logic
) -> list[dict[str, Any]]:
    """
    Generates a master metadata file by scanning .dat files for potential samples.

    Args:
        dat_files_list: List of paths to .dat files.
        output_json_path: Path to save the master JSON metadata file.
        sample_time: Duration of each sample in seconds.
        step_time: Time step between consecutive samples in seconds.
        sampling_rate: Sampling rate of the signals.
        fft_size: FFT size for processing parameters.
        num_fft_spec: Number of FFTs for processing parameters.


    Returns:
        A list of metadata dictionaries for all potential samples.
    """
    master_metadata = []

    print(f"Scanning {len(dat_files_list)} .dat files to generate master metadata...")
    for dat_file_path in tqdm(dat_files_list, desc="Generating Master Metadata"):
        try:
            # Use memmap to get signal size without loading the whole file
            mmap_signal = np.memmap(dat_file_path, dtype=np.complex64, mode='r')
            signal_total_points = mmap_signal.size
            del mmap_signal # Release memmap
            file_meta = get_info(file_path=dat_file_path, from_path=True)

            sampling_rate = max(file_meta.get('sampling_rate') or 0, file_meta.get('sampling_rate_2') or 0)
            sample_size_points = int(sample_time * sampling_rate)
            step_size_points = int(step_time * sampling_rate)

            if signal_total_points < sample_size_points:
                print(f"Warning: File {dat_file_path} is too short ({signal_total_points} points) for sample size ({sample_size_points} points). Skipping.")
                continue

            for i in range(0, signal_total_points - sample_size_points + 1, step_size_points):
                meta_entry = file_meta.copy()  # Copy the file metadata to each entry
                start_index = i
                end_index = i + sample_size_points

                # Basic length correction and truncation check (conceptual)
                # This part ensures the segment *could* be processed by STFT logic
                # Actual processing happens in __getitem__
                temp_segment_size = end_index - start_index
                
                # Simulate length_corrector effect for determining if a valid STFT can be made
                # This is a simplified check; actual correction is in __getitem__
                num_blocks_for_stft = (temp_segment_size - fft_size) // (fft_size - 128) + 1 # Assuming overlap 128 for STFT
                
                if num_blocks_for_stft <= 0 : # Not enough data for even one STFT window after potential correction
                    # This check might need refinement based on exact STFT needs
                    # print(f"Debug: Segment {start_index}-{end_index} in {dat_file_path} too short for STFT after initial check. Skipping.")
                    continue


                meta_entry['dat_file_path'] = os.path.abspath(dat_file_path)
                meta_entry['start_index'] = start_index
                meta_entry['end_index'] = end_index
                meta_entry['time_start_sec'] = start_index / sampling_rate
                meta_entry['time_end_sec'] = end_index / sampling_rate
                meta_entry['fft_size'] = fft_size
                meta_entry['num_fft_spec'] = num_fft_spec
                meta_entry['label'] = [meta_entry.get(device) for device in ['device', 'device_2'] if meta_entry.get(device) is not None]
                
                master_metadata.append(meta_entry)

        except FileNotFoundError:
            print(f"Error: .dat file not found: {dat_file_path}. Skipping.")
            continue
        except Exception as e:
            print(f"Error processing {dat_file_path} for metadata: {e}")
            traceback.print_exc()
            continue

    if master_metadata:
        save_to_json(master_metadata, output_json_path)
        print(f"Master metadata saved to {output_json_path} with {len(master_metadata)} potential samples.")
    else:
        print("No metadata generated.")
    return master_metadata


"""
Image to Metadata Matching Utilities

This module provides utilities for matching image file paths to metadata entries.
It handles various naming convention inconsistencies and transformations needed
to map between image file names and their corresponding metadata entries.
"""


def normalize_image_paths_for_matching(img_short_names: List[str]) -> List[str]:
    """
    Normalizes image file paths to match the format used in metadata 'spect_path' entries.
    
    This function applies a series of transformations to handle naming inconsistencies
    between image file names and metadata entries, including:
    - Removing _RC* suffixes (e.g., _RC, _RC-m, _RC-s)
    - Removing "-armed" suffixes
    - Converting background class names back to original drone names
    - Fixing various spelling/naming inconsistencies (HolyStone vs HollyStone, etc.)
    
    Args:
        img_short_names: List of short image paths (format: "class_name/filename.ext")
        
    Returns:
        List of normalized image paths that can be matched against metadata 'spect_path' entries
        
    Example:
        >>> img_short_names = ["DJI_Phantom4/sample_123_RC.png", "background/HolyStone_HS110G_1.2_3.4.png"]
        >>> normalize_image_paths_for_matching(img_short_names)
        ["DJI_Phantom4/sample_123.png", "HollyStone_HS110G/HolyStone_HS110G_1.2_3.4.png"]
    """
    img_spect_path = img_short_names.copy()
    
    # Remove _RC* suffixes from names (e.g., _RC, _RC-m, _RC-s)
    img_spect_path = [re.sub(r'_RC(-[a-z])?$', '', name) for name in img_spect_path]
    
    # Remove "-armed" suffixes
    img_spect_path = [name.replace('-armed', '') for name in img_spect_path]
    
    # Change background to original name
    # Background images are stored as "background/DroneModel_params.png" but metadata has "DroneModel/..."
    for i, name in enumerate(img_spect_path):
        if name.startswith('background'):
            components = name.split(os.path.sep)[1].split("_")
            first_float_index = find_first_float_index(components)
            if components[0].lower() != 'skydio':
                img_spect_path[i] = name.replace(f'background{os.path.sep}', f'{"_".join(components[:first_float_index])}{os.path.sep}')
            else:
                img_spect_path[i] = name.replace(f'background{os.path.sep}', f'{components[0]}_{components[1]}{os.path.sep}')
    
    # Fix naming inconsistencies
    # HolyStone_HS110G -> HollyStone_HS110G (spelling fix)
    img_spect_path = [name.replace(f'HolyStone_HS110G{os.path.sep}', f'HollyStone_HS110G{os.path.sep}') for name in img_spect_path]
    
    # HolyStone_HS720E -> Holystone_HS720E (capitalization fix)
    img_spect_path = [name.replace(f'HolyStone_HS720E{os.path.sep}', f'Holystone_HS720E{os.path.sep}') for name in img_spect_path]
    
    # RadioMaster_TX16S -> RadioMaster_TX16S_NA (add region suffix)
    img_spect_path = [name.replace(f'RadioMaster_TX16S{os.path.sep}', f'RadioMaster_TX16S_NA{os.path.sep}') for name in img_spect_path]
    
    # Autel_EXOII -> Autel-EXOII (underscore to dash)
    img_spect_path = [name.replace(f'Autel_EXOII{os.path.sep}', f'Autel-EXOII{os.path.sep}') for name in img_spect_path if 'Autel-EXOII' in name]
    
    return img_spect_path


def create_image_short_names(img_files: List[str]) -> List[str]:
    """
    Extracts short names from full image file paths.
    
    Args:
        img_files: List of full image file paths
        
    Returns:
        List of short names in format "class_name/filename.ext"
        
    Example:
        >>> img_files = ["/path/to/dataset/train/images/DJI_Phantom4/sample_123.png"]
        >>> create_image_short_names(img_files)
        ["DJI_Phantom4/sample_123.png"]
    """
    return [os.path.join(os.path.basename(os.path.dirname(f)), os.path.basename(f)) for f in img_files]


def match_images_to_metadata(
    img_files: List[str],
    meta_data: List[dict],
    verbose: bool = True
) -> tuple[List[str], List[str], set[str]]:
    """
    Matches image files to metadata entries based on 'spect_path'.
    
    This function creates short names from image files, normalizes them to match
    metadata format, and returns the mappings needed to link images with their metadata.
    
    Args:
        img_files: List of full image file paths
        meta_data: List of metadata dictionaries (each with a 'spect_path' key)
        verbose: Whether to log matching statistics and unmatched files
        
    Returns:
        Tuple of (img_short_names, img_spect_path, matched_img_paths):
        - img_short_names: Original short names (class_name/filename.ext)
        - img_spect_path: Normalized paths that match metadata format
        - matched_img_paths: Set of spect_paths that were successfully matched
        
    Example:
        >>> img_files = ["/path/to/dataset/train/images/DJI_Phantom4/sample_123_RC.png"]
        >>> meta_data = [{"spect_path": "DJI_Phantom4/sample_123.png", ...}]
        >>> short_names, spect_paths, matched = match_images_to_metadata(img_files, meta_data)
        >>> # short_names: ["DJI_Phantom4/sample_123_RC.png"]
        >>> # spect_paths: ["DJI_Phantom4/sample_123.png"]
        >>> # matched: {"DJI_Phantom4/sample_123.png"}
    """
    # Create short names and normalize them
    img_short_names = create_image_short_names(img_files)
    img_spect_path = normalize_image_paths_for_matching(img_short_names)
    
    if verbose:
        logger.debug(f"Matching {len(img_spect_path)} unique image paths against {len(meta_data)} metadata entries.")
        logger.debug(f"\nSample normalized paths:\n{img_spect_path[:3]}")
    
    # Create set of available spect_paths in metadata for fast lookup
    available_spect_paths = {sample['spect_path'] for sample in meta_data}
    
    # Track which image paths were matched
    matched_img_paths = set()
    for spect_path in img_spect_path:
        if spect_path in available_spect_paths:
            matched_img_paths.add(spect_path)
    
    if verbose:
        # Log matching statistics
        match_rate = len(matched_img_paths) / len(img_spect_path) * 100 if img_spect_path else 0
        logger.info(f"Matched {len(matched_img_paths)} out of {len(img_spect_path)} image files ({match_rate:.1f}%)")
        
        # Log unmatched files if any
        unmatched_paths = [path for path in img_spect_path if path not in matched_img_paths]
        if unmatched_paths:
            logger.warning(f"Could not find metadata for {len(unmatched_paths)} image files")
            logger.debug(f"First 10 unmatched paths:\n{unmatched_paths[:10]}")
            
            # Group unmatched by class/model
            from collections import defaultdict
            unmatched_by_class = defaultdict(int)
            for path in unmatched_paths:
                class_name = os.path.dirname(path)
                unmatched_by_class[class_name] += 1
            
            logger.debug("Unmatched files by class:")
            for class_name, count in sorted(unmatched_by_class.items(), key=lambda x: x[1], reverse=True):
                logger.debug(f"  {class_name}: {count} files")
    
    return img_short_names, img_spect_path, matched_img_paths


if __name__ == "__main__":
    data_files_path = '../../dataset/Relate work from AeroDefense/Data/Raw data/site_survey_recording/'
    dat_files_list: list[str] = glob(os.path.join(data_files_path, '**','*.dat'), recursive=True)
    len(dat_files_list)
    master_meta = create_master_metadata(
        dat_files_list= dat_files_list,
        output_json_path= 'test.json',
        sample_time=0.5,
        step_time=0.1,
        fft_size=1024,
        num_fft_spec=1500
)