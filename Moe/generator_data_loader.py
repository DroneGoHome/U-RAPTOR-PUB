# Python library imports
import os
import sys
import time
import random
import logging
import traceback
from glob import glob
from typing import List, Dict, Any, Optional

# Standard library imports
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.signal import stft
import torchvision.transforms as transforms


# Torch imports
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms # Import transforms


# --- Start of sys.path modification ---
# Get the directory of the current script (e.g., /mnt/d/Rowan/AeroDefence/src/pytorch)
_current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up two levels to get to the project root (e.g., /mnt/d/Rowan/AeroDefence/)
# This directory contains the 'src' package.
_project_root = os.path.abspath(os.path.join(_current_script_dir, '..', '..'))

# Add the project root to sys.path if it's not already there.
# This allows Python to find the 'src' package.
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
# --- End of sys.path modification ---

# Update these imports to match your actual file structure
try:
    from Moe.src.dataset import load_json, save_to_json, get_drone_model, get_drone_center_freq, get_drone_manufacture, extract_center_freq_from_filename, get_color_img
    from Moe.src.constants import FFT_SIZE, SAMPLING_RATE, NUM_FFT_SPEC
    from Moe.src.processing import length_corrector, get_stft_spectrogram, get_power_spectrogram_db, add_awgn_noise
except ImportError:
    # If the above doesn't work, try without the Moe prefix
    try:
        from src.dataset import load_json, save_to_json, get_drone_model, get_drone_center_freq, get_drone_manufacture, extract_center_freq_from_filename, get_color_img
        from src.constants import FFT_SIZE, SAMPLING_RATE, NUM_FFT_SPEC
        from src.processing import length_corrector, get_stft_spectrogram, get_power_spectrogram_db, add_awgn_noise
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Please check your file structure and import paths")
        # You might want to define placeholder functions or raise an error here
        raise

def create_master_metadata(
    dat_files_list: List[str],
    output_json_path: str,
    sample_time: float,
    step_time: float,
    sampling_rate: int, # Assuming constant SR for now, or fetch per file
    fft_size: int, # Needed for length_corrector and truncation logic
    num_fft_spec: int # Needed for truncation logic
) -> List[Dict[str, Any]]:
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
    sample_size_points = int(sample_time * sampling_rate)
    step_size_points = int(step_time * sampling_rate)

    if sample_size_points <= 0:
        print("Error: sample_time results in non-positive sample_size_points.")
        return []
    if step_size_points <= 0:
        print("Error: step_time results in non-positive step_size_points.")
        return []

    print(f"Scanning {len(dat_files_list)} .dat files to generate master metadata...")
    for dat_file_path in tqdm(dat_files_list, desc="Generating Master Metadata"):
        try:
            # Use memmap to get signal size without loading the whole file
            mmap_signal = np.memmap(dat_file_path, dtype=np.complex64, mode='r')
            signal_total_points = mmap_signal.size
            del mmap_signal # Release memmap

            if signal_total_points < sample_size_points:
                print(f"Warning: File {dat_file_path} is too short ({signal_total_points} points) for sample size ({sample_size_points} points). Skipping.")
                continue

            drone_manufacture = get_drone_manufacture(dat_file_path).lower()
            
            # Define a set of strings that indicate "no drone"
            no_drone_identifiers = {'nodrone', 'no_drone', 'no drone'}
            no_drone = any(identifier in drone_manufacture for identifier in no_drone_identifiers)

            drone_model = 'no_drone' if no_drone else get_drone_model(dat_file_path)
            center_freq = extract_center_freq_from_filename(dat_file_path) if no_drone else get_drone_center_freq(dat_file_path)

            for i in range(0, signal_total_points - sample_size_points + 1, step_size_points):
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


                meta_entry = {
                    'dat_file_path': os.path.abspath(dat_file_path),
                    'start_index': start_index,
                    'end_index': end_index,
                    'label': drone_model if drone_model else "Unknown",
                    'center_freq': center_freq, # Store if needed
                    'sampling_rate': sampling_rate, # Store if it can vary
                    # Add any other static info you need per sample
                }
                master_metadata.append(meta_entry)

        except FileNotFoundError:
            print(f"Error: .dat file not found: {dat_file_path}. Skipping.")
        except Exception as e:
            print(f"Error processing {dat_file_path} for metadata: {e}")
            traceback.print_exc()

    if master_metadata:
        save_to_json(master_metadata, output_json_path)
        print(f"Master metadata saved to {output_json_path} with {len(master_metadata)} potential samples.")
    else:
        print("No metadata generated.")
    return master_metadata

class SpectogramDataset(Dataset):
    def __init__(self,
                 master_meta_data_path: str,
                 snr_list: List[int] = [None],
                 fft_size: int = FFT_SIZE, num_fft_spec: int = NUM_FFT_SPEC,
                 stft_overlap: int = 128,
                 transform: callable = transforms.ToTensor(),
                 classify_type: str = "label"):
        """
        Args:
            master_meta_data_path (str): Path to the master JSON metadata file.
            snr_list (list): List of SNR values in dB to pick from (e.g., [-10, 0, 10, None]).
                             None means no added noise.
            fft_size (int): FFT window size for STFT.
            num_fft_spec (int): Number of FFTs for truncation logic, determining max segment length.
            stft_overlap (int): Overlap for STFT calculation.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.master_metadata = load_json(master_meta_data_path)
        if not self.master_metadata:
            raise ValueError(f"Could not load or metadata is empty from {master_meta_data_path}")

        self.snr_list = snr_list
        self.transform = transform
        self.fft_size = fft_size
        self.num_fft_spec = num_fft_spec # For truncation logic
        self.stft_overlap = stft_overlap
        
        self.classify_type = classify_type  

        if self.classify_type == "modulation":
            all_string_targets = [sample['modulation'] for sample in self.master_metadata]
        elif self.classify_type == "protocol":
            all_string_targets = [sample['protocol'] for sample in self.master_metadata]
        elif self.classify_type == "label":
            all_string_targets = [sample['label'] for sample in self.master_metadata]
        else:
            raise ValueError(f"Invalid classify_type: {self.classify_type}. Must be 'modulation', 'protocol', or 'label'.")
        
        self.unique_labels = sorted(list(set(all_string_targets)))
        self.label_to_int = {label: i for i, label in enumerate(self.unique_labels)}
        self.int_to_label = {i: label for label, i in self.label_to_int.items()}
        self.classes = self.unique_labels  # For compatibility with torchvision
        self.num_classes = len(self.unique_labels)

        self.merge_list = sorted(list(set([sample['dat_file_path'] for sample in self.master_metadata if sample['label'] != 'no_drone'])))

    def __len__(self):
        return len(self.master_metadata)

    def _truncate_segment(self, start_idx: int, end_idx: int) -> np.ndarray:
        """
        Applies length correction and truncates the segment.

        The segment is first processed by `length_corrector`. If the corrected
        segment's size exceeds `self.num_fft_spec * self.fft_size`, it is
        truncated to this maximum length.

        Args:
            segment (np.ndarray): The input signal segment.

        Returns:
            np.ndarray: The processed and potentially truncated segment.
        """
        # Apply length correction
        number_of_vectors = (end_idx - start_idx) // self.fft_size
        new_end_idx = start_idx + (number_of_vectors * self.fft_size)
        
        # Truncate if segment is too long based on num_fft_spec and fft_size
        max_len = self.num_fft_spec * self.fft_size
        if (new_end_idx - start_idx) > max_len:
            return start_idx + max_len
        return new_end_idx

    def _merge_stft_matrices(self, input_stft_matrix: np.ndarray, background_stft_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Merges input_stft_matrix with a background STFT matrix.
        If background_stft_matrix is provided, it's used. Otherwise, a random one is chosen.
        """
        if background_stft_matrix is not None:
            return input_stft_matrix + background_stft_matrix
        
        if not self.merge_list:
            logging.warning("Merge list is empty. Returning original STFT matrix without merging.")
            return input_stft_matrix
        
        chosen_file = random.choice(self.merge_list)
        available_samples = [s for s in self.master_metadata if s['dat_file_path'] == chosen_file]
        
        if not available_samples:
            logging.warning(f"No available samples for merging from {chosen_file}. Returning input STFT matrix.")
            return input_stft_matrix
        
        sample = random.choice(available_samples)
        try:
            # Load segment for background
            segment_bg = self._load_segment_from_meta(sample)
            # background_stft_matrix_to_add = get_stft_spectrogram(signal_sample=segment_bg, window_size=self.fft_size, overlap=self.stft_overlap)
            _,_, background_stft_matrix_to_add = stft(segment_bg, fs=SAMPLING_RATE, nperseg=self.fft_size, noverlap=self.stft_overlap, window= 'hann', return_onesided=False)
            background_stft_matrix_to_add = np.fft.fftshift(background_stft_matrix_to_add, axes=0) # Shift along the frequency axis (axis 0)

            return input_stft_matrix + background_stft_matrix_to_add
        except Exception as e:
            logging.warning(f"Error loading segment for merging from {sample['dat_file_path']} [{sample['start_index']}:{sample['end_index']}]: {e}. Returning input STFT matrix.")
            return input_stft_matrix

    def _convert_to_pil(self, spectrum_db: np.ndarray) -> Image.Image:
        """Converts a spectrogram (numpy array) to a PIL Image."""
        rgb_image_np = get_color_img(spectrum_db=spectrum_db, colormap='viridis')
        rgb_image_uint8 = (rgb_image_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(rgb_image_uint8, 'RGB')
        return pil_image

    def _load_segment_from_meta(self, meta_entry: Dict[str, Any]) -> np.ndarray:
        """Loads a signal segment from a .dat file using metadata."""
        dat_file_path = meta_entry['dat_file_path']
        start_idx = meta_entry['start_index']
        end_idx = meta_entry['end_index']
        end_idx = self._truncate_segment(start_idx, end_idx)  # Ensure truncation logic is applied
        try:
            mmap_array = np.memmap(dat_file_path, dtype=np.complex64, mode='r')
            if not (0 <= start_idx < end_idx <= mmap_array.size):
                logging.critical(f"Invalid indices [{start_idx}:{end_idx}] for file {dat_file_path} of size {mmap_array.size}.")
                raise IndexError(f"Invalid indices for {dat_file_path}")
            
            signal_segment = mmap_array[start_idx:end_idx]
            del mmap_array
            return signal_segment
        except FileNotFoundError:
            logging.error(f"File not found: {dat_file_path}")
            raise
        except IndexError as e:
            logging.error(f"Index error for {dat_file_path} [{start_idx}:{end_idx}]: {e}")
            raise
        except Exception as e:
            logging.error(f"Error loading segment from {dat_file_path} [{start_idx}:{end_idx}]: {e}")
            raise

    def _get_signal_plus_noise_and_pure_noise(self, base_signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, Optional[int]]:
        """
        Generates a noisy version of the base signal and the pure noise component.
        Returns: (signal_with_noise, pure_noise_component, selected_snr)
        """
        selected_snr = None
        if self.snr_list:
            selected_snr = random.choice(self.snr_list)
        
        if selected_snr is not None:
            # Assuming add_awgn_noise returns only the noise component
            pure_noise_component = add_awgn_noise(base_signal, snr_db=selected_snr)
            signal_with_noise = base_signal + pure_noise_component
        else:
            pure_noise_component = np.zeros_like(base_signal)
            signal_with_noise = base_signal.copy() # Return a copy to ensure consistency
        
        return signal_with_noise, pure_noise_component, selected_snr

    def _create_spectrogram_image(self,
                                  signal_data: np.ndarray,
                                  sampling_rate: int,
                                  center_freq: float,
                                  should_merge_background: bool = False) -> Image.Image:
        """
        Converts signal data into an RGB PIL Image of its spectrogram, optionally merging background.
        """
        # stft_matrix = get_stft_spectrogram(signal_sample=signal_data,
        #                                    window_size=self.fft_size,
        #                                    overlap=self.stft_overlap)
        
        _,_, stft_matrix = stft(signal_data, fs=SAMPLING_RATE, nperseg=self.fft_size, noverlap=self.stft_overlap, window= 'hann', return_onesided=False)
        stft_matrix = np.fft.fftshift(stft_matrix, axes=0) 
        
        if should_merge_background:
            stft_matrix = self._merge_stft_matrices(input_stft_matrix=stft_matrix, 
                                                    background_stft_matrix=None)

        spectrum_db = get_power_spectrogram_db(stft_matrix=stft_matrix,
                                               sampling_rate=sampling_rate,
                                               center_freq=center_freq,
                                               window_size=self.fft_size,
                                               overlap=self.stft_overlap)
        
        return self._convert_to_pil(spectrum_db=spectrum_db)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        meta_entry = self.master_metadata[idx]

        dat_file_path = meta_entry['dat_file_path']

        if self.classify_type == "modulation":
            target_label_str = meta_entry['modulation']
        elif self.classify_type == "protocol":
            target_label_str = meta_entry['protocol']
        elif self.classify_type == "label":
            target_label_str = meta_entry['label']
        else:
            raise ValueError(f"Invalid classify_type: {self.classify_type}. Must be 'modulation', 'protocol', or 'label'.")

        target_int = self.label_to_int[target_label_str]
        sampling_rate = meta_entry.get('sampling_rate', SAMPLING_RATE)
        
        center_freq = extract_center_freq_from_filename(dat_file_path) if target_label_str == "no_drone" \
                      else meta_entry.get('center_freq', get_drone_center_freq(dat_file_path))

        # 1. Load and truncate the clean signal segment
        truncated_clean_signal = self._load_segment_from_meta(meta_entry)

        # 2. Generate the main input signal (potentially noisy) and the pure noise signal
        main_input_signal, pure_noise_signal, selected_snr = \
            self._get_signal_plus_noise_and_pure_noise(truncated_clean_signal)

        # 3. Create the main spectrogram image (merged with background)
        main_spectrogram_image = self._create_spectrogram_image(
            signal_data=main_input_signal,
            sampling_rate=sampling_rate,
            center_freq=center_freq,
            should_merge_background=False,
        )

        # 4. Create the noise spectrogram image (if noise was added, not merged)
        if selected_snr is not None:
            noise_spectrogram_image = self._create_spectrogram_image(
                signal_data=pure_noise_signal,
                sampling_rate=sampling_rate,
                center_freq=center_freq,
                should_merge_background=False
            )
            noise_spectrogram_tensor = transforms.ToTensor()(noise_spectrogram_image)
        else:
            width, height = main_spectrogram_image.size
            num_channels = 3 # For RGB PIL images
            noise_spectrogram_tensor = torch.zeros((num_channels, height, width), dtype=torch.float32)

        # 5. Apply transformations to the main spectrogram image
        final_main_tensor = self.transform(main_spectrogram_image) if self.transform else transforms.ToTensor()(main_spectrogram_image)
            
        # return final_main_tensor, noise_spectrogram_tensor, target_int, snr_to_return
        return final_main_tensor, target_int

if __name__ == "__main__":
    # Define your .dat files (e.g., by globbing)
    all_dat_files = []
    base_data_path = '/mnt/d/Raptor/Moe_data/raw/'
    patterns_to_glob = ['*.dat', os.path.join("*", "*.dat")]
    for pattern in patterns_to_glob:
        all_dat_files.extend(glob(os.path.join(base_data_path, pattern), recursive=True))

    # no_drone_files_list = [
	# 		'../../data/Relate work from AeroDefense/Data/Raw data/NoDrone_indoor/NoDrone_Band24GHz_Freq2442_SampRate20MHz_ReqTime30sec-002.dat',
	# 		'../../data/Relate work from AeroDefense/Data/Raw data/NoDrone_indoor/NoDrone_Band900MHz_Freq918_SampRate20MHz_ReqTime30sec-002.dat',
	# 		'../../data/Relate work from AeroDefense/Data/Raw data/NoDrone_outdoor/NoDrone905_SampRate20MHz.dat',
	# 		'../../data/Relate work from AeroDefense/Data/Raw data/NoDrone_outdoor/NoDrone2442_SampRate20MHz.dat'
	# 	]
    # all_dat_files.extend(no_drone_files_list) # Add no_drone files too

    unique_dat_files = sorted(list(set(all_dat_files)))

    master_meta_path = '/mnt/d/Raptor/Moe_data/master_meta_data.json'

    create_master_metadata(
        dat_files_list=unique_dat_files,
        output_json_path=master_meta_path,
        sample_time=0.5,  # Corresponds to your main function's sample_time
        step_time=0.1,    # Corresponds to your main function's step_time
        sampling_rate=SAMPLING_RATE,
        fft_size=FFT_SIZE,
        num_fft_spec=NUM_FFT_SPEC
    )
