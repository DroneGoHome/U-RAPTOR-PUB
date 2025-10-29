# %%
# Python library imports
import os
import sys
import time
import random
import logging
import traceback
from glob import glob
from typing import List, Dict, Any, Optional, Literal

# Standard library imports
from click import Choice
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

# Custom imports
from src.data.dataset import load_json, save_to_json, get_drone_model, get_drone_center_freq, get_drone_manufacture, extract_center_freq_from_filename, get_color_img
from src.data.constants import FFT_SIZE, SAMPLING_RATE, NUM_FFT_SPEC
from data.processing import get_power_spectrogram_db, add_awgn_noise

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
                 snr_list: List[int] = [None, -20, -15, -10, -5, 0, 5, 10, 15, 20],
                 fft_size: int = FFT_SIZE, num_fft_spec: int = NUM_FFT_SPEC,
                 stft_overlap: int = 128,
                 mode: Literal['label', 'multi-label', 'multi-modal', 'binary'] = 'label',
                 background_list: Optional[List[str]] = None,
                 colormap: str='viridis',
                 transform: callable = transforms.ToTensor()):
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

        self.mode = mode
        if (self.mode =='multi-label') and (background_list is None):
            raise ValueError("For 'multi-label' mode, a background_list must be provided.")
        elif (background_list is None) and (self.mode != 'multi-label'):
            background_list = ['no_drone']  # Ensure only 'no_drone' is used for other modes
        self.background_list = background_list

        self.colormap = colormap # for multi-modal we might want to use different colormaps
        self.snr_list = snr_list
        self.transform = transform
        self.fft_size = fft_size
        self.num_fft_spec = num_fft_spec # For truncation logic
        self.stft_overlap = stft_overlap

        all_string_targets = [sample['label'] for sample in self.master_metadata]
        self.unique_labels = sorted(list(set(all_string_targets)))
        self.label_to_int = {label: i for i, label in enumerate(self.unique_labels)}
        self.int_to_label = {i: label for label, i in self.label_to_int.items()}
        self.classes = self.unique_labels  # For compatibility with torchvision
        self.num_classes = len(self.unique_labels)

        self.no_drone_indexes = [i for i, sample in enumerate(self.master_metadata) if sample['label'] != 'no_drone']

        self.mmem_map_files = {f"{sample['dat_file_path']}":np.memmap(sample['dat_file_path'], dtype=np.complex64, mode='r') for sample in self.master_metadata}

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

    def _convert_to_pil(self, spectrum_db: np.ndarray, colormap='viridis') -> Image.Image:
        """Converts a spectrogram (numpy array) to a PIL Image."""
        rgb_image_np = get_color_img(spectrum_db=spectrum_db, colormap=colormap)
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
                
            # if not (0 <= start_idx < end_idx <= mmap_array.size):
            #     logging.critical(f"Invalid indices [{start_idx}:{end_idx}] for file {dat_file_path} of size {mmap_array.size}.")
            #     raise IndexError(f"Invalid indices for {dat_file_path}")

            signal_segment = self.mmem_map_files[dat_file_path][start_idx:end_idx].copy()  # Copy to avoid memmap issues
            # del mmap_array
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
                                  colormap: str = None) -> Image.Image:
        """
        Converts signal data into an RGB PIL Image of its spectrogram, optionally merging background.
        """

        colormap = self.colormap if colormap is None else colormap
        
        _,_, stft_matrix = stft(signal_data, fs=SAMPLING_RATE, nperseg=self.fft_size, noverlap=self.stft_overlap, window= 'hann', return_onesided=False)
        stft_matrix = np.fft.fftshift(stft_matrix, axes=0) 
        
        spectrum_db = get_power_spectrogram_db(stft_matrix=stft_matrix,
                                               sampling_rate=sampling_rate,
                                               center_freq=center_freq,
                                               window_size=self.fft_size,
                                               overlap=self.stft_overlap)
        
        return self._convert_to_pil(spectrum_db=spectrum_db, colormap=colormap)

    def _get_sample(self, index=None) -> Dict[str, Any]:
        
        index = random.randint(0, self.len() - 1) if index is None else index
        meta_entry = self.master_metadata[index]

        dat_file_path = meta_entry['dat_file_path']
        target_label_str = meta_entry['label']
        target_int = self.label_to_int[target_label_str]
        sampling_rate = meta_entry.get('sampling_rate', SAMPLING_RATE)

        center_freq = extract_center_freq_from_filename(dat_file_path) if target_label_str == "no_drone" \
                      else meta_entry.get('center_freq', get_drone_center_freq(dat_file_path))

        return meta_entry, target_label_str, target_int, sampling_rate, center_freq

    def _get_random_no_drone_sample(self, index=None) -> Dict[str, Any]:
        background_idx = random.choice(self.no_drone_indexes)
        meta_entry, target_label_str, target_int, sampling_rate, center_freq = self._get_sample(index=background_idx)

        return meta_entry, target_label_str, target_int, sampling_rate, center_freq

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        # 1. Load and truncate the clean signal segment
        meta_entry, target_label_str, target_int, sampling_rate, center_freq = self._get_sample(index=idx)
        truncated_clean_signal = self._load_segment_from_meta(meta_entry)

        targets = [target_int]
        # 2. Load and truncate background signal segment
        truncated_background_signal = np.zeros_like(truncated_clean_signal)
        for background_target in self.background_list:
            background_meta_entry, background_target_label_str, background_target_int, background_sampling_rate, background_center_freq = self._get_random_no_drone_sample() if background_target=='random' else self._get_sample(index=idx)
            truncated_background_signal += self._load_segment_from_meta(background_meta_entry)
            targets.append(background_target_int)

        # 3. Generate the main input signal (potentially noisy) and the pure noise signal
        main_input_signal, pure_noise_signal, selected_snr = \
            self._get_signal_plus_noise_and_pure_noise(truncated_clean_signal+truncated_background_signal)

        # 4. Create the main spectrogram image (merged with background)
        main_spectrogram_image = self._create_spectrogram_image(
            signal_data=main_input_signal,
            sampling_rate=sampling_rate,
            center_freq=center_freq,
            colormap=None, # colormap not used for multi-modal
        )

        # 5. Create the noise spectrogram image (if noise was added, not merged)
        if (selected_snr is not None) and (self.mode == 'multi-modal'):
            noise_spectrogram_image = self._create_spectrogram_image(
                signal_data=pure_noise_signal,
                sampling_rate=sampling_rate,
                center_freq=center_freq,
                colormap=None, # colormap not used for multi-modal
            )
            noise_spectrogram_tensor = transforms.ToTensor()(noise_spectrogram_image)
        elif (self.mode == 'multi-modal'):
            width, height = main_spectrogram_image.size
            num_channels = 3 # For RGB PIL images
            noise_spectrogram_tensor = torch.zeros((num_channels, height, width), dtype=torch.float32)

        # 6. Apply transformations to the main spectrogram image
        final_main_tensor = self.transform(main_spectrogram_image)
            
        if self.mode == 'multi-modal':
            snr_to_return = selected_snr if selected_snr is not None else 100
            return final_main_tensor, noise_spectrogram_tensor, target_int, snr_to_return
        elif self.mode == 'multi-label':
            return final_main_tensor, targets, background_target_int
        elif self.mode == 'binary':
            return final_main_tensor, 0 if target_label_str != 'no_drone' else 1
        return final_main_tensor, target_int
# %%
if __name__ == "__main__":
    # Define your .dat files (e.g., by globbing)
    all_dat_files = []
    base_data_path = '../../data/Relate work from AeroDefense/Data/Raw data/SR20M_G50_cage_RT15/'
    patterns_to_glob = ['*.dat', os.path.join("*", "*.dat")]
    for pattern in patterns_to_glob:
        all_dat_files.extend(glob(os.path.join(base_data_path, pattern), recursive=True))

    no_drone_files_list = [
			'../../data/Relate work from AeroDefense/Data/Raw data/NoDrone_indoor/NoDrone_Band24GHz_Freq2442_SampRate20MHz_ReqTime30sec-002.dat',
			'../../data/Relate work from AeroDefense/Data/Raw data/NoDrone_indoor/NoDrone_Band900MHz_Freq918_SampRate20MHz_ReqTime30sec-002.dat',
			'../../data/Relate work from AeroDefense/Data/Raw data/NoDrone_outdoor/NoDrone905_SampRate20MHz.dat',
			'../../data/Relate work from AeroDefense/Data/Raw data/NoDrone_outdoor/NoDrone2442_SampRate20MHz.dat'
		]
    all_dat_files.extend(no_drone_files_list) # Add no_drone files too
    unique_dat_files = sorted(list(set(all_dat_files)))

    master_meta_path = '../../data/Related work from Rowan/new_dataset/master_meta_data.json'

    # create_master_metadata(
    #     dat_files_list=unique_dat_files,
    #     output_json_path=master_meta_path,
    #     sample_time=0.5,  # Corresponds to your main function's sample_time
    #     step_time=0.1,    # Corresponds to your main function's step_time
    #     sampling_rate=SAMPLING_RATE,
    #     fft_size=FFT_SIZE,
    #     num_fft_spec=NUM_FFT_SPEC
    # )

# %%
    # 1. First, ensure 'master_meta_data.json' is generated by running the
    #    example call for `create_master_metadata` (shown in dataset.py modifications)
    #    or a similar script.
    
    # Check if master metadata exists
    if not os.path.exists(master_meta_path):
        print(f"Master metadata file not found at: {master_meta_path}")
        print("Please generate it first using `create_master_metadata` function.")
    else:
        print(f"Attempting to load dataset using metadata from: {master_meta_path}")
        
        snr_options = [-10, -5, 0, 5, 10, 15, 20, None] # Example SNR list

        # Example transform (e.g., if your spectrograms are treated as images)
        # Add a channel dimension if needed by CNNs: HxW -> 1xHxW
        # The spectrograms from get_power_spectrogram_db are 2D (freq_bins, time_bins)
    

        custom_dataset = SpectogramDataset(
            master_meta_data_path=master_meta_path,
            snr_list=snr_options,
            background_list=['no_drone'],  # Use 'no_drone' for binary or multi-label
            mode='multi-modal',  # Change to 'multi-label' or 'binary' as needed
            colormap='viridis',  # or 'gray' for grayscale spectrograms
            transform=transforms.Compose([
                transforms.ToTensor(),  # Convert PIL Image to Tensor
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                # Normalize if needed
            ]),
            fft_size=FFT_SIZE, # Pass relevant params
            num_fft_spec=1500, # from constants.NUM_FFT_SPEC
            stft_overlap=128
        )

        print("Dataset initialized.")
        print(f"Number of samples in dataset: {len(custom_dataset)}")
        print(f"Class mapping: {custom_dataset.label_to_int}")

        if len(custom_dataset) > 0:
            print("\nFetching first sample directly from dataset:")
            try:
                samples_batch, noise_batch, targets_batch, snr_batch = custom_dataset[0]
                print(samples_batch[0])
                print(f"Sample (noisy/original) spectrogram type: {type(samples_batch)}")
                print(f"Sample spectrogram shape: {samples_batch.shape}") # Expected: (C, H, W) after transform
                print(f"Sample spectrogram dtype: {samples_batch.dtype}")

                print(f"Noise spectrogram type: {type(noise_batch)}")
                print(f"Noise spectrogram shape: {noise_batch.shape}") # Should be 2D before transform
                print(f"Noise spectrogram dtype: {noise_batch.dtype}")

                print(f"Target type: {type(targets_batch)}")
                print(f"Target (int): {targets_batch}")
                print(f"Target (label): {custom_dataset.int_to_label[targets_batch]}")

            except Exception as e:
                print(f"Error fetching first sample: {e}")
                import traceback
                traceback.print_exc()

            print("\nTesting DataLoader...")
            try:
                data_loader = DataLoader(custom_dataset, batch_size=4, shuffle=True, num_workers=22) # num_workers > 0 can speed up
                
                samples_batch, noise_batch, targets_batch, snr_batch = next(iter(data_loader))
                
                print("Fetched one batch from DataLoader:")
                print(f"Samples batch type: {type(samples_batch)}")
                print(f"Samples batch shape: {samples_batch.shape}")
                print(f"Samples batch dtype: {samples_batch.dtype}")

                print(f"Noise batch type: {type(noise_batch)}")
                print(f"Noise batch shape: {noise_batch.shape}")
                print(f"Noise batch dtype: {noise_batch.dtype}")

                print(f"Targets batch type: {type(targets_batch)}")
                print(f"Targets batch: {targets_batch}")
                print(f"Targets batch shape: {targets_batch.shape}")

                print(f"SNR batch type: {type(snr_batch)}")
                print(f"SNR batch: {snr_batch}")
                print(f"SNR batch shape: {snr_batch.shape}")
            except Exception as e:
                print(f"Error during DataLoader test: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Dataset is empty, cannot test fetching or DataLoader.")


# %%
    # 1. First, ensure 'master_meta_data.json' is generated by running the
    #    example call for `create_master_metadata` (shown in dataset.py modifications)
    #    or a similar script.
    
    # Check if master metadata exists
    if not os.path.exists(master_meta_path):
        print(f"Master metadata file not found at: {master_meta_path}")
        print("Please generate it first using `create_master_metadata` function.")
    else:
        print(f"Attempting to load dataset using metadata from: {master_meta_path}")

        # --- Scenario 1: With effective caching ---
        print("\n--- Testing with effective caching")
        iterate = 10
        batch_size=32
        if len(custom_dataset) > 0:
            start_time = time.perf_counter()
            for i in range(iterate*batch_size):
                _ = custom_dataset[i % len(custom_dataset)] # Access samples
            end_time = time.perf_counter()
            print(f"Time to access {iterate*batch_size} samples (direct access, cached): {end_time - start_time:.4f} seconds")
            
            # DataLoader test (cached)
            # For more stable benchmark, use shuffle=False
            loader_cached = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False, num_workers=22, pin_memory=True)
            counter = 0
            start_time_dl = time.perf_counter()
            for _ in loader_cached: # Iterate through all batches
                counter += 1
                if counter == iterate: # Print every 10 batches
                    break
            
            end_time_dl = time.perf_counter()
            print(f"Time to iterate through DataLoader (cached, num_workers=22): {end_time_dl - start_time_dl:.4f} seconds")