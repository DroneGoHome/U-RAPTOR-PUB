import sys
import os

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions with specific aliases to avoid name conflicts
from .processing import (
    load_dat_file,
    length_corrector,
    get_sampling_rate,
    visualize_signal,
    get_spectrogram,
    get_stft_spectrogram,
    get_power_spectrogram_db,
    get_color_img,
    add_awgn_noise,
    save_color_img,
    save_spectrogram,
)

from .constants import (
        FFT_SIZE,
        SAMPLING_RATE,
        CENTER_FREQ,
        NUM_FFT_SPEC,
        RECORDING_TIME,
)

from .dataset import (
    save_to_json,
    load_json,
    load_jsonl,
    save_to_jsonl,
    save_processed_data,
    create_folder,
    get_starting_index,
    get_drone_model,
    get_drone_manufacture,
    get_drone_bandwidth,
    get_drone_center_freq,
    get_drone_operation_mode,
    get_drone_info,
    get_samples_from_recording,
    save_samples_from_recording,
    make_dataset,
)

__all__ = [
    # From data.processing
    'load_dat_file',
    'length_corrector',
    'get_sampling_rate',
    'visualize_signal',
    'get_spectrogram',
    'get_stft_spectrogram',
    'get_power_spectrogram_db',
    'get_color_img',
    'add_awgn_noise',
    'save_color_img',
    'save_spectrogram',
    
    # From data.dataset
    'save_to_json',
    'load_jsonl',
    'load_json',
    'save_to_jsonl',
    'save_processed_data',
    'create_folder',
    'get_starting_index',
    'get_drone_model',
    'get_drone_manufacture',
    'get_drone_bandwidth',
    'get_drone_center_freq',
    'get_drone_operation_mode',
    'get_drone_info',
    'get_samples_from_recording',
    'save_samples_from_recording',
    'make_dataset',
    
    
    # From data.constants
    'FFT_SIZE',
    'SAMPLING_RATE',
    'CENTER_FREQ',
    'NUM_FFT_SPEC',
    'RECORDING_TIME',
]