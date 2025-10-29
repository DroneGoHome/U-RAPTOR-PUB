# u-raptor

## Data Module (`src/data`)

This module is responsible for handling all aspects of data loading, processing, and dataset creation.

### Purpose

The primary goal of this module is to take raw signal recordings (`.dat` files) and transform them into a structured dataset suitable for analysis or model training. This involves:
*   Loading raw signal data.
*   Extracting metadata from filenames (e.g., drone model, frequency).
*   Segmenting the raw signals into smaller, manageable samples.
*   Generating spectrogram representations of these samples.
*   Saving the processed samples, spectrograms, and associated metadata.

### Key Components

*   **`constants.py`**: Defines global constants used throughout the data processing pipeline, such as `SAMPLING_RATE`, `FFT_SIZE`, `CENTER_FREQ`, etc.
*   **`processing.py`**: Contains low-level functions for signal processing tasks:
    *   `load_dat_file`: Loads raw complex signal data from `.dat` files using memory mapping for efficiency.
    *   `length_corrector`: Adjusts signal length based on processing requirements.
    *   `get_spectrogram`: Generates a spectrogram image from a signal segment using Matplotlib.
    *   `save_spectrogram`: Saves the generated spectrogram figure to a file.
    *   `visualize_signal`: Utility function to plot signal magnitude over time.
*   **`dataset.py`**: Orchestrates the dataset creation process and provides utility functions:
    *   **File I/O**: Functions (`save_to_json`, `load_jsonl`, `save_to_jsonl`, `save_processed_data`) for handling metadata (JSON, JSONL) and processed data (pickle).
    *   **Metadata Extraction**: Functions (`get_drone_model`, `get_drone_manufacture`, etc.) to parse filenames and extract relevant information about the recording.
    *   **Sampling**: `get_samples_from_recording` segments a long recording into smaller, potentially overlapping samples.
    *   **Dataset Creation**:
        *   `save_samples_from_recording`: Processes a single signal, extracts samples, saves raw samples (.pkl) and spectrograms (.png), and logs metadata to a temporary file.
        *   `make_dataset`: The main function that finds all raw data files, processes each one using `save_samples_from_recording`, organizes the output into subdirectories based on drone models, and compiles the final metadata file (`meta_data.json`).

### Workflow (`make_dataset`)

1.  Scans the specified `base_path` for raw `.dat` files using provided `patterns`.
2.  For each `.dat` file:
    *   Loads the signal data.
    *   Extracts drone metadata from the filename.
    *   Creates a dedicated output folder within `dataset_path` based on the drone model.
    *   Calls `save_samples_from_recording` to:
        *   Segment the signal into samples based on `sample_time` and `step_time`.
        *   For each sample:
            *   Save the raw signal segment as a `.pkl` file.
            *   Generate a spectrogram using `get_spectrogram`.
            *   Save the spectrogram as a `.png` file.
            *   Append metadata (file paths, time interval, drone info) to a temporary JSON Lines file (`tmp.json`).
3.  After processing all files, loads all entries from `tmp.json`.
4.  Saves the aggregated metadata list into `meta_data.json` in the `dataset_path`.
5.  Removes the temporary `tmp.json` file.