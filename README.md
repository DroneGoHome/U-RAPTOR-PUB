# **CageDroneRF: A Large-Scale RF Benchmark and Toolkit for Drone Perception**

This repository contains the official data processing modules, tooling, and data access information for the [CageDroneRF (CDRF) dataset](https://arxiv.org/abs/2601.03302). CDRF is a large-scale benchmark for Radio-Frequency (RF) drone detection and identification, built from real-world captures and systematically generated synthetic variants.

## 🗄️ Data Access

You can access the full CageDroneRF dataset using the links below:
* **Official Request Portal:** [AeroDefense Data Request](https://aerodefense.tech/u-raptor-data-request)

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

## 📝 Citation

If you find our dataset, code, or trained models helpful in your research, please consider citing our work. If you share the dataset, its metadata, or released models, or adaptations of them, citing this paper is **required** as attribution under [`LICENSE-DATA.md`](LICENSE-DATA.md):

```bibtex
@article{rostami2026cagedronerf,
  title={CageDroneRF: A Large-Scale RF Benchmark and Toolkit for Drone Perception},
  author={Rostami, Mohammad and Faysal, Atik and Xia, Hongtao and Kasasbeh, Hadi and Gao, Ziang and Wang, Huaxia},
  journal={arXiv preprint arXiv:2601.03302},
  year={2026}
}
```

## ⚖️ License

This project is **source-available for noncommercial use only**. Except for
the AGPL-3.0 YOLO code noted below, it is not licensed for commercial use.

| What | License | File |
|---|---|---|
| Source code and documentation (everything not listed below) | [PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) | [`LICENSE`](LICENSE) |
| CageDroneRF dataset, metadata files (`*meta*.json`), and trained models (except YOLO checkpoints, see `LICENSE-DATA.md`) | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) | [`LICENSE-DATA.md`](LICENSE-DATA.md) |
| YOLO training code in `src/yolo/` (code only, not its metadata files) and `notebooks/yolo.ipynb` | [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html), because it extends [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) | [`src/yolo/LICENSE`](src/yolo/LICENSE) |

In short: you may use, study, and modify this code and data for research,
education, and other noncommercial purposes, with attribution. You may not
sell it, build commercial products or services on it, or train commercial
models on the data. The AGPL-3.0 YOLO code may be used under the AGPL's own
terms: if you distribute it, or run a modified version that users interact
with over a network, you must offer those recipients or users its source
under the AGPL.

**Commercial licensing:** contact AeroDefense via
[aerodefense.tech](https://aerodefense.tech).
