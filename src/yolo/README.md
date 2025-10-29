# YOLO Training Pipeline

This directory contains a set of scripts to download, process, augment, and train YOLO models on a custom spectrogram dataset.

## Pipeline Overview

The entire process is orchestrated by the `download_and_train_yolo.sh` bash script. It automates the following steps:

1.  **Download**: Fetches the specified dataset version from Roboflow.
2.  **Clean**: Reorganizes the downloaded dataset into a structured format.
3.  **Augment**: Creates an augmented version of the dataset with various signal processing techniques.
4.  **Train**: Trains the selected YOLO models on both the cleaned and the augmented datasets for comparison.

---

## Environment Setup

To run this pipeline, you need to set up a Python environment with all the necessary dependencies. You can use either Conda (recommended) or a standard Python virtual environment with pip.

### Using Conda (Recommended)

Conda is the recommended method as it creates a completely isolated environment, which helps prevent conflicts with other Python projects. The `environment.yml` file is provided to automate this process.

1.  **Create the Conda Environment**:
    This command reads the `environment.yml` file, creates a new environment named `yolo`, and installs all the specified packages from both Conda and Pip channels.
    ```bash
    conda env create -f environment.yml
    ```

2.  **Activate the Environment**:
    Before running the scripts, you must activate the newly created environment.
    ```bash
    conda activate yolo
    ```

### Using Pip and a Virtual Environment

If you prefer not to use Conda, you can use `venv` to create a virtual environment and `pip` to install the dependencies from `requirements.txt`.

1.  **Create a Virtual Environment**:
    ```bash
    python -m venv .venv
    ```

2.  **Activate the Environment**:
    -   On **Linux/macOS**:
        ```bash
        source .venv/bin/activate
        ```
    -   On **Windows**:
        ```bash
        .venv\Scripts\activate
        ```

3.  **Install Dependencies**:
    This command installs all the required packages, including PyTorch with GPU support if a compatible CUDA version is detected on your system.
    ```bash
    pip install -r requirements.txt
    ```

---

## Configuration (`download_and_train_yolo.sh`)

All parameters for the pipeline are defined as variables at the top of the `download_and_train_yolo.sh` script. Below is a detailed explanation of each variable.

### General & Paths
-   `VERSION`: The version number of the dataset to download from Roboflow.
-   `CURRENT_DATE`: Automatically gets the current date in `YYYY-MM-DD` format. Used to create a uniquely named folder for each run.
-   `BASE_DATASET_PATH`: The root directory where the new dataset folder will be created (e.g., `../../dataset/RoboFlow-2025-09-11`).
-   `ORIGINAL_DIR`, `CLEANED_DIR`, `AUGMENTED_DIR`: Paths to the subdirectories for the raw, cleaned, and augmented datasets, respectively.
-   `OLD_META_FILE`, `NEW_META_FILE`, `TEMP_META_FILE`: Paths to JSON files containing metadata about the raw signal (`.dat`) files, which is essential for the augmentation script.

### Augmentation Schedule
These parameters control the class balancing feature of the augmentation script.
-   `BASE_AUGMENTATIONS`: The minimum number of times an image from any class will be augmented.
-   `MAX_AUGMENTATIONS`: The maximum number of augmentations for any single image, capping the augmentation for under-represented classes.

### Augmentation Parameters
These variables control the specifics of the signal-based augmentation process.
-   `NUM_AUGMENTATIONS_PER_IMAGE`: The target number of augmented samples to generate for each original image. The final number is determined by the augmentation schedule.
-   `MAX_ABS_SHIFT_HZ`: The maximum absolute frequency shift (in Hz) to apply to a signal.
-   `CENTER_DC`: If `true`, the spectrogram is centered around the DC (0 Hz) component.
-   `MIN_BOX_HEIGHT_RATIO`: The minimum percentage of a bounding box's height that must remain in the frame after a frequency shift for it to be kept.
-   `MIN_SIGNAL_WEIGHT`, `MAX_SIGNAL_WEIGHT`: The weight range for combining two signals during the mixing augmentation. A value of `0.5` means equal contribution.
-   `ADD_SECOND_SAMPLE_PROB`: The probability (0.0 to 1.0) of mixing a second, random signal into the primary signal.
-   `FREQUENCY_SHIFT_PROB`: The probability of applying a frequency shift to a signal.
-   `CHANNEL_EFFECTS_PROB`: The probability of applying channel effects (AWGN noise or fading).
-   `MAX_CHANNEL_EFFECTS`: The maximum number of channel effects to apply to a single sample.
-   `SNR_DB_MIN`, `SNR_DB_MAX`: The range (in dB) for the Signal-to-Noise Ratio when adding AWGN noise.
-   `RICIAN_PROB`: The probability of using Rician fading. If not chosen, Rayleigh fading is used.
-   `RICIAN_K_DB_MIN`, `RICIAN_K_DB_MAX`: The range (in dB) for the Rician K-factor, which determines the severity of the fading.
--   `SEED`: A random seed to ensure that augmentations are reproducible.
-   `VISUALIZE`: If `true`, the script will display a plot showing an example of an augmented spectrogram and its annotations at the end of the process.

### Training Configuration
-   `MODELS_DIR`: The directory where the base pre-trained YOLO model weights (e.g., `yolo11n.pt`) are stored.
-   `MODEL_SELECTION`: An array of YOLO model names to be trained (e.g., `'yolo11n'`, `'yolo11s'`). The script will loop through this array and train each one.
-   `EPOCHS`: The total number of training epochs.
-   `IMG_SIZE`: The image size (in pixels) to which all images will be resized for training.
-   `BATCH_SIZE`: The number of images to process in each batch. Adjust based on your GPU's VRAM.
-   `PATIENCE`: The number of epochs to wait for an improvement in the validation metric before stopping the training early.

---

## Modules

### 1. `download_and_train_yolo.sh`

This is the main entry point for the entire pipeline.

**Functionality:**
-   **Configuration**: All parameters for the pipeline are defined as variables at the top of the script. This includes dataset paths, augmentation parameters, and training hyperparameters.
-   **Execution Flow**: It calls the Python scripts in the correct order, passing the configured parameters as command-line arguments.
-   **Dual Training**: It orchestrates two separate training runs: one on the `cleaned` dataset and another on the `augmented` dataset, saving the results in different experiment folders.

**Usage:**
Simply run the script from your terminal:
```bash
bash download_and_train_yolo.sh
```
You can modify the variables at the top of the script to customize the pipeline.

### 2. `dataset_cleaner.py`

This script processes the raw dataset downloaded from Roboflow.

**Functionality:**
-   **File Renaming & Restructuring**: It renames the image and label files based on their metadata and organizes them into a directory structure grouped by drone model. This makes the dataset more interpretable and easier to work with.
-   **YAML Updates**: It updates the `data.yaml` file to reflect the new directory structure, ensuring compatibility with the YOLO training format.
-   **Logging**: Uses `loguru` for colored, informative logging of the cleaning process.

**Usage (from `download_and_train_yolo.sh`):**
```bash
python dataset_cleaner.py \
    --dataset_dir "$ORIGINAL_DIR" \
    --distenation_dir "$CLEANED_DIR"
```

### 3. `augment_annotations.py`

This script generates a new, augmented dataset from the cleaned one.

**Functionality:**
-   **Signal-Based Augmentation**: Instead of traditional image augmentation, this script operates on the raw signal data (`.dat` files) to create more realistic and diverse spectrograms.
-   **Techniques**:
    -   **Frequency Shifting**: Shifts the signal in the frequency domain.
    -   **Signal Mixing**: Combines two different signal samples.
    -   **Channel Effects**: Adds AWGN (noise) and simulates Rayleigh or Rician fading.
-   **Annotation Handling**: It intelligently recalculates YOLO bounding box annotations to match the applied augmentations, including handling cases where a bounding box wraps around the spectrogram edges.
-   **Balanced Augmentation**: It calculates an "augmentation schedule" to generate more samples for classes that have fewer examples, helping to balance the dataset.
-   **Logging**: Uses `loguru` for detailed progress and debugging information.

**Usage (from `download_and_train_yolo.sh`):**
The script is called with numerous arguments to control the augmentation process, such as shift amounts, noise levels, and probabilities for applying different effects.

### 4. `train_yolo.py`

This script handles the training of the YOLO models.

**Functionality:**
-   **Model Iteration**: It can train multiple YOLO models (e.g., `yolo12x`, `yolo11n`) in a single run.
-   **Hyperparameter Control**: All key training parameters, such as epochs, image size, batch size, and patience, are configurable via command-line arguments.
-   **Clear Logging**: Uses `loguru` to provide clear, colored logs for the start and end of each training session.

**Usage (from `download_and_train_yolo.sh`):**
The script is called twice: once for the cleaned dataset and once for the augmented dataset, with the `project_name` argument pointing to different output directories.
