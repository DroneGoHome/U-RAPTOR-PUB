# -*- coding: utf-8 -*-
# %%
import os
import json
import shutil
import pickle
import logging
import traceback
import coloredlogs
from glob import glob
from typing import Any, Union, Dict, List, Optional

from tqdm import tqdm  # type: ignore
import numpy as np

from constants import SAMPLING_RATE, FFT_SIZE, CENTER_FREQ, NUM_FFT_SPEC, RECORDING_TIME
from processing import load_dat_file, length_corrector, get_spectrogram, save_spectrogram, get_centered_stft_matrix, get_power_spectrogram_db, get_color_img, add_awgn_noise, save_color_img

# Setup colored logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='WARNING', logger=logger, fmt='%(asctime)s %(levelname)s %(message)s')


def save_to_json(data: Union[Dict[Any, Any], List[Any]], file_path: str) -> None:
    """
    Saves a Python dictionary or list to a JSON file with indentation.

    Args:
        data: The Python dictionary or list to save.
        file_path: The path to the output JSON file.

    Returns:
        None
    """
    try:
        # Open the file in write mode ('w')
        with open(file_path, 'w') as f:
            # Use json.dump to write the data to the file
            # indent=2 makes the JSON file human-readable
            json.dump(data, f, indent=2)
        logger.info(f"Successfully saved data to JSON file: {file_path}")
    except (IOError, TypeError) as e:
        # Handle potential errors during file writing or JSON serialization
        logger.error(f"Error saving data to JSON file {file_path}: {e}")
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred while saving to {file_path}: {e}")

def load_json(file_path: str) -> Optional[Dict[Any, Any]]:
    """
    Loads data from a JSON file.

    Args:
        file_path: The path to the JSON file.

    Returns:
        A Python dictionary containing the loaded data, or None if an error occurs.
    """
    try:
        # Open the file in read mode ('r')
        with open(file_path, 'r') as f:
            # Use json.load to read the data from the file
            loaded_data = json.load(f)
        logger.info(f"Successfully loaded data from JSON file: {file_path}")
        return loaded_data
    except (IOError, json.JSONDecodeError) as e:
        # Handle potential errors during file reading or JSON parsing
        logger.error(f"Error loading JSON file {file_path}: {e}")
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred while loading {file_path}: {e}")
    return None

def load_jsonl(file_path: str) -> List[Any]:
    """
    Loads data from a JSON Lines (.jsonl) file.

    Each line in the file is expected to be a valid JSON object.
    Lines that cannot be parsed as JSON are skipped, and an error message is printed.

    Args:
        file_path: The path to the JSON Lines file.

    Returns:
        A list containing the Python objects decoded from each line of the file.
        Returns an empty list if the file is not found or if an error occurs during file reading.
    """
    data: List[Any] = []  # Initialize an empty list to store the loaded data
    try:
        # Open the file in read mode ('r')
        with open(file_path, 'r') as f:
            # Iterate over each line in the file
            for line_number, line in enumerate(f, 1):
                try:
                    # Remove leading/trailing whitespace and parse the JSON string
                    stripped_line = line.strip()
                    if stripped_line: # Avoid trying to parse empty lines
                        data.append(json.loads(stripped_line))
                except json.JSONDecodeError as e:
                    # Handle errors if a line is not valid JSON
                    logger.error(f"Error decoding JSON from line {line_number} in {file_path}: '{stripped_line}' - {e}")
                except Exception as e: # Catch other potential errors during line processing
                    logger.error(f"An unexpected error occurred processing line {line_number} in {file_path} ('{stripped_line}'): {e}")
        logger.info(f"Successfully loaded {len(data)} entries from JSONL file: {file_path}")
    except FileNotFoundError:
        # Handle the case where the file does not exist
        logger.error(f"Error: File not found at {file_path}")
        return [] # Return empty list as the file couldn't be opened
    except IOError as e:
        # Handle potential I/O errors during file reading (e.g., permission issues)
        logger.error(f"Error reading file {file_path}: {e}")
        return [] # Return empty list as there was an issue reading the file
    except Exception as e:
        # Catch any other unexpected errors during file opening or iteration
        logger.error(f"An unexpected error occurred while loading {file_path}: {e}")
        return [] # Return empty list in case of other errors
    return data

def load_pickle(file_path: str) -> Optional[Any]:
    """
    Loads data from a pickle file.

    Args:
        file_path: The path to the pickle file.

    Returns:
        The Python object loaded from the pickle file, or None if an error occurs.
    """
    try:
        # Open the file in binary read mode ('rb')
        with open(file_path, 'rb') as f:
            # Use pickle.load to read the data from the file
            loaded_data = pickle.load(f)
        logger.info(f"Successfully loaded data from pickle file: {file_path}")
        return loaded_data
    except (IOError, pickle.UnpicklingError) as e:
        # Handle potential errors during file reading or unpickling
        logger.error(f"Error loading pickle file {file_path}: {e}")
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred while loading {file_path}: {e}")
    return None

def save_to_jsonl(data: Dict[Any, Any], file_path: str) -> None:
    """
    Appends a Python dictionary as a JSON string to a file, one object per line (JSON Lines format).

    This function is designed for efficiently logging or saving data where each entry
    is a separate JSON object on its own line. It creates the file if it doesn't exist.

    Args:
        data: The Python dictionary to append to the file.
        file_path: The path to the target JSON Lines file.

    Returns:
        None

    Raises:
        Prints an error message to the console if issues occur during JSON serialization
        or file writing (e.g., permission errors, invalid data types).
    """
    try:
        # Convert the Python dictionary to a JSON formatted string.
        # No indentation is used as each entry is on a single line.
        json_string = json.dumps(data)
        # Open the file in append mode ('a').
        # This creates the file if it doesn't exist, otherwise appends to the end.
        # 'utf-8' encoding is commonly used and recommended for text files.
        with open(file_path, 'a', encoding='utf-8') as f:
            # Write the JSON string followed by a newline character
            # to ensure each JSON object is on its own line.
            f.write(json_string + '\n')
        # logger.info(f"Successfully appended data to JSONL file: {file_path}") # Potentially too verbose for append
    except TypeError as e:
        # Handle errors specifically related to JSON serialization
        # (e.g., data contains non-serializable types).
        logger.error(f"Error serializing data to JSON for file {file_path}: {e}")
    except IOError as e:
        # Handle errors related to file system operations
        # (e.g., file permissions, disk full).
        logger.error(f"Error writing to JSONL file {file_path}: {e}")
    except Exception as e:
        # Catch any other unexpected errors during the process.
        logger.error(f"An unexpected error occurred while saving to {file_path}: {e}")

def save_processed_data(data: Any, file_path: str) -> None:
    """
    Saves arbitrary Python data to a file using pickle serialization.

    Args:
        data: The Python object to be saved. Can be any object that
              the pickle module can handle.
        file_path: The path to the file where the data will be saved.
                   The file will be overwritten if it already exists.

    Returns:
        None

    Raises:
        Prints an error message to the console if issues occur during
        pickling or file writing (e.g., permission errors, disk full,
        unpicklable object).
    """
    try:
        # Open the file in binary write mode ('wb').
        # Pickle requires binary mode for serialization.
        with open(file_path, 'wb') as f:
            # Use pickle.dump to serialize the data object and write it to the file.
            pickle.dump(data, f)
        logger.info(f"Successfully saved data to pickle file: {file_path}")
    except (IOError, pickle.PicklingError) as e:
        # Handle potential errors during file writing (IOError)
        # or data serialization (PicklingError).
        logger.error(f"Error saving data to pickle file {file_path}: {e}")
    except Exception as e:
        # Catch any other unexpected errors during the process.
        logger.error(f"An unexpected error occurred while saving to {file_path}: {e}")


def save_numpy_array_compressed(array: np.ndarray, file_path: str, array_name: str = 'data') -> None:
    """
    Saves a NumPy array to a .npz file using compression for smaller file size.

    Args:
        array: The NumPy array to save.
        file_path: The path to the output .npz file. The .npz extension
                   will be added if not present, or it's good practice to include it.
        array_name: The name under which the array will be saved within the .npz file.
                    Defaults to 'data'. This name is used when loading the array.

    Returns:
        None

    Raises:
        Prints an error message to the console if issues occur during saving.
    """
    file_path_npz = file_path
    try:
        # Ensure the filepath ends with .npz for clarity, though savez_compressed handles it
        if not file_path.endswith('.npz'):
            file_path_npz = file_path + '.npz'
        
        # np.savez_compressed saves the array in a compressed zip format.
        # You can save multiple arrays by passing them as keyword arguments.
        np.savez_compressed(file_path_npz, **{array_name: array})
        logger.info(f"Successfully saved compressed NumPy array to {file_path_npz}")
    except IOError as e:
        logger.error(f"Error saving compressed NumPy array to {file_path_npz}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving compressed NumPy array to {file_path_npz}: {e}")

def load_numpy_array_compressed(file_path: str, array_name: str = 'data') -> Optional[np.ndarray]:
    """
    Loads a NumPy array from a .npz file that was saved with np.savez_compressed.

    Args:
        file_path: The path to the .npz file.
        array_name: The name under which the array was saved. Defaults to 'data'.

    Returns:
        The loaded NumPy array, or None if an error occurs.
    """
    file_path_npz = file_path
    try:
        if not file_path.endswith('.npz'):
             file_path_npz = file_path + '.npz'
        
        with np.load(file_path_npz) as loaded_data:
            if array_name in loaded_data:
                logger.info(f"Successfully loaded compressed NumPy array '{array_name}' from {file_path_npz}")
                return loaded_data[array_name]
            else:
                logger.error(f"Error: Array name '{array_name}' not found in {file_path_npz}. Available names: {list(loaded_data.keys())}")
                return None
    except FileNotFoundError:
        logger.error(f"Error: File not found at {file_path_npz}")
        return None
    except IOError as e:
        logger.error(f"Error loading NumPy array from {file_path_npz}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading NumPy array from {file_path_npz}: {e}")
        return None

def create_folder(folder_path: str, verbose: bool = True, overwrite: bool = False) -> None:
    """
    Creates a folder at the specified path, optionally overwriting if it exists.

    Args:
        folder_path: The path where the folder should be created.
        verbose: If True, print messages about actions taken (default: True).
        overwrite: If True and the folder exists, remove the existing folder
                   and its contents before creating the new one. (default: False).

    Returns:
        None

    Raises:
        Prints an error message to the console if issues occur during
        directory creation or deletion (e.g., permission errors).
    """
    try:
        # Check if the target path already exists
        if os.path.isdir(folder_path):
            if verbose:
                logger.info(f"Path already exists: {folder_path}")

            # Handle the case where the folder exists
            if not overwrite:
                if verbose:
                    logger.info("Overwrite is False. Skipping creation.")
                return  # Exit if folder exists and overwrite is False

            # If overwrite is True, attempt to remove the existing directory and its contents
            if verbose:
                logger.info(f"Overwrite is True. Attempting to remove existing folder: {folder_path}")
            try:
                # Use shutil.rmtree to recursively remove the directory
                shutil.rmtree(folder_path)
                if verbose:
                    logger.info(f"Successfully removed existing folder: {folder_path}")
            except OSError as e:
                # Handle errors specifically during the removal process (e.g., permissions)
                logger.error(f"Error removing existing folder {folder_path}: {e}")
                return # Stop execution if removal fails
            except Exception as e:
                # Catch other unexpected errors during removal
                logger.error(f"An unexpected error occurred while removing {folder_path}: {e}")
                return # Stop execution if removal fails

        # Create the folder (and any necessary parent directories)
        # This part runs if the folder didn't exist initially, or if it existed and was successfully removed
        os.makedirs(folder_path)
        if verbose:
            logger.info(f"Successfully created folder: {folder_path}")

    except OSError as e:
        # Handle errors during os.makedirs (e.g., permission denied)
        # This will catch errors if makedirs fails after a successful rmtree or if the path didn't exist initially
        logger.error(f"Error creating folder {folder_path}: {e}")
    except Exception as e:
        # Catch any other unexpected errors during the process (e.g., issues with os.path.exists)
        logger.error(f"An unexpected error occurred while processing folder {folder_path}: {e}")

def get_starting_index(folder_path: str, pattern: str) -> int:
    """
    Determines the next available integer index for a file pattern within a specified folder.

    This function checks for existing files matching the pattern '{pattern}_{index}'
    (e.g., 'sample_0', 'sample_1', etc.) in the given folder_path. It increments
    the index until it finds an index for which no corresponding file exists.

    Args:
        folder_path: The path to the directory where files are checked.
        pattern: The base name pattern for the files (e.g., 'sample').

    Returns:
        The first integer index `i` such that the file path
        `os.path.join(folder_path, f"{pattern}_{i}")` does not exist.
        Returns 0 if no files matching the pattern exist yet.

    Raises:
        Prints an error message to the console if an OSError occurs during
        path checking (e.g., invalid path, permission errors) or if any
        other unexpected exception occurs. In case of an error, it might
        return 0 as a default, although the primary goal is to report the error.
    """
    index = 0
    try:
        png_in_folder = glob(os.path.join(folder_path, "*.png"))
        patterns_in_folder = [path[path.find(pattern):] for path in png_in_folder]
        # Construct the potential file path using the current index
        pattern_to_check = f"{pattern}_{index}.png" # Check against a common extension like .png

        # Loop while a file with the current index exists
        # Note: 
        # For simplicity, we check for a common extension like .png as an example.
        # If multiple file types (.pkl, .png) use the same index, checking one is usually sufficient.
        while pattern_to_check in patterns_in_folder:
            index += 1 # Increment the index to check the next number
            # Update the file path for the next iteration
            pattern_to_check = f"{pattern}_{index}.png"

    except OSError as e:
        # Handle potential OS-level errors during path checking (e.g., invalid path format, permissions)
        logger.error(f"Error checking file existence in {folder_path} for pattern {pattern}: {e}")
        # Depending on requirements, you might want to return a specific error code (e.g., -1)
        # or re-raise the exception. Returning 0 might be misleading.
        # For now, we print the error and return the index found so far (which might be 0).
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred in get_starting_index: {e}")
        # Similar to OSError, consider the best way to signal an error.

    # Return the first index for which a file was not found
    return index

def get_drone_model(file_path: str) -> Optional[str]:
    """
    Extracts the drone model from the filename.

    Assumes the filename format is like 'Manufacture_Model_Bandwidth_Freq_Mode.ext'.
    The model is expected to be the second part when split by '_'.

    Args:
        file_path: The full path to the file.

    Returns:
        The extracted drone model as a string, or None if parsing fails
        (e.g., filename doesn't contain enough '_' separators).
    """
    try:
        # Get the base name of the file (e.g., 'file.txt' from '/path/to/file.txt')
        base_name = os.path.basename(file_path)
        # Split the base name by the underscore character
        parts = base_name.split('_')
        # The model is expected to be the second element (index 1)
        if len(parts) > 1:
            return parts[1]
        else:
            logger.warning(f"Could not extract drone model from filename (not enough parts): {base_name}")
            return None
    except IndexError:
        # This handles cases where splitting might occur but the expected index is out of bounds
        # Although the length check above should prevent this specific error here.
        logger.warning(f"Index error while extracting drone model from filename: {file_path}")
        return None
    except Exception as e:
        # Catch any other unexpected errors during processing
        logger.error(f"An unexpected error occurred in get_drone_model for {file_path}: {e}")
        return None

def get_drone_manufacture(file_path: str) -> Optional[str]:
    """
    Extracts the drone manufacturer from the filename.

    Assumes the filename format is like 'Manufacture_Model_Bandwidth_Freq_Mode.ext'.
    The manufacturer is expected to be the first part when split by '_'.

    Args:
        file_path: The full path to the file.

    Returns:
        The extracted drone manufacturer as a string, or None if parsing fails
        (e.g., filename is empty or doesn't contain '_').
    """
    try:
        # Get the base name of the file
        base_name = os.path.basename(file_path)
        # Split the base name by the underscore character
        parts = base_name.split('_')
        # The manufacturer is expected to be the first element (index 0)
        if len(parts) > 0:
            return parts[0]
        else:
            # This case is unlikely if basename returns a non-empty string, but good practice.
            logger.warning(f"Could not extract drone manufacturer from filename (no parts): {base_name}")
            return None
    except IndexError:
        # Should not happen with index 0 if parts list is not empty, but included for safety.
        logger.warning(f"Index error while extracting drone manufacturer from filename: {file_path}")
        return None
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred in get_drone_manufacture for {file_path}: {e}")
        return None

def get_drone_bandwidth(file_path: str) -> Optional[str]:
    """
    Extracts the drone bandwidth information from the filename.

    Assumes the filename format is like 'Manufacture_Model_Bandwidth_Freq_Mode.ext'.
    The bandwidth is expected to be the third part when split by '_'.

    Args:
        file_path: The full path to the file.

    Returns:
        The extracted drone bandwidth as a string, or None if parsing fails
        (e.g., filename doesn't contain enough '_' separators).
    """
    try:
        # Get the base name of the file
        base_name = os.path.basename(file_path)
        # Split the base name by the underscore character
        parts = base_name.split('_')
        # The bandwidth is expected to be the third element (index 2)
        if len(parts) > 2:
            try:
                return int(parts[2])
            except ValueError:
                return parts[2]  # If conversion to int fails, return the string part
        else:
            logger.warning(f"Could not extract drone bandwidth from filename (not enough parts): {base_name}")
            return None
    except IndexError:
        logger.warning(f"Index error while extracting drone bandwidth from filename: {file_path}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred in get_drone_bandwidth for {file_path}: {e}")
        return None

def get_drone_center_freq(file_path: str) -> Optional[float]:
    """
    Extracts the drone center frequency from the filename and converts it to a float.

    Assumes the filename format is like 'Manufacture_Model_Bandwidth_Freq_Mode.ext'.
    The frequency is expected to be the fourth part when split by '_' and should be numeric.

    Args:
        file_path: The full path to the file.

    Returns:
        The extracted drone center frequency as a float, or None if parsing fails
        (e.g., filename doesn't contain enough '_' separators, or the frequency part is not a valid number).
    """
    try:
        # Get the base name of the file
        base_name = os.path.basename(file_path)
        # Split the base name by the underscore character
        parts = base_name.split('_')
        # The frequency is expected to be the fourth element (index 3)
        if len(parts) > 3:
            # Attempt to convert the frequency part to a float (handles both int and float values)
            part = parts[3].replace('.dat', '')
            return float(part)
        else:
            logger.warning(f"Could not extract drone center frequency from filename (not enough parts): {base_name}")
            return None
    except IndexError:
        logger.warning(f"Index error while extracting drone center frequency from filename: {file_path}")
        return None
    except ValueError:
        # Handle cases where the frequency part is not a valid number
        logger.warning(f"Could not convert center frequency part to number in filename: {file_path}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred in get_drone_center_freq for {file_path}: {e}")
        return None

def get_drone_operation_mode(file_path: str) -> Optional[str]:
    """
    Extracts the drone operation mode from the filename.

    Assumes the filename format is like 'Manufacture_Model_Bandwidth_Freq_Mode1_Mode2.ext'.
    The operation mode consists of all parts from the fifth onwards (split by '_'),
    joined by '-', after removing the file extension.

    Args:
        file_path: The full path to the file.

    Returns:
        The extracted drone operation mode as a string, or None if parsing fails
        (e.g., filename doesn't contain enough '_' separators or has no extension).
        Returns an empty string if there are no mode parts after the frequency.
    """
    try:
        # Get the base name of the file
        base_name = os.path.basename(file_path)
        # Split the filename from its extension
        name_part, _ = os.path.splitext(base_name)
        # Split the name part by the underscore character
        parts = name_part.split('_')
        # The operation mode parts start from the fifth element (index 4)
        if len(parts) >= 4:
            # Join the remaining parts with a hyphen
            return '-'.join(parts[4:])
        else:
            # Handle case where there are fewer than 4 parts before the extension
            logger.warning(f"Could not extract drone operation mode from filename (not enough parts): {base_name}")
            return None
    except IndexError:
        # This might occur if splitext fails unexpectedly, though unlikely.
        logger.warning(f"Index error while extracting drone operation mode from filename: {file_path}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred in get_drone_operation_mode for {file_path}: {e}")
        return None

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

def get_drone_info(file_path: str, base_path: str) -> Dict[str, Optional[Union[str, int]]]:
    """
    Aggregates drone information extracted from the filename into a dictionary.

    Calls individual getter functions to parse different parts of the filename.
    Handles potential None return values from getter functions if parsing fails for a part.

    Args:
        file_path: The full path to the file.

    Returns:
        A dictionary containing the extracted drone information:
        - 'dat_file_path': The input file path.
        - 'model': Drone model (str) or None.
        - 'manufacture': Drone manufacturer (str) or None.
        - 'bandwidth': Drone bandwidth info (str) or None.
        - 'center_freq': Drone center frequency (float) or None.
        - 'operation_mode': Drone operation mode (str) or None.
    """
    sub_folders = file_path.replace(base_path, '').split(os.path.sep)

    if len(sub_folders) > 2:
        logger.warning(f"Unexpected folder structure in file path: {file_path}")
        return {}

    infos = os.path.basename(file_path)[:-4].split('_')
    first_index = find_first_float_index(infos)
    
    if sub_folders[0].lower() == 'non-drone':
        # Handle non-drone sub-folder case
        drone_info: Dict[str, Optional[Union[str, int]]] = {
        'dat_file_path': file_path,
        'model': '_'.join(infos[:first_index]),
        'manufacture': '',
        'label': "_".join(infos[:first_index]),
        'bandwidth': infos[first_index],
        'center_freq': infos[first_index+1],
        'operation_mode': '_'.join(infos[first_index+2:])
    }
        return drone_info

    elif sub_folders[0].lower() == 'multi-drone':
        # Handle multi-drone sub-folder case
        drone_info: Dict[str, Optional[Union[str, int]]] = {
        'dat_file_path': file_path,
        'model': '_'.join(infos[:first_index]),
        'manufacture': '',
        'label': "_".join(infos[:first_index]),
        'bandwidth': '_'.join(infos[first_index:first_index+2]),
        'center_freq': '_'.join(infos[first_index+2:first_index+4]),
        'operation_mode': '_'.join(infos[first_index+4:])
    }
        return drone_info


    # Call each getter function and store the results
    # The dictionary values can be None if the respective getter fails
    drone_info: Dict[str, Optional[Union[str, int]]] = {
        'dat_file_path': os.path.abspath(file_path),
        'model': get_drone_model(file_path),
        'manufacture': get_drone_manufacture(file_path),
        'label': "_".join(infos[:first_index]),
        'bandwidth': get_drone_bandwidth(file_path),
        'center_freq': get_drone_center_freq(file_path),
        'operation_mode': get_drone_operation_mode(file_path)
    }

    if drone_info['label'] == 'Skydio':
        # Special case for Skydio, where the model is not in the filename
        drone_info['label'] = 'Skydio_2'
    return drone_info

def get_samples_from_recording(
    signal: np.ndarray,
    sampling_rate: int = SAMPLING_RATE,
    sample_time: float = 1.0,
    step_time: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Generates overlapping samples (windows) from a signal using a sliding window approach.

    Args:
        signal: The input signal as a NumPy array.
        sampling_rate: The sampling rate of the signal in Hz. Defaults to SAMPLING_RATE.
        sample_time: The duration of each sample window in seconds. Defaults to 1.0.
        step_time: The time step (overlap) between consecutive windows in seconds.
                   Defaults to 0.5 (meaning 50% overlap if sample_time is 1.0).

    Returns:
        A dictionary where keys are strings representing the time interval
        (e.g., "0.0-1.0", "0.5-1.5") and values are the corresponding
        NumPy array segments (samples) from the signal. Returns an empty
        dictionary if an error occurs or if the signal is too short for
        even one sample.

    Raises:
        Prints an error message to the console if any exception occurs during processing.
    """
    samples: Dict[str, np.ndarray] = {}
    try:
        # Calculate the number of data points in each sample window
        sample_size: int = int(sample_time * sampling_rate)
        # Calculate the number of data points to slide the window forward for the next sample
        window_step_size: int = int(step_time * sampling_rate)

        # Ensure window step size is at least 1 to avoid infinite loops
        if window_step_size <= 0:
            logger.error("step_time must result in a window step size greater than 0.")
            return {}
        # Ensure sample size is valid
        if sample_size <= 0:
            logger.error("sample_time must result in a sample size greater than 0.")
            return {}
        # Check if the signal is long enough for at least one sample
        if signal.size < sample_size:
            logger.warning("Signal length is smaller than the requested sample size.")
            return {}

        # Iterate through the signal with a sliding window
        # The loop starts at index 0 and increments by window_step_size
        # It stops when the remaining signal is shorter than sample_size
        for i in range(0, signal.size - sample_size + 1, window_step_size):
            # Calculate the start and end time for the current window for the dictionary key
            start_t = i / sampling_rate
            end_t = start_t + sample_time
            # Format the key string
            time_key = f"{start_t:.3f}-{end_t:.3f}" # Using .3f for milliseconds precision

            # Extract the signal segment (sample)
            sample_segment = signal[i : i + sample_size]
            # Store the sample in the dictionary
            samples[time_key] = sample_segment

    except TypeError as e:
        logger.error(f"Type error during sample generation. Check input types. Details: {e}")
        return {} # Return empty dict on type error
    except AttributeError as e:
        logger.error(f"Attribute error during sample generation. Is 'signal' a NumPy array? Details: {e}")
        return {} # Return empty dict on attribute error
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred during sample generation: {e}")
        # Optionally, re-raise the exception if needed: raise e
        return {} # Return empty dict on other errors

    return samples

def save_samples_from_recording(
    signal: np.ndarray,
    save_path: str,
    drone_info: Dict[str, Any],
    file_pattern: str = 'sample',
    tmp_file: str = 'tmp.json',
    snr_db: Optional[float] = None,
) -> None:
    """
    Processes a signal, extracts samples, saves each sample and its spectrogram,
    and logs metadata for each sample to a JSON Lines file.

    This function iterates through the input signal using a sliding window defined
    by 'sample_size' and 'window_size' (step) found within the `drone_info`
    dictionary. For each window (sample):
    1. The raw signal sample is saved as a .pkl file.
    2. The sample is potentially truncated based on FFT parameters.
    3. A spectrogram is generated and saved as a .png file.
    4. Metadata about the sample (including file paths, time interval, and
       original drone info) is appended to a temporary JSON Lines file.

    Args:
        signal: The input signal as a NumPy array.
        save_path: The directory path where the sample (.pkl) and
                   spectrogram (.png) files will be saved.
        drone_info: A dictionary containing metadata about the drone recording
                    and processing parameters. Expected keys include:
                    'sample_size' (int): Number of data points per sample.
                    'window_size' (int): Step size for the sliding window.
                    'fft_size' (int): FFT size for spectrogram generation.
                    'num_fft_spec' (int): Number of FFTs for spectrogram.
                    'sampling_rate' (int): Signal sampling rate.
                    'center_freq' (float): Center frequency for spectrogram.
                    'sample_time' (float): Duration of each sample in seconds.
                    'step_time' (float): Time step between samples in seconds.
                    (Other keys from get_drone_info are also expected).
        file_pattern: The base name pattern for the output files (e.g., 'sample').
                      Files will be named '{pattern}_{index}.pkl' and
                      '{pattern}_{index}.png'. Defaults to 'sample'.
        tmp_file: The path to the temporary JSON Lines file where metadata for
                  each processed sample will be appended. Defaults to 'tmp.json'.

    Returns:
        None. Files are saved to disk, and metadata is appended to `tmp_file`.

    Raises:
        Prints error messages to the console if:
        - Required keys are missing from `drone_info`.
        - Errors occur during file I/O (saving .pkl or .png).
        - Errors occur during signal processing (spectrogram generation).
        - Errors occur during metadata serialization or writing to the JSONL file.
    """
    try:
        # Retrieve necessary parameters from drone_info, handling potential KeyErrors
        sample_size: int = drone_info['sample_size']
        window_size: int = drone_info['window_size']
        fft_size: int = drone_info['fft_size']
        num_fft_spec: int = drone_info['num_fft_spec']
        sampling_rate: int = drone_info['sampling_rate']
        center_freq: int = drone_info['center_freq']

        # Ensure window size is positive to prevent infinite loops
        if window_size <= 0:
            logger.error(f"Invalid 'window_size' ({window_size}) in drone_info. Must be positive.")
            return
        # Ensure sample size is positive
        if sample_size <= 0:
            logger.error(f"Invalid 'sample_size' ({sample_size}) in drone_info. Must be positive.")
            return
        # Check if signal is long enough for at least one sample
        if signal.size < sample_size:
            logger.warning(f"Signal length ({signal.size}) is smaller than sample size ({sample_size}). No samples generated.")
            return

    except KeyError as e:
        logger.error(f"Missing required key in 'drone_info' dictionary: {e}")
        return
    except Exception as e:
        logger.error(f"Error retrieving parameters from drone_info: {e}")
        return

    # Determine the starting index for file naming to avoid overwriting existing files
    try:
        index: int = get_starting_index(folder_path=save_path, pattern=file_pattern)
    except Exception as e:
        logger.error(f"Error determining starting index in {save_path}: {e}")
        return # Cannot proceed without a valid starting index

    # Iterate through the signal using a sliding window
    # The loop stops when the remaining signal is shorter than sample_size
    for i in range(0, signal.size - sample_size + 1, window_size):
        current_index = index # Store index for this iteration in case of errors
        file_name_base = os.path.join(save_path, f'{os.path.basename(drone_info["dat_file_path"])[:-4]}_{file_pattern}_{current_index}')

        try:
            # 1. Extract the raw signal sample
            signal_sample = np.array(signal[i : i + sample_size])

            # 2. Prepare sample for spectrogram (length correction/truncation)
            # Correct length if needed (though length_corrector might need review based on its exact purpose)
            corrected_sample = length_corrector(signal_sample, fft_size)
            # Truncate the sample to the specified number of FFTs if needed
            if corrected_sample.size > num_fft_spec*fft_size:
                corrected_sample = corrected_sample[:num_fft_spec*fft_size]
        
            # Add complex Gaussian noise based on SNR
            if snr_db is not None:
                gaussian_noise = add_awgn_noise(corrected_sample, snr_db)
            else:
                gaussian_noise = np.zeros_like(corrected_sample)

            spectrogram_input = corrected_sample + gaussian_noise
    
            # 3. Generate and save the spectrogram
            # spectrum, fig = get_spectrogram(spectrogram_input, Fs=sampling_rate, NFFT=fft_size, Fc=center_freq)
            stft_matrix = get_centered_stft_matrix(spectrogram_input, sampling_rate=sampling_rate, fft_size=fft_size)
            spectrum = get_power_spectrogram_db(stft_matrix, sampling_rate=sampling_rate, center_freq=center_freq, window_size=fft_size, overlap=128)
            rgb_image = get_color_img(spectrum, colormap='viridis')
            save_color_img(rgb_image, filename=f'{file_name_base}.png', dpi=300, verbose=False, overwrite=False)
            # save_numpy_array_compressed(stft_matrix, f'{file_name_base}.npz') # Save the spectrogram data as a .pkl file
            if snr_db is not None:
                # Generate and save the noise spectrogram
                # spectrum_noise, fig_noise = get_spectrogram(gaussian_noise, Fs=sampling_rate, NFFT=fft_size, Fc=center_freq)

                stft_matrix_noise = get_centered_stft_matrix(gaussian_noise, sampling_rate=sampling_rate, fft_size=fft_size)
                spectrum_noise = get_power_spectrogram_db(stft_matrix_noise, sampling_rate=sampling_rate, center_freq=center_freq, window_size=fft_size, overlap=128)
                rgb_image_noise = get_color_img(spectrum_noise, colormap='viridis')
                save_color_img(rgb_image_noise, filename=f'{file_name_base}_noise.png', dpi=300, verbose=False, overwrite=False)
                # save_numpy_array_compressed(stft_matrix_noise, f'{file_name_base}_noise.npz') # Save the spectrogram data as a .pkl file

        except (IOError, pickle.PicklingError) as e:
            logger.error(f"Error saving data for sample {current_index} (path: {file_name_base}): {e}")
            # continue # ? Decide if you want to skip this sample or not
        except Exception as e:
            # Catch other potential errors during processing (e.g., in get_spectrogram, length_corrector)
            logger.error(f"Error processing sample {current_index} (path: {file_name_base}): {e}")
            logger.error(traceback.format_exc()) # Print detailed traceback for debugging
            # continue # ? Decide if you want to skip this sample or not

        finally:
            # 4. Log metadata regardless of processing success (unless 'continue' was used above)
            # Create a copy to avoid modifying the original drone_info dict
            index_drone_info = drone_info.copy()
            # Add sample-specific information
            index_drone_info['index'] = current_index
            index_drone_info['start_index'] = i
            index_drone_info['end_index'] = i + sample_size
            # Calculate time based on sample index 'i' and sampling rate/times
            # Note: The original calculation seemed off. Using 'i' (start index) and sampling rate is more direct.
            start_time_sec = i / sampling_rate
            end_time_sec = (i + sample_size) / sampling_rate # End time of the sample window
            index_drone_info['time_start_sec'] = start_time_sec
            index_drone_info['time_end_sec'] = end_time_sec
            # Store relative paths for portability
            # index_drone_info['signal_path'] = os.path.relpath(f'{file_name_base}.pkl', os.path.dirname(tmp_file)) if os.path.dirname(tmp_file) else f'{file_name_base}.pkl'
            index_drone_info['spect_path'] = os.path.relpath(f'{file_name_base}.png', os.path.dirname(tmp_file)) if os.path.dirname(tmp_file) else f'{file_name_base}.png'

            try:
                # Append the metadata for this sample to the JSON Lines file
                save_to_jsonl(index_drone_info, tmp_file)
            except Exception as e:
            # Catch errors during JSONL saving
                logger.error(f"Failed to save metadata for sample {current_index} to {tmp_file}: {e}")

            # Increment index for the next file, only after successfully processing (or attempting) the current one
            index += 1

def make_dataset(
    base_path: str = '../data/Relate work from AeroDefense/Data/Raw data/SR20M_G50_cage_RT15/',
    patterns: List[str] = ['*.dat', os.path.join("*", "*.dat")],
    dataset_path: str = '../data/Related work from Rowan/dataset/',
    sampling_rate: int = SAMPLING_RATE,
    sample_time: float = 1.0,
    step_time: float = 0.5,
    fft_size: int = FFT_SIZE,
    num_fft_spec: int = NUM_FFT_SPEC,
    snr_db: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Processes raw signal data files (.dat) to create a structured dataset of signal samples and spectrograms.

    This function searches for .dat files within the specified `base_path` using the provided `patterns`.
    For each found file, it:
    1. Loads the signal data.
    2. Extracts drone metadata from the filename.
    3. Calculates processing parameters (sample size, window size).
    4. Creates a subdirectory within `dataset_path` named after the drone model.
    5. Calls `save_samples_from_recording` to extract, process, and save signal samples
       and their corresponding spectrograms into the model-specific subdirectory.
       Metadata for each sample is temporarily stored in a JSON Lines file ('tmp.json').
    6. After processing all files, it loads the aggregated metadata from 'tmp.json'
       and saves it into a final 'meta_data.json' file in the `dataset_path`.
    7. Removes the temporary 'tmp.json' file.

    Args:
        base_path: The root directory containing the raw .dat files or subdirectories with .dat files.
        patterns: A list of glob patterns to find .dat files relative to `base_path`.
        dataset_path: The root directory where the processed dataset (samples, spectrograms, metadata) will be saved.
        sampling_rate: The sampling rate of the signal data in Hz.
        sample_time: The duration of each extracted sample in seconds.
        step_time: The time step (overlap) between consecutive samples in seconds.
        fft_size: The FFT size used for spectrogram generation.
        num_fft_spec: The number of FFT segments to include in each spectrogram.

    Returns:
        A list of dictionaries, where each dictionary contains the metadata for one processed sample,
        loaded from the final 'meta_data.json'. Returns an empty list if no files are processed or
        if critical errors occur during metadata loading/saving.

    Raises:
        Prints error messages to the console for issues like:
        - Failure to create directories.
        - Errors loading .dat files.
        - Errors during sample processing or saving.
        - Errors reading/writing metadata files.
    """
    # TO DO: resume from where it left off if interrupted
    # Construct full paths for glob patterns
    path_patterns: List[str] = [os.path.join(base_path, pattern) for pattern in patterns]
    dat_files: List[str] = []

    # Find all .dat files matching the patterns
    try:
        for pattern in path_patterns:
            found_files = glob(pattern, recursive=True) # Use recursive=True if patterns include '**'
            dat_files.extend(found_files)
        if not dat_files:
            logger.warning(f"No .dat files found matching patterns in {base_path}. Exiting.")
            return []
        logger.info(f"Found {len(dat_files)} .dat files to process.")
    except Exception as e:
        logger.error(f"Error finding .dat files using patterns in {base_path}: {e}")
        return []

    # Ensure the main dataset directory exists
    try:
        create_folder(folder_path=dataset_path, verbose=True, overwrite=False) # create_folder might have its own logging
    except Exception as e:
        logger.error(f"Error creating base dataset directory {dataset_path}: {e}. Cannot proceed.")
        return []

    # Define the path for the temporary metadata file
    tmp_metadata_file = os.path.join(dataset_path, 'tmp.json')
    # Ensure the temporary file does not exist from a previous failed run
    # if os.path.exists(tmp_metadata_file): # ? Decide if you want to remove it or not
    #     try:
    #         os.remove(tmp_metadata_file)
    #         logger.info(f"Removed existing temporary metadata file: {tmp_metadata_file}")
    #     except OSError as e:
    #         logger.warning(f"Could not remove existing temporary file {tmp_metadata_file}: {e}")
    #         # Depending on requirements, might want to exit here

    total_iter = len(dat_files)
    processed_files_count = 0

    # Process each found .dat file
    for i, dat_file in tqdm(enumerate(dat_files), total=total_iter, desc="Processing files", unit="file"):
        try:
            # logger.info(f"\nProcessing file {i+1}/{total_iter}: {dat_file}")
            # Load the signal data from the .dat file
            signal = load_dat_file(dat_file)
            if signal is None or signal.size == 0:
                logger.warning(f"Skipping file {dat_file} due to loading error or empty signal.")
                continue

            # Extract drone information from the filename
            drone_info = get_drone_info(dat_file, base_path=base_path)
            if drone_info.get('model') is None: # Check if model extraction failed
                logger.warning(f"Skipping file {dat_file} because drone model could not be determined from filename.")
                continue

            # Add processing parameters to the drone_info dictionary
            drone_info['sampling_rate'] = sampling_rate
            drone_info['sample_time'] = sample_time
            drone_info['step_time'] = step_time
            drone_info['fft_size'] = fft_size
            drone_info['num_fft_spec'] = num_fft_spec
            # Calculate sample size and step size in data points
            drone_info['sample_size'] = int(sample_time * sampling_rate)
            drone_info['window_size'] = int(step_time * sampling_rate)

            # Define and create the save path specific to the drone model
            save_path = os.path.join(dataset_path, str(drone_info['label'])) # Ensure model is string for path
            create_folder(folder_path=save_path, verbose=False, overwrite=False) # verbose=False to reduce console clutter

            # Process the signal: extract samples, generate spectrograms, save files, and log metadata
            save_samples_from_recording(
                signal,
                save_path,
                drone_info,
                file_pattern='sample',
                tmp_file=tmp_metadata_file,
                snr_db=snr_db,
            )
            processed_files_count += 1

        except FileNotFoundError:
            logger.error(f"File not found during processing: {dat_file}. Skipping.")
        except IOError as e:
            logger.error(f"Error reading or writing file during processing of {dat_file}: {e}. Skipping.")
        except ValueError as e:
            logger.error(f"Error processing data (e.g., invalid format) for {dat_file}: {e}. Skipping.")
        except Exception as e:
            # Catch any other unexpected errors during the processing of a single file
            logger.error(f"An unexpected error occurred while processing file {dat_file}: {e}")
            logger.exception("Traceback for the error above:") # Print traceback for debugging
            # Decide whether to continue with the next file or stop

    logger.info(f"Finished processing {processed_files_count} out of {total_iter} files.")

    # Finalize metadata: load from tmp file and save to final JSON file
    final_metadata_file = os.path.join(dataset_path, 'meta_data.json')
    meta_data: List[Dict[str, Any]] = []
    if os.path.exists(tmp_metadata_file):
        try:
            logger.info(f"Loading metadata from temporary file: {tmp_metadata_file}")
            meta_data = load_jsonl(tmp_metadata_file) # Assuming load_jsonl uses logger or is fine as is
            if meta_data:
                logger.info(f"Saving final metadata ({len(meta_data)} entries) to: {final_metadata_file}")
                save_to_json(meta_data, final_metadata_file) # Assuming save_to_json uses logger or is fine
            else:
                logger.warning("Temporary metadata file was empty or failed to load.")

            # Clean up the temporary file
            try:
                os.remove(tmp_metadata_file)
                logger.info(f"Removed temporary metadata file: {tmp_metadata_file}")
            except OSError as e:
                logger.warning(f"Could not remove temporary metadata file {tmp_metadata_file}: {e}")

        except Exception as e:
            logger.error(f"Error finalizing metadata from {tmp_metadata_file} to {final_metadata_file}: {e}")
            # Keep tmp file in case of error for inspection
    else:
        logger.warning("Temporary metadata file not found. No metadata saved.")

    # Return the loaded metadata (might be empty if errors occurred)
    return meta_data

def extract_center_freq_from_filename(file_path: str) -> int:
    """
    Parses a filename to extract an integer associated with 'Freq' or 'NoDrone'.

    Args:
        file_path: The full path to the file.

    Returns:
        An integer if a numeric part associated with 'Freq' or 'NoDrone'
        is found, otherwise None.
    """
    base_name = os.path.basename(file_path)
    name_without_ext = os.path.splitext(base_name)[0]
    
    parts = name_without_ext.split('_')
    
    for part_str in parts:
        if 'Freq' in part_str:
            numeric_candidate_str = part_str.replace('Freq', '')
            try:
                return int(numeric_candidate_str)
            except ValueError:
                # If 'Freq' is present but not followed by a valid int,
                # continue searching in other parts.
                pass
        elif 'NoDrone' in part_str:
            numeric_candidate_str = part_str.replace('NoDrone', '')
            try:
                return int(numeric_candidate_str)
            except ValueError:
                # If 'NoDrone' is present but not followed by a valid int (e.g. "NoDroneABC" or just "NoDrone"),
                # continue searching in other parts.
                pass
                
    return None

def make_dataset_for_no_drone(
    dat_files: List[str],
    dataset_path: str = '../data/Related work from Rowan/dataset/',
    sampling_rate: int = SAMPLING_RATE,
    sample_time: float = 1.0,
    step_time: float = 0.5,
    fft_size: int = FFT_SIZE,
    num_fft_spec: int = NUM_FFT_SPEC,
    snr_db: Optional[float] = None,):

    # Ensure the main dataset directory exists
    try:
        create_folder(folder_path=dataset_path, verbose=True, overwrite=False) # verbose=True can be handled by create_folder's own logging
    except Exception as e:
        logger.error(f"Error creating base dataset directory {dataset_path}: {e}. Cannot proceed.")
        return []

    # Define the path for the temporary metadata file
    tmp_metadata_file = os.path.join(dataset_path, 'no_drone_tmp.jsonl')
    total_iter = len(dat_files)
    processed_files_count = 0

    # Process each found .dat file
    for i, dat_file in tqdm(enumerate(dat_files), total=total_iter, desc="Processing files", unit="file"):
        try:
            # logger.info(f"\nProcessing file {i+1}/{total_iter}: {dat_file}") # Optional: for very verbose logging
            # Load the signal data from the .dat file
            signal = load_dat_file(dat_file)
            if signal is None or signal.size == 0:
                logger.warning(f"Skipping file {dat_file} due to loading error or empty signal.")
                continue

            # Extract drone information from the filename
            drone_info = {'dat_file_path': dat_file,
                        'model': 'no_drone',
                        'manufacture': None,
                        'bandwidth': None,
                        'center_freq': extract_center_freq_from_filename(dat_file),
                        'operation_mode': None}

            # Add processing parameters to the drone_info dictionary
            drone_info['sampling_rate'] = sampling_rate
            drone_info['sample_time'] = sample_time
            drone_info['step_time'] = step_time
            drone_info['fft_size'] = fft_size
            drone_info['num_fft_spec'] = num_fft_spec
            # Calculate sample size and step size in data points
            drone_info['sample_size'] = int(sample_time * sampling_rate)
            drone_info['window_size'] = int(step_time * sampling_rate)

            # Define and create the save path specific to the drone model
            save_path = os.path.join(dataset_path, str(drone_info['model'])) # Ensure model is string for path
            create_folder(folder_path=save_path, verbose=False, overwrite=False) # verbose=False to reduce console clutter

            # Process the signal: extract samples, generate spectrograms, save files, and log metadata
            save_samples_from_recording(
                signal,
                save_path,
                drone_info,
                file_pattern='sample',
                tmp_file=tmp_metadata_file,
                snr_db=snr_db,
            )
            processed_files_count += 1

        except FileNotFoundError:
            logger.error(f"File not found during processing: {dat_file}. Skipping.")
        except IOError as e:
            logger.error(f"Error reading or writing file during processing of {dat_file}: {e}. Skipping.")
        except ValueError as e:
            logger.error(f"Error processing data (e.g., invalid format) for {dat_file}: {e}. Skipping.")
        except Exception as e:
            # Catch any other unexpected errors during the processing of a single file
            logger.error(f"An unexpected error occurred while processing file {dat_file}: {e}")
            logger.exception("Traceback for the error above:") # This will include the traceback
            # Decide whether to continue with the next file or stop

    logger.info(f"Finished processing {processed_files_count} out of {total_iter} files.")

    # Finalize "no drone" metadata: load from tmp file and save to a dedicated JSON file
    no_drone_specific_metadata_filepath = os.path.join(dataset_path, 'no_drone_meta_data.json')
    no_drone_entries: List[Dict[str, Any]] = [] # Will hold metadata for "no drone" files

    # tmp_metadata_file is 'no_drone_tmp.jsonl' as defined earlier in make_dataset_for_no_drone
    if os.path.exists(tmp_metadata_file):
        try:
            logger.info(f"Loading 'no drone' metadata from temporary file: {tmp_metadata_file}")
            no_drone_entries = load_jsonl(tmp_metadata_file)
            if no_drone_entries:
                logger.info(f"Saving 'no drone' specific metadata ({len(no_drone_entries)} entries) to: {no_drone_specific_metadata_filepath}")
                save_to_json(no_drone_entries, no_drone_specific_metadata_filepath)
            else:
                logger.warning(f"Temporary 'no drone' metadata file ({tmp_metadata_file}) was empty or failed to load.")

            # Clean up the temporary "no drone" metadata file
            try:
                os.remove(tmp_metadata_file)
                logger.info(f"Removed temporary 'no drone' metadata file: {tmp_metadata_file}")
            except OSError as e:
                logger.warning(f"Could not remove temporary 'no drone' metadata file {tmp_metadata_file}: {e}")

        except Exception as e:
            logger.error(f"Error finalizing 'no drone' metadata from {tmp_metadata_file} to {no_drone_specific_metadata_filepath}: {e}")
            # Consider keeping tmp_metadata_file for inspection if saving to no_drone_specific_metadata_filepath failed
    else:
        logger.warning(f"Temporary 'no drone' metadata file ({tmp_metadata_file}) not found. No 'no drone' specific metadata saved.")
        
    # Define file paths for combining metadata
    # Source file for general metadata (e.g., from drone recordings processed by make_dataset)
    general_metadata_source_filepath = os.path.join(dataset_path, 'meta_data.json')
    # Destination file for all combined metadata
    combined_all_metadata_filepath = os.path.join(dataset_path, 'all_meta_data.json')

    # 1. Load existing general metadata (e.g., from drone files)
    general_metadata_entries: List[Dict[str, Any]] = []
    try:
        logger.info(f"Loading general metadata from: {general_metadata_source_filepath}")
        # Assuming load_json is a function you have that might also use logging or print
        # For this example, I'll keep the direct file open, but ideally load_json would be used.
        with open(general_metadata_source_filepath, 'r') as f:
            loaded_content = json.load(f) # Make sure json is imported
        
        if isinstance(loaded_content, list):
            general_metadata_entries = loaded_content
            logger.info(f"Loaded {len(general_metadata_entries)} entries from {general_metadata_source_filepath}.")
        else:
            logger.warning(f"Data in {general_metadata_source_filepath} is not a list. Initializing general metadata as an empty list.")
    except FileNotFoundError:
        logger.warning(f"General metadata file {general_metadata_source_filepath} not found. Assuming no prior general metadata.")
    except json.JSONDecodeError:
        logger.warning(f"Could not decode JSON from {general_metadata_source_filepath}. Assuming no prior general metadata and starting with an empty list.")

    # 2. Combine the loaded general metadata with the "no drone" metadata
    # 'no_drone_entries' contains the metadata for "no drone" files processed in this function.
    # 'general_metadata_entries' contains metadata from other sources (e.g., drone files from 'meta_data.json').
    combined_entries = general_metadata_entries + no_drone_entries # Creates a new list

    # 3. Save the combined data to the final 'all_meta_data.json' file
    logger.info(f"Saving combined metadata ({len(combined_entries)} entries) to: {combined_all_metadata_filepath}")
    save_to_json(combined_entries, combined_all_metadata_filepath) # Assuming save_to_json is available
    logger.info(f"Successfully saved combined metadata to '{combined_all_metadata_filepath}'.")
    # More detailed summary if desired:
    # logger.info(f"Combined {len(general_metadata_entries)} general entries and {len(no_drone_entries)} 'no drone' entries into '{combined_all_metadata_filepath}'.")

    # Return the combined metadata list
    return combined_entries


def main(snr_range,
         dataset_path='../../dataset/unannotated_cage_dataset/',
         base_path='../../dataset/Relate work from AeroDefense/Data/Raw data/SR20M_G50_cage_RT15/',
         patterns=[os.path.join("**", "*.dat")],
         sampling_rate=SAMPLING_RATE,
         sample_time=0.5,
         step_time=0.1,
         fft_size=FFT_SIZE,
         num_fft_spec=NUM_FFT_SPEC,
        add_original=True):

    snr_list = [None] if add_original else []  # Include original data if specified
    snr_list.extend(list(snr_range))
    for snr_db in snr_list:
        logger.info(f"Creating dataset with SNR: {snr_db} dB")
        # Create the dataset path if it doesn't exist
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
            logger.info(f"Created dataset directory: {dataset_path}")
        else:
            logger.info(f"Dataset directory already exists: {dataset_path}")

        snr_dataset_path = os.path.join(dataset_path, f'{snr_db}db' if snr_db is not None else 'original')
        if not os.path.exists(snr_dataset_path):
            os.makedirs(snr_dataset_path)
            logger.info(f"Created SNR dataset directory: {snr_dataset_path}")
        else:
            logger.info(f"SNR dataset directory already exists: {snr_dataset_path}")
        
        # Create the dataset with the specified SNR
        dataset = make_dataset(
            base_path=base_path,
            patterns=patterns,
            dataset_path=snr_dataset_path,
            sampling_rate=sampling_rate,
            sample_time=sample_time,
            step_time=step_time,
            fft_size=fft_size,
            num_fft_spec=num_fft_spec,
            snr_db=snr_db,
        )

        # no_drone_files = [
        #     '../../data/Relate work from AeroDefense/Data/Raw data/NoDrone_indoor/NoDrone_Band24GHz_Freq2442_SampRate20MHz_ReqTime30sec-002.dat',
        #     '../../data/Relate work from AeroDefense/Data/Raw data/NoDrone_indoor/NoDrone_Band900MHz_Freq918_SampRate20MHz_ReqTime30sec-002.dat',
        #     '../../data/Relate work from AeroDefense/Data/Raw data/NoDrone_outdoor/NoDrone905_SampRate20MHz.dat',
        #     '../../data/Relate work from AeroDefense/Data/Raw data/NoDrone_outdoor/NoDrone2442_SampRate20MHz.dat'
        # ]

        # dataset = make_dataset_for_no_drone(
        #     dat_files=no_drone_files,
        #     dataset_path=snr_dataset_path,
        #     sampling_rate=sampling_rate,
        #     sample_time=sample_time,
        #     step_time=step_time,
        #     fft_size=fft_size,
        #     num_fft_spec=num_fft_spec,
        #     snr_db=snr_db,
        # )

        logger.info(f"Dataset created for {snr_db}db with {len(dataset)} entries.")


# %%
if __name__ == "__main__":
    # Define the SNR range for dataset creation
    # snr_range = range(-20, 21, 5)  # Example: from -20 dB to 20 dB in steps of 5 dB
    snr_range = []
    dataset_path='../../dataset/SiteSurveyRecord/2025-10-07/'
    base_path='../../dataset/SiteSurveyRecord/2025-10-07/'
    # Call the main function to create the dataset
    main(snr_range,
         dataset_path=dataset_path,
         base_path=base_path, 
         add_original=True)  # Set add_original=True to include the original data without noise
# %%
