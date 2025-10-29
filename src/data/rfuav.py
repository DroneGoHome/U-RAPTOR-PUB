import logging
import coloredlogs
from typing import Optional
import zipfile
import os

from datasets import load_dataset

# Setup colored logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='WARNING', logger=logger, fmt='%(asctime)s %(levelname)s %(message)s')

def download_rfuav_dataset(cache_dir: Optional[str] = None):
    """
    Downloads and loads the kitofrank/RFUAV dataset from Hugging Face.

    Args:
        cache_dir: Optional directory to cache the dataset. If None, uses the default Hugging Face cache.

    Returns:
        The loaded dataset.
    """
    try:
        logger.info("Downloading kitofrank/RFUAV dataset from Hugging Face...")
        dataset = load_dataset("kitofrank/RFUAV", cache_dir=cache_dir)
        logger.info("Successfully downloaded and loaded the dataset.")
        print(dataset)
        return dataset
    except Exception as e:
        logger.error(f"Failed to download or load the dataset: {e}")
        return None

def unzip_file(zip_path: str, extract_to: str):
    """
    Unzips a file to a specified directory.

    Args:
        zip_path: The path to the zip file.
        extract_to: The directory to extract the files to.
    """
    try:
        if not os.path.exists(extract_to):
            os.makedirs(extract_to)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"Successfully unzipped {zip_path} to {extract_to}")
    except zipfile.BadZipFile:
        logger.error(f"Error: {zip_path} is not a zip file or is corrupted.")
    except FileNotFoundError:
        logger.error(f"Error: {zip_path} not found.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    download_rfuav_dataset(cache_dir='../../dataset/rfuav')