# %%
import os
import re
import sys
import yaml
import shutil
from glob import glob
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from PIL import Image
from loguru import logger
from tqdm import tqdm

from metadata import find_first_float_index

# MAYBE TO DO: YOU CAN MAKE THIS OUT OF YAML FILE
MANUFACTURES = {
'MavicAir':    'DJI_MavicAir',       
'Mavic3':      'DJI_Mavic3',       
'Phantom4':    'DJI_Phantom4',
'FPV':         'DJI_FPV',       
'Mavic2Pro':   'DJI_Mavic2Pro',
'MavicPro':    'DJI_MavicPro',
'Mini3':       'DJI_Mini3',
'Inspire2':    'DJI_Inspire2',
'Inspire1':    'DJI_Inspire1',
'PhantomAdv3': 'DJI_PhantomAdv3',
'Tello':       'DJI_Tello',
'F11GIM':      'Ruko_F11GIM',
'EXOII':       'Autel_EXOII',
'Xstar':       'Autel_Xstar',   
'Q500-HD':     'Yuneec_Q500-HD',    
'Anafi':       'Parrot_Anafi',
'HS720E':      'HolyStone_HS720E', 
'KY601S':      'Quadcopter_KY601S',
'HS110G':      'HolyStone_HS110G',
'2':           'Skydio_2',
'X4':          'Hubsan_X4_Air',
'MavicMini4':  'DJI_MavicMini4'
}

# Configure Loguru for colored output
logger.remove()
logger.add(sys.stderr, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")



def load_yaml_file(yaml_path: str) -> dict:
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def save_yaml_file(yaml_path: str, data: dict) -> None:
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)

def get_info_by_filename(filename: str, class_names: dict, filter_classes: set = None) -> dict:
    """
    Extract drone information and filtered annotations from filename.
    
    Expected formats:
    - "2_sample_0_png.rf.4994904fa08fbaf4ac41f2a6b9807f1f.jpg" -> manufacture: 'Skydio', model': '2', 'filename': f"2_sample_0"
    - "Autel_EXOII_10_2457_sample_163_png.rf.496935e7dead28761bc9f638ebd81ca3.jpg" -> manufacture: 'Autel', model': 'EXOII', 'filename': f"Autel_EXOII_10_2457_sample_163"
    - "DJI_Mini3_10_2442_sample_144_png.rf.b11c9fecb332fdf979ecc50d6f41bdd5.jpg" -> manufacture: 'DJI', model': 'Mini3', 'filename': f"DJI_Mini3_10_2442_sample_144"
    - "DJI_Phantom4_10_2442_sample_144_png.rf.b11c9fecb332fdf979ecc50d6f41bdd5.jpg" -> model': 'background', 'filename': f"DJI_Phantom4_10_2442_sample_144"

    Args:
        filename: Path to image file
        class_names: Dict mapping class IDs to drone names (from data.yaml)
        filter_classes: Optional set of class names to keep. If None, keeps all classes.
    
    Returns:
        Dict with 'manufacture', 'model', 'filename', and 'annotations' (list of filtered annotation lines)
        Files with no annotations after filtering are treated as background.
    """
    try:
        ff = os.path.basename(filename)
        # clean roboflow random name assignment
        ff = ff[:ff.find('_png')]
        # remove extension if it exists; Should never happen
        if ff.endswith('.jpg') or filename.endswith('.png'):
            ff = ff[:-4]
        
        annotate_file_path = filename.replace('.jpg', '.txt').replace(f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}')

        if os.path.exists(annotate_file_path):
            with open(annotate_file_path, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                # Empty annotation file - treat as background
                return {'manufacture': 'background', 'model': None, 'filename': ff, 'annotations': []}
            
            # Filter annotations based on class filter
            filtered_lines = []
            for line in lines:
                try:
                    class_id = int(line.split(' ')[0])
                    class_name = class_names[class_id]
                    if filter_classes is None or class_name in filter_classes:
                        filtered_lines.append(line)
                except (ValueError, IndexError, KeyError):
                    logger.warning(f"Skipping malformed annotation line in {annotate_file_path}: {line.strip()}")
                    continue
            
            if not filtered_lines:
                # No annotations left after filtering - treat as background
                return {'manufacture': 'background', 'model': None, 'filename': ff, 'annotations': []}
            
            # Use first remaining annotation to determine drone type
            first_class_id = int(filtered_lines[0].split(' ')[0])
            components = class_names[first_class_id].split('_')
            
            return {
                'manufacture': components[0],
                'model': "_".join(components[1:]),
                'filename': ff,
                'annotations': filtered_lines
            }
        else:
            # No annotation file - treat as background
            return {'manufacture': 'background', 'model': None, 'filename': ff, 'annotations': []}
            
    except Exception as e:
        logger.error(f"Error getting info by filename: {e}")
        logger.error(f"Filename: {filename}")
        raise e

def process_image_file(img_file: str, dataset_dir: str, destination_dir: str, class_names: dict, filter_classes: set = None) -> tuple:
    """
    Process a single image file: extract info, convert to PNG, and save.
    Returns (success: bool, error_msg: str or None)
    """
    try:
        img_info = get_info_by_filename(img_file, class_names, filter_classes)
        temp_path = img_file.replace(dataset_dir, destination_dir)
        temp_path = os.path.dirname(temp_path)
        basename = f"{img_info['filename']}.png"
        
        drone_manufacture = f"{img_info['manufacture']}_{img_info['model']}" if img_info['model'] is not None else "background"
        temp_path = os.path.join(temp_path, drone_manufacture, basename)
        
        # Create directory with race condition handling
        dir_path = os.path.dirname(temp_path)
        try:
            os.makedirs(dir_path, exist_ok=True)
        except (FileExistsError, OSError):
            os.makedirs(dir_path, exist_ok=True)
        
        # Open, convert, and immediately close to reduce memory usage
        with Image.open(img_file) as img:
            # Using a temporary file in the destination to avoid issues with cross-device moves
            temp_img_path = os.path.join(os.path.dirname(temp_path), f"temp_{os.path.basename(temp_path)}")
            img.save(temp_img_path, 'PNG')
        
        shutil.move(temp_img_path, temp_path)
        return (True, None)
    except Exception as e:
        return (False, f"Failed to process image file {img_file}: {e}")

def process_text_file(txt_file: str, dataset_dir: str, destination_dir: str, class_names: dict, filter_classes: set = None, class_id_mapping: dict = None) -> tuple:
    """
    Process a single text file: extract info and write filtered annotations with remapped class IDs.
    Returns (success: bool, error_msg: str or None)
    """
    try:
        txt_info = get_info_by_filename(txt_file, class_names, filter_classes)
        temp_path = txt_file.replace(dataset_dir, destination_dir)
        temp_path = os.path.dirname(temp_path)
        basename = f"{txt_info['filename']}.txt"
        
        drone_manufacture = f"{txt_info['manufacture']}_{txt_info['model']}" if txt_info['model'] is not None else "background"
        temp_path = os.path.join(temp_path, drone_manufacture, basename)
        
        # Create directory with race condition handling
        dir_path = os.path.dirname(temp_path)
        try:
            os.makedirs(dir_path, exist_ok=True)
        except (FileExistsError, OSError):
            os.makedirs(dir_path, exist_ok=True)
        
        # Remap class IDs if mapping is provided
        remapped_annotations = []
        if class_id_mapping:
            for line in txt_info['annotations']:
                parts = line.strip().split()
                if parts:
                    old_class_id = int(parts[0])
                    new_class_id = class_id_mapping.get(old_class_id)
                    if new_class_id is not None:
                        parts[0] = str(new_class_id)
                        remapped_annotations.append(' '.join(parts) + '\n')
        else:
            remapped_annotations = txt_info['annotations']
        
        # Write filtered and remapped annotations to new file
        with open(temp_path, 'w') as f:
            f.writelines(remapped_annotations)
        
        return (True, None)
    except Exception as e:
        return (False, f"Failed to process label file {txt_file}: {e}")

def fix_roboflow_dataset(dataset_dir: str, destination_dir: str, filter_classes: list = None):
    """
    Fix the Roboflow dataset by moving images and labels to images/model_name/sample_index.jpg and labels/model_name/sample_index.txt.
    
    Args:
        dataset_dir: Source directory containing the original Roboflow dataset
        destination_dir: Destination directory for the cleaned dataset
        filter_classes: Optional list of class names to keep. If None, keeps all classes.
    """
    logger.info(f"Starting to fix Roboflow dataset from '{dataset_dir}' to '{destination_dir}'.")
    
    # load yaml data for extra info
    yaml_data_path = os.path.join(dataset_dir, 'data.yaml')
    if not os.path.exists(yaml_data_path):
        logger.error(f"data.yaml not found in {dataset_dir}. Aborting.")
        return
        
    yaml_data = load_yaml_file(yaml_data_path)

    class_names = yaml_data.get('names', {})
    # If class_names is a list, convert it to a dict with indices as keys
    if isinstance(class_names, list):
        class_names = {i: name for i, name in enumerate(class_names)}
    logger.info(f"Loaded {len(class_names)} classes from data.yaml.")
    
    # Convert filter_classes to a set for faster lookup
    filter_classes_set = set(filter_classes) if filter_classes else None
    
    # Create class ID mapping if filtering
    class_id_mapping = None
    filtered_class_names = {}
    if filter_classes_set:
        logger.info(f"Filtering to keep only these classes: {', '.join(sorted(filter_classes_set))}")
        
        # Create mapping from old class IDs to new class IDs
        class_id_mapping = {}
        new_id = 0
        for old_id, class_name in sorted(class_names.items()):
            if class_name in filter_classes_set:
                class_id_mapping[old_id] = new_id
                filtered_class_names[new_id] = class_name
                new_id += 1
        
        logger.info(f"Created class ID mapping: {len(class_id_mapping)} classes kept out of {len(class_names)}")
        logger.info(f"Class ID mapping: {class_id_mapping}")
    else:
        filtered_class_names = class_names

    img_files = glob(f"{dataset_dir}/**/*.jpg", recursive=True)
    txt_files = glob(f"{dataset_dir}/**/*.txt", recursive=True)
    txt_files = [f for f in txt_files if 'README.roboflow.txt' not in f and 'classes.txt' not in f]
    
    # Filter out files in test folder
    img_files = [f for f in img_files if f'{dataset_dir.removesuffix(os.sep)}{os.sep}test{os.sep}' not in f]
    txt_files = [f for f in txt_files if f'{dataset_dir.removesuffix(os.sep)}{os.sep}test{os.sep}' not in f]
    
    logger.info(f"Found {len(img_files)} image files and {len(txt_files)} label files to process (excluding test folder).")

    # Determine number of processes to use - reduce to avoid memory issues
    num_processes = min(12, max(1, os.cpu_count() - 4))  # Use max 4 processes to avoid OOM
    logger.info(f"Using {num_processes} processes for parallel processing.")

    # Process images in parallel using ProcessPoolExecutor (more WSL-compatible)
    process_img_partial = partial(process_image_file, dataset_dir=dataset_dir, destination_dir=destination_dir, class_names=class_names, filter_classes=filter_classes_set)
    
    results = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {executor.submit(process_img_partial, img_file): img_file for img_file in img_files}
        
        for future in tqdm(as_completed(futures), total=len(img_files), desc="Processing images", unit="file"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                img_file = futures[future]
                logger.error(f"Exception processing {img_file}: {e}")
                results.append((False, f"Exception processing {img_file}: {e}"))
    
    # Log any errors from image processing
    img_errors = [err for success, err in results if not success]
    if img_errors:
        for error in img_errors:
            logger.error(error)
    
    # Process text files in parallel
    process_txt_partial = partial(process_text_file, dataset_dir=dataset_dir, destination_dir=destination_dir, class_names=class_names, filter_classes=filter_classes_set, class_id_mapping=class_id_mapping)
    
    results = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {executor.submit(process_txt_partial, txt_file): txt_file for txt_file in txt_files}
        
        for future in tqdm(as_completed(futures), total=len(txt_files), desc="Processing labels", unit="file"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                txt_file = futures[future]
                logger.error(f"Exception processing {txt_file}: {e}")
                results.append((False, f"Exception processing {txt_file}: {e}"))
    
    # Log any errors from text processing
    txt_errors = [err for success, err in results if not success]
    if txt_errors:
        for error in txt_errors:
            logger.error(error)

    logger.info("Updating data.yaml with filtered classes...")
    
    # Update class names in yaml_data
    if filter_classes_set:
        yaml_data['names'] = filtered_class_names
        yaml_data['nc'] = len(filtered_class_names)
        logger.info(f"Updated data.yaml: {yaml_data['nc']} classes")
    
    # rename valid to val
    valid_dir = os.path.join(destination_dir , "valid")
    if os.path.exists(valid_dir):
        logger.info("Renaming 'valid' directory to 'val'.")
        os.rename(valid_dir, os.path.join(destination_dir , "val"))

    # change valid to val in yaml
    if 'val' in yaml_data and os.path.exists(os.path.join(destination_dir , 'val')):
        logger.info("Updating 'valid' to 'val' in data.yaml.")
        yaml_data['val'] = yaml_data.pop('val').replace('valid', 'val')
    elif 'val' in yaml_data:
        logger.warning("'val' key found in data.yaml but 'val' directory does not exist. Please check manually.")
        yaml_data.pop('val', None)
    # if 'train' in yaml_data and not os.path.exists(os.path.join(destination_dir , 'train')):
    #     yaml_data.pop('train', None)
    if 'test' in yaml_data and not os.path.exists(os.path.join(destination_dir , 'test')):
        yaml_data.pop('test', None)
    
    # Save updated yaml
    save_yaml_file(os.path.join(destination_dir , "data.yaml"), yaml_data)
    
    # Copy README if exists
    readme_path = os.path.join(dataset_dir, "README.roboflow.txt")
    if os.path.exists(readme_path):
        shutil.copy(readme_path, os.path.join(destination_dir , "README.roboflow.txt"))
    
    logger.success(f"Dataset cleaning complete. Output at: {destination_dir }")



    

# %%
if __name__ == '__main__':
    base_dir = './dataset/Roboflow-2025-10-12/'
    original_dir = f'{base_dir}original/'
    clean_dir = f'{base_dir}cleaned/'

    # Only load filter classes if using default directory and file exists
    filter_classes = None
    yaml_file = f'{original_dir}/data.yaml'
    if os.path.exists(yaml_file):
        data = yaml.safe_load(open(yaml_file))
        filter_classes = [name for name in data['names'] if not re.search(r'_RC(-[a-z])?$', name)]

    parser = argparse.ArgumentParser(description='Clean and reorganize Roboflow dataset.')
    parser.add_argument('--dataset_dir', type=str, default=original_dir, help='Directory of the original Roboflow dataset.')
    parser.add_argument('--destination_dir', type=str, default=clean_dir, help='Directory to save the cleaned dataset.')
    parser.add_argument('--filter_classes', type=str, nargs='*', default=filter_classes, help='Optional list of class names to keep (e.g., DJI_Mavic3 DJI_Phantom4). If not specified, keeps all classes.')
    args = parser.parse_args()

    fix_roboflow_dataset(args.dataset_dir, args.destination_dir, args.filter_classes)
# %%