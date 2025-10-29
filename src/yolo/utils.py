import cv2
import numpy as np
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from loguru import logger
from ultralytics.engine.results import Results

def draw_detections(image: np.ndarray, results: Results) -> np.ndarray:
    """
    Draws bounding boxes and labels on an image from detection results.
    This function uses a two-pass approach to ensure labels are drawn on top
    of all boxes and assigns a unique, deterministic color to each class.

    Args:
        image (np.ndarray): The image to draw on.
        results (Results): The detection results from an Ultralytics model.

    Returns:
        np.ndarray: The image with detections drawn on it.
    """
    annotated_image = image.copy()
    boxes = results.boxes
    names = results.names
    img_h, img_w, _ = annotated_image.shape

    # --- 1. Define a list of deterministic, distinct colors ---
    # Using a larger, more distinct color palette
    colors = [
        (255, 56, 56), (255, 157, 151), (255, 112, 255), (178, 0, 255),
        (0, 102, 255), (0, 204, 255), (51, 255, 153), (102, 255, 0),
        (255, 255, 0), (255, 153, 0), (255, 102, 0), (153, 76, 0)
    ]

    if boxes is not None and len(boxes) > 0:
        sorted_indices = np.argsort(boxes.xyxy[:, 1].cpu().numpy())
        
        # --- 2. First Pass: Draw all bounding boxes ---
        for i in sorted_indices:
            box = boxes[i]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_index = int(box.cls[0])
            color = colors[class_index % len(colors)]
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

        # --- 3. Second Pass: Draw all labels with smart placement ---
        drawn_label_boxes = []
        for i in sorted_indices:
            box = boxes[i]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_index = int(box.cls[0])
            class_name = names[class_index]
            confidence = float(box.conf[0])
            color = colors[class_index % len(colors)]

            label = f"{class_name} {confidence:.2f}"
            if hasattr(boxes, 'probs') and boxes.probs is not None:
                class_probability = boxes.probs[i][class_index].item()
                label += f" P:{class_probability:.2f}"

            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            print(label)

            # --- Robust Label Placement Logic ---
            def is_valid(bbox, drawn_boxes):
                x1, y1, x2, y2 = bbox
                if not (x1 >= 0 and y1 >= 0 and x2 <= img_w and y2 <= img_h): return False
                for drawn in drawn_boxes:
                    if not (x2<drawn[0] or x1>drawn[2] or y2<drawn[1] or y1>drawn[3]): return False
                return True

            above_bbox = (x1, y1-label_height-baseline, x1+label_width, y1)
            below_bbox = (x1, y2, x1+label_width, y2+label_height+baseline)

            if is_valid(above_bbox, drawn_label_boxes):
                pos = (x1, y1 - baseline)
                bg_pos = ((x1, y1-label_height-baseline), (x1+label_width, y1))
                drawn_label_boxes.append(above_bbox)
            elif is_valid(below_bbox, drawn_label_boxes):
                pos = (x1, y2 + label_height)
                bg_pos = ((x1, y2), (x1 + label_width, y2 + label_height + baseline))
                drawn_label_boxes.append(below_bbox)
            else: # Fallback: inside
                pos = (x1 + 2, y1 + label_height)
                bg_pos = ((x1, y1), (x1 + label_width, y1 + label_height + baseline))

            cv2.rectangle(annotated_image, bg_pos[0], bg_pos[1], color, cv2.FILLED)
            cv2.putText(annotated_image, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # This code builds on the previous snippet
    if boxes is not None and boxes.format == 'xyxy_conf_cls_classconf':
        print("\n--- Detailed Class Probabilities per Detection ---")
        for i, box_data in enumerate(boxes.data):
            predicted_class_index = int(box_data[5])
            print(f"\nDetection {i+1} (Predicted: '{names[predicted_class_index]}'):")
            
            # The class probabilities start from column index 6
            all_class_probs = box_data[6 : 6 + boxes.num_classes]
            
            for cls_idx, prob in enumerate(all_class_probs):
                class_name = names[cls_idx]
                print(f"  - {class_name}: {prob:.4f}")

    return annotated_image


def copy_file_pair(src_img: str, dest_img: str, src_label: str = None, dest_label: str = None) -> Tuple[bool, Optional[str]]:
    """
    Copy an image file and optionally its corresponding label file.
    
    Args:
        src_img: Source image file path
        dest_img: Destination image file path
        src_label: Optional source label file path
        dest_label: Optional destination label file path
        
    Returns:
        Tuple of (success: bool, error_msg: str or None)
    """
    try:
        # Create destination directory if needed
        os.makedirs(os.path.dirname(dest_img), exist_ok=True)
        
        # Copy image
        shutil.copy2(src_img, dest_img)
        
        # Copy label if provided
        if src_label and dest_label:
            os.makedirs(os.path.dirname(dest_label), exist_ok=True)
            shutil.copy2(src_label, dest_label)
        
        return True, None
    except Exception as e:
        return False, f"Error copying {src_img}: {str(e)}"


def parallel_copy_files(file_pairs: List[Tuple[str, str]], 
                       label_pairs: List[Tuple[str, str]] = None,
                       max_workers: int = 8,
                       desc: str = "Copying files") -> List[str]:
    """
    Copy multiple files in parallel using ThreadPoolExecutor.
    
    Args:
        file_pairs: List of (source, destination) tuples for image files
        label_pairs: Optional list of (source, destination) tuples for label files (must match file_pairs length)
        max_workers: Number of parallel workers (default: 8)
        desc: Description for progress bar
        
    Returns:
        List of error messages (empty if all successful)
    """
    if label_pairs and len(file_pairs) != len(label_pairs):
        raise ValueError("file_pairs and label_pairs must have the same length")
    
    # Prepare copy tasks
    copy_tasks = []
    for i, (src_img, dest_img) in enumerate(file_pairs):
        if label_pairs:
            src_label, dest_label = label_pairs[i]
            copy_tasks.append((src_img, dest_img, src_label, dest_label))
        else:
            copy_tasks.append((src_img, dest_img, None, None))
    
    # Execute copies in parallel
    errors = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(copy_file_pair, src_img, dest_img, src_label, dest_label): (src_img, dest_img)
            for src_img, dest_img, src_label, dest_label in copy_tasks
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="file"):
            success, error = future.result()
            if not success:
                errors.append(error)
    
    if errors:
        logger.warning(f"Encountered {len(errors)} errors during parallel copying")
        for error in errors[:5]:  # Log first 5 errors
            logger.error(error)
        if len(errors) > 5:
            logger.error(f"... and {len(errors) - 5} more errors")
    
    return errors


def parallel_copy_directory(src_dir: str, dest_dir: str, max_workers: int = 8) -> List[str]:
    """
    Recursively copy entire directory tree in parallel.
    
    Args:
        src_dir: Source directory path
        dest_dir: Destination directory path
        max_workers: Number of parallel workers (default: 8)
        
    Returns:
        List of error messages (empty if all successful)
    """
    logger.info(f"Copying directory tree from {src_dir} to {dest_dir}...")
    
    # Create destination directory
    os.makedirs(dest_dir, exist_ok=True)
    
    # Collect all files to copy
    file_pairs = []
    for root, dirs, files in os.walk(src_dir):
        # Create corresponding directory structure
        rel_path = os.path.relpath(root, src_dir)
        dest_root = os.path.join(dest_dir, rel_path) if rel_path != '.' else dest_dir
        
        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_root, file)
            file_pairs.append((src_file, dest_file))
    
    logger.info(f"Found {len(file_pairs)} files to copy")
    
    # Copy all files in parallel
    return parallel_copy_files(file_pairs, max_workers=max_workers, desc="Copying directory")
