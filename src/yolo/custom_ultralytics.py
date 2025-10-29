import numpy as np
from ultralytics.utils.checks import LOGGER
from copy import copy
import albumentations as A
import torch
import csv
from pathlib import Path

# Import original classes
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import DetMetrics as OriginalDetMetrics
from ultralytics.utils.torch_utils import strip_optimizer
from ultralytics.utils.instance import Instances


class AlbumentationsWrapper:
    """
    Wrapper for Albumentations transforms to make them compatible with the YOLOv8 pipeline.
    This class handles the conversion between YOLO's `Instances` object and the format
    required by Albumentations.
    """
    def __init__(self):
        self.p = 0.5
        self.transform = A.Compose([
            A.AdvancedBlur(blur_limit=(7, 13), sigma_x_limit=(7, 13), sigma_y_limit=(7, 13), rotate_limit=(-90, 90), beta_limit=(0.5, 8), noise_limit=(2, 10), p=self.p),
            A.CLAHE(clip_limit=3, tile_grid_size=(13, 13), p=self.p),
            A.ColorJitter(brightness=(0.5, 1.5), contrast=(1, 1), saturation=(1, 1), hue=(-0, 0), p=self.p),
            A.GaussNoise(std_range=(0.1, 0.5), mean_range=(0, 0), p=self.p),
            A.ISONoise(intensity=(0.2, 0.5), color_shift=(0.01, 0.05), p=self.p),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1), p=self.p)
        ])

    def __call__(self, data):
        """Applies the augmentations to the image only (no bbox modification needed)."""
        im = data['img']
        
        # Apply image-only transformations (bboxes remain unchanged)
        transformed = self.transform(image=im)
        
        # Update the image, but keep all other data unchanged since these are image-only transforms
        data['img'] = transformed['image']
        return data


class CustomDetMetrics(OriginalDetMetrics):
    """
    A custom detection metrics class that extends the base DetMetrics to include
    mAP at higher IoU thresholds (0.75, 0.90, 0.95) and an average over 0.90-0.95.
    """

    @property
    def map75(self):
        """mAP@0.75 (index 5 in 0.05 steps from 0.5)."""
        return self.box.all_ap[:, 5].mean() if len(self.box.all_ap) else 0.0

    @property
    def map90(self):
        """mAP@0.90 (index 8)."""
        return self.box.all_ap[:, 8].mean() if len(self.box.all_ap) else 0.0

    @property
    def map95(self):
        """mAP@0.95 (index 9)."""
        return self.box.all_ap[:, 9].mean() if len(self.box.all_ap) else 0.0

    @property
    def map90_95(self):
        """Average mAP@0.90-0.95."""
        return self.box.all_ap[:, 8:10].mean() if len(self.box.all_ap) else 0.0

    @property
    def keys(self) -> list:
        """Return a list of keys for accessing specific metrics, including new ones for CSV export."""
        original_keys = super().keys
        return original_keys + ["metrics/mAP75(B)", "metrics/mAP90(B)", "metrics/mAP95(B)", "metrics/mAP90-95(B)"]

    def mean_results(self) -> list:
        """Return mean of results, including the new custom metrics."""
        original_results = super().mean_results()
        return original_results + [self.map75, self.map90, self.map95, self.map90_95]

    def class_result(self, i):
        """Return class-aware results, including AP for new custom metrics."""
        results = list(super().class_result(i))
        ap75 = self.box.all_ap[i, 5]
        ap90 = self.box.all_ap[i, 8]
        ap95 = self.box.all_ap[i, 9]
        ap90_95 = self.box.all_ap[i, 8:10].mean()
        return tuple(results + [ap75, ap90, ap95, ap90_95])

    @property
    def results_dict(self):
        """A dictionary of results formatted for YOLOv8 model logging, including custom metrics."""
        results = super().results_dict
        results.update(
            {
                "metrics/mAP75(B)": self.map75,
                "metrics/mAP90(B)": self.map90,
                "metrics/mAP95(B)": self.map95,
                "metrics/mAP90-95(B)": self.map90_95,
            }
        )
        # Overwrite the default fitness with our custom metric
        results['fitness'] = self.map90_95
        # LOGGER.info(f"Passing fitness to trainer: {results['fitness']:.4f} (from mAP90-95)")
        return results


class CustomValidator(DetectionValidator):
    """
    A custom validator that uses CustomDetMetrics and correctly prints the new results.
    """

    def init_metrics(self, model):
        """Initialize our patched metrics class instead of the original."""
        super().init_metrics(model)
        # Re-initialize our custom metrics, passing along the names from the base validator
        self.metrics = CustomDetMetrics(names=self.names)
        self.metrics.confusion_matrix = self.confusion_matrix
        # Attach the names to the confusion matrix object itself for our plotting patch to use
        self.metrics.confusion_matrix.names = self.names

    def get_desc(self):
        """Return a formatted string for the results table header, including new metrics."""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95",
            "mAP75",
            "mAP90",
            "mAP95",
            "mAP90-95",
        )

    def print_results(self):
        """Prints training/validation set metrics per class with the correct format."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * 8  # 2 for counts, 8 for metrics
        LOGGER.info(pf % ("all", self.seen, self.metrics.nt_per_class.sum(), *self.metrics.mean_results()))

        if self.metrics.nt_per_class.sum() == 0:
            LOGGER.warning(f"WARNING ⚠️ No labels found in {self.args.task} set, can not compute metrics without labels.")
            return

        if self.args.verbose and not self.training and self.nc > 1 and len(self.metrics.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf
                    % (
                        self.names[c],
                        getattr(self.metrics, "nt_per_image", [0])[c] if hasattr(self.metrics, "nt_per_image") else 0,
                        self.metrics.nt_per_class[c],
                        *self.metrics.class_result(i),
                    )
                )
            
            # Save per-class metrics to CSV
            self.save_per_class_metrics()

    def save_per_class_metrics(self):
        """Save per-class metrics to a CSV file."""
        if not self.training and self.nc > 1 and len(self.metrics.stats):
            # Define the CSV file path
            csv_path = Path(self.save_dir) / 'per_class_metrics.csv'
            
            # Prepare data
            rows = []
            
            # Header row
            header = ['Class', 'Images', 'Instances', 'Box(P)', 'R', 'mAP50', 'mAP50-95', 'mAP75', 'mAP90', 'mAP95', 'mAP90-95']
            rows.append(header)
            
            # Overall metrics row
            overall_row = [
                'all',
                self.seen,
                int(self.metrics.nt_per_class.sum()),
                *[f'{x:.3g}' for x in self.metrics.mean_results()]
            ]
            rows.append(overall_row)
            
            # Per-class metrics rows
            for i, c in enumerate(self.metrics.ap_class_index):
                class_row = [
                    self.names[c],
                    getattr(self.metrics, "nt_per_image", [0])[c] if hasattr(self.metrics, "nt_per_image") else 0,
                    int(self.metrics.nt_per_class[c]),
                    *[f'{x:.3g}' for x in self.metrics.class_result(i)]
                ]
                rows.append(class_row)
            
            # Write to CSV
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            
            LOGGER.info(f"Per-class metrics saved to {csv_path}")


class CustomTrainer(DetectionTrainer):
    """
    A custom trainer that uses a custom validator and a custom fitness function.
    """

    def build_dataset(self, img_path, mode='train', batch=None):
        """Build YOLO Dataset with custom augmentations."""
        dataset = super().build_dataset(img_path, mode, batch)
        if mode == 'train':
            # Insert the custom Albumentations wrapper into the transform pipeline.
            # It's inserted before the final formatting transforms.
            # NOTE: This is a fragile approach and may break with ultralytics updates.
            # The Mosaic, CopyPaste, and Format transforms are typically at the end.
            insert_index = -3  # A common position before final formatting
            if len(dataset.transforms.transforms) < abs(insert_index):
                # Handle cases with fewer transforms than expected
                insert_index = len(dataset.transforms.transforms)
            dataset.transforms.transforms.insert(insert_index, AlbumentationsWrapper())
            LOGGER.info(f"Added custom Albumentations to the training pipeline at index {insert_index}.")
        return dataset

    def get_validator(self):
        """Returns a custom validator."""
        return CustomValidator(
            dataloader=self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )


from ultralytics import YOLO as OriginalYOLO

class CustomYOLO(OriginalYOLO):
    """
    A custom YOLO class that uses a custom trainer for the 'detect' task.
    """
    @property
    def task_map(self):
        """A dictionary mapping tasks to models, trainers, validators, and predictors."""
        # Get the original task map
        task_map = super().task_map
        # Override the trainer for the 'detect' task
        task_map['detect']['trainer'] = CustomTrainer
        return task_map
