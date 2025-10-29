"""
A patch to display numerical values within the cells of the confusion matrix plot.
By importing this file, it will automatically patch the ConfusionMatrix.plot()
method in ultralytics.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import seaborn as sn
from ultralytics.utils import metrics as metrics_module
from ultralytics.utils.checks import LOGGER

# Store the original plot method for unpatching
_original_plot = metrics_module.ConfusionMatrix.plot

def patched_plot(self, normalize=True, save_dir='.', names=(), on_plot=None, show_values=True):
    """
    Plots a confusion matrix with numerical values in each cell if show_values is True.
    Filters out classes that have no samples (no predictions and no ground truth).
    """
    if not show_values:
        # Call the original plot method if values are not to be shown
        _original_plot(self, normalize, save_dir, names, on_plot)
        return

    try:
        array = self.matrix.copy()
        
        # Filter out classes with no samples (both rows and columns are all zeros)
        # Keep class if it has any predictions OR any ground truth samples
        row_sums = array.sum(axis=1)  # Ground truth samples per class
        col_sums = array.sum(axis=0)  # Predicted samples per class
        
        # Keep classes that have either ground truth OR predictions (excluding background)
        # Last row/col is typically background, always keep it
        keep_mask = (row_sums > 0) | (col_sums > 0)
        keep_indices = np.where(keep_mask)[0]
        
        # Filter the confusion matrix
        array = array[np.ix_(keep_indices, keep_indices)]
        
        if normalize:
            sums = array.sum(axis=1, keepdims=True)
            # Perform division only where the sum is not zero to avoid warnings
            array = np.divide(array, sums, out=np.zeros_like(array, dtype=float), where=sums != 0)

        # Use the names attached directly to the confusion matrix object, falling back to the argument
        names_to_use = getattr(self, 'names', names)

        # Prepare class names for plotting, ensuring they are sorted by class index
        plot_names = []
        if names_to_use:
            if isinstance(names_to_use, dict):
                # Sort the dictionary by key (class index) to ensure correct order
                all_names = [names_to_use[i] for i in sorted(names_to_use.keys())]
            else:
                all_names = list(names_to_use)

            # Add background class if it's missing and the matrix dimensions expect it
            if len(all_names) == self.matrix.shape[0] - 1:
                all_names.append('background')
            
            # Filter names to match the filtered classes
            plot_names = [all_names[i] for i in keep_indices if i < len(all_names)]

        fig = plt.figure(figsize=(12, 9), tight_layout=True)
        sn.set(font_scale=1.0 if len(plot_names) < 50 else 0.8)
        # Check if we have the correct number of names for the matrix dimensions
        labels = (0 < len(plot_names) < 99) and len(plot_names) == array.shape[0]

        with sn.axes_style('white'):
            ax = sn.heatmap(array,
                            annot=True,
                            annot_kws={'size': 8},
                            fmt='.2f' if normalize else '.0f',
                            cmap='Blues',
                            xticklabels=plot_names if labels else "auto",
                            yticklabels=plot_names if labels else "auto")

        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')

        filename = 'confusion_matrix_normalized.png' if normalize else 'confusion_matrix.png'
        if on_plot:
            on_plot(Path(save_dir) / filename, fig)

        fig.savefig(Path(save_dir) / filename, dpi=200)
        plt.close(fig)

    except Exception as e:
        LOGGER.warning(f"WARNING ⚠️ Error plotting confusion matrix: {e}")

# Apply the patch
def patch_confusion_matrix():
    """Applies the patch to the ultralytics ConfusionMatrix."""
    metrics_module.ConfusionMatrix.plot = patched_plot
    LOGGER.info("✅ Ultralytics ConfusionMatrix patch applied. Plot will now include numerical values.")

def unpatch_confusion_matrix():
    """Restores the original ultrralytics ConfusionMatrix."""
    metrics_module.ConfusionMatrix.plot = _original_plot
    LOGGER.info("↩️ Ultralytics ConfusionMatrix restored to original.")

# Auto-apply the patch on import
patch_confusion_matrix()