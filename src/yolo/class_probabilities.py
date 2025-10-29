"""
Patch Ultralytics to add class probabilities support. This patch modifies the
detection post-processing pipeline to retain and expose the likelihood scores
for all classes for each bounding box.
"""
import time
from typing import List, Dict

import torch
import torchvision

import ultralytics.engine.results
import ultralytics.utils.ops
from ultralytics.utils.checks import LOGGER
from ultralytics.models.yolo.detect import predict as detect_predict
from ultralytics.engine.results import Results, Boxes, BaseTensor
from ultralytics.models.yolo.detect.predict import DetectionPredictor

# Store original methods and properties to allow for unpatching
_original_Boxes_init = Boxes.__init__
_original_Boxes_conf = Boxes.conf
_original_Boxes_cls = Boxes.cls
_original_postprocess = DetectionPredictor.postprocess
_original_construct_result = DetectionPredictor.construct_result
# Check if nms_rotated exists, otherwise provide an alternative implementation or placeholder
try:
    from ultralytics.utils.ops import nms_rotated
except ImportError:
    try:
        from ultralytics.utils.metrics import nms_rotated
    except ImportError:
        # Simple placeholder to avoid errors if not used
        def nms_rotated(*args, **kwargs):
            LOGGER.warning("nms_rotated not available in this version of ultralytics")
            return torch.zeros(0, device=args[0].device).long() if args else None

def init(self, boxes, orig_shape) -> None:
    """
    Initialize the Boxes class with detection box data and the original image shape.

    Args:
        boxes (torch.Tensor | np.ndarray): A tensor or numpy array with detection boxes.
            Shape can be (num_boxes, 6), (num_boxes, 7), or (num_boxes, 6 + num_classes).
            Columns should contain [x1, y1, x2, y2, confidence, class, (optional) track_id, (optional) class_conf_1, class_conf_2, ...].
        orig_shape (tuple): The original image shape as (height, width). Used for normalization.

    Returns:
        (None)
    """

    if boxes.ndim == 1:
        boxes = boxes[None, :]
    n = boxes.shape[-1]
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.from_numpy(boxes)

    # Bypassing the original Boxes.__init__ which has a strict assertion for columns.
    # We call the parent's __init__ directly to support our custom box format.
    BaseTensor.__init__(self, boxes, orig_shape)
    self.is_track = False
    self.num_classes = 0

    if n == 6:
        self.format = 'xyxy_conf_cls'
    elif n == 7:
        self.format = 'xyxy_conf_cls_track'
        self.is_track = True
    else:
        self.format = 'xyxy_conf_cls_classconf'
        self.num_classes = n - 6

@property
def patched_conf(self):
    """
    Return the confidence of the box. If class probabilities are available,
    it returns the maximum probability.
    """
    if self.format == 'xyxy_conf_cls_classconf' and self.num_classes > 0:
        conf, _ = torch.max(self.data[:, 6 : 6 + self.num_classes], dim=1)
        return conf
    return self.data[:, 4]

@property
def patched_cls(self):
    """
    Return the class of the box. If class probabilities are available,
    it returns the class with the maximum probability.
    """
    if self.format == 'xyxy_conf_cls_classconf' and self.num_classes > 0:
        _, j = torch.max(self.data[:, 6 : 6 + self.num_classes], dim=1)
        return j.float()
    return self.data[:, 5]

def postprocess(self, preds, img, orig_imgs):
    """    
    This patched postprocess method ensures that the custom non_max_suppression
    function is called, which retains all class probabilities.
    """
    preds = non_max_suppression(preds,
                                self.args.conf,
                                self.args.iou,
                                agnostic=self.args.agnostic_nms,
                                max_det=self.args.max_det,
                                classes=self.args.classes)

    results = []
    for i, pred in enumerate(preds):
        orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
        if not isinstance(orig_img, torch.Tensor):
            pred[:, :4] = ultralytics.utils.ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        path = self.batch[0]
        img_path = path[i] if isinstance(path, list) else path
        results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
    return results

def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,  # number of classes (optional)
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
    in_place=True,
    rotated=False,
    end2end=False,
    return_idxs=False,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.
    This version returns confidences for all classes.
    """
    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    # Create a copy to avoid modifying the original prediction tensor, which can cause issues in training loops
    prediction = prediction.clone()

    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    # End-to-end models have a different output format, handle them separately
    if end2end or prediction.shape[-1] == 6:
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        if return_idxs:
            return output, [torch.arange(len(o), device=o.device) for o in output]
        return output

    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    extra = prediction.shape[1] - nc - 4  # number of extra columns
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    if not rotated:
        # This is an in-place operation, but we already cloned prediction, so it's safe
        prediction[..., :4] = ultralytics.utils.ops.xywh2xyxy(prediction[..., :4])

    t = time.time()
    output = [torch.zeros((0, 6 + nc + extra), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, preds
        # Apply constraints
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + extra + 4), device=x.device)
            v[:, :4] = ultralytics.utils.ops.xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx(6 + nc + extra) (xyxy, conf, cls, cls_confs, masks)
        box, cls_conf, mask = x.split((4, nc, extra), 1)

        if multi_label:
            i, j = torch.where(cls_conf > conf_thres)
            # This part is complex and may not be needed for your use case.
            # For now, we focus on the single-label case which is more common.
            # The line below is a placeholder for multi-label logic.
            x = torch.cat((box[i], cls_conf[i, j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls_conf.max(1, keepdim=True)
            # Create the final output tensor with all class probabilities
            x = torch.cat((box, conf, j.float(), cls_conf, mask), 1)

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == classes).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores
        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
            i = nms_rotated(boxes, scores, iou_thres)
        else:
            boxes = x[:, :4] + c  # boxes (offset by class)
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        output[xi] = x[i]

        if (time.time() - t) > time_limit:
            LOGGER.warning(f"NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    # This patch does not support `return_idxs`
    if return_idxs:
        LOGGER.warning("`return_idxs=True` is not supported with this patch. Returning empty indices.")
        return output, [torch.zeros(0, device=prediction.device)] * bs

    return output

def construct_result(self, pred, img, orig_img, img_path):
    """Construct a single Results object from one image prediction."""
    pred[:, :4] = ultralytics.utils.ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
    return Results(orig_img, path=img_path, names=self.model.names, boxes=pred)

@property
def sorted_probs(self):
    """
    Returns the class probabilities sorted in descending order for each detection,
    along with their original class indices.
    """
    if self.format == 'xyxy_conf_cls_classconf' and self.num_classes > 0:
        all_class_probs = self.data[:, 6 : 6 + self.num_classes]
        return torch.sort(all_class_probs, dim=1, descending=True)
    return None, None

@property
def probs(self):
    """
    Returns the raw class probabilities for each detection.
    """
    if self.format == 'xyxy_conf_cls_classconf' and self.num_classes > 0:
        return self.data[:, 6 : 6 + self.num_classes]
    return None

def to_dict_list(self, names: Dict[int, str]) -> List[Dict[str, float]]:
    """
    Returns a list of dictionaries with class probabilities for each detection.
    """
    if self.probs is not None:
        all_detections_probs = []
        for p in self.probs:  # self.probs is the tensor from the property we just added
            class_probs = {}
            for cls_idx, prob in enumerate(p):
                class_name = names[cls_idx]
                class_probs[class_name] = prob.item()  # .item() gets the float value
            all_detections_probs.append(class_probs)
        return all_detections_probs
    return []

# --- Patching and Unpatching Functions ---
def patch_class_probabilities():
    """Applies the patch to enable class probability extraction."""
    Boxes.__init__ = init
    Boxes.conf = patched_conf
    Boxes.cls = patched_cls
    DetectionPredictor.postprocess = postprocess
    DetectionPredictor.construct_result = construct_result
    # Add new properties to the Boxes class
    setattr(Boxes, 'sorted_probs', sorted_probs)
    setattr(Boxes, 'probs', probs)
    setattr(Boxes, 'to_dict_list', to_dict_list)
    LOGGER.info("✅ Class probabilities patch applied. Test with model.predict() to verify.")

def unpatch_class_probabilities():
    """Restores the original ultralytics methods and properties."""
    Boxes.__init__ = _original_Boxes_init
    Boxes.conf = _original_Boxes_conf
    Boxes.cls = _original_Boxes_cls
    DetectionPredictor.postprocess = _original_postprocess
    DetectionPredictor.construct_result = _original_construct_result
    # Remove the added properties
    if hasattr(Boxes, 'sorted_probs'):
        delattr(Boxes, 'sorted_probs')
    if hasattr(Boxes, 'probs'):
        delattr(Boxes, 'probs')
    if hasattr(Boxes, 'to_dict_list'):
        delattr(Boxes, 'to_dict_list')
    LOGGER.info("↩️ Ultralytics class probabilities patch restored to original.")
    
# Auto-apply the patch on import
patch_class_probabilities()