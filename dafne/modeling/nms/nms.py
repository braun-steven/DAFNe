import torch
from dafne.utils.sort_corners import sort_quadrilateral
import numpy as np
import detectron2.utils.comm as comm
from detectron2.structures import Instances
from poly_nms import poly_gpu_nms
from poly_overlaps import poly_overlaps


def ml_nms(boxlist, nms_thresh, max_proposals=-1):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Args:
        boxlist (detectron2.structures.Boxes):
        nms_thresh (float):
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str):
    """
    if nms_thresh <= 0:
        return boxlist
    if boxlist.scores.shape[0] == 0:
        return boxlist
    polys = boxlist.pred_corners
    scores = boxlist.scores
    labels = boxlist.pred_classes
    keep = batched_nms_poly(polys, scores, labels, nms_thresh)
    if max_proposals > 0:
        keep = keep[:max_proposals]
    boxlist = boxlist[keep]
    return boxlist



def batched_nms_poly(boxes, scores, idxs, iou_threshold):
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes (Tensor[N, 8]):
           boxes where NMS will be performed. They
           are expected to be in (x0,y0,x1,y1,x2,y2,x3,y3) format
        scores (Tensor[N]):
           scores for each one of the boxes
        idxs (Tensor[N]):
           indices of the categories for each one of the boxes.
        iou_threshold (float):
           discards all overlapping boxes
           with IoU < iou_threshold

    Returns:
        Tensor:
            int64 tensor with the indices of the elements that have been kept
            by NMS, sorted in decreasing order of scores
    """
    assert boxes.shape[-1] == 8

    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    # Strategy: in order to perform NMS independently per class,
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap

    # Note that batched_nms in torchvision/ops/boxes.py only uses max_coordinate,
    # which won't handle negative coordinates correctly.
    # Here by using min_coordinate we can make sure the negative coordinates are
    # correctly handled.
    max_coordinate = boxes.max()
    min_coordinate = boxes.min()

    # HACK: treat small-vehicle (idx: 4) and large-vehicle (idx: 5) class as the same
    idxs = idxs.clone()
    idxs[idxs == 5] = 4

    offsets = idxs.to(boxes) * (max_coordinate - min_coordinate + 1)
    boxes_for_nms = boxes.clone()  # avoid modifying the original values in boxes
    boxes_for_nms[:, 0:8] += offsets[:, None]

    # convert to numpy before calculate
    boxes_np = boxes_for_nms.data.cpu().numpy()
    score_np = scores.data.cpu().numpy()

    # Stack [N, 8] polygon and [N, 1] scores to the expected [N, 9] data array
    boxes_np = np.hstack((boxes_np, score_np.reshape(-1, 1)))
    keep = poly_gpu_nms(boxes_np, iou_threshold, comm.get_local_rank())
    return keep


