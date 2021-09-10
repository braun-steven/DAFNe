#!/usr/bin/env python3

from detectron2.data.detection_utils import filter_empty_instances
from detectron2.structures.instances import Instances
import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import DatasetMapper
from dafne.utils.sort_corners import sort_quadrilateral


class DAFNeDatasetMapper(DatasetMapper):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._cfg = cfg

    def __call__(self, dataset_dict):
        result = super().__call__(dataset_dict)
        if "instances" in result:
            instances = result["instances"]


            # Iterate over polygons and check that they are still valid
            if instances.has("gt_masks") and len(instances.gt_masks.polygons) > 0:
                for i, ps in enumerate(instances.gt_masks):
                    instances.gt_masks.polygons[i] = [p for p in ps if p.shape[0] == 8]


            instances = filter_empty_instances(instances, by_mask=True)

            if instances.has("gt_masks") and len(instances.gt_masks) > 0:
                gt_masks = np.stack(instances.gt_masks).squeeze(1)
                gt_corners = torch.tensor(gt_masks, dtype=instances.gt_boxes.tensor.dtype)

                # Sort corners
                if self._cfg.MODEL.DAFNE.SORT_CORNERS_DATALOADER:
                    gt_corners = sort_quadrilateral(gt_corners)

                instances.gt_corners = gt_corners
                instances.gt_corners_area = instances.gt_masks.area().float()
            else:
                instances.gt_corners = torch.empty(0, 8)
                instances.gt_corners_area = torch.empty(0)
            result["instances"] = instances

        return result
