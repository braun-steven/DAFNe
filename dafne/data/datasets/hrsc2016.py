from detectron2.data.datasets import register_coco_instances
from dafne.utils.sort_corners import sort_quadrilateral
from detectron2.utils.colormap import colormap
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    DatasetMapper,
    transforms as T,
)
import xml.etree.ElementTree as ET

from detectron2.structures import BoxMode, PolygonMasks, RotatedBoxes
from detectron2.data import detection_utils as utils
import copy
import torch
import contextlib
import datetime
import io
import json
import logging
import numpy as np
import os
import pycocotools.mask as mask_util
from fvcore.common.file_io import PathManager, file_lock
from fvcore.common.timer import Timer
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


import os

logger = logging.getLogger(__name__)


def xywha2xy4(xywha):  # a represents the angle(degree), clockwise, a=0 along the X axis
    x, y, w, h, a = xywha
    corner = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    # a = np.deg2rad(a)
    transform = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return transform.dot(corner.T).T + [x, y]


def norm_angle(angle, range=[-np.pi / 4, np.pi]):
    return (angle - range[0]) % range[1] + range[0]


NAMES = ["ship"]

label2name = dict((label, name) for label, name in enumerate(NAMES))
name2label = dict((name, label) for label, name in enumerate(NAMES))


def load_hrsc(root, image_set, cfg):
    image_sets = [image_set] if isinstance(image_set, str) else image_set
    dataset_dicts = []
    for image_set in image_sets:
        # Read lines in image set file
        with open(os.path.join(root, "ImageSets", f"{image_set}.txt")) as f:
            lines = f.read().splitlines()

        if cfg.DEBUG.OVERFIT_NUM_IMAGES > 0:
            # Select the first N images
            lines = lines[: cfg.DEBUG.OVERFIT_NUM_IMAGES]

        for img_id in lines:
            img_id = int(img_id)
            # Construct image and annotation path
            img_path = os.path.join(root, "images", f"{img_id}.bmp")
            anno_path = os.path.join(root, "labelXml", f"{img_id}.xml")

            # Create new data record for each image
            record = {}
            record["file_name"] = img_path
            record["image_id"] = img_id

            # Parse annotation
            anno_tree = ET.parse(anno_path)
            anno_root = anno_tree.getroot()

            record["width"] = int(anno_root.find("Img_SizeWidth").text)
            record["height"] = int(anno_root.find("Img_SizeHeight").text)

            # # Skip invalid paths
            # if not anno_path:
            #     continue

            # Collect annotations
            objs = []
            for obj in anno_root.findall("HRSC_Objects")[0].findall("HRSC_Object"):
                label = name2label["ship"]
                difficult = int(obj.find("difficult").text)
                bbox = []
                for key in ["mbox_cx", "mbox_cy", "mbox_w", "mbox_h", "mbox_ang"]:
                    bbox.append(obj.find(key).text)
                # TODO: check whether it is necessary to use int
                # Coordinates may be float type
                cx, cy, w, h, a = list(map(float, bbox))
                # set w the long side and h the short side
                # new_w, new_h = max(w, h), min(w, h)
                # # adjust angle
                # a = a if w > h else a + np.pi / 2
                # normalize angle to [-np.pi/4, pi/4*3]
                # a = norm_angle(a)
                bbox = [cx, cy, w, h, a]

                obj = {}
                obbox = xywha2xy4(bbox).reshape(1, -1).tolist()
                obj["segmentation"] = obbox
                obj["category_id"] = label
                obj["difficult"] = difficult

                bbox = np.array(obbox)
                xmin, xmax = bbox[:, 0::2].min(), bbox[:, 1::2].max()
                ymin, ymax = bbox[:, 1::2].min(), bbox[:, 1::2].max()
                w = np.abs(xmax - xmin)
                h = np.abs(ymax - ymin)
                obj["bbox"] = [xmin, ymin, w, h]
                obj["bbox_mode"] = BoxMode.XYWH_ABS
                obj["area"] = w * h

                objs.append(obj)

            record["annotations"] = objs
            dataset_dicts.append(record)

    return dataset_dicts


def register_hrsc_instances(name, split, metadata, image_root, cfg):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(image_root, (str, os.PathLike)), image_root

    DatasetCatalog.register(
        name,
        lambda: load_hrsc(
            root=metadata["root_dir"],
            image_set=split,
            cfg=cfg,
        ),
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(image_root=image_root, evaluator_type="hrsc", **metadata)


class HrscDatasetMapper(DatasetMapper):
    def __call__(self, dataset_dict):
        result = super().__call__(dataset_dict)
        if "instances" in result:
            instances = result["instances"]
            if instances.has("gt_masks") and len(instances.gt_masks) > 0:
                gt_masks = np.stack(instances.gt_masks).squeeze(1)
                gt_corners = torch.tensor(gt_masks, dtype=instances.gt_boxes.tensor.dtype)

                # Sort corners
                gt_corners = sort_quadrilateral(gt_corners)

                instances.gt_corners = gt_corners
                instances.gt_corners_area = instances.gt_masks.area().float()
            else:
                instances.gt_corners = torch.empty(0, 8)
                instances.gt_corners_area = torch.empty(0)
            result["instances"] = instances

        return result


def _make_datasets_dict():
    datasets_dict = {}
    # Construct datasets dict from currently available datasets
    for split in ["train", "val", "test", "trainval"]:
        name = f"hrsc_{split}"
        datasets_dict[name] = {
            "root_dir": "hrsc",
            "img_dir": "images",
            "ann_file": f"ImageSets/{split}.txt",
            "split": split,
            "is_test": "test" in name,
        }

    return datasets_dict


def register_hrsc(cfg):
    """Setup method to register the hrsc dataset."""
    datasets_dict = _make_datasets_dict()

    # Get the data directory
    data_dir = os.environ["DAFNE_DATA_DIR"]
    colors = colormap(rgb=True, maximum=255)
    for dataset_name, d in datasets_dict.items():

        def reg(name):
            register_hrsc_instances(
                name=name,
                metadata={
                    "is_test": d["is_test"],
                    "root_dir": os.path.join(data_dir, d["root_dir"]),
                    "thing_colors": colors,
                },
                image_root=os.path.join(data_dir, d["root_dir"], d["img_dir"]),
                split=d["split"],
                cfg=cfg,
            )

        # Register normal version
        reg(dataset_name)
