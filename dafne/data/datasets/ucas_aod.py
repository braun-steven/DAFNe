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
import cv2
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


def load_annotation(root_dir, img_id):
    filename = os.path.join(root_dir, "Annotations", img_id + ".txt")

    boxes, gt_classes = [], []
    with open(filename, "r", encoding="utf-8-sig") as f:
        content = f.read()
        objects = content.split("\n")
        for obj in objects:
            if len(obj) != 0:
                class_name = obj.split()[0]
                box = obj.split()[1:9]
                label = name2label[class_name]
                box = [eval(x) for x in box]
                boxes.append(box)
                gt_classes.append(label)
    return {"boxes": np.array(boxes, dtype=np.int32), "gt_classes": np.array(gt_classes)}


def xywha2xy4(xywha):  # a represents the angle(degree), clockwise, a=0 along the X axis
    x, y, w, h, a = xywha
    corner = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    # a = np.deg2rad(a)
    transform = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return transform.dot(corner.T).T + [x, y]


def norm_angle(angle, range=[-np.pi / 4, np.pi]):
    return (angle - range[0]) % range[1] + range[0]


NAMES = ["__background__", "car", "airplane"]

label2name = dict((label, name) for label, name in enumerate(NAMES))
name2label = dict((name, label) for label, name in enumerate(NAMES))


def parse_annotation(img_id: str, root: str):
    anno = load_annotation(root_dir=root, img_id=img_id)

    # Construct image and annotation path
    img_path = os.path.join(root, "AllImages", f"{img_id}.png")
    # anno_path = os.path.join(root, "Annotations", f"{img_id}.txt")

    # Create new data record for each image
    record = {}
    record["file_name"] = img_path
    record["image_id"] = img_id[1:]  # Strip starting letter "P"

    img = cv2.imread(img_path)
    record["width"] = img.shape[1]
    record["height"] = img.shape[0]

    # Collect annotations
    objs = []
    num_objects = anno["boxes"].shape[0]
    for i in range(num_objects):
        obj = {}
        obbox = anno["boxes"][i]
        label = anno["gt_classes"][i] - 1

        if label == -1:
            # Skip background
            continue

        bbox = np.array(obbox).reshape(1, -1)
        xmin, xmax = bbox[:, 0::2].min(), bbox[:, 0::2].max()
        ymin, ymax = bbox[:, 1::2].min(), bbox[:, 1::2].max()
        w = np.abs(xmax - xmin)
        h = np.abs(ymax - ymin)

        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        is_valid_box = (w > 2) & (h > 2) & (ar < 30)
        if not is_valid_box:
            continue

        obj["bbox"] = [xmin, ymin, w, h]
        obj["bbox_mode"] = BoxMode.XYWH_ABS
        obj["area"] = w * h
        area = w * h
        bbox = np.array([xmin, ymin, xmax, ymax])

        obj["segmentation"] = obbox.reshape(1, -1).tolist()
        obj["category_id"] = label
        obj["bbox"] = bbox
        obj["bbox_mode"] = BoxMode.XYXY_ABS
        obj["difficult"] = 0
        obj["area"] = area
        objs.append(obj)
    record["annotations"] = objs
    return record


def load_ucas_aod(root, image_set, cfg):
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
            record = parse_annotation(img_id, root)
            dataset_dicts.append(record)

    return dataset_dicts


def register_ucas_aod_instances(name, split, metadata, image_root, cfg):
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
        lambda: load_ucas_aod(
            root=metadata["root_dir"],
            image_set=split,
            cfg=cfg,
        ),
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(image_root=image_root, evaluator_type="ucas_aod", **metadata)


def _make_datasets_dict():
    datasets_dict = {}
    # Construct datasets dict from currently available datasets
    for split in ["train", "val", "test", "trainval"]:
        name = f"ucas_aod_{split}"
        datasets_dict[name] = {
            "root_dir": "UCAS-AOD",
            "img_dir": "images",
            "ann_file": f"ImageSets/{split}.txt",
            "split": split,
            "is_test": "test" in name,
        }

    return datasets_dict


def register_ucas_aod(cfg):
    """Setup method to register the ucas_aod dataset."""
    datasets_dict = _make_datasets_dict()

    # Get the data directory
    data_dir = os.environ["DAFNE_DATA_DIR"]
    colors = colormap(rgb=True, maximum=255)
    for dataset_name, d in datasets_dict.items():

        def reg(name):
            register_ucas_aod_instances(
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
