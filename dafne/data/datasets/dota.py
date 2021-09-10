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

from scipy import stats


import os

logger = logging.getLogger(__name__)


def plot_hist(data, output_dir, dataset_name, var_name):
    plt.figure()
    plt.title(f"{dataset_name}: {var_name}")
    sns.kdeplot(data)
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_{var_name}"))





def load_dota_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None, cfg=None):
    """
    ---
    Code used from detectron2.data.datasets.coco.load_coco_json to adopt for DOTA.
    ---

    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.
        cfg (ConfigNode): Configuration node object.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO

    # Check if this should be the mini version
    if dataset_name.endswith("_mini"):
        dataset_name = dataset_name[: -len("_mini")]
        is_mini_set = True
    else:
        is_mini_set = False

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]

        # # Remove container-crane to make DOTA 1.5 compatible wiith DOTA 1.0
        # if "container-crane" in thing_classes:
        #     thing_classes.remove("container-crane")

        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())

    if cfg.DEBUG.OVERFIT_NUM_IMAGES > 0:
        # Select the first N images
        img_ids = img_ids[: cfg.DEBUG.OVERFIT_NUM_IMAGES]

    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["bbox", "category_id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0
    count_skipped_boxes = 0
    area_skipped = []
    w_skipped = []
    h_skipped = []

    areas = []
    ws = []
    hs = []

    count_skipped_container_crane = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}

            # # Skip container-crane in dota_1_5 to make 1.5 compatible with 1.0
            # if obj["category_id"] == 16:
            #     count_skipped_container_crane += 1
            #     continue

            x, y, w, h = obj["bbox"]
            area = obj["area"]
            areas.append(area)
            ws.append(w)
            hs.append(h)
            # TODO: make threshold configurable
            if obj["area"] <= cfg.INPUT.MIN_AREA or max(w, h) < cfg.INPUT.MIN_SIDE:
                count_skipped_boxes += 1
                area_skipped += [obj["area"]]
                w_skipped += [w]
                h_skipped += [h]
                # Skip way too small object
                continue

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance

                # Filter out segmentations where two corners overlap
                seg = np.array(segm[0]).reshape(4, 2)
                has_overlapping_corners = False
                for i in range(4):
                    if has_overlapping_corners:
                        break
                    for j in range(i, 4):
                        if i == j:
                            continue
                        seg_i = seg[i]
                        seg_j = seg[j]
                        if np.sum(np.abs(seg_i - seg_j)) < 1e-2:
                            has_overlapping_corners = True
                            break

                # Skip this object if there are overlapping corners
                if has_overlapping_corners:
                    continue

                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                obj["category_id"] = id_map[obj["category_id"]]
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process. "
            "A valid polygon should be a list[float] with even length >= 6."
        )

    wqs = stats.mstats.mquantiles(ws, prob=[1/5, 2/5, 3/5, 4/5])
    hqs = stats.mstats.mquantiles(hs, prob=[1/5, 2/5, 3/5, 4/5])
    logger.info(f"Width quantiles: {wqs}")
    logger.info(f"Height quantiles: {hqs}")
    area_hist, _ = np.histogram(areas, bins=[0] + cfg.MODEL.DAFNE.SIZES_OF_INTEREST + [np.inf])
    width_hist, _ = np.histogram(ws, bins=[0] + cfg.MODEL.DAFNE.SIZES_OF_INTEREST + [np.inf])
    height_hist, _ = np.histogram(hs, bins=[0] + cfg.MODEL.DAFNE.SIZES_OF_INTEREST + [np.inf])

    logger.info(f"Area hist: {area_hist}")
    logger.info(f"Width hist: {width_hist}")
    logger.info(f"Height hist: {height_hist}")

    # logger.info(
    #     f"Skipped {count_skipped_container_crane} annotations with the label 'container-crane'. This is to make Dota 1.5 usable in conjunction with Dota 1.0."
    # )
    logger.warn(f"Skipped {count_skipped_boxes} annotations with too small area or width/height.")

    # If this is the mini set, only sample a random 5% subset
    if is_mini_set:
        n = len(dataset_dicts)
        p = 0.05
        n_mini = int(n * p)
        n_mini = max(10, n_mini)
        dataset_dicts = np.random.choice(dataset_dicts, n_mini).tolist()

    return dataset_dicts


def register_dota_instances(name, metadata, json_file, image_root, cfg):
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
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root

    DatasetCatalog.register(
        name,
        lambda: load_dota_json(
            json_file,
            image_root,
            name,
            extra_annotation_keys=["segmentation", "area"],
            cfg=cfg,
        ),
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="dota", **metadata
    )


def _make_datasets_dict():
    ds_name_template = "dota_{version}_{split}_{size}"
    root_dir_template = "dota_{version}_split/{split}{size}"
    ann_file_template = "DOTA{version}_{split}{size}.json"

    datasets_dict = {}
    # Construct datasets dict from currently available datasets
    for version in ["1", "1_5"]:
        for split in ["train", "val", "test"]:
            for size in ["600", "800", "1024", "1300", "1600", "2048"]:
                name = ds_name_template.format(version=version, split=split, size=size)
                root_dir = root_dir_template.format(version=version, split=split, size=size)
                ann_file = ann_file_template.format(version=version, split=split, size=size)

                datasets_dict[name] = {
                    "root_dir": root_dir,
                    "img_dir": "images",
                    "ann_file": ann_file,
                    "is_test": "test" in name,
                }

    return datasets_dict


def register_dota(cfg):
    """Setup method to register the dota dataset."""
    datasets_dict = _make_datasets_dict()

    # Get the data directory
    data_dir = os.environ["DAFNE_DATA_DIR"]
    colors = colormap(rgb=True, maximum=255)
    for dataset_name, d in datasets_dict.items():

        def reg(name):
            register_dota_instances(
                name=name,
                metadata={
                    "is_test": d["is_test"],
                    "root_dir": os.path.join(data_dir, d["root_dir"]),
                    "thing_colors": colors,
                },
                json_file=os.path.join(data_dir, d["root_dir"], d["ann_file"]),
                image_root=os.path.join(data_dir, d["root_dir"], d["img_dir"]),
                cfg=cfg,
            )

        # Register normal version
        reg(dataset_name)

        # Register mini version
        reg(dataset_name + "_mini")
