from detectron2.utils.visualizer import _create_text_labels
from detectron2.structures.masks import PolygonMasks
from detectron2.utils.colormap import colormap
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.structures import BoxMode, Instances, Boxes
import cv2
from detectron2.utils.visualizer import Visualizer
from bs4 import BeautifulSoup as bs

import copy
from collections import OrderedDict, defaultdict
from fvcore.common.file_io import PathManager
import itertools
from detectron2.utils import comm
import os

import torch
import logging

from detectron2.data import DatasetCatalog, MetadataCatalog

classnames = ["ship"]

from .voc_eval import voc_eval

from .dafne_evaluator import DafneEvaluator


class HrscEvaluator(DafneEvaluator):
    def _eval_predictions(self, predictions):
        do_hrsc_evaluation(
            dataset_name=self._dataset_name,
            metadata=self._metadata,
            predictions=predictions,
            output_folder=self._output_dir,
            logger=self._logger,
            results=self._results,
            cfg=self._cfg,
        )


# --------------------------------------------------------
# hrsc_evaluation_task1
# Licensed under The MIT License [see LICENSE for details]
# Written by Jian Ding, based on code from Bharath Hariharan
# --------------------------------------------------------

"""
    To use the code, users should to config detpath, annopath and imagesetfile
    detpath is the path for 15 result files, for the format, you can refer to "http://captain.whu.edu.cn/HRSCweb/tasks.html"
    search for PATH_TO_BE_CONFIGURED to config the paths
    Note, the evaluation is on the large scale images
"""
from pprint import pformat
from dafne.utils.ResultMerge_multi_process import mergebypoly
import torch
import xml.etree.ElementTree as ET
import os
import zipfile

import glob
import seaborn as sns

sns.set(style="whitegrid")

# import cPickle
import numpy as np
import matplotlib.pyplot as plt
import polyiou
from functools import partial

import logging

logger = logging.getLogger(__name__)


def xywha2xy4(xywha):  # a represents the angle(degree), clockwise, a=0 along the X axis
    x, y, w, h, a = xywha
    corner = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    # a = np.deg2rad(a)
    transform = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return transform.dot(corner.T).T + [x, y]


def parse_gt(filename):
    objects = []
    tree = ET.parse(filename)
    root = tree.getroot()
    for obj in root.findall("HRSC_Objects")[0].findall("HRSC_Object"):
        object_struct = {}
        object_struct["name"] = "ship"
        object_struct["difficult"] = int(obj.find("difficult").text)
        bbox = []
        for key in ["mbox_cx", "mbox_cy", "mbox_w", "mbox_h", "mbox_ang"]:
            bbox.append(obj.find(key).text)
        # Coordinates may be float type
        cx, cy, w, h, a = list(map(float, bbox))
        bbox = [cx, cy, w, h, a]
        poly = xywha2xy4(bbox).reshape(-1)
        object_struct["bbox"] = poly.tolist()
        objects.append(object_struct)

    return objects


def _generate_task_1_files(metadata, predictions, output_folder, task1_dir, cfg):
    """Construct files according to Task1: https://captain-whu.github.io/DOAI2019/tasks.html"""

    fp_classes = {}
    cls_name = classnames[0]
    fp_classes[0] = open(os.path.join(task1_dir, f"Task1_{cls_name}.txt"), "w")

    fname_set = set()

    logger.info("Collecting predictions into Task1_<class-name>.txt files")

    # Iterate over images
    for prediction in predictions:
        fname = prediction["file_name"]
        fname = fname.split("/")[-1][:-4]
        fname_set.add(fname)
        corners = prediction["corners"]

        labels = prediction["labels"]
        scores = prediction["scores"]

        # Remove centerness from score to get back original class confidences
        if cfg.MODEL.DAFNE.CENTERNESS != "none" and not cfg.MODEL.DAFNE.CENTERNESS_USE_IN_SCORE:
            centerness = prediction["centerness"]
            scores = scores ** 2
            scores = scores / centerness

        # Iterate over boxes
        num_boxes = corners.shape[0]

        for i in range(num_boxes):
            prediction = corners[i]

            x1, y1, x2, y2, x3, y3, x4, y4 = prediction.unbind()
            label = labels[i].item()
            score = scores[i]

            # Add line to the correct class file
            line = f"{fname} {score:.4f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {x3:.2f} {y3:.2f} {x4:.2f} {y4:.2f}\n"
            fp_classes[label].write(line)

    # Close files
    for _, fp in fp_classes.items():
        fp.close()

    # Create the imageset file
    with open(os.path.join(output_folder, "imageset.txt"), "w") as f:
        lines = "\n".join(list(fname_set))
        f.write(lines)


def plot_pr_curve(rec, prec, ap, classname, pr_dir):
    # Plot PR Curve
    plt.figure(figsize=(10, 6))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(f"PR-Curve {classname} (AP={ap * 100:2.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.plot(rec, prec)
    plt.tight_layout()
    plt.savefig(os.path.join(pr_dir, f"pr-curve_{classname}.png"))


def run_merge(src, dst):
    logger.info("Starting task1 result merge ...")
    mergebypoly(src, dst)
    logger.info("task1 result merge done ...")


def create_zip(output_dir, task1_merged_dir):
    logger.info("Creating zip file ...")
    zfile = zipfile.ZipFile(
        file=os.path.join(output_dir, "task1_merged.zip"),
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    )
    for fname in glob.glob(task1_merged_dir + "/Task1_*.txt"):
        arcname = fname.split("/")[-1]  # Avoid directory structure in zip file
        zfile.write(fname, arcname=arcname)
    zfile.close()
    logger.info("Zip file done ...")


def create_instances(predictions, image_size, conf_threshold):
    ret = Instances(image_size)

    if len(predictions) == 0:
        ret.scores = torch.empty(0, 1)
        # ret.pred_boxes = torch.empty(0, 5)
        ret.pred_corners = torch.empty(0, 8)
        ret.pred_classes = torch.empty(0, 1)

    score = torch.cat([x["scores"] for x in predictions], dim=0)
    chosen_mask = score > conf_threshold
    score = score[chosen_mask]

    corners = torch.cat([p["corners"] for p in predictions], dim=0)
    corners = corners[chosen_mask]

    labels = torch.cat([p["labels"] for p in predictions], dim=0)
    labels = labels[chosen_mask]

    ret.scores = score
    ret.pred_classes = labels
    ret.pred_corners = corners

    return ret


def make_sample_plots(metadata, dataset_name, predictions, output_dir, conf_threshold):

    pred_by_image = defaultdict(list)
    for p in predictions:
        pred_by_image[p["image_id"]].append(p)

    valid_image_ids = pred_by_image.keys()
    dicts = list(DatasetCatalog.get(dataset_name))
    dicts = [d for d in dicts if d["image_id"] in valid_image_ids]

    samples_dir = os.path.join(output_dir, "samples", f"{conf_threshold:0.1f}")
    os.makedirs(samples_dir, exist_ok=True)

    colors = colormap(rgb=True, maximum=1)
    count = 0
    for dic in dicts:
        img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
        basename = os.path.basename(dic["file_name"])

        instances = create_instances(
            pred_by_image[dic["image_id"]],
            img.shape[:2],
            conf_threshold=conf_threshold,
        )

        vis = Visualizer(img, metadata)
        assigned_colors = [colors[i] for i in instances.pred_classes]
        labels = _create_text_labels(instances.pred_classes, instances.scores, classnames)

        vis_pred = vis.overlay_instances(
            labels=labels,
            masks=PolygonMasks([[poly] for poly in instances.pred_corners]),
            assigned_colors=assigned_colors,
        ).get_image()

        vis = Visualizer(img, metadata)

        annos = dic["annotations"]

        # Skip images without annotations
        if len(annos) == 0:
            continue

        masks = [x["segmentation"] for x in annos]

        labels = [x["category_id"] for x in annos]
        names = classnames
        assigned_colors = [colors[i] for i in labels]
        if names:
            labels = [names[i] for i in labels]
        vis_gt = vis.overlay_instances(
            labels=labels, masks=masks, assigned_colors=assigned_colors
        ).get_image()

        concat = np.concatenate((vis_pred, vis_gt), axis=1)
        assert basename[-4:] == ".bmp"
        basename_png = basename[:-4] + ".png"  # Remove ".bmp"
        cv2.imwrite(os.path.join(samples_dir, basename_png), concat[:, :, ::-1])

        count += 1
        if count > 19:
            break


def do_hrsc_evaluation(
    dataset_name,
    metadata,
    predictions: dict,
    output_folder: str,
    logger,
    results: dict,
    cfg,
):

    task1_dir = os.path.join(output_folder, "Task1")
    os.makedirs(task1_dir, exist_ok=True)
    _generate_task_1_files(metadata, predictions, output_folder, task1_dir, cfg)

    # Create sample plots
    make_sample_plots(metadata, dataset_name, predictions, output_folder, conf_threshold=0.4)

    dataset_base_path = metadata.root_dir

    detpath = os.path.join(task1_dir, r"Task1_{:s}.txt")
    annopath = os.path.join(dataset_base_path, "labelXml", r"{:s}.xml")
    imagesetfile = os.path.join(output_folder, "imageset.txt")

    mean_ap = 0.0
    pr_dir = os.path.join(output_folder, "pr-curves/")
    PathManager.mkdirs(pr_dir)
    task_results = OrderedDict()

    data_scores_overlap = []
    for classname in classnames:
        # Skip container crane
        # if classname == "container-crane":
        #     continue
        # Compute class AP
        rec, prec, ap, data_scores_overlap_per_cls = voc_eval(
            detpath,
            annopath,
            imagesetfile,
            classname,
            ovthresh=cfg.TEST.IOU_TH,
            use_07_metric=True,
            parse_gt=parse_gt,
        )
        mean_ap = mean_ap + ap
        task_results[classname] = ap

        plot_pr_curve(rec, prec, ap, classname, pr_dir)
        data_scores_overlap += data_scores_overlap_per_cls

    # Save score to overlap tuples
    np.savetxt(
        fname=os.path.join(output_folder, "scores_overlap.csv"),
        X=data_scores_overlap,
        delimiter=",",
        fmt="%s"
    )

    mean_ap = mean_ap / len(classnames)
    task_results["map"] = mean_ap

    results["task1"] = task_results

    logger.info("\n" + pformat(results))
    if comm.is_main_process():
        torch.save(results, os.path.join(output_folder, "results.pth"))
        with open(os.path.join(output_folder, "results.txt"), "w") as f:
            for key, value in results["task1"].items():
                f.write(f"{key: <18}: {value:2.4f}\n")
