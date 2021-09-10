# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2_backbone.config import add_backbone_config
import shutil
from matplotlib import pyplot as plt
from pathlib import Path
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from dafne.modeling.tta import OneStageRCNNWithTTA
from detectron2.utils.colormap import colormap
from detectron2.structures import PolygonMasks
import numpy as np
import argparse
import atexit
import bisect
import glob
import multiprocessing as mp
import os
import time
from collections import deque

import cv2
import torch

# import cv2
import tqdm

from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
import detectron2.data.transforms as T


from dafne.config import get_cfg

classnames = [
    "plane",
    "baseball-diamond",
    "bridge",
    "ground-track-field",
    "small-vehicle",
    "large-vehicle",
    "ship",
    "tennis-court",
    "basketball-court",
    "storage-tank",
    "soccer-ball-field",
    "roundabout",
    "harbor",
    "swimming-pool",
    "helicopter",
]


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_backbone_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.DAFNE.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def _create_text_labels(classes, scores, fpn_levels, class_names, is_crowd=None):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        fpn_levels (list[int] or None):
        class_names (list[str] or None):
        is_crowd (list[bool] or None):

    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None and class_names is not None and len(class_names) > 0:
        labels = [class_names[i] for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%, {}".format(l, s * 100, lvl + 3) for l, s, lvl in zip(labels, scores, fpn_levels)]
    if is_crowd is not None:
        labels = [l + ("|crowd" if crowd else "") for l, crowd in zip(labels, is_crowd)]
    return labels


class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.

    If you'd like to do anything more fancy, please refer to its source code
    as examples to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            features = self.model.proposal_generator.feature_cache
            return predictions, features


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image_path, output_dir):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """

        image = read_image(image_path, format="BGR")
        vis_output = None
        predictions, features = self.predictor(image)


        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer_with_labels = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        visualizer_without_labels = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        visualizer_without_labels_hbox = Visualizer(image, self.metadata, instance_mode=self.instance_mode)

        colors = colormap(rgb=True, maximum=1)
        instances = predictions["instances"].to(self.cpu_device)

        # if instances.pred_classes.shape[0] < 40:
        if (instances.pred_classes == 0).sum() < 10:
            return predictions

        assigned_colors = [colors[i] for i in instances.pred_classes]
        labels = _create_text_labels(
            instances.pred_classes, instances.scores, instances.fpn_levels, classnames
        )

        visualized_output_with_labels = visualizer_with_labels.overlay_instances(
            labels=labels,
            masks=PolygonMasks([[poly] for poly in instances.pred_corners]),
            assigned_colors=assigned_colors,
            alpha=0.1,
        )
        visualized_output_without_labels = visualizer_without_labels.overlay_instances(
            # labels=labels,
            masks=PolygonMasks([[poly] for poly in instances.pred_corners]),
            assigned_colors=assigned_colors,
            alpha=0.0,
        )

        _t: torch.Tensor = instances.pred_boxes.tensor
        xmin, ymin, xmax, ymax = _t.unbind(-1)
        hbox_polymask = torch.stack((xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin), 1)

        visualized_output_without_labels_hbox = visualizer_without_labels_hbox.overlay_instances(
            # labels=labels,
            # masks=PolygonMasks([[poly] for poly in instances.pred_corners]),
            masks=PolygonMasks([[poly] for poly in hbox_polymask]),
            # boxes=instances.pred_boxes,
            assigned_colors=assigned_colors,
            alpha=0.00,
        )
        assert os.path.isdir(args.output), args.output
        out_filename = os.path.join(args.output, os.path.basename(path))

        plt.imsave(out_filename, image)
        visualized_output_with_labels.save(out_filename.replace(".png", "_pred.png"))
        visualized_output_without_labels.save(out_filename.replace(".png", "_pred-no-label.png"))
        visualized_output_without_labels_hbox.save(out_filename.replace(".png", "_pred-no-label_hbox.png"))

        # # Add location dots
        # for cls, loc in zip(instances.pred_classes, instances.locations):
        #     visualizer.draw_circle((loc[0], loc[1]), color=colors[cls], radius=5)


        # fname_without_suffix = out_filename.split(".")[-2]
        # for level_name, feat_dict in features.items():
        #     for tower_name, features in feat_dict.items():
        #         base_dir = os.path.join(fname_without_suffix, tower_name)
        #         Path(base_dir).mkdir(parents=True, exist_ok=True)

        #         # For each channel
        #         features.squeeze_(0)

        #         features_np = features.cpu().numpy()
        #         np.save(os.path.join(base_dir, level_name + ".npy"), features_np)


        # vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)


    Path(args.output).mkdir(parents=True, exist_ok=True)

    if len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"
    for path in tqdm.tqdm(args.input, disable=not args.output):
        # use PIL, to be consistent with evaluation
        start_time = time.time()
        predictions = demo.run_on_image(path, args.output)
        logger.info(
            "{}: {} in {:.2f}s".format(
                path,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )
