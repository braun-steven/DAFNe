# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from dafne.modeling.tta import OneStageRCNNWithTTA
from detectron2.utils.colormap import colormap
from detectron2.structures import PolygonMasks
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
from detectron2.utils.visualizer import ColorMode, Visualizer, _create_text_labels


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
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    # cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    # cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
    #     args.confidence_threshold
    # )
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


class TTADefaultPredictor(DefaultPredictor):
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
        super().__init__(cfg)

        self.model = OneStageRCNNWithTTA(cfg, self.model)


    # def __call__(self, original_image):
    #     """
    #     Args:
    #         original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

    #     Returns:
    #         predictions (dict):
    #             the output of the model for one image only.
    #             See :doc:`/tutorials/models` for details about the format.
    #     """
    #     with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
    #         # Apply pre-processing to image.
    #         if self.input_format == "RGB":
    #             # whether the model expects BGR inputs or RGB
    #             original_image = original_image[:, :, ::-1]
    #         height, width = original_image.shape[:2]
    #         image = self.aug.get_transform(original_image).apply_image(original_image)
    #         image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

    #         inputs = {"image": image, "height": height, "width": width}
    #         predictions = self.model([inputs])[0]
    #         return predictions


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
        self.predictor = TTADefaultPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)

        colors = colormap(rgb=True, maximum=1)
        instances = predictions["instances"].to(self.cpu_device)

        assigned_colors = [colors[i] for i in instances.pred_classes]
        labels = _create_text_labels(
            instances.pred_classes, instances.scores, classnames
        )

        vis_output = visualizer.overlay_instances(
            labels=labels,
            masks=PolygonMasks([[poly] for poly in instances.pred_corners]),
            assigned_colors=assigned_colors,
            alpha=0.1
        )

        # vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"
    for path in tqdm.tqdm(args.input, disable=not args.output):
        # use PIL, to be consistent with evaluation
        img = read_image(path, format="BGR")
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)
        logger.info(
            "{}: {} in {:.2f}s".format(
                path,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )

        if os.path.isdir(args.output):
            assert os.path.isdir(args.output), args.output
            out_filename = os.path.join(args.output, os.path.basename(path))
        else:
            assert len(args.input) == 1, "Please specify a directory with args.output"
            out_filename = args.output

        visualized_output.save(out_filename)
