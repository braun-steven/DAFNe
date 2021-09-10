#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.data import transforms as T
from detectron2.data.transforms.augmentation_impl import RandomContrast, RandomLighting, ResizeShortestEdge
from dafne.data.datasets.dota import DotaDatasetMapper
import argparse
import os
from itertools import chain
import cv2
import tqdm
from dafne.data.datasets import dota

from dafne.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data.build import filter_images_with_few_keypoints
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.colormap import colormap


def setup(args):
    cfg = get_cfg()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description="Visualize ground-truth data")
    parser.add_argument(
        "--source",
        choices=["annotation", "dataloader"],
        required=True,
        help="visualize the annotations or the data loader (with pre-processing)",
    )
    parser.add_argument("--config-file", metavar="FILE", help="path to config file")
    parser.add_argument("--output-dir", default="./", help="path to output directory")
    parser.add_argument("--show", action="store_true", help="show output in a window")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args(in_args)


if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup(args)

    dota.register_dota(cfg)
    dirname = args.output_dir
    os.makedirs(dirname, exist_ok=True)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    def output(vis, fname):
        if args.show:
            print(fname)
            cv2.imshow("window", vis.get_image()[:, :, ::-1])
            cv2.waitKey()
        else:
            filepath = os.path.join(dirname, fname)
            print("Saving to {} ...".format(filepath))
            vis.save(filepath)

    scale = 2.0 if args.show else 1.0
    if args.source == "dataloader":

        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        resize_aug = T.ResizeShortestEdge(min_size, max_size, sample_style)

        mapper = DotaDatasetMapper(
            cfg,
            is_train=True,
            use_instance_mask=True,
            augmentations=[
                # T.RandomLighting(scale=1.0),
                # T.RandomBrightness(intensity_min=0.5, intensity_max=1.5),
                # T.RandomContrast(intensity_min=0.5, intensity_max=1.5),
                # T.RandomSaturation(intensity_min=0.5, intensity_max=1.5),
                resize_aug,
                # T.RandomFlip(prob=0.5, horizontal=True, vertical=False),  # HFLIP
                # T.RandomFlip(prob=0.5, horizontal=False, vertical=True),  # VFLIP
                # T.RandomRotation(sample_style="choice", angle=[0.0, 90.0, 180.0, 270.0]),
                # T.RandomCrop(crop_type="relative", crop_size=(0.5, 0.5)),
                # Random Crop doesn't work as it changes the polygon sizes (e.g. 10 instead of 8)
            ],
        )

        train_data_loader = build_detection_train_loader(cfg, mapper=mapper)

        colors = colormap(rgb=True, maximum=1)
        for batch in train_data_loader:
            for per_image in batch:
                # Pytorch tensor is in (C, H, W) format
                img = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()
                img = utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)

                visualizer = Visualizer(img, metadata=metadata, scale=scale)
                target_fields = per_image["instances"].get_fields()
                labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
                assigned_colors = [colors[i] for i in target_fields["gt_classes"]]
                vis = visualizer.overlay_instances(
                    labels=labels,
                    # boxes=target_fields.get("gt_boxes", None),
                    masks=target_fields.get("gt_masks", None),
                    keypoints=target_fields.get("gt_keypoints", None),
                    assigned_colors=assigned_colors
                )
                output(vis, str(per_image["image_id"]) + ".jpg")
    else:
        dicts = list(chain.from_iterable([DatasetCatalog.get(k) for k in cfg.DATASETS.TRAIN]))
        if cfg.MODEL.KEYPOINT_ON:
            dicts = filter_images_with_few_keypoints(dicts, 1)
        for dic in tqdm.tqdm(dicts):
            img = utils.read_image(dic["file_name"], "RGB")
            visualizer = Visualizer(img, metadata=metadata, scale=scale)
            vis = visualizer.draw_dataset_dict(dic)
            output(vis, os.path.basename(dic["file_name"]))