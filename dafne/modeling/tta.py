import copy
from itertools import count
from dafne.modeling.dafne.dafne_outputs import ml_nms

import numpy as np
import torch
from fvcore.transforms import HFlipTransform, NoOpTransform
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from detectron2.data.detection_utils import read_image
from detectron2.data.transforms import (
    RandomFlip,
    ResizeShortestEdge,
    Resize,
    ResizeTransform,
    apply_augmentations,
)
from detectron2.data.transforms.augmentation_impl import RandomRotation
from detectron2.structures import Boxes, Instances
from dafne.modeling.one_stage_detector import OneStageDetector

# from dafne.data.transforms.transform import RandomRotation


__all__ = ["DotaDatasetMapperTTA", "OneStageRCNNWithTTA"]


class DotaDatasetMapperTTA:
    """
    Implement test-time augmentation for detection data.
    It is a callable which takes a dataset dict from a detection dataset,
    and returns a list of dataset dicts where the images
    are augmented from the input image by the transformations defined in the config.
    This is used for test-time augmentation.
    """

    def __init__(self, cfg):
        self.min_sizes = cfg.TEST.AUG.MIN_SIZES
        self.max_size = cfg.TEST.AUG.MAX_SIZE
        self.resize_type = cfg.INPUT.RESIZE_TYPE
        self.vflip = cfg.TEST.AUG.VFLIP
        self.hflip = cfg.TEST.AUG.HFLIP
        self.rotation_angles = cfg.TEST.AUG.ROTATION_ANGLES
        self.image_format = cfg.INPUT.FORMAT
        self.cfg = cfg

    def __call__(self, dataset_dict):
        """
        Args:
            dict: a dict in standard model input format. See tutorials for details.

        Returns:
            list[dict]:
                a list of dicts, which contain augmented version of the input image.
                The total number of dicts is ``len(min_sizes) * (2 if flip else 1)``.
                Each dict has field "transforms" which is a TransformList,
                containing the transforms that are used to generate this image.
        """
        numpy_image = dataset_dict["image"].permute(1, 2, 0).numpy()
        shape = numpy_image.shape
        orig_shape = (dataset_dict["height"], dataset_dict["width"])
        if shape[:2] != orig_shape:
            # It transforms the "original" image in the dataset to the input image
            pre_tfm = ResizeTransform(orig_shape[0], orig_shape[1], shape[0], shape[1])
        else:
            pre_tfm = NoOpTransform()

        # Create all combinations of augmentations to use
        aug_candidates = []  # each element is a list[Augmentation]
        for min_size in self.min_sizes:
            if self.resize_type == "shortest-edge":
                resize = ResizeShortestEdge(
                    min_size, self.max_size
                )
            elif self.resize_type == "both":
                # Get original resize height/width for testing
                h_test = self.cfg.INPUT.RESIZE_HEIGHT_TEST
                w_test = self.cfg.INPUT.RESIZE_WIDTH_TEST

                # Find scale of current min_size w.r.t. to test width
                scale = min_size / w_test

                # Scale height accordingly
                h = int(h_test * scale)
                w = min_size
                resize = Resize(shape=(h, w))
                # resize = ResizeTransform(shape[0], shape[1], h, w)
            else:
                raise RuntimeError(f"Invalid resize-type: {self.cfg.INPUT.RESIZE_TYPE}")

            aug_candidates.append([resize])  # resize only

            if len(self.rotation_angles) == 0:
                # Horizontal Flip
                if self.hflip:
                    flip = RandomFlip(prob=1.0, horizontal=True, vertical=False)
                    aug_candidates.append([resize, flip])  # resize + hflip

                # Horizontal Flip
                if self.vflip:
                    flip = RandomFlip(prob=1.0, horizontal=False, vertical=True)
                    aug_candidates.append([resize, flip])  # resize + vflip

            else:
                for angle in self.rotation_angles:
                    # Choose specific rotation augmentation (choice from a single element)
                    rot = RandomRotation(angle=[angle], sample_style="choice")

                    # Horizontal Flip
                    if self.hflip:
                        flip = RandomFlip(prob=1.0, horizontal=True, vertical=False)
                        aug_candidates.append([resize, rot, flip])  # resize + rot + hflip

                    # Vertical Flip
                    # NOTE: VLIP is unnecessary since HFLIP + ROT(180) = VFLIP
                    # if self.vflip:
                    #     flip = RandomFlip(prob=1.0, horizontal=False, vertical=True)
                    #     aug_candidates.append([resize, rot, flip])  # resize + rot + vflip

                    # If no flip flag was active, just add resize + rot
                    if not self.hflip:
                        aug_candidates.append([resize, rot])

        # Apply all the augmentations
        ret = []
        for aug in aug_candidates:
            new_image, tfms = apply_augmentations(aug, np.copy(numpy_image))
            torch_image = torch.from_numpy(np.ascontiguousarray(new_image.transpose(2, 0, 1)))

            dic = copy.deepcopy(dataset_dict)
            dic["transforms"] = pre_tfm + tfms
            dic["image"] = torch_image
            ret.append(dic)
        return ret


class OneStageRCNNWithTTA(nn.Module):
    """
    A OneStageDetectorWithTTA with test-time augmentation enabled.
    Its :meth:`__call__` method has the same interface as :meth:`GeneralizedRCNN.forward`.
    """

    def __init__(self, cfg, model, tta_mapper=None, batch_size=3):
        """
        Args:
            cfg (CfgNode):
            model (GeneralizedRCNN): a GeneralizedRCNN to apply TTA on.
            tta_mapper (callable): takes a dataset dict and returns a list of
                augmented versions of the dataset dict. Defaults to
                `DatasetMapperTTA(cfg)`.
            batch_size (int): batch the augmented images into this batch size for inference.
        """
        super().__init__()
        if isinstance(model, DistributedDataParallel):
            model = model.module
        assert isinstance(
            model, OneStageDetector
        ), "TTA is only supported on OneStageDetector. Got a model of type {}".format(type(model))
        self.cfg = cfg.clone()
        assert not self.cfg.MODEL.KEYPOINT_ON, "TTA for keypoint is not supported yet"
        assert (
            not self.cfg.MODEL.LOAD_PROPOSALS
        ), "TTA for pre-computed proposals is not supported yet"

        self.model = model

        if tta_mapper is None:
            tta_mapper = DotaDatasetMapperTTA(cfg)
        self.tta_mapper = tta_mapper
        self.batch_size = batch_size

    def _batch_inference(self, batched_inputs, detected_instances=None):
        """
        Execute inference on a list of inputs,
        using batch size = self.batch_size, instead of the length of the list.

        Inputs & outputs have the same format as :meth:`GeneralizedRCNN.inference`
        """
        if detected_instances is None:
            detected_instances = [None] * len(batched_inputs)

        outputs = []
        inputs, instances = [], []
        for idx, input, instance in zip(count(), batched_inputs, detected_instances):
            inputs.append(input)
            instances.append(instance)
            if len(inputs) == self.batch_size or idx == len(batched_inputs) - 1:
                outputs.extend(
                    self.model.inference(
                        inputs,
                        instances if instances[0] is not None else None,
                        do_postprocess=False,
                    )
                )
                inputs, instances = [], []
        return outputs

    def __call__(self, batched_inputs):
        """
        Same input/output format as :meth:`GeneralizedRCNN.forward`
        """

        def _maybe_read_image(dataset_dict):
            ret = copy.copy(dataset_dict)
            if "image" not in ret:
                image = read_image(ret.pop("file_name"), self.tta_mapper.image_format)
                image = torch.from_numpy(image).permute(2, 0, 1)  # CHW
                ret["image"] = image
            if "height" not in ret and "width" not in ret:
                ret["height"] = image.shape[1]
                ret["width"] = image.shape[2]
            return ret

        return [self._inference_one_image(_maybe_read_image(x)) for x in batched_inputs]

    def _inference_one_image(self, input):
        """
        Args:
            input (dict): one dataset dict with "image" field being a CHW tensor

        Returns:
            dict: one output dict
        """
        orig_shape = (input["height"], input["width"])
        augmented_inputs, tfms = self._get_augmented_inputs(input)
        instances = self._get_augmented_corners(augmented_inputs, tfms)
        merged_instances = self._merge_detections(instances)
        return {"instances": merged_instances}

    def _get_augmented_inputs(self, input):
        augmented_inputs = self.tta_mapper(input)
        tfms = [x.pop("transforms") for x in augmented_inputs]
        return augmented_inputs, tfms

    def _get_augmented_corners(self, augmented_inputs, tfms):
        # 1: forward with all augmented images
        outputs = self._batch_inference(augmented_inputs)
        # 2: union the results
        # all_corners = []
        # all_scores = []
        # all_classes = []
        instances_list = []
        for output, tfm in zip(outputs, tfms):
            # Need to inverse the transforms on boxes, to obtain results on original image
            instances = output["instances"]
            pred_corners = instances.pred_corners
            N, C = pred_corners.shape
            assert C == 8
            pred_corners = pred_corners.view(-1, 2)
            pred_corners_np = pred_corners.cpu().numpy()
            original_pred_corners = tfm.inverse().apply_coords(pred_corners_np)
            original_pred_corners = original_pred_corners.reshape(N, C)
            inst = Instances(instances.image_size)
            inst.scores = instances.scores
            inst.centerness = instances.centerness
            inst.pred_corners = torch.from_numpy(original_pred_corners).to(
                pred_corners.device, dtype=pred_corners.dtype
            )
            inst.pred_classes = instances.pred_classes
            instances_list.append(inst)
        return Instances.cat(instances_list)

    def _merge_detections(self, instances):
        merged_instances = self.model.proposal_generator.dafne_outputs.select_over_all_levels(
            [instances]
        )[0]
        return merged_instances
