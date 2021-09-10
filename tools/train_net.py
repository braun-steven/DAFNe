#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import traceback
from fvcore.common.file_io import PathManager
from detectron2.data.detection_utils import build_augmentation
from detectron2.utils.logger import setup_logger
import contextlib
import time
from typing import List
import logging
import os
from collections import OrderedDict
from detectron2.engine.train_loop import HookBase
import torch
import shutil
from dafne.utils.mail import notify_mail, send_mail_error, send_mail_success

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from detectron2.evaluation import DatasetEvaluators, verify_results
from detectron2.utils.comm import synchronize
from dafne.config import get_cfg
from dafne.hooks import RTPTHook
from dafne.data.datasets.dota import DotaDatasetMapper, register_dota
from dafne.evaluation.dota_evaluation import DotaEvaluator
from dafne.modeling.tta import OneStageRCNNWithTTA

from torch.cuda.amp import GradScaler, autocast


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__(cfg)

        # Init AMP if given
        logger = logging.getLogger(__name__)
        if cfg.SOLVER.AMP.ENABLED:
            logger.info("Using AMP")
            self.scaler = GradScaler()
        else:
            logger.info("Not using AMP")

    def build_train_loader(self, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """

        # Build resize augmentation
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING

        augmentations = [
            T.ResizeShortestEdge(min_size, max_size, sample_style),
            # T.RandomLighting(scale=1.0),
            # T.RandomBrightness(intensity_min=0.5, intensity_max=1.5),
            # T.RandomContrast(intensity_min=0.5, intensity_max=1.5),
            # T.RandomSaturation(intensity_min=0.5, intensity_max=1.5),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),  # HFLIP
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),  # VFLIP
            T.RandomRotation(sample_style="choice", angle=[0.0, 90.0, 180.0, 270.0]),
        ]
        mapper = DotaDatasetMapper(
            cfg,
            is_train=True,
            use_instance_mask=True,
            augmentations=augmentations,
        )
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """

        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
        augmentations = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
        mapper = DotaDatasetMapper(
            cfg,
            is_train="_train_" in dataset_name or "_val_" in dataset_name,
            use_instance_mask=True,
            augmentations=augmentations,
        )

        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        evaluator_list = []

        # Construct the DOTA evaluator object
        dota_evaluator = DotaEvaluator(
            dataset_name=dataset_name,
            cfg=cfg,
            distributed=True,
            output_dir=output_folder,
        )
        evaluator_list.append(dota_evaluator)
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Performs super().test() and sends results via email.

        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.

        Returns:
            dict: a dict of result metrics
        """
        results = super().test(cfg, model, evaluators)
        if not is_debug_session(cfg):
            send_mail_success(cfg, results)
        return results

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("dafne.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = OneStageRCNNWithTTA(cfg, model)

        evaluators = [
            cls.build_evaluator(
                cfg,
                name,
                output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA", name),
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})

        if not is_debug_session(cfg):
            send_mail_success(cfg, res)
        return res

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        if self.cfg.SOLVER.AMP.ENABLED:
            self.run_step_amp()
        else:
            return super().run_step()

    def run_step_amp(self):
        """Perform step() with automatic mixed precision."""

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        with autocast():
            loss_dict = self.model(data)
            losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        self.scaler.scale(losses).backward()

        # use a new stream so the ops don't wait for DDP
        with torch.cuda.stream(
            torch.cuda.Stream()
        ) if losses.device.type == "cuda" else contextlib.nullcontext():
            metrics_dict = loss_dict
            metrics_dict["data_time"] = data_time
            self._write_metrics(metrics_dict)
            self._detect_anomaly(losses, loss_dict)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        if self.cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
            # Unscales the gradients of optimizer's assigned params in-place
            self.scaler.unscale_(self.optimizer)

        self.scaler.step(self.optimizer)
        self.scaler.update()


def backup_config_file(cfg):
    if comm.is_main_process():
        path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
        path_backup = os.path.join(cfg.OUTPUT_DIR, "config_orig.yaml")
        shutil.copy2(path, path_backup)


def restore_config_file(cfg):
    if comm.is_main_process():
        path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
        path_backup = os.path.join(cfg.OUTPUT_DIR, "config_orig.yaml")
        shutil.move(path_backup, path)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Even in eval_only mode, `default_setup` will overwrite the original config.yaml in the output_dir.
    # Therefore, it is necessary to back it up such that an evaluation does not change the original
    # saved config file.
    if args.eval_only:
        backup_config_file(cfg)

    default_setup(cfg, args)

    # Restore the original config file
    if args.eval_only:
        restore_config_file(cfg)

    # Setup logger for "dafne" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="dafne")
    return cfg


def setup_hooks(cfg, trainer) -> List[HookBase]:
    hook_list = []

    # Update process title hook (only on main process)
    hook_list.append(RTPTHook())

    # Autograd profiler in iteration 10-20
    # hook_list.append(
    #     hooks.AutogradProfiler(
    #         lambda trainer: trainer.iter > 10 and trainer.iter < 20, cfg.OUTPUT_DIR
    #     )
    # )

    # TODO
    # evaluators = [
    #     trainer.build_evaluator(
    #         cfg,
    #         name,
    #         output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_mini", name),
    #     )
    #     for name in cfg.DATASETS.TEST
    # ]

    # hook_list.append(
    #     hooks.EvalHook(cfg.SOLVER.MAX_ITER // 10, lambda: trainer.test(cfg, ))
    # )

    # Add EvalHook with TTA if test time augmentation is enabled
    if cfg.TEST.AUG.ENABLED:
        hook_list.append(
            hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))
        )

    return hook_list


def is_debug_session(cfg) -> bool:
    if cfg.DEBUG.OVERFIT_NUM_IMAGES > 0:
        return True
    if "debug" in cfg.OUTPUT_DIR.lower():
        return True
    if cfg.SOLVER.MAX_ITER < 10000:
        return True
    # TODO: Add more cases where debugging is enabled
    return False


def main(args):
    cfg = setup(args)
    logger = logging.getLogger(__name__)

    # Register the DOTA dataset
    register_dota(cfg)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    try:
        trainer = Trainer(cfg)

        # Add hooks to trainer
        trainer.register_hooks(setup_hooks(cfg, trainer))

        trainer.resume_or_load(resume=args.resume)
        return trainer.train()

    except KeyboardInterrupt:  # Catch keyboard interruptions
        logger.error("Keyboard interruption catched. Removing output directory")
        if comm.is_main_process():
            shutil.rmtree(cfg.OUTPUT_DIR)

    except Exception as e:
        # Log error message
        tbstr = "".join(traceback.extract_tb(e.__traceback__).format())
        errormsg = f"Traceback:\n{tbstr}\nError: {e}"
        logger.error(errormsg)

        # Rename output dir
        src = cfg.OUTPUT_DIR
        dst = src + "_errror"
        shutil.move(src, dst)

        logger.error(f"Moving output directory from")
        logger.error(src)
        logger.error("to")
        logger.error(dst)
        raise e


if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
