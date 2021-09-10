#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""
from pprint import pformat
from dafne.evaluation.hrsc_evaluation import HrscEvaluator
from dafne.data.datasets.hrsc2016 import register_hrsc
from dafne.data.datasets.dafne_dataset_mapper import DAFNeDatasetMapper
from setproctitle import setproctitle
import contextlib
import logging
import os
import shutil
import time
import traceback
from collections import OrderedDict
from typing import Any, Dict, List, Set
from detectron2.utils.comm import is_main_process, synchronize
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from pathlib import Path

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import CfgNode, get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)

from detectron2.data import transforms as T
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from detectron2.engine.defaults import DefaultTrainer
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.utils.logger import setup_logger
from dafne.config import get_cfg
from dafne.data.datasets.dota import register_dota
from dafne.evaluation.dota_evaluation import DotaEvaluator
from dafne.modeling.tta import OneStageRCNNWithTTA
from dafne.utils.mail import send_mail_error, send_mail_success
from dafne.utils.rtpt import RTPT

logger = logging.getLogger("detectron2")


def build_optimizer_custom(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module in model.modules():
        for key, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if isinstance(module, norm_module_types):
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM
            elif key == "bias":
                # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                # hyperparameters are by default exactly the same as for regular
                # weights.
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER == "sgd":
        optimizer = torch.optim.SGD(
            params,
            cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            nesterov=cfg.SOLVER.NESTEROV,
        )
    elif cfg.SOLVER.OPTIMIZER == "adam":
        optimizer = torch.optim.Adam(params, cfg.SOLVER.BASE_LR)
    else:
        raise RuntimeError(f"Invalid optimizer selection ({cfg.SOLVER.OPTIMIZER})")
    return optimizer


def detect_anomaly(losses, loss_dict, iter):
    if not torch.isfinite(losses).all():
        raise FloatingPointError(
            "Loss became infinite or NaN at iteration={}!\nloss_dict = {}".format(iter, loss_dict)
        )


def write_metrics(storage, metrics_dict: dict):
    """
    Args:
        metrics_dict (dict): dict of scalar metrics
    """
    metrics_dict = {
        k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
        for k, v in metrics_dict.items()
    }
    # gather metrics among all workers for logging
    # This assumes we do DDP-style training, which is currently the only
    # supported method in detectron2.
    all_metrics_dict = comm.gather(metrics_dict)

    if comm.is_main_process():
        if "data_time" in all_metrics_dict[0]:
            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

        # average the rest metrics
        metrics_dict = {
            k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())

        storage.put_scalar("loss/total_loss", total_losses_reduced)
        if len(metrics_dict) > 1:
            storage.put_scalars(**metrics_dict)


def get_evaluator(cfg, dataset_name, output_folder=None):
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
    if "dota" in dataset_name.lower():
        evaluator = DotaEvaluator(
            dataset_name=dataset_name,
            cfg=cfg,
            distributed=True,
            output_dir=output_folder,
        )
    elif "hrsc" in dataset_name.lower():
        evaluator = HrscEvaluator(
            dataset_name=dataset_name,
            cfg=cfg,
            distributed=True,
            output_dir=output_folder,
        )
    else:
        raise RuntimeError()

    evaluator_list.append(evaluator)
    return DatasetEvaluators(evaluator_list)


def build_train_loader(cfg):
    """
    Returns:
        iterable

    It now calls :func:`detectron2.data.build_detection_train_loader`.
    Overwrite it if you'd like a different data loader.
    """

    # Start list with default augmentations
    augmentations = [
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),  # HFLIP
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),  # VFLIP
        # T.RandomCrop(crop_type="relative_range", crop_size=(0.66, 0.66))
    ]

    # Choose between shortest-edge and rescale resizing
    if cfg.INPUT.RESIZE_TYPE == "shortest-edge":
        # Build resize augmentation
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        augmentations.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    elif cfg.INPUT.RESIZE_TYPE == "both":
        h = cfg.INPUT.RESIZE_HEIGHT_TRAIN
        w = cfg.INPUT.RESIZE_WIDTH_TRAIN
        augmentations.append(T.Resize(shape=(h, w)))
    else:
        raise RuntimeError(f"Invalid resize-type: {cfg.INPUT.RESIZE_TYPE}")

    # Use Rotations augmentation if aug_angles is given
    if len(cfg.INPUT.ROTATION_AUG_ANGLES) > 0:
        augmentations.append(
            T.RandomRotation(
                sample_style=cfg.INPUT.ROTATION_AUG_SAMPLE_STYLE,
                angle=cfg.INPUT.ROTATION_AUG_ANGLES,
            )
        )

    # Add color augmentations ifenabled
    if cfg.INPUT.USE_COLOR_AUGMENTATIONS:
        augmentations.extend(
            [
                T.RandomLighting(scale=1.0),
                T.RandomBrightness(intensity_min=0.5, intensity_max=1.5),
                T.RandomContrast(intensity_min=0.5, intensity_max=1.5),
                T.RandomSaturation(intensity_min=0.5, intensity_max=1.5),
            ]
        )

    # Create datasetmapper
    mapper = DAFNeDatasetMapper(
        cfg,
        is_train=True,
        use_instance_mask=True,
        augmentations=augmentations,
    )

    return build_detection_train_loader(cfg, mapper=mapper)


def build_test_loader(cfg, dataset_name):
    """
    Returns:
        iterable

    It now calls :func:`detectron2.data.build_detection_test_loader`.
    Overwrite it if you'd like a different data loader.
    """

    # Collect augmentations
    augmentations = []

    # Choose between shortest-edge and rescale resizing
    if cfg.INPUT.RESIZE_TYPE == "shortest-edge":
        # Build resize augmentation
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
        augmentations.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    elif cfg.INPUT.RESIZE_TYPE == "both":
        h = cfg.INPUT.RESIZE_HEIGHT_TRAIN
        w = cfg.INPUT.RESIZE_WIDTH_TRAIN
        augmentations.append(T.Resize(shape=(h, w)))
    else:
        raise RuntimeError(f"Invalid resize-type: {cfg.INPUT.RESIZE_TYPE}")

    mapper = DAFNeDatasetMapper(
        cfg,
        is_train=True,  # TODO: Shouldn't this be False?
        use_instance_mask=True,
        augmentations=augmentations,
    )

    return build_detection_test_loader(cfg, dataset_name, mapper=mapper)


def do_test(cfg, model, evaluators: Dict = None):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        exp_name = cfg.EXPERIMENT_NAME
        setproctitle(f"@SL_{exp_name}|eval:{dataset_name}")
        logger.info(f'Starting testing on dataset "{dataset_name}"')
        data_loader = build_test_loader(cfg, dataset_name)
        if evaluators is None:
            evaluator = get_evaluator(
                cfg,
                dataset_name,
                os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name),
            )
        else:
            evaluator = evaluators[dataset_name]
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        logger.info(f"Evaluation results for dataset {dataset_name}")
        logger.info("\n" + pformat(results_i))

    return results


def do_test_with_TTA(cfg, model):
    logger = logging.getLogger("dafne.trainer")
    # In the end of training, run an evaluation with TTA
    # Only support some R-CNN models.
    logger.info("Running inference with test-time augmentation ...")
    model = OneStageRCNNWithTTA(cfg, model)

    evaluators = {
        name: get_evaluator(
            cfg,
            name,
            output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA", name),
        )
        for name in cfg.DATASETS.TEST
    }
    res = do_test(cfg, model, evaluators)
    res = OrderedDict({k + "_TTA": v for k, v in res.items()})

    return res


def save_test_results(results, cfg, iteration):
    """Evaluate the model at a specific iteration and save mAP results."""
    if comm.is_main_process():
        for dataset_name, dataset_result in results.items():

            if not "task1" in dataset_result:
                continue

            task_1_results = dataset_result["task1"]
            d = os.path.join(cfg.OUTPUT_DIR, "map_evaluations")
            Path(d).mkdir(exist_ok=True)
            fname = os.path.join(d, dataset_name + ".csv")
            with open(fname, "a") as f:
                map_value = task_1_results["map"]
                f.write(f"{iteration},{map_value}\n")


def setup_rtpt(experiment_name, max_iter, start_iter):
    rtpt = RTPT(
        name_initials="SL",
        experiment_name=experiment_name,
        max_iterations=max_iter,
        iteration_start=start_iter,
        update_interval=50,
        moving_avg_window_size=50,
    )
    rtpt.start()
    return rtpt


def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer_custom(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )

    resume_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )

    if resume:
        start_iter = resume_iter
    else:
        start_iter = 0

    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    # Setup process name updater
    rtpt = setup_rtpt(
        experiment_name=cfg.EXPERIMENT_NAME,
        max_iter=max_iter,
        start_iter=start_iter,
    )

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    data_loader_train = build_train_loader(cfg)

    # Create a config clone where the test dataset is the first original test set but as "mini" version
    # This set is used to evaluate the performance during the training
    # cfg_mini_test = cfg.clone()
    # cfg_mini_test.defrost()
    # cfg_mini_test.DATASETS.TEST = (cfg.DATASETS.TEST[0] + "_mini",)
    # cfg_mini_test.freeze()



    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        data_time = time.perf_counter()

        for data, iteration in zip(data_loader_train, range(start_iter, max_iter)):
            data_time = time.perf_counter() - data_time
            iteration = iteration + 1
            storage.step()

            loss_dict = model(data)
            losses = sum(loss_dict.values())

            optimizer.zero_grad()

            # Backward pass
            losses.backward()

            # use a new stream so the ops don't wait for DDP
            with torch.cuda.stream(
                torch.cuda.Stream()
            ) if losses.device.type == "cuda" else contextlib.nullcontext():
                metrics_dict = loss_dict
                metrics_dict["data_time"] = data_time
                write_metrics(storage, metrics_dict)
                detect_anomaly(losses, loss_dict, iteration)

            optimizer.step()

            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

            scheduler.step()


            if (
                cfg.TEST.EVAL_PERIOD > 0
                and iteration % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter
            ):
                results = do_test(cfg, model)
                save_test_results(results=results, cfg=cfg, iteration=iteration)

            if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

            # Reset data time
            data_time = time.perf_counter()

            # Update process title
            progress = iteration / max_iter * 100
            rtpt.step(subtitle=f"[{progress:0>2.0f}%]")



def is_debug_session(cfg) -> bool:
    """TODO: move to utils"""
    if cfg.DEBUG.OVERFIT_NUM_IMAGES > 0:
        return True
    if "debug" in cfg.OUTPUT_DIR.lower():
        return True
    if cfg.SOLVER.MAX_ITER < 10000:
        return True
    # TODO: Add more cases where debugging is enabled
    return False


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

def custom_auto_scale_workers(cfg, num_workers):
    cfg = cfg.clone()
    frozen = cfg.is_frozen()
    cfg.defrost()
    old_world_size = cfg.SOLVER.REFERENCE_WORLD_SIZE
    scale = num_workers / old_world_size

    if frozen:
        cfg.freeze()

    return cfg

def main(args):
    cfg = setup(args)
    # Scale config according to number of GPUs
    cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())


    logger = logging.getLogger(__name__)

    # Register the datasets
    register_dota(cfg)
    register_hrsc(cfg)

    model = build_model(cfg)

    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )

        results = do_test(cfg, model)
        # results = {}

        # Run testing with test-time-augmentation
        if cfg.TEST.AUG.ENABLED:
            results_tta = do_test_with_TTA(cfg, model)
            results.update(results_tta)

        return results

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
        )

    try:
        # Start training + testing
        do_train(cfg, model, resume=args.resume)

        results = do_test(cfg, model)
        save_test_results(results, cfg, iteration=cfg.SOLVER.MAX_ITER)

        if cfg.TEST.AUG.ENABLED:
            results_tta = do_test_with_TTA(cfg, model)
            results.update(results_tta)

        if not is_debug_session(cfg):
            send_mail_success(cfg, results)

        return results

    except KeyboardInterrupt:  # Catch keyboard interruptions

        # Rename output dir
        src = cfg.OUTPUT_DIR
        dst = src + "_interrupted"

        logger.error(f"Keyboard interruption catched.")
        logger.error(f"Moving output directory from")
        logger.error(src)
        logger.error("to")
        logger.error(dst)

        if comm.is_main_process():
            shutil.move(src, dst)

    except Exception as e:
        # Log error message
        tbstr = "".join(traceback.extract_tb(e.__traceback__).format())
        errormsg = f"Traceback:\n{tbstr}\nError: {e}"

        logger.error(errormsg)

        # Rename output dir
        src = cfg.OUTPUT_DIR
        dst = src + "_error"

        if comm.is_main_process():
            # send_mail_error(cfg, args, errormsg)

            # Write error to separate file
            with open(os.path.join(cfg.OUTPUT_DIR, "error.txt"), "w") as f:
                f.write(errormsg)

            shutil.move(src, dst)


        logger.error("Moving output directory from")
        logger.error(src)
        logger.error("to")
        logger.error(dst)
        raise e
    finally:
        synchronize()


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
