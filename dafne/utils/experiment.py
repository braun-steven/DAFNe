#!/usr/bin/env python3
from typing import List
import argparse
import datetime
import json
import logging
import os
import random
import sys
import time

import numpy as np

import torch
import torchvision
from torch import nn

from dafne_core.utils.comm import is_main_process
import pdb

logger = logging.getLogger(__name__)


def dafne_breakpoint():
    if is_main_process():
        pdb.set_trace()


def time_delta_now(t_start: float) -> str:
    """
    Convert the difference of the given timestamp and now into a human readable timestring.
    Args:
        t_start (float): Start timestamp.

    Returns:
        Human readable timestring of time passed between `t_start` and now.
    """
    a = t_start
    b = time.time()  # current epoch time
    c = b - a  # seconds
    days = round(c // 86400)
    hours = round(c // 3600 % 24)
    minutes = round(c // 60 % 60)
    seconds = round(c % 60)
    millisecs = round(c % 1 * 1000)
    return "{} days, {} hours, {} minutes, {} seconds, {} milliseconds".format(
        days, hours, minutes, seconds, millisecs
    )


def ensure_dir(path: str):
    """
    Ensure that a directory exists.

    For 'foo/bar/baz.csv' the directories 'foo' and 'bar' will be created if not already present.

    Args:
        path (str): Directory path.
    """
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)


def count_params(model: torch.nn.Module) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model (torch.nn.Module): PyTorch model.

    Returns:
        int: Number of learnable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_run_base_dir(
    result_dir: str, timestamp: int = None, tag: str = None, sub_dirs: List[str] = None
) -> str:
    """
    Generate a base directory for each experiment run.
    Looks like this: result_dir/date_tag/sub_dir_1/.../sub_dir_n
    Args:
        result_dir (str): Experiment output directory.
        timestamp (int): Timestamp which will be inlcuded in the form of '%y%m%d_%H%M'.
        tag (str): Tag after timestamp.
        sub_dirs (List[str]): List of subdirectories that should be created.

    Returns:
        str: Directory name.
    """
    if timestamp is None:
        timestamp = time.time()

    if sub_dirs is None:
        sub_dirs = []

    # Convert time
    date = datetime.datetime.fromtimestamp(timestamp)
    date_str = date.strftime("%y-%m-%d_%H:%M")

    # Append tag if given
    if tag is None:
        base_dir = date_str
    else:
        base_dir = date_str + "_" + tag

    # Create directory
    base_dir = os.path.join(result_dir, base_dir, *sub_dirs) + "/"
    ensure_dir(base_dir)
    return base_dir


def setup_logging(log_file_path: str = "log.txt", level: str = "INFO"):
    """
    Setup global loggers. Logs to stdout and the given log file.

    Args:
        log_file_path (str): Log file destination.
        level (str): Log level.
    """
    # Make sure the directory actually exists
    ensure_dir(os.path.dirname(log_file_path))

    # Check if previous log exists since logging.FileHandler only appends
    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    logging.basicConfig(
        level=logging.getLevelName(level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(stream=sys.stdout),
            logging.FileHandler(filename=log_file_path),
        ],
    )


def set_seed(seed: int):
    """
    Set the seed globally for python, numpy and torch.

    Args:
        seed (int): Seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_cuda_device(cuda_device_id: List[int]):
    """
    Set the `CUDA_VISIBLE_DEVICES` environment variable to the list of device ids.

    Reminder: torch's cuda device index will index the given list. That is,
    if `cuda_device_id` == [3, 4, 5], then torch.device('cuda:0') will point to
    the physical GPU #3.

    Warning is logged if `CUDA_VISIBLE_DEVICES` has already been set in the environment.

    Args:
        cuda_device_id (List[int]): List of physical cuda device ids.

    """
    key = "CUDA_VISIBLE_DEVICES"
    new_val = ",".join([str(x) for x in cuda_device_id])
    if key in os.environ:
        val = os.environ[key]
        logger.warning(
            'Environment variable "{}" exists with value "{}". Overwriting with "{}".'.format(
                key, val, new_val
            )
        )
    os.environ[key] = new_val


def make_multi_gpu(model: torch.nn.Module, cuda_device_id: List):
    """
    Distribute a given model across the cuda device list.

    If cuda_device_id == [-1], all available cuda devices will be selected.

    Args:
        model (torch.nn.Module): Model which is to be distributed across the cuda devices.
        cuda_device_id (List): List of physical cuda device ids.
    """
    multi_gpu = len(cuda_device_id) > 1 or cuda_device_id[0] == -1

    # Check if multiple cuda devices are selected
    if multi_gpu:
        logger.info("Using multiple gpus")
        logger.info("cuda_device_id=%s" % cuda_device_id)
        num_cuda_devices = torch.cuda.device_count()

        if cuda_device_id[0] == -1:
            # Select all devices
            cuda_device_id_vis = list(range(num_cuda_devices))
        else:
            # Select all visible devices
            cuda_device_id_vis = list(range(len(cuda_device_id)))

        # Check if multiple cuda devices are available
        if num_cuda_devices > 1:
            logger.info("Running experiment on the following GPUs: %s" % cuda_device_id)

            # Transform model into data parallel model on all selected cuda deviecs
            model = torch.nn.DataParallel(model, device_ids=cuda_device_id_vis)
        else:
            logger.info(
                "Attempted to run the experiment on multiple GPUs while only a single GPU was available"
            )
    return model


def load_args(base_dir: str, filename="args.txt") -> argparse.Namespace:
    """
    Load the commandline arguments.

    Args:
        base_dir (str): Directory in which the arguments are stored.
        filename (str): Filename of the stored arguments. Defaults to "args.txt".

    Returns:
        argparse.Namespace: Argparse namespace object that is constructed from the loaded arguments.

    """
    parser = argparse.ArgumentParser()
    args = parser.parse_args(args=[])
    with open(os.path.join(base_dir, filename), "r") as f:
        args.__dict__ = json.load(f)
    return args


def save_args(args: argparse.Namespace, base_dir: str):
    """
    Save the commandline arguments.

    Args:
        args (argparse.Namespace): Commandline arguments.
        base_dir (str): Directory to store the arguments into.

    """
    with open(os.path.join(base_dir, "args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)


def clone_args(args: argparse.Namespace) -> argparse.Namespace:
    """
    Clone the given namespace object.

    Args:
        args (argparse.Namespace): Namespace object which is to be cloned.

    Returns:
        argparse.Namespace: Cloned input namespace object.
    """
    parser = argparse.ArgumentParser()
    tmp_args = parser.parse_args(args=[])
    tmp_args.__dict__ = args.__dict__.copy()
    return tmp_args


def plot_samples(x: torch.Tensor, y: torch.Tensor):
    """
    Plot a single sample witht the target and prediction in the title.

    Args:
        x (torch.Tensor): Batch of input images. Has to be shape: [N, C, H, W].
        y (torch.Tensor): Target.
        y_pred: Target prediction.
        loss: Loss value.
    """
    import matplotlib.pyplot as plt

    # Normalize in valid range
    x = (x - x.min()) / (x.max() - x.min())
    tensors = torchvision.utils.make_grid(x, nrow=8, padding=1)

    # Permute channels and h/w for matplotlib
    plt.figure()
    plt.imshow(tensors.permute(1, 2, 0))
    plt.title("y={}".format(y.squeeze().numpy()))
    plt.show()
