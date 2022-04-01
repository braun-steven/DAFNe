#!/usr/bin/env python3
import argparse
import datetime
import os
import random
import time
from os.path import join as joinpath
from pathlib import Path
from typing import List


def cmd(command, split=False, interactive=False):
    if interactive:
        print(command)
        os.system(command)
        return

    print(command)
    out = os.popen(command).read()
    if split:
        return out.split("\n")
    else:
        return out


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
    base_dir = os.path.join(result_dir, base_dir, *sub_dirs)
    return base_dir


def mkdir(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run script wrapper to start DAFNe docker container.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--use-docker-root-user",
        action="store_true",
        help="Enter the docker container as a root user. This can be helpful if some files or directories have wrong permissions which need to be corrected. Only applicable if --no-run is set.",
    )
    parser.add_argument(
        "-g", "--gpus", help="Comma-separated list of GPU indices (e.g. 0,1,2,3).", required=True
    )
    parser.add_argument(
        "--opts",
        help="Additional configuration key-value pairs.",
        default="",
    )
    parser.add_argument(
        "--eval-only",
        help="Don't train and only evaluate with the given model weights.",
        action="store_true",
    )
    parser.add_argument("--resume", help="Resume training", action="store_true")
    parser.add_argument(
        "--config-file",
        help="Config file to use.",
    )
    parser.add_argument(
        "--iter-scale",
        help="Scale factor to scale up/down the number of training iterations",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--data-dir",
        help="Set the (host) data directory.",
        default=joinpath(os.environ["HOME"], "data")
    )
    parser.add_argument(
        "--output-dir",
        help="Set a fixed output directory. If None is given, a new one will be generated.",
    )
    parser.add_argument(
        "-t",
        "--tag",
        help="Tag to identify the experiment in the output dir, as well as the docker container",
        default=f"default-{int(time.time())}",
    )
    parser.add_argument(
        "-d",
        "--detach",
        default=False,
        action="store_true",
        help="Run the docker container in detach mode.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Enable debug mode (fewew iterations, le).",
    )
    parser.add_argument(
        "--shm-size",
        default=1024,
        type=int,
        help="Docker shared memory size.",
    )
    parser.add_argument(
        "--no-run",
        default=None,
        help="Don't run the python script and only start the container in a shell. Or run the given argument in a shell in the container (e.g. '... --no-run nvidia-smi').",
    )

    args = parser.parse_args()

    args.gpus = parse_gpu_arg(args.gpus)

    # Some assertions
    if args.eval_only:
        assert args.output_dir is not None

    if args.resume:
        assert args.output_dir is not None

    return args


def parse_gpu_arg(arg):
    # Parse --gpus:
    # --gpus 1,2,3  --> [1,2,3]
    # --gpus 1,3-8,10  --> [1,3,4,5,6,7,8,10]
    gpus = []
    for tag in arg.split(","):
        if "-" in tag:
            lower, upper = tag.split("-")
            lower, upper = int(lower), int(upper)
            for x in range(lower, upper + 1):
                gpus.append(str(x))
        else:
            tag = str(int(tag))  # First convert to int to assert that its correct
            gpus.append(tag)
    return ",".join(gpus)


def get_num_gpus():
    """Get the number of gpus."""
    return len(ARGS.gpus.split(","))


def is_docker_rootless():
    """Check if the docker daemon runs in rootless-mode"""
    out = cmd("docker info | grep 'rootless'").strip()
    return "rootless" in out

def get_docker_gpu_flag():
    """Get the proper GPU flag for the docker run command. Depending on whether
    the docker nvidia runtime can be found or not, a different flag will be set."""
    out = cmd("docker info | grep 'Runtimes.*nvidia'").strip()
    if out == "":
        gpu_flag = f"--gpus '\"device={ARGS.gpus}\"'"
    else:
        gpu_flag = f"--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES={ARGS.gpus}"

    return gpu_flag

def get_main_model_cache_dir():
    home_dir = os.environ["HOME"]
    path = f"{home_dir}/.torch/detectron2"
    mkdir(path)
    return path

def get_docker_volumes():
    # Get current working directory
    pwd = os.getcwd()

    home_dir = os.environ["HOME"]
    data_dir = ARGS.data_dir
    assert os.path.isdir(data_dir), f"Argument --data-dir ({ARGS.data_dir}) is not a valid directory."
    models_dir = joinpath(home_dir, "models")
    results_dir = get_results_dir()
    model_cache_dir = get_main_model_cache_dir()

    # Define volumes
    # NOTE: Volume dirs must already exist on the host file or else the docker container root user
    # will create them with root owner and permissions
    # NOTE: ":z" is necessary for read/write right in docker rootless mode
    mapping = {
        # Mount project code
        f"{pwd}": "/app/dafne:z",
        # Mount dataset, results and saved models
        f"{data_dir}": "/app/data/:z",
        f"{models_dir}": "/app/models/:z",
        f"{results_dir}": "/app/results/:z",
        f"{model_cache_dir}": "/app/.torch/detectron2:z",
    }


    # Collect
    volumes = []
    for src, dst in mapping.items():
        volumes.append(f"--volume {src}:{dst}")

    # Join into a single string
    volumes_arg = " ".join(volumes)
    return volumes_arg


def get_docker_env_vars():
    var_dict = {
        "DAFNE_DATA_DIR": "/app/data",
        "PYTHONPATH": "./",
        # "TORCH_HOME": "/app/.torch",
        "FVCORE_CACHE": "/app/.torch",
        "EMAIL_CREDENTIALS": "/app/dafne/.mail",
    }

    variables = []
    for key, value in var_dict.items():
        variables.append(f"-e {key}={value}")

    # Join into a single string
    variables_arg = " ".join(variables)
    return variables_arg


def get_docker_options():
    name = ARGS.tag.replace("=", "_").replace("+", "_")
    gpu_cmd = get_docker_gpu_flag()
    user_id = cmd("id -u", split=True)[0]
    user_group = cmd("id -g", split=True)[0]

    opts = [
        # Necessary for faster communication between processes
        # "--ipc=host",
        "--shm-size=1024m",
        # Interactive
        "--interactive",
        # Allocate pseudo tty
        "-t",
        # Remove the container after it has finished running
        "--rm",
        # Name the container (replace = with _ in tag)
        f"--name dafne_{name}",
        # Set GPU options
        gpu_cmd,
    ]

    if not (ARGS.use_docker_root_user or  is_docker_rootless()):
        opts.append(
            # Set correct user
            f"--user {user_id}:{user_group}",
        )
    else:
        # Defaults to root user
        pass


    if ARGS.detach:
        opts.append("-d")

    opts_str = " ".join(opts)
    return opts_str


def get_additional_opts():
    # Arguments passed via the command line
    opts = ARGS.opts

    if not (abs(ARGS.iter_scale - 1.0) < 1e-4):
        s = ARGS.iter_scale
        if "SOLVER.MAX_ITER" not in opts:
            opts += f" SOLVER.MAX_ITER {int(90000 * s)} "

        if "SOLVER.STEPS" not in opts:
            opts += f" SOLVER.STEPS ({int(60000 * s)},{int(80000 * s)}) "

        if "SOLVER.WARMUP_ITERS" not in opts:
            opts += f" SOLVER.WARMUP_ITERS {int(500 * s)} "

        if "TEST.EVAL_PERIOD" not in opts:
            opts += f" TEST.EVAL_PERIOD {int(9000 * s)} "


    # If debug is enabled, set some specific arguments that
    # reduce the runtime
    if ARGS.debug:
        debug_args = [
            "DEBUG.OVERFIT_NUM_IMAGES 8",
            "SOLVER.MAX_ITER 20",
            "DATALOADER.NUM_WORKERS 0",
            "MODEL.WEIGHTS ''",
        ]
        debug_args = " ".join(debug_args)
        opts += " " + debug_args

    # Escape parenthesis and ticks
    opts = opts.replace("(", r"\(")
    opts = opts.replace(")", r"\)")
    opts = opts.replace('"', r"\\\"")
    opts = opts.replace("'", r"\'")

    return opts


def get_docker_image_tag():
    """Tag of the image that should be run."""
    return "dafne"


def get_docker_arguments():
    docker_args = [
        get_docker_options(),
        get_docker_volumes(),
        get_docker_env_vars(),
        get_docker_image_tag(),
    ]
    docker_args_str = " ".join(docker_args)
    return docker_args_str


def run_no_python():
    assert ARGS.no_run is not None
    cmd(f"docker run {DOCKER_ARGS} {ARGS.no_run}", interactive=True)


def get_config_file():
    if ARGS.resume:
        output_dir = get_relative_output_dir()
        return joinpath(output_dir, "config.yaml")

    if ARGS.eval_only:
        if ARGS.output_dir is not None:
            rel = get_relative_output_dir()
            return joinpath(rel, "config.yaml")
        else:
            return ARGS.config_file

    return ARGS.config_file


def run_train_resume():
    num_gpus = get_num_gpus()
    opts = get_additional_opts()

    config_file = get_config_file()

    output_dir = get_relative_output_dir()

    lines = [
        f"docker run {DOCKER_ARGS}",
        "python ./tools/plain_train_net.py",
        f"--config-file {config_file}",
        f"--num-gpus {num_gpus}",
        "--resume",
        # f"--tag {ARGS.tag}",
        f"OUTPUT_DIR {output_dir}",
        opts,
    ]

    run_command = " ".join(lines)
    cmd(run_command, interactive=True)


def get_results_dir():
    home_dir = os.environ["HOME"]
    results_dir = joinpath(home_dir, "results")
    mkdir(results_dir)
    return results_dir


def run_train():
    num_gpus = get_num_gpus()
    opts = get_additional_opts()
    config_file = get_config_file()

    output_dir = generate_run_base_dir(result_dir="/app/results/dafne", tag=ARGS.tag)

    lines = [
        f"docker run {DOCKER_ARGS}",
        "python ./tools/plain_train_net.py",
        f"--config-file {config_file}",
        f"--num-gpus {num_gpus}",
        opts,
        f"OUTPUT_DIR {output_dir}",
        f"EXPERIMENT_NAME {ARGS.tag}",
    ]

    run_command = " ".join(lines)
    cmd(run_command, interactive=True)


def get_relative_output_dir():
    output_dir = ARGS.output_dir

    # Remove trailing slash
    if output_dir[-1] == "/":
        output_dir = output_dir[:-1]
    output_dir_rel = output_dir.split("/")[-1]
    return joinpath("/app", "results", "dafne", output_dir_rel)


def run_test():
    num_gpus = get_num_gpus()
    opts = get_additional_opts()
    config_file = get_config_file()

    output_dir = get_relative_output_dir()

    lines = [
        f"docker run {DOCKER_ARGS}",
        "python tools/plain_train_net.py",
        "--eval-only",
        "--resume" if ARGS.resume else "",
        f"--config-file {config_file}",
        f"--num-gpus {num_gpus}",
        opts,
        f"OUTPUT_DIR {output_dir}",
    ]

    run_command = " ".join(lines)
    cmd(run_command, interactive=True)


if __name__ == "__main__":
    # Make commandline arguments globally available
    ARGS = parse_args()

    # Get docker arguments
    DOCKER_ARGS = get_docker_arguments()

    # Run the appropriate docker command
    if ARGS.no_run is not None:
        run_no_python()
        exit(0)

    if ARGS.eval_only:
        run_test()
        exit(0)

    if ARGS.resume:
        run_train_resume()
        exit(0)

    # If nothing of the above was specified, just train
    run_train()
