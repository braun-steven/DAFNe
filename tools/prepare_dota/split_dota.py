"""Modified from: https://github.com/dingjiansw101/AerialDetection/tree/master/DOTA_devkit"""

import utils as util
import os
import ImgSplit_multi_process
import SplitOnlyImage_multi_process
import shutil
from multiprocessing import Pool
from DOTA2COCO import DOTA2COCOTest, DOTA2COCOTrain
import argparse


def parse_args():

    parser = argparse.ArgumentParser(
        description="Split the DOTA 1.0/1.5 raw data into image patches of the given size.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--srcpath", help="DOTA 1.0/1.5 dataset raw data directory (source).", required=True
    )
    parser.add_argument(
        "--dstpath", help="DOTA 1.0/1.5 dataset split data directory (destination).", required=True
    )
    parser.add_argument(
        "--dota-version",
        required=True,
        choices=["1.0", "1.5"],
        help="DOTA dataset version, can be one of [1.0, 1.5].",
    )
    parser.add_argument("--patchsize", type=int, help="Patchsize of each image patch.", default=1024)
    parser.add_argument("--overlap", type=int, help="Overlap between image patches.", default=200)
    args = parser.parse_args()

    return args


def single_copy(src_dst_tuple):
    shutil.copyfile(*src_dst_tuple)


def filecopy(srcpath, dstpath, num_process=32):
    pool = Pool(num_process)
    filelist = util.GetFileFromThisRootDir(srcpath)

    name_pairs = []
    for file in filelist:
        basename = os.path.basename(file.strip())
        dstname = os.path.join(dstpath, basename)
        name_tuple = (file, dstname)
        name_pairs.append(name_tuple)

    pool.map(single_copy, name_pairs)


def singel_move(src_dst_tuple):
    shutil.move(*src_dst_tuple)


def filemove(srcpath, dstpath, num_process=32):
    pool = Pool(num_process)
    filelist = util.GetFileFromThisRootDir(srcpath)

    name_pairs = []
    for file in filelist:
        basename = os.path.basename(file.strip())
        dstname = os.path.join(dstpath, basename)
        name_tuple = (file, dstname)
        name_pairs.append(name_tuple)

    pool.map(filemove, name_pairs)


def getnamelist(srcpath, dstfile):
    filelist = util.GetFileFromThisRootDir(srcpath)
    with open(dstfile, "w") as f_out:
        for file in filelist:
            basename = util.mybasename(file)
            f_out.write(basename + "\n")


def prepare(srcpath, dstpath, patchsize, overlap):
    """
    :param srcpath: train, val, test
          train --> trainval1024, val --> trainval1024, test --> test1024
    :return:
    """

    if not os.path.exists(os.path.join(dstpath, f"test{patchsize}")):
        os.mkdir(os.path.join(dstpath, f"test{patchsize}"))
    if not os.path.exists(os.path.join(dstpath, f"train{patchsize}")):
        os.mkdir(os.path.join(dstpath, f"train{patchsize}"))
    if not os.path.exists(os.path.join(dstpath, f"val{patchsize}")):
        os.mkdir(os.path.join(dstpath, f"val{patchsize}"))

    srcpath_train = os.path.join(srcpath, "train")
    assert os.path.isdir(srcpath_train), f"Directory '{srcpath_train}' does not exist. Please make sure, that the DOTA dataset is properly downloaded and extracted to '{srcpath}'."
    split_train = ImgSplit_multi_process.splitbase(
        srcpath_train,
        os.path.join(dstpath, f"train{patchsize}"),
        gap=overlap,
        patchsize=patchsize,
        num_process=32,
    )
    split_train.splitdata(1)

    srcpath_val = os.path.join(srcpath, "val")
    assert os.path.isdir(srcpath_val), f"Directory '{srcpath_val}' does not exist. Please make sure, that the DOTA dataset is properly downloaded and extracted to '{srcpath}'."
    split_val = ImgSplit_multi_process.splitbase(
        srcpath_val,
        os.path.join(dstpath, f"val{patchsize}"),
        gap=overlap,
        patchsize=patchsize,
        num_process=32,
    )
    split_val.splitdata(1)

    srcpath_test = os.path.join(srcpath, "test", "images")
    assert os.path.isdir(srcpath_test), f"Directory '{srcpath_test}' does not exist. Please make sure, that the DOTA dataset is properly downloaded and extracted to '{srcpath}'."
    split_test = SplitOnlyImage_multi_process.splitbase(
        srcpath_test,
        os.path.join(dstpath, f"test{patchsize}", "images"),
        gap=overlap,
        patchsize=patchsize,
        num_process=32,
    )
    split_test.splitdata(1)

    DOTA2COCOTrain(
        os.path.join(dstpath, f"train{patchsize}"),
        os.path.join(dstpath, f"train{patchsize}", f"DOTA1_train{patchsize}.json"),
        classnames,
        difficult="-1",
    )
    DOTA2COCOTrain(
        os.path.join(dstpath, f"val{patchsize}"),
        os.path.join(dstpath, f"val{patchsize}", f"DOTA1_val{patchsize}.json"),
        classnames,
        difficult="-1",
    )
    DOTA2COCOTest(
        os.path.join(dstpath, f"test{patchsize}"),
        os.path.join(dstpath, f"test{patchsize}", f"DOTA1_test{patchsize}.json"),
        classnames,
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


def get_class_names(use_dota_1_5: bool):
    names = [
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

    # Add container-craine if 1.5 is selected
    if use_dota_1_5 == "1.5":
        names.append("container-crane")

    return names


if __name__ == "__main__":
    args = parse_args()
    srcpath = args.srcpath

    # Check if srcpath exists
    assert os.path.exists(srcpath), f"Data source path '{srcpath}' does not exist."

    dstpath = args.dstpath
    ensure_dir(dstpath + "/")

    patchsize = args.patchsize
    overlap = args.overlap

    classnames = get_class_names(args.dota_version == "1.5")
    prepare(srcpath, dstpath, patchsize, overlap)
