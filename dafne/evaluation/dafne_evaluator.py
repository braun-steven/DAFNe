import copy
from collections import OrderedDict, defaultdict
from fvcore.common.file_io import PathManager
import itertools
import os

from detectron2.utils import comm
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.data import DatasetCatalog, MetadataCatalog

import torch

import logging

logger = logging.getLogger(__name__)


class DafneEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name,
        cfg,
        distributed,
        output_dir: str = None,
    ):

        self._cfg = cfg
        self._distributed = distributed
        self._output_dir = output_dir
        self._dataset_name = dataset_name

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)

        # Collect predictions during proces(...) in this array
        self._predictions = []

    def reset(self):
        self._predictions = []
        return super().reset()

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {
                "image_id": input["image_id"],
                "file_name": input["file_name"],
                "height": input["height"],
                "width": input["width"],
            }
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["labels"] = instances.pred_classes
                prediction["scores"] = instances.scores
                prediction["corners"] = instances.pred_corners
                prediction["centerness"] = instances.centerness
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[DafneEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        self._eval_predictions(predictions)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)
