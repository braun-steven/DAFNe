import logging
from detectron2.layers.deform_conv import DeformConv
import math
from typing import Dict, List
import numpy as np

# import e2cnn
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from dafne.layers.deform_conv import DFConv2d, DFConv2dNoOffset, ltrb_to_offset_mask


from .dafne_outputs import DAFNeOutputs

from detectron2.layers import (
    NaiveSyncBatchNorm,
    ShapeSpec,
)


logger = logging.getLogger(__name__)

__all__ = ["DAFNe"]

INF = 100000000


class Mish(nn.Module):
    """Mish activation function"""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def compute_locations(h, w, stride, device):
    shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32, device=device)
    shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.float32, device=device)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class ModuleListDial(nn.ModuleList):
    def __init__(self, modules=None):
        super(ModuleListDial, self).__init__(modules)
        self.cur_position = 0

    def forward(self, x):
        result = self[self.cur_position](x)
        self.cur_position += 1
        if self.cur_position >= len(self):
            self.cur_position = 0
        return result


@PROPOSAL_GENERATOR_REGISTRY.register()
class DAFNe(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self.in_features = cfg.MODEL.DAFNE.IN_FEATURES
        self.fpn_strides = cfg.MODEL.DAFNE.FPN_STRIDES
        self.yield_proposal = cfg.MODEL.DAFNE.YIELD_PROPOSAL

        # Construct head
        shape_list = [input_shape[f] for f in self.in_features]
        self.dafne_head = DAFNeHead(cfg, shape_list)

        self.in_channels_to_top_module = self.dafne_head.in_channels_to_top_module
        self.dafne_outputs = DAFNeOutputs(cfg)

    def forward_head(self, features, top_module=None):
        features = [features[f] for f in self.in_features]
        return self.dafne_head(features, top_module, self.yield_proposal)

    def forward(self, images, features, gt_instances=None, top_module=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        features = [features[f] for f in self.in_features]
        locations = self.compute_locations(features)
        (
            logits_pred,
            corners_reg_pred,
            center_reg_pred,
            ltrb_reg_pred,
            ctrness_pred,
            top_feats,
            towers,
        ) = self.dafne_head(images, features, top_module, self.yield_proposal)

        results = {}
        if self.yield_proposal:
            results["features"] = {}
            for i, level_name in enumerate(self.in_features):
                results["features"][level_name] = {}
                for k, v in towers.items():
                    if len(v) > 0:
                        results["features"][level_name][k] = v[i]
            self.feature_cache = results["features"]

        if self.training:
            results, losses = self.dafne_outputs.losses(
                logits_pred,
                corners_reg_pred,
                center_reg_pred,
                ltrb_reg_pred,
                ctrness_pred,
                locations,
                gt_instances,
                top_feats,
            )

            if self.yield_proposal:
                with torch.no_grad():
                    results["proposals"] = self.dafne_outputs.predict_proposals(
                        logits_pred,
                        corners_reg_pred,
                        ctrness_pred,
                        locations,
                        images.image_sizes,
                        top_feats,
                    )
            return results, losses
        else:
            results = self.dafne_outputs.predict_proposals(
                logits_pred,
                corners_reg_pred,
                ctrness_pred,
                locations,
                images.image_sizes,
                top_feats,
            )

            return results, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = compute_locations(h, w, self.fpn_strides[level], feature.device)
            locations.append(locations_per_level)
        return locations


class DAFNeHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg.MODEL.DAFNE.NUM_CLASSES
        self.fpn_strides = cfg.MODEL.DAFNE.FPN_STRIDES
        norm = None if cfg.MODEL.DAFNE.NORM == "none" else cfg.MODEL.DAFNE.NORM
        self.num_levels = len(input_shape)
        centerness_mode = cfg.MODEL.DAFNE.CENTERNESS
        assert centerness_mode in ["none", "plain", "oriented"]
        self.use_centerness = centerness_mode != "none"
        self.merge_corner_center_pred = cfg.MODEL.DAFNE.MERGE_CORNER_CENTER_PRED

        self.corner_prediction_strategy = cfg.MODEL.DAFNE.CORNER_PREDICTION
        self.corner_tower_on_center_tower = cfg.MODEL.DAFNE.CORNER_TOWER_ON_CENTER_TOWER
        assert self.corner_prediction_strategy in [
            "direct",
            "iterative",
            "center-to-corner",
            "offset",
            "angle",
        ]

        # Get in_channels
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        self.in_channels_to_top_module = in_channels

        in_channels_cls = in_channels

        modules_init_list = []

        # Construct the towers for each objective
        self.make_towers(in_channels, in_channels_cls, norm, modules_init_list)

        # Make prediction modules for class, centerness, corners and centers
        self.cls_logits = nn.Conv2d(
            in_channels_cls, self.num_classes, kernel_size=3, stride=1, padding=1
        )

        if self.use_centerness:
            self.ctrness = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)
            modules_init_list.append(self.ctrness)

        # Construct "corners" prediction from in-channels->8
        if self.corner_prediction_strategy in ["direct", "center-to-corner", "offset"]:
            self.corners_pred = nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1)
            modules_init_list.append(self.corners_pred)

        # Construct xywha prediction
        if self.corner_prediction_strategy == "angle":
            self.xywha_pred = nn.Conv2d(in_channels, 5, kernel_size=3, stride=1, padding=1)
            modules_init_list.append(self.xywha_pred)

        # Construct "center" prediction from in-channels->2
        if self.corner_prediction_strategy == "center-to-corner":
            self.center_pred = nn.Conv2d(in_channels, 2, kernel_size=3, stride=1, padding=1)
            modules_init_list.append(self.center_pred)

        if self.corner_prediction_strategy == "offset":
            # Construct base corners
            self.base_corners = nn.Parameter(
                torch.tensor([-2.0, 2.0, 2.0, 2.0, 2.0, -2.0, -2.0, -2.0]).view(1, 8, 1, 1),
                requires_grad=False,
            )

        if self.corner_prediction_strategy == "iterative":

            def _conv(_in_channels):
                return nn.Conv2d(_in_channels, 2, kernel_size=3, stride=1, padding=1)

            self.c0_pred = _conv(in_channels)
            self.c1_pred = _conv(in_channels + 2)
            self.c2_pred = _conv(in_channels + 4)
            self.c3_pred = _conv(in_channels + 6)
            modules_init_list.extend(
                [
                    self.c0_pred,
                    self.c1_pred,
                    self.c2_pred,
                    self.c3_pred,
                ]
            )

        if cfg.MODEL.DAFNE.USE_SCALE:
            self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(self.num_levels)])
        else:
            self.scales = None

        # Add predictions modules to the module list
        modules_init_list.extend(
            [
                self.cls_logits,
            ]
        )

        # Initialize modules
        for modules in modules_init_list:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d) or isinstance(l, DeformConv):
                    torch.nn.init.normal_(l.weight, std=0.01)

                    # Bias might be disabled if conv is followeb by batchnorm
                    if l.bias is not None:
                        torch.nn.init.constant_(l.bias, 0)

                if isinstance(l, DFConv2dNoOffset):
                    torch.nn.init.normal_(l.conv.weight, std=0.01)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.DAFNE.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def make_towers(self, in_channels, in_channels_cls, norm, modules_init_list):
        head_configs = {
            "cls": (
                self.cfg.MODEL.DAFNE.NUM_CLS_CONVS,
                in_channels_cls,
                self.cfg.MODEL.DAFNE.USE_DEFORMABLE,
            ),
            "corners": (
                self.cfg.MODEL.DAFNE.NUM_BOX_CONVS,
                in_channels,
                self.cfg.MODEL.DAFNE.USE_DEFORMABLE,
            ),
            "share": (self.cfg.MODEL.DAFNE.NUM_SHARE_CONVS, in_channels, False),
        }

        if self.corner_prediction_strategy == "center-to-corner":
            if not self.merge_corner_center_pred:
                head_configs["center"] = (
                    self.cfg.MODEL.DAFNE.NUM_BOX_CONVS,
                    in_channels,
                    self.cfg.MODEL.DAFNE.USE_DEFORMABLE,
                )

        for head in head_configs:
            tower = []
            num_convs, head_in_channels, use_deformable = head_configs[head]
            for i in range(num_convs):
                if use_deformable and i == num_convs - 1:
                    conv_func = DFConv2d
                    bias = False
                else:
                    conv_func = nn.Conv2d
                    bias = True
                tower.append(
                    conv_func(
                        head_in_channels,
                        head_in_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=bias,
                    )
                )
                if norm == "GN":
                    tower.append(nn.GroupNorm(head_in_channels // 8, head_in_channels))
                elif norm == "BN":
                    tower.append(
                        ModuleListDial(
                            [nn.BatchNorm2d(head_in_channels) for _ in range(self.num_levels)]
                        )
                    )
                elif norm == "SyncBN":
                    tower.append(
                        ModuleListDial(
                            [NaiveSyncBatchNorm(head_in_channels) for _ in range(self.num_levels)]
                        )
                    )
                tower.append(nn.ReLU())

            module_name = f"{head}_tower"
            self.add_module(module_name, nn.Sequential(*tower))
            modules_init_list.append(getattr(self, module_name))

    def forward(self, images, x, top_module=None, yield_corners_towers=False):
        logits = []
        corners_reg = []
        ltrb_reg = []
        center_reg = []
        ctrness = []
        top_feats = []
        corners_towers = []
        center_towers = []
        cls_towers = []

        # For each feature level
        for level, feature in enumerate(x):

            # Apply sharing tower if set, else no-op
            feature = self.share_tower(feature)

            # Apply cls tower on FPN features
            cls_tower = self.cls_tower(feature)
            cls_towers.append(cls_tower)

            # Direct corner prediction
            if self.corner_prediction_strategy == "direct":
                corners_tower = self.corners_tower(feature)
                reg_corners = self.corners_pred(corners_tower)

                if self.cfg.MODEL.DAFNE.USE_SCALE:
                    reg_corners = self.scales[level](reg_corners)
            elif self.corner_prediction_strategy == "iterative":
                corners_tower = self.corners_tower(feature)
                c0 = self.c0_pred(corners_tower)
                c1 = self.c1_pred(torch.cat((corners_tower, c0), dim=1))
                c2 = self.c2_pred(torch.cat((corners_tower, c0, c1), dim=1))
                c3 = self.c3_pred(torch.cat((corners_tower, c0, c1, c2), dim=1))
                reg_corners = torch.cat((c0, c1, c2, c3), dim=1)

                if self.cfg.MODEL.DAFNE.USE_SCALE:
                    reg_corners = self.scales[level](reg_corners)
            elif self.corner_prediction_strategy == "center-to-corner":

                if self.merge_corner_center_pred:
                    corners_tower = self.corners_tower(feature)

                    # center-to-corner vectors
                    reg_corners_delta = self.corners_pred(corners_tower)
                    reg_center = self.center_pred(corners_tower)
                    reg_corners = reg_center.repeat(1, 4, 1, 1) + reg_corners_delta
                else:
                    center_tower = self.center_tower(feature)

                    if self.corner_tower_on_center_tower:
                        corners_tower = self.corners_tower(center_tower)
                    else:
                        corners_tower = self.corners_tower(feature)

                    reg_center = self.center_pred(center_tower)
                    reg_corners_delta = self.corners_pred(corners_tower)
                    reg_corners = reg_center.repeat(1, 4, 1, 1) + reg_corners_delta

                if self.cfg.MODEL.DAFNE.USE_SCALE:
                    reg_corners = self.scales[level](reg_corners)
                    reg_center = self.scales[level](reg_center)

                # Collect center regression
                center_reg.append(reg_center)
            elif self.corner_prediction_strategy == "offset":
                corners_tower = self.corners_tower(feature)
                reg_corners_delta = self.corners_pred(corners_tower)

                # Use regression as offset to obtain the actual prediction
                reg_corners = self.base_corners + reg_corners_delta

                if self.cfg.MODEL.DAFNE.USE_SCALE:
                    reg_corners = self.scales[level](reg_corners)
            elif self.corner_prediction_strategy == "angle":
                corners_tower = self.corners_tower(feature)
                reg_xywha = self.xywha_pred(corners_tower)

                x, y, w, h, alpha = reg_xywha.unbind(1)

                # Note: x,y are the top-left corner
                c0 = torch.stack((x, y), dim=-1)
                c1 = torch.stack((x, y + h), dim=-1)
                c2 = torch.stack((x + w, y + h), dim=-1)
                c3 = torch.stack((x + w, y), dim=-1)
                corners = torch.stack((c0, c1, c2, c3), dim=-2)

                # Fix alpha to range
                alpha = torch.sigmoid(alpha) * np.pi - np.pi / 2

                # Construct rotation matrix
                sin = torch.sin(alpha)
                cos = torch.cos(alpha)
                R = torch.stack([torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -1)

                # Perform rotate the proposed box by alph
                reg_corners_mean = corners.mean(-2, keepdim=True)
                reg_corners = corners - reg_corners_mean
                reg_corners = reg_corners @ R
                reg_corners = reg_corners + reg_corners_mean
                shape = reg_corners.shape
                reg_corners = reg_corners.view(*shape[0:-2], -1)
                reg_corners = reg_corners.permute(0, 3, 1, 2)

                if self.cfg.MODEL.DAFNE.USE_SCALE:
                    reg_corners = self.scales[level](reg_corners)
            else:
                raise RuntimeError("Invalid corner prediction strategy.")

            corners_reg.append(reg_corners)
            corners_towers.append(corners_tower)

            reg_cls_logits = self.cls_logits(cls_tower)
            logits.append(reg_cls_logits)

            # Predict centerness if enabled
            if self.use_centerness:
                if self.cfg.MODEL.DAFNE.CTR_ON_REG:
                    reg_ctrness = self.ctrness(corners_tower)
                else:
                    reg_ctrness = self.ctrness(cls_tower)
                ctrness.append(reg_ctrness)
            else:
                reg_ctrness = torch.ones(
                    (feature.shape[0], 1, *feature.shape[2:]),
                    device=feature.device,
                    dtype=feature.dtype,
                )
                ctrness.append(reg_ctrness)

            if top_module is not None:
                top_feats.append(top_module(corners_tower))
        return (
            logits,
            corners_reg,
            center_reg,
            ltrb_reg,
            ctrness,
            top_feats,
            {
                "corners_towers": corners_towers,
                "center_towers": center_towers,
                "cls_towers": cls_towers,
            },
        )
