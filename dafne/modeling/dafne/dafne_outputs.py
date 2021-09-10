import logging

import torch
import torch.nn.functional as F
from fvcore.nn import sigmoid_focal_loss_jit
from torch import distributed as dist
from torch import nn

from detectron2.layers import cat
from detectron2.structures import Instances
from detectron2.structures.boxes import Boxes
from detectron2.utils.comm import get_world_size
from dafne.modeling.losses.smooth_l1 import ModulatedEightPointLoss, SmoothL1Loss
from dafne.modeling.nms.nms import ml_nms
from dafne.utils.sort_corners import sort_quadrilateral

logger = logging.getLogger(__name__)

INF = 100000000

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    L: number of feature maps per image on which RPN is run
    Hi, Wi: height and width of the i-th feature map
    8: size of the box parameterization

Naming convention:

    labels: refers to the ground-truth class of an position.

    reg_targets: refers to the 4-d (left, top, right, bottom) distances that parameterize the ground-truth box.

    logits_pred: predicted classification scores in [-inf, +inf];

    reg_pred: the predicted (left, top, right, bottom), corresponding to reg_targets

    ctrness_pred: predicted centerness scores

"""


def reduce_sum(tensor):
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def dist_point_to_line(p1, p2, x0, y0):
    """
    https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    Line defined by P1=(x1,y1), P2=(x2,y2)
    Point defined by P0=(x0, y0)
    """
    x1, y1 = p1.unbind(2)
    x2, y2 = p2.unbind(2)
    nom = torch.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denom = torch.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

    return nom / denom


def compute_abcd(corners, xs_ext, ys_ext):
    num_locs = len(xs_ext)
    num_targets = corners.shape[0]
    corners_rep = corners[None].repeat(num_locs, 1, 1)
    c0, c1, c2, c3 = corners_rep.view(num_locs, num_targets, 4, 2).unbind(2)

    left = torch.stack((c0, c1, c2, c3), dim=-1)
    right = torch.stack((c1, c2, c3, c0), dim=-1)
    abcd = dist_point_to_line(left, right, xs_ext[..., None], ys_ext[..., None])
    return abcd


def compute_ctrness_targets(reg_targets, alpha):
    if len(reg_targets) == 0:
        return reg_targets.new_zeros(len(reg_targets))
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    ctrness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
        top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]
    )

    ctrness = ctrness ** (1 / alpha)

    # Set critical cases where the ctrness computation was not possible to zero
    ctrness[torch.isnan(ctrness)] = 0.0

    return ctrness


def _cross2d(x, y):
    """Cross product in 2D."""
    return x[:, :, 0] * y[:, :, 1] - x[:, :, 1] * y[:, :, 0]


def area_triangle(a, b, c):
    """Area of a triangle"""
    x = a - c
    y = b - c
    crs = 1 / 2 * torch.abs(_cross2d(x, y))
    return crs


def is_in_quadrilateral(c0, c1, c2, c3, poly_area, loc):
    """Check if loc is in the given quadrilateral.
    Assumes, that the quadrilateral is sorted."""
    # Compute area between edges and loc
    a = area_triangle(c0, c1, loc)
    b = area_triangle(c1, c2, loc)
    c = area_triangle(c2, c3, loc)
    d = area_triangle(c3, c0, loc)
    sum_area_to_loc = a + b + c + d

    return ~(sum_area_to_loc > (poly_area + 1e-3))  # 1e-3 is some epsilon to avoid equality


class DAFNeOutputs(nn.Module):
    def __init__(self, cfg):
        super(DAFNeOutputs, self).__init__()
        self.cfg = cfg

        self.focal_loss_alpha = cfg.MODEL.DAFNE.LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.DAFNE.LOSS_GAMMA
        self.center_sample = cfg.MODEL.DAFNE.CENTER_SAMPLE
        self.radius = cfg.MODEL.DAFNE.POS_RADIUS
        self.pre_nms_thresh_train = cfg.MODEL.DAFNE.INFERENCE_TH_TRAIN
        self.pre_nms_topk_train = cfg.MODEL.DAFNE.PRE_NMS_TOPK_TRAIN
        self.post_nms_topk_train = cfg.MODEL.DAFNE.POST_NMS_TOPK_TRAIN
        self.sort_corners = cfg.MODEL.DAFNE.SORT_CORNERS

        logspace = cfg.MODEL.DAFNE.ENABLE_LOSS_LOG
        beta = cfg.MODEL.DAFNE.LOSS_SMOOTH_L1_BETA

        if cfg.MODEL.DAFNE.ENABLE_LOSS_MODULATION:
            self.corners_loss_func = ModulatedEightPointLoss(
                beta=beta,
                reduction="sum",
                logspace=logspace,
            )
        else:
            self.corners_loss_func = SmoothL1Loss(
                beta=beta,
                reduction="sum",
                logspace=logspace,
            )

        self.center_loss_func = SmoothL1Loss(
            beta=beta,
            reduction="sum",
            logspace=logspace,
        )

        self.pre_nms_thresh_test = cfg.MODEL.DAFNE.INFERENCE_TH_TEST
        self.pre_nms_topk_test = cfg.MODEL.DAFNE.PRE_NMS_TOPK_TEST
        self.post_nms_topk_test = cfg.MODEL.DAFNE.POST_NMS_TOPK_TEST
        self.nms_thresh = cfg.MODEL.DAFNE.NMS_TH
        self.thresh_with_ctr = cfg.MODEL.DAFNE.THRESH_WITH_CTR
        self.centerness_mode = cfg.MODEL.DAFNE.CENTERNESS
        self.centerness_alpha = cfg.MODEL.DAFNE.CENTERNESS_ALPHA
        self.has_centerness = self.centerness_mode != "none"
        assert self.centerness_mode in ["none", "plain", "oriented"]
        self.corner_prediction_strategy = cfg.MODEL.DAFNE.CORNER_PREDICTION
        self.has_center_reg = self.corner_prediction_strategy == "center-to-corner"
        self.num_classes = cfg.MODEL.DAFNE.NUM_CLASSES
        self.strides = cfg.MODEL.DAFNE.FPN_STRIDES

        # Lambdas
        self.lambda_cls = cfg.MODEL.DAFNE.LOSS_LAMBDA.CLS
        self.lambda_ctr = cfg.MODEL.DAFNE.LOSS_LAMBDA.CTR
        self.lambda_corners = cfg.MODEL.DAFNE.LOSS_LAMBDA.CORNERS
        self.lambda_center = cfg.MODEL.DAFNE.LOSS_LAMBDA.CENTER
        self.lambda_ltrb = cfg.MODEL.DAFNE.LOSS_LAMBDA.LTRB
        lambda_normalize = cfg.MODEL.DAFNE.LOSS_LAMBDA_NORM

        if lambda_normalize:
            self.normalize_lambdas()

        # generate sizes of interest
        soi = []
        prev_size = -1
        for s in cfg.MODEL.DAFNE.SIZES_OF_INTEREST:
            soi.append([prev_size, s])
            prev_size = s
        soi.append([prev_size, INF])
        self.sizes_of_interest = soi

    def normalize_lambdas(self):
        # Make them sum up to one
        lambda_sum = self.lambda_cls + self.lambda_corners
        if self.has_centerness:
            lambda_sum += self.lambda_ctr

        if self.has_center_reg:
            lambda_sum += self.lambda_center


        self.lambda_cls = self.lambda_cls / lambda_sum
        self.lambda_ctr = self.lambda_ctr / lambda_sum
        self.lambda_corners = self.lambda_corners / lambda_sum
        self.lambda_center = self.lambda_center / lambda_sum
        self.lambda_ltrb = self.lambda_ltrb / lambda_sum

    def update_lambdas(
        self,
        lambda_cls=None,
        lambda_ctr=None,
        lambda_corners=None,
        lambda_center=None,
        normalize=False,
    ):
        if lambda_cls is not None:
            self.lambda_cls = lambda_cls
        else:
            self.lambda_cls = self.cfg.MODEL.DAFNE.LOSS_LAMBDA.CLS

        if lambda_ctr is not None:
            self.lambda_ctr = lambda_ctr
        else:
            self.lambda_ctr = self.cfg.MODEL.DAFNE.LOSS_LAMBDA.CTR

        if lambda_corners is not None:
            self.lambda_corners = lambda_corners
        else:
            self.lambda_corners = self.cfg.MODEL.DAFNE.LOSS_LAMBDA.CORNERS

        if lambda_center is not None:
            self.lambda_center = lambda_center
        else:
            self.lambda_center = self.cfg.MODEL.DAFNE.LOSS_LAMBDA.CENTER

        if normalize:
            self.normalize_lambdas()

    def _transpose(self, training_targets, num_loc_list):
        """
        This function is used to transpose image first training targets to level first ones
        :return: level first training targets
        """
        for im_i in range(len(training_targets)):
            training_targets[im_i] = torch.split(training_targets[im_i], num_loc_list, dim=0)

        targets_level_first = []
        for targets_per_level in zip(*training_targets):
            targets_level_first.append(torch.cat(targets_per_level, dim=0))
        return targets_level_first

    def _get_ground_truth(self, locations, gt_instances):
        num_loc_list = [len(loc) for loc in locations]

        # compute locations to size ranges
        loc_to_size_range = []
        for l, loc_per_level in enumerate(locations):
            loc_to_size_range_per_level = loc_per_level.new_tensor(self.sizes_of_interest[l])
            loc_to_size_range.append(loc_to_size_range_per_level[None].expand(num_loc_list[l], -1))

        loc_to_size_range = torch.cat(loc_to_size_range, dim=0)
        locations = torch.cat(locations, dim=0)

        training_targets = self.compute_targets_for_locations(
            locations, gt_instances, loc_to_size_range, num_loc_list
        )

        training_targets["locations"] = [locations.clone() for _ in range(len(gt_instances))]
        training_targets["im_inds"] = [
            locations.new_ones(locations.size(0), dtype=torch.long) * i
            for i in range(len(gt_instances))
        ]

        # transpose im first training_targets to level first ones
        training_targets = {
            k: self._transpose(v, num_loc_list) for k, v in training_targets.items()
        }

        training_targets["fpn_levels"] = [
            loc.new_ones(len(loc), dtype=torch.long) * level
            for level, loc in enumerate(training_targets["locations"])
        ]

        # we normalize reg_targets by FPN's strides here
        reg_targets_corners = training_targets["reg_targets_corners"]
        reg_targets_ltrb = training_targets["reg_targets_ltrb"]
        reg_targets_abcd = training_targets["reg_targets_abcd"]

        if self.cfg.MODEL.DAFNE.ENABLE_FPN_STRIDE_NORM:
            for l in range(len(reg_targets_corners)):
                reg_targets_corners[l] = reg_targets_corners[l] / float(self.strides[l])
                reg_targets_ltrb[l] = reg_targets_ltrb[l] / float(self.strides[l])
                reg_targets_abcd[l] = reg_targets_abcd[l] / float(self.strides[l])

        return training_targets

    def get_sample_region(
        self, boxes, strides, num_loc_list, loc_xs, loc_ys, bitmasks=None, radius=1
    ):
        if bitmasks is not None:
            _, h, w = bitmasks.size()

            ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
            xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

            m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
            m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
            m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
            center_x = m10 / m00
            center_y = m01 / m00
        else:
            center_x = boxes[..., [0, 2]].sum(dim=-1) * 0.5
            center_y = boxes[..., [1, 3]].sum(dim=-1) * 0.5

        num_gts = boxes.shape[0]
        K = len(loc_xs)
        boxes = boxes[None].expand(K, num_gts, 4)
        center_x = center_x[None].expand(K, num_gts)
        center_y = center_y[None].expand(K, num_gts)
        center_gt = boxes.new_zeros(boxes.shape)
        # no gt
        if center_x.numel() == 0 or center_x[..., 0].sum() == 0:
            return loc_xs.new_zeros(loc_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, num_loc in enumerate(num_loc_list):
            end = beg + num_loc
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(
                xmin > boxes[beg:end, :, 0], xmin, boxes[beg:end, :, 0]
            )
            center_gt[beg:end, :, 1] = torch.where(
                ymin > boxes[beg:end, :, 1], ymin, boxes[beg:end, :, 1]
            )
            center_gt[beg:end, :, 2] = torch.where(
                xmax > boxes[beg:end, :, 2], boxes[beg:end, :, 2], xmax
            )
            center_gt[beg:end, :, 3] = torch.where(
                ymax > boxes[beg:end, :, 3], boxes[beg:end, :, 3], ymax
            )
            beg = end
        left = loc_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - loc_xs[:, None]
        top = loc_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - loc_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def compute_targets_for_locations(self, locations, targets, size_ranges, num_loc_list):
        labels = []
        reg_targets_corners = []
        reg_targets_ltrb = []
        reg_targets_abcd = []
        target_inds = []
        xs, ys = locations[:, 0], locations[:, 1]


        K = len(xs)
        num_targets = 0
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            bboxes = targets_per_im.gt_boxes.tensor
            num_gts = bboxes.shape[0]
            corners = targets_per_im.gt_corners
            area = targets_per_im.gt_corners_area
            labels_per_im = targets_per_im.gt_classes
            locations_to_gt_area = area[None].repeat(K, 1)

            # no gt
            if bboxes.numel() == 0:
                labels.append(labels_per_im.new_zeros(locations.size(0)) + self.num_classes)
                reg_targets_ltrb.append(locations.new_zeros((locations.size(0), 4)))
                reg_targets_abcd.append(locations.new_zeros((locations.size(0), 4)))
                reg_targets_corners.append(locations.new_zeros((locations.size(0), 8)))
                target_inds.append(labels_per_im.new_zeros(locations.size(0)) - 1)
                continue

            xs_ext = xs[:, None]
            ys_ext = ys[:, None]

            # Generate ltrb values
            l = xs_ext - bboxes[:, 0][None]
            t = ys_ext - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs_ext
            b = bboxes[:, 3][None] - ys_ext
            reg_targets_ltrb_per_im = torch.stack([l, t, r, b], dim=2)

            reg_targets_abcd_per_im = compute_abcd(corners, xs_ext, ys_ext)

            # Compute corner w.r.t. locations (expand for each location)
            x0_centered = corners[:, 0][None] - xs_ext
            y0_centered = corners[:, 1][None] - ys_ext
            x1_centered = corners[:, 2][None] - xs_ext
            y1_centered = corners[:, 3][None] - ys_ext
            x2_centered = corners[:, 4][None] - xs_ext
            y2_centered = corners[:, 5][None] - ys_ext
            x3_centered = corners[:, 6][None] - xs_ext
            y3_centered = corners[:, 7][None] - ys_ext

            reg_targets_corners_per_im = torch.stack(
                [
                    x0_centered,
                    y0_centered,
                    x1_centered,
                    y1_centered,
                    x2_centered,
                    y2_centered,
                    x3_centered,
                    y3_centered,
                ],
                dim=2,
            )

            if self.center_sample:
                if targets_per_im.has("gt_bitmasks_full"):
                    bitmasks = targets_per_im.gt_bitmasks_full
                else:
                    bitmasks = None
                is_in_boxes_center_sampling = self.get_sample_region(
                    bboxes,
                    self.strides,
                    num_loc_list,
                    xs,
                    ys,
                    bitmasks=bitmasks,
                    radius=self.radius,
                )
            else:
                is_in_boxes_center_sampling = reg_targets_ltrb_per_im.min(dim=2)[0] > 0

            # is_in_boxes = is_in_boxes_ltrb

            if self.cfg.MODEL.DAFNE.CENTER_SAMPLE_ONLY:
                # Only use center sampling
                is_in_boxes = is_in_boxes_center_sampling
            else:
                # IS_IN_BOXES for quadrilateral
                corners_rep = corners[None].repeat(K, 1, 1)
                is_in_boxes_quad = is_in_quadrilateral(
                    corners_rep[..., 0:2],
                    corners_rep[..., 2:4],
                    corners_rep[..., 4:6],
                    corners_rep[..., 6:8],
                    locations_to_gt_area,
                    locations[:, None],
                )

                # Combine center_sampling + is_in_quadrilateral with logical and
                if self.cfg.MODEL.DAFNE.COMBINE_CENTER_SAMPLE:
                    is_in_boxes = is_in_boxes_center_sampling & is_in_boxes_quad
                else:
                    # Only use box-check sampling
                    is_in_boxes = is_in_boxes_quad

            max_reg_targets_per_im = reg_targets_ltrb_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = (max_reg_targets_per_im >= size_ranges[:, [0]]) & (
                max_reg_targets_per_im <= size_ranges[:, [1]]
            )

            if self.cfg.MODEL.DAFNE.ENABLE_IN_BOX_CHECK:
                locations_to_gt_area[is_in_boxes == 0] = INF

            if self.cfg.MODEL.DAFNE.ENABLE_LEVEL_SIZE_FILTERING:
                locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_ltrb_per_im = reg_targets_ltrb_per_im[
                range(len(locations)), locations_to_gt_inds
            ]
            reg_targets_abcd_per_im = reg_targets_abcd_per_im[
                range(len(locations)), locations_to_gt_inds
            ]
            reg_targets_corners_per_im = reg_targets_corners_per_im[
                range(len(locations)), locations_to_gt_inds
            ]
            target_inds_per_im = locations_to_gt_inds + num_targets
            num_targets += len(targets_per_im)

            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = self.num_classes

            labels.append(labels_per_im)
            reg_targets_ltrb.append(reg_targets_ltrb_per_im)
            reg_targets_abcd.append(reg_targets_abcd_per_im)
            reg_targets_corners.append(reg_targets_corners_per_im)
            target_inds.append(target_inds_per_im)

        return {
            "labels": labels,
            "reg_targets_ltrb": reg_targets_ltrb,
            "reg_targets_abcd": reg_targets_abcd,
            "reg_targets_corners": reg_targets_corners,
            "target_inds": target_inds,
        }

    def losses(
        self,
        logits_pred,
        corners_reg_pred,
        center_reg_pred,
        ltrb_reg_pred,
        ctrness_pred,
        locations,
        gt_instances,
        top_feats=None,
    ):
        """
        Return the losses from a set of DAFNE predictions and their associated ground-truth.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
        """

        training_targets = self._get_ground_truth(locations, gt_instances)

        # Collect all logits and regression predictions over feature maps
        # and images to arrive at the same shape as the labels and targets
        # The final ordering is L, N, H, W from slowest to fastest axis.

        instances = Instances((0, 0))
        instances.labels = cat(
            [
                # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
                x.reshape(-1)
                for x in training_targets["labels"]
            ],
            dim=0,
        )
        instances.gt_inds = cat(
            [
                # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
                x.reshape(-1)
                for x in training_targets["target_inds"]
            ],
            dim=0,
        )
        instances.im_inds = cat([x.reshape(-1) for x in training_targets["im_inds"]], dim=0)
        instances.reg_targets_corners = cat(
            [
                # Reshape: (N, Hi, Wi, 8) -> (N*Hi*Wi, 8)
                x.reshape(-1, 8)
                for x in training_targets["reg_targets_corners"]
            ],
            dim=0,
        )
        instances.reg_targets_ltrb = cat(
            [
                # Reshape: (N, Hi, Wi, 4) -> (N*Hi*Wi, 4)
                x.reshape(-1, 4)
                for x in training_targets["reg_targets_ltrb"]
            ],
            dim=0,
        )
        instances.reg_targets_abcd = cat(
            [
                # Reshape: (N, Hi, Wi, 4) -> (N*Hi*Wi, 4)
                x.reshape(-1, 4)
                for x in training_targets["reg_targets_abcd"]
            ],
            dim=0,
        )

        instances.locations = cat([x.reshape(-1, 2) for x in training_targets["locations"]], dim=0)
        instances.fpn_levels = cat([x.reshape(-1) for x in training_targets["fpn_levels"]], dim=0)

        instances.logits_pred = cat(
            [
                # Reshape: (N, C, Hi, Wi) -> (N, Hi, Wi, C) -> (N*Hi*Wi, C)
                x.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
                for x in logits_pred
            ],
            dim=0,
        )
        instances.corners_reg_pred = cat(
            [
                # Reshape: (N, B, Hi, Wi) -> (N, Hi, Wi, B) -> (N*Hi*Wi, B)
                x.permute(0, 2, 3, 1).reshape(-1, 8)
                for x in corners_reg_pred
            ],
            dim=0,
        )
        if self.has_center_reg:
            instances.center_reg_pred = cat(
                [x.permute(0, 2, 3, 1).reshape(-1, 2) for x in center_reg_pred],
                dim=0,
            )


        if self.has_centerness:
            instances.ctrness_pred = cat(
                [
                    # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
                    x.permute(0, 2, 3, 1).reshape(-1)
                    for x in ctrness_pred
                ],
                dim=0,
            )

        if len(top_feats) > 0:
            instances.top_feats = cat(
                [
                    # Reshape: (N, -1, Hi, Wi) -> (N*Hi*Wi, -1)
                    x.permute(0, 2, 3, 1).reshape(-1, x.size(1))
                    for x in top_feats
                ],
                dim=0,
            )

        return self.dafne_losses(instances)

    def dafne_losses(self, instances):
        num_classes = instances.logits_pred.size(1)
        assert num_classes == self.num_classes

        labels = instances.labels.flatten()

        pos_inds = torch.nonzero(labels != num_classes).squeeze(1)
        num_pos_local = pos_inds.numel()
        num_gpus = get_world_size()
        total_num_pos = reduce_sum(pos_inds.new_tensor([num_pos_local])).item()
        num_pos_avg = max(total_num_pos / num_gpus, 1.0)

        # prepare one_hot
        class_target = torch.zeros_like(instances.logits_pred)
        class_target[pos_inds, labels[pos_inds]] = 1

        class_loss = (
            sigmoid_focal_loss_jit(
                instances.logits_pred,
                class_target,
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="sum",
            )
            / num_pos_avg
        )

        instances = instances[pos_inds]
        instances.pos_inds = pos_inds

        if self.centerness_mode == "oriented":
            ctrness_targets = compute_ctrness_targets(
                instances.reg_targets_abcd, self.centerness_alpha
            )
        elif self.centerness_mode == "plain":
            ctrness_targets = compute_ctrness_targets(
                instances.reg_targets_ltrb, self.centerness_alpha
            )
        else:
            ctrness_targets = compute_ctrness_targets(
                instances.reg_targets_abcd, self.centerness_alpha
            )
            ctrness_targets[:] = 1.0

        ctrness_targets_sum = ctrness_targets.sum()
        loss_denorm = max(reduce_sum(ctrness_targets_sum).item() / num_gpus, 1e-6)
        instances.gt_ctrs = ctrness_targets

        if pos_inds.numel() > 0:

            # Sort corners if flag is set
            # NOTE: targets are sorted in the datasetmapper
            if self.sort_corners:
                instances.corners_reg_pred = sort_quadrilateral(instances.corners_reg_pred)

            corners_reg_loss = (
                self.corners_loss_func(
                    instances.corners_reg_pred,
                    instances.reg_targets_corners,
                    ctrness_targets,
                )
                / loss_denorm
            )

            reg_targets_center = instances.reg_targets_corners.view(-1, 4, 2).mean(1)
            if self.has_center_reg:
                center_reg_loss = (
                    self.center_loss_func(
                        instances.center_reg_pred,
                        reg_targets_center,
                        ctrness_targets,
                    )
                    / loss_denorm
                )


            if self.has_centerness:
                ctrness_loss = (
                    F.binary_cross_entropy_with_logits(
                        instances.ctrness_pred, ctrness_targets, reduction="sum"
                    )
                    / num_pos_avg
                )
        else:
            corners_reg_loss = instances.corners_reg_pred.sum() * 0
            if self.has_center_reg:
                center_reg_loss = instances.center_reg_pred.sum() * 0

            if self.has_centerness:
                ctrness_loss = instances.ctrness_pred.sum() * 0


        # Apply lambdas
        class_loss = class_loss * self.lambda_cls
        corners_reg_loss = corners_reg_loss * self.lambda_corners

        losses = {
            "loss/cls": class_loss,
            "loss/corners": corners_reg_loss,
        }

        # Add center reg
        if self.has_center_reg:
            losses["loss/center"] = center_reg_loss * self.lambda_center

        # Add centerness if not none
        if self.has_centerness:
            losses["loss/ctr"] = ctrness_loss * self.lambda_ctr


        extras = {"instances": instances, "loss_denorm": loss_denorm}
        return extras, losses

    def predict_proposals(
        self,
        logits_pred,
        corners_reg_pred,
        ctrness_pred,
        locations,
        image_sizes,
        top_feats=None,
    ):
        if self.training:
            self.pre_nms_thresh = self.pre_nms_thresh_train
            self.pre_nms_topk = self.pre_nms_topk_train
            self.post_nms_topk = self.post_nms_topk_train
        else:
            self.pre_nms_thresh = self.pre_nms_thresh_test
            self.pre_nms_topk = self.pre_nms_topk_test
            self.post_nms_topk = self.post_nms_topk_test

        sampled_boxes = []

        bundle = {
            "l": locations,
            "o": logits_pred,
            "rc": corners_reg_pred,
            "c": ctrness_pred,
            "s": self.strides,
        }

        if len(top_feats) > 0:
            bundle["t"] = top_feats

        for i, per_bundle in enumerate(zip(*bundle.values())):
            # get per-level bundle
            per_bundle = dict(zip(bundle.keys(), per_bundle))
            # recall that during training, we normalize regression targets with FPN's stride.
            # we denormalize them here.
            l = per_bundle["l"]
            o = per_bundle["o"]
            if self.cfg.MODEL.DAFNE.ENABLE_FPN_STRIDE_NORM:
                rc = per_bundle["rc"] * per_bundle["s"]
            else:
                rc = per_bundle["rc"]
            c = per_bundle["c"]
            t = per_bundle["t"] if "t" in bundle else None

            sampled_boxes.append(self.forward_for_single_feature_map(l, o, rc, c, image_sizes, t))

            for per_im_sampled_boxes in sampled_boxes[-1]:
                per_im_sampled_boxes.fpn_levels = (
                    l.new_ones(len(per_im_sampled_boxes), dtype=torch.long) * i
                )

        boxlists = list(zip(*sampled_boxes))

        boxlists = [Instances.cat(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)

        return boxlists

    def forward_for_single_feature_map(
        self,
        locations,
        logits_pred,
        corners_reg_pred,
        ctrness_pred,
        image_sizes,
        top_feat=None,
    ):
        N, C, H, W = logits_pred.shape

        # put in the same format as locations
        logits_pred = logits_pred.view(N, C, H, W).permute(0, 2, 3, 1)
        cls_pred = logits_pred.reshape(N, -1, C).sigmoid()
        box_regression_corners = corners_reg_pred.view(N, 8, H, W).permute(0, 2, 3, 1)
        box_regression_corners = box_regression_corners.reshape(N, -1, 8)
        ctrness_pred = ctrness_pred.view(N, 1, H, W).permute(0, 2, 3, 1)
        ctrness_pred = ctrness_pred.reshape(N, -1)

        # Only apply sigmoid if centerness is enabled, else keep dummy "1.0" values
        if self.has_centerness:
            ctrness_pred = ctrness_pred.sigmoid()

        if top_feat is not None:
            top_feat = top_feat.view(N, -1, H, W).permute(0, 2, 3, 1)
            top_feat = top_feat.reshape(N, H * W, -1)

        # if self.thresh_with_ctr is True, we multiply the classification
        # scores with centerness scores before applying the threshold.
        if self.has_centerness and self.thresh_with_ctr:
            cls_pred = torch.sqrt(cls_pred * ctrness_pred[:, :, None])

        candidate_inds = cls_pred > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.reshape(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_topk)

        if self.has_centerness and not self.thresh_with_ctr:
            cls_pred = torch.sqrt(cls_pred * ctrness_pred[:, :, None])

        results = []
        for i in range(N):
            per_box_cls = cls_pred[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]

            per_box_regression_corners = box_regression_corners[i]
            per_box_regression_corners = per_box_regression_corners[per_box_loc]
            per_locations = locations[per_box_loc]
            per_box_centerness = ctrness_pred[i, per_box_loc]
            if top_feat is not None:
                per_top_feat = top_feat[i]
                per_top_feat = per_top_feat[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression_corners = per_box_regression_corners[top_k_indices]
                per_locations = per_locations[top_k_indices]
                per_box_centerness = per_box_centerness[top_k_indices]
                if top_feat is not None:
                    per_top_feat = per_top_feat[top_k_indices]

            detections_poly = torch.stack(
                [
                    per_locations[:, 0] + per_box_regression_corners[:, 0],
                    per_locations[:, 1] + per_box_regression_corners[:, 1],
                    per_locations[:, 0] + per_box_regression_corners[:, 2],
                    per_locations[:, 1] + per_box_regression_corners[:, 3],
                    per_locations[:, 0] + per_box_regression_corners[:, 4],
                    per_locations[:, 1] + per_box_regression_corners[:, 5],
                    per_locations[:, 0] + per_box_regression_corners[:, 6],
                    per_locations[:, 1] + per_box_regression_corners[:, 7],
                ],
                dim=1,
            )

            # Sort quadrilateral to have a canonical representation
            if self.sort_corners:
                detections_poly = sort_quadrilateral(detections_poly)

            if type(image_sizes[i]) == torch.Tensor:
                image_size = tuple(image_sizes[i].tolist())
            else:
                image_size = image_sizes[i]
            boxlist = Instances(image_size)

            # Generate surrounding hboxes from corners
            if detections_poly.shape[0] > 0:
                xmin = torch.min(detections_poly[:, 0::2], dim=1).values
                xmax = torch.max(detections_poly[:, 0::2], dim=1).values
                ymin = torch.min(detections_poly[:, 1::2], dim=1).values
                ymax = torch.max(detections_poly[:, 1::2], dim=1).values
                hbboxes = torch.stack((xmin, ymin, xmax, ymax), dim=1)
            else:
                hbboxes = detections_poly.new_empty(0, 4)

            boxlist.pred_boxes = Boxes(hbboxes)
            boxlist.pred_corners = detections_poly
            boxlist.scores = per_box_cls
            boxlist.centerness = per_box_centerness
            # boxlist.scores = torch.sqrt(per_box_cls)
            boxlist.pred_classes = per_class
            boxlist.locations = per_locations
            if top_feat is not None:
                boxlist.top_feat = per_top_feat
            results.append(boxlist)

        return results

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.post_nms_topk > 0:
                cls_scores = result.scores
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(), number_of_detections - self.post_nms_topk + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results
