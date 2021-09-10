"""
Source: https://github.com/aim-uofa/AdelaiDet/blob/master/adet/layers/deform_conv.py
"""


import torch
from torch import nn

from detectron2.layers import Conv2d


class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None

def ltrb_to_offset_mask(ltrb: torch.Tensor) -> torch.Tensor:
    """
    Convert the corner regression to an offset mask.

    Generate a 3x3 offset mask:

    off_0 = tl | off_1 | off_2 = tr
    -----------|-------|-----------
    off_3      | off_4 | off_5
    -----------|-------|-----------
    off_6 = bl | off_7 | off_8 = br

    tl = top-left, tr = top-right, bl = bottom-left, br = bottom-right.

    Where the corners off_0, off_2, off_6 and off_8 are equal to the original corners,
    off_4 is the mean of all corners (center), and off_1, off_3, off_5, off_7 are
    middle-points on the edge between the corners accordingly.

    Args:
        corners: Tensor of shape [N, 4, H, W]
    Returns:
        torch.Tensor: Offset tensor of shape [N, 18, H, W].
    """
    N, C, H, W = ltrb.shape

    l, t, r, b = ltrb.unbind(1)
    xmin, ymin, xmax, ymax = -l, -t, r, b

    # Compute corners from hbox xmin,ymin,xmax,ymax
    tl = torch.stack((ymin, xmin), dim=1)
    bl = torch.stack((ymax, xmin), dim=1)
    br = torch.stack((ymax, xmax), dim=1)
    tr = torch.stack((ymin, xmax), dim=1)

    # Corners
    off_0 = tl
    off_2 = tr
    off_8 = br
    off_6 = bl

    # Center
    off_4 = (tl + tr + br + bl) / 4

    # Middle-points on edges
    off_1 = (tl + tr) / 2
    off_5 = (tr + br) / 2
    off_7 = (bl + br) / 2
    off_3 = (tl + bl) / 2

    # Cat in column-major order
    offset = torch.cat(
        (off_0, off_1, off_2, off_3, off_4, off_5, off_6, off_7, off_8), dim=1
    )
    return offset

def hbox_to_offset_mask(hbox: torch.Tensor) -> torch.Tensor:
    """
    Convert the corner regression to an offset mask.

    Generate a 3x3 offset mask:

    off_0 = tl | off_1 | off_2 = tr
    -----------|-------|-----------
    off_3      | off_4 | off_5
    -----------|-------|-----------
    off_6 = bl | off_7 | off_8 = br

    tl = top-left, tr = top-right, bl = bottom-left, br = bottom-right.

    Where the corners off_0, off_2, off_6 and off_8 are equal to the original corners,
    off_4 is the mean of all corners (center), and off_1, off_3, off_5, off_7 are
    middle-points on the edge between the corners accordingly.

    Args:
        corners: Tensor of shape [N, 4, H, W]
    Returns:
        torch.Tensor: Offset tensor of shape [N, 18, H, W].
    """
    N, C, H, W = hbox.shape

    xmin, ymin, xmax, ymax = hbox.unbind(1)

    # Compute corners from hbox xmin,ymin,xmax,ymax
    tl = torch.stack((ymin, xmin), dim=1)
    bl = torch.stack((ymax, xmin), dim=1)
    br = torch.stack((ymax, xmax), dim=1)
    tr = torch.stack((ymin, xmax), dim=1)

    # Corners
    off_0 = tl
    off_2 = tr
    off_8 = br
    off_6 = bl

    # Center
    off_4 = (tl + tr + br + bl) / 4

    # Middle-points on edges
    off_1 = (tl + tr) / 2
    off_5 = (tr + br) / 2
    off_7 = (bl + br) / 2
    off_3 = (tl + bl) / 2

    # Cat in column-major order
    offset = torch.cat(
        (off_0, off_1, off_2, off_3, off_4, off_5, off_6, off_7, off_8), dim=1
    )
    return offset


def center_to_offset_mask(center: torch.Tensor) -> torch.Tensor:
    """
    Convert the corner regression to an offset mask.

    Generate a 3x3 offset mask that shifts the convolution sampling
    by the given center value.


    Args:
        corners: Tensor of shape [N, 8, H, W]
    Returns:
        torch.Tensor: Offset tensor of shape [N, 18, H, W].
    """
    offset = center.repeat(1, 3 * 3, 1, 1)
    return offset


def corners_to_offset_mask(corners: torch.Tensor) -> torch.Tensor:
    """
    Convert the corner regression to an offset mask.

    Generate a 3x3 offset mask:

    off_0 = c0 | off_1 | off_2 = c3
    -----------|-------|-----------
    off_3      | off_4 | off_5
    -----------|-------|-----------
    off_6 = c1 | off_7 | off_8 = c2

    Where the corners off_0, off_2, off_6 and off_8 are equal to the original corners,
    off_4 is the mean of all corners (center), and off_1, off_3, off_5, off_7 are
    middle-points on the edge between the corners accordingly.

    Args:
        corners: Tensor of shape [N, 8, H, W]
    Returns:
        torch.Tensor: Offset tensor of shape [N, 18, H, W].
    """
    N, C, H, W = corners.shape
    # Swap (x, y) coordinates since the offset arrays expects (y, x) format
    corners = corners[:, [1, 0, 3, 2, 5, 4, 7, 6], :, :]
    c0, c1, c2, c3 = corners.view(N, 4, 2, H, W).unbind(1)

    # Corners
    off_0 = c0
    off_2 = c3
    off_8 = c2
    off_6 = c1

    # Center
    off_4 = (c0 + c1 + c2 + c3) / 4

    # Middle-points on edges
    off_1 = (off_0 + off_2) / 2
    off_5 = (off_2 + off_8) / 2
    off_7 = (off_6 + off_8) / 2
    off_3 = (off_0 + off_6) / 2

    # Cat in column-major order
    offset = torch.cat(
        (off_0, off_1, off_2, off_3, off_4, off_5, off_6, off_7, off_8), dim=1
    )
    return offset


class DFConv2dNoOffset(nn.Module):
    """
    Deformable convolutional layer with configurable
    deformable groups, dilations and groups.

    Does not learn the offset mask internally but accepts an offset mask in the
    forward pass.

    Code is from:
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/layers/misc.py
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        with_modulated_dcn=False,
        kernel_size=3,
        stride=1,
        groups=1,
        dilation=1,
        deformable_groups=1,
        bias=False,
        padding=None,
    ):
        super(DFConv2dNoOffset, self).__init__()
        if isinstance(kernel_size, (list, tuple)):
            assert isinstance(stride, (list, tuple))
            assert isinstance(dilation, (list, tuple))
            assert len(kernel_size) == 2
            assert len(stride) == 2
            assert len(dilation) == 2
            padding = (
                dilation[0] * (kernel_size[0] - 1) // 2,
                dilation[1] * (kernel_size[1] - 1) // 2,
            )
        else:
            padding = dilation * (kernel_size - 1) // 2
        if with_modulated_dcn:
            from detectron2.layers.deform_conv import ModulatedDeformConv

            conv_block = ModulatedDeformConv
        else:
            from detectron2.layers.deform_conv import DeformConv

            conv_block = DeformConv
        self.conv = conv_block(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            deformable_groups=deformable_groups,
            bias=bias,
        )
        self.with_modulated_dcn = with_modulated_dcn
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x, offset):
        if x.numel() > 0:
            if not self.with_modulated_dcn:
                x = self.conv(x, offset)
            else:
                offset_mask = self.offset(x)
                offset = offset_mask[:, : self.offset_split, :, :]
                mask = offset_mask[:, self.offset_split :, :, :].sigmoid()
                x = self.conv(x, offset, mask)
            return x
        # get output shape
        output_shape = [
            (i + 2 * p - (di * (k - 1) + 1)) // d + 1
            for i, p, di, k, d in zip(
                x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
            )
        ]
        output_shape = [x.shape[0], self.conv.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class DFConv2d(nn.Module):
    """
    Deformable convolutional layer with configurable
    deformable groups, dilations and groups.
    Code is from:
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/layers/misc.py
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        with_modulated_dcn=True,
        kernel_size=3,
        stride=1,
        groups=1,
        dilation=1,
        deformable_groups=1,
        bias=False,
        padding=None,
    ):
        super(DFConv2d, self).__init__()
        if isinstance(kernel_size, (list, tuple)):
            assert isinstance(stride, (list, tuple))
            assert isinstance(dilation, (list, tuple))
            assert len(kernel_size) == 2
            assert len(stride) == 2
            assert len(dilation) == 2
            padding = (
                dilation[0] * (kernel_size[0] - 1) // 2,
                dilation[1] * (kernel_size[1] - 1) // 2,
            )
            offset_base_channels = kernel_size[0] * kernel_size[1]
        else:
            padding = dilation * (kernel_size - 1) // 2
            offset_base_channels = kernel_size * kernel_size
        if with_modulated_dcn:
            from detectron2.layers.deform_conv import ModulatedDeformConv

            offset_channels = offset_base_channels * 3  # default: 27
            conv_block = ModulatedDeformConv
        else:
            from detectron2.layers.deform_conv import DeformConv

            offset_channels = offset_base_channels * 2  # default: 18
            conv_block = DeformConv
        self.offset = Conv2d(
            in_channels,
            deformable_groups * offset_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=1,
            dilation=dilation,
        )
        for l in [
            self.offset,
        ]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            torch.nn.init.constant_(l.bias, 0.0)
        self.conv = conv_block(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            deformable_groups=deformable_groups,
            bias=bias,
        )
        self.with_modulated_dcn = with_modulated_dcn
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.offset_split = offset_base_channels * deformable_groups * 2

    def forward(self, x, return_offset=False):
        if x.numel() > 0:
            if not self.with_modulated_dcn:
                offset_mask = self.offset(x)
                x = self.conv(x, offset_mask)
            else:
                offset_mask = self.offset(x)
                offset = offset_mask[:, : self.offset_split, :, :]
                mask = offset_mask[:, self.offset_split :, :, :].sigmoid()
                x = self.conv(x, offset, mask)
            if return_offset:
                return x, offset_mask
            return x
        # get output shape
        output_shape = [
            (i + 2 * p - (di * (k - 1) + 1)) // d + 1
            for i, p, di, k, d in zip(
                x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
            )
        ]
        output_shape = [x.shape[0], self.conv.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)
