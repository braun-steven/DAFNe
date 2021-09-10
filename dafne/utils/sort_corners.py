#!/usr/bin/env python3
import torch


def _cross2d(a, b):
    """Cross product in 2D."""
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def _remove(T, idx_remove):
    """Remove an element from the list of points for each batch element."""
    num_boxes = T.shape[0]
    num_points_left = T.shape[1]

    # Define which elements to keep
    keep = torch.ones(num_boxes, num_points_left, dtype=bool, device=T.device)
    keep[range(num_boxes), idx_remove] = False

    # Apply mask
    T = T[keep]

    # Put back in correct shape
    return T.view(num_boxes, num_points_left - 1, 2)


def sort_quadrilateral(bboxes):
    """Algorithm according to Alg. 1 in 'Learning Modulated Loss for Rotated Object Detection'.

    Sequence ordering of quadrilateral corners
    """
    assert bboxes.dim() == 2
    num_boxes = bboxes.shape[0]

    # If no boxes are present, return
    if num_boxes == 0:
        return bboxes

    device = bboxes.device
    S = bboxes.view(num_boxes, 4, 2)

    p2_ = bboxes.new_zeros(num_boxes, 2)
    p3_ = bboxes.new_zeros(num_boxes, 2)
    p4_ = bboxes.new_zeros(num_boxes, 2)

    # Find leftmost vertext
    leftmost_idx = S[:, :, 0].min(dim=1).indices  # TODO what if two have the same xmin
    p1_ = S[range(num_boxes), leftmost_idx]

    # Remove leftmost from set
    S = _remove(S, leftmost_idx)
    # S: [3, 2]

    # Track which batch elements have already been set
    done = bboxes.new_zeros(num_boxes, dtype=bool)
    S_new = bboxes.new_zeros(num_boxes, 2, 2)
    for i in range(S.shape[1]):
        s1 = S[:, i]
        S_ = _remove(S, i)
        # S_: [2, 2]
        s2, s3 = S_[:, 0], S_[:, 1]

        l = _cross2d(s1 - p1_, s2 - p1_)
        r = _cross2d(s1 - p1_, s3 - p1_)

        cond = (l * r) < 0.0
        cond = cond & ~done
        p3_[cond] = s1[cond]
        S_new[cond] = torch.stack((s2[cond], s3[cond]), dim=1)
        done = done | cond

        # Stop early if all are already done
        if torch.all(done):
            break

    S = S_new

    done = bboxes.new_zeros(num_boxes, dtype=bool)
    for i in range(S.shape[1]):
        s1 = S[:, i]
        S_ = _remove(S, i)
        s2 = S_[:, 0]

        cond = _cross2d(p3_ - p1_, s1 - p1_) > 0.0
        cond = cond & ~done
        p2_[cond] = s1[cond]
        p4_[cond] = s2[cond]
        p2_[~cond] = s2[~cond]
        p4_[~cond] = s1[~cond]
        done = done | cond

    bbox_sorted = torch.stack((p1_, p2_, p3_, p4_), dim=1).view(num_boxes, -1)
    return bbox_sorted


def cross2d(a, b):
    return a[0] * b[1] - a[1] * b[0]


def remove(T, i):
    mask = [True] * T.shape[0]
    mask[i] = False
    return T[mask]


def sort(bboxes):
    """Algorithm according to Alg. 1 in 'Learning Modulated Loss for Rotated Object Detection'.

    Sequence ordering of quadrilateral corners
    """
    num_boxes = bboxes.shape[0]
    bboxes_sorted = []
    device = bboxes.device
    for box_idx in range(num_boxes):
        S = bboxes[box_idx].view(-1, 2)

        p2_ = p3_ = p4_ = torch.zeros(2, device=device)

        # Find leftmost vertext
        leftmost_idx = S[:, 0].min(dim=0).indices  # TODO what if two have the same xmin
        p1_ = S[leftmost_idx]

        # Remove leftmost from set
        S = remove(S, leftmost_idx)
        # S: [3, 2]

        for j, s1 in enumerate(S):
            S_ = remove(S, j)
            # S_: [2, 2]
            s2, s3 = S_[0], S_[1]

            l = cross2d(s1 - p1_, s2 - p1_)
            r = cross2d(s1 - p1_, s3 - p1_)
            if l * r < 0.0:
                p3_ = s1
                S = torch.stack((s2, s3), dim=0)
                break

        for j, s1 in enumerate(S):
            S_ = remove(S, j)
            s2 = S_[0]

            if cross2d(p3_ - p1_, s1 - p1_) > 0.0:
                p2_ = s1
                p4_ = s2
            else:
                p2_ = s2
                p4_ = s1

        bbox_sorted = torch.stack((p1_, p2_, p3_, p4_), dim=0).view(-1)
        bboxes_sorted.append(bbox_sorted)

    bboxes_sorted = torch.stack((bboxes_sorted), dim=0)
    return bboxes_sorted

if __name__ == '__main__':
    import timeit
    import numpy as np


    def _run(func, label, num_boxes, device):
        bboxes = torch.randn(num_boxes, 8, device=device)

        def _func():
            func(bboxes)

        N = 1
        R = 1
        ts = timeit.repeat(_func, number=N, repeat=R)
        ts = np.array(ts) / N
        print(f"label: {label}")
        print(f"device: {device}")
        print(f"nboxes: {num_boxes}")
        mean = np.mean(ts)
        std = np.std(ts)
        print(f"mean: {mean}")
        print(f"std: {std}")
        return mean, std


    data = []
    for num_boxes in [1, 10, 100, 1000, 10000, 100000]:
        for device in ["cpu", "cuda"]:
            for label, func in [("seq", sort), ("vec", sort_quadrilateral)]:
                print()
                mean, std = _run(func=func, label=label, num_boxes=num_boxes, device=device)
                data.append([device, label, mean, std])

    print(data)
