# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications: Add prepare_debug_out
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image

from .transforms import denorm

uavid_color_map = np.array([
    (0, 0, 0),  # Background clutter
    (128, 0, 0),  # Building
    (128, 64, 128),  # Road
    (0, 128, 0),  # Tree
    (128, 128, 0),  # Low vegetation
    (64, 0, 128),  # Moving car
    (192, 0, 128),  # Static car
    (64, 64, 0)  # Human
], dtype=np.uint8)

udd_color_map = np.array([
    (107, 142, 35),  # Vegetation
    (102, 102, 156),  # Facde
    (128, 64, 128),  # Road
    (0, 0, 142),  # Vehicle
    (70, 70, 70),  # Roof
    (0, 0, 0),  # Other
], dtype=np.uint8)


def colorize(mask, color_map):
    color_mask = color_map[mask]
    img = Image.fromarray(color_mask)
    return img


def calc_entropy(prob):
    """
    :param prob: softmax of the score
    :return:
    """
    entropy_map = torch.sum(-prob * torch.log(prob + 1e-7), dim=1)
    return entropy_map


def get_segmentation_error_vis(seg, gt):
    error_mask = seg != gt
    error_mask[gt == 255] = 0
    out = seg.copy()
    out[error_mask == 0] = 255
    return out


def is_integer_array(a):
    return np.all(np.equal(np.mod(a, 1), 0))


def prepare_debug_out(title, out, mean, std):
    if len(out.shape) == 4 and out.shape[0] == 1:
        out = out[0]
    if len(out.shape) == 2:
        out = np.expand_dims(out, 0)
    assert len(out.shape) == 3
    if out.shape[0] == 3:
        if mean is not None:
            out = torch.clamp(denorm(out, mean, std), 0, 1)[0]
        out = dict(title=title, img=out)
    elif out.shape[0] > 3:
        out = torch.softmax(torch.from_numpy(out), dim=0).numpy()
        out = np.argmax(out, axis=0)
        out = dict(title=title, img=out, cmap='cityscapes')
    elif out.shape[0] == 1:
        if is_integer_array(out) and np.max(out) > 1:
            out = dict(title=title, img=out[0], cmap='cityscapes')
        elif np.min(out) >= 0 and np.max(out) <= 1:
            out = dict(title=title, img=out[0], cmap='viridis', vmin=0, vmax=1)
        else:
            out = dict(
                title=title, img=out[0], cmap='viridis', range_in_title=True)
    else:
        raise NotImplementedError(out.shape)
    return out


def subplotimg(ax,
               img,
               title=None,
               range_in_title=False,
               palette=uavid_color_map,
               **kwargs):
    if img is None:
        return
    with torch.no_grad():
        if torch.is_tensor(img):
            img = img.cpu()
        if len(img.shape) == 2:
            if torch.is_tensor(img):
                img = img.numpy()
        elif img.shape[0] == 1:
            if torch.is_tensor(img):
                img = img.numpy()
            img = img.squeeze(0)
        elif img.shape[0] == 3:
            img = img.permute(1, 2, 0)
            if not torch.is_tensor(img):
                img = img.numpy()
        if kwargs.get('cmap', '') == 'uavid':
            kwargs.pop('cmap')
            if torch.is_tensor(img):
                img = img.numpy()
            img = colorize(img, color_map=uavid_color_map)
        if kwargs.get('cmap', '') == 'udd':
            kwargs.pop('cmap')
            if torch.is_tensor(img):
                img = img.numpy()
            img = colorize(img, color_map=udd_color_map)

    if range_in_title:
        vmin = np.min(img)
        vmax = np.max(img)
        title += f' {vmin:.3f}-{vmax:.3f}'

    ax.imshow(img, **kwargs)
    if title is not None:
        ax.set_title(title)
