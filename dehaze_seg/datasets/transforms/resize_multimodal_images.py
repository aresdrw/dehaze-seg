""" 
-*- coding: utf-8 -*-
    @Time    : 2025/4/11  10:53
    @Author  : AresDrw
    @File    : load_multimodal_image_from_file.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
import mmcv
from mmcv import Resize
from mmcv.transforms.builder import TRANSFORMS


@TRANSFORMS.register_module()
class ResizeMultiModal(Resize):
    def __init__(self, scale, keys, **kwargs):
        self.keys = keys
        super().__init__(scale, **kwargs)

    def _resize_img(self, results: dict) -> None:
        """Resize images with ``results['scale']``."""

        for key in self.keys:
            if results.get(key, None) is not None:
                if self.keep_ratio:
                    img, scale_factor = mmcv.imrescale(
                        results[key],
                        results['scale'],
                        interpolation=self.interpolation,
                        return_scale=True,
                        backend=self.backend)
                    # the w_scale and h_scale has minor difference
                    # a real fix should be done in the mmcv.imrescale in the future
                    new_h, new_w = img.shape[:2]
                    h, w = results[key].shape[:2]
                    w_scale = new_w / w
                    h_scale = new_h / h
                else:
                    img, w_scale, h_scale = mmcv.imresize(
                        results[key],
                        results['scale'],
                        interpolation=self.interpolation,
                        return_scale=True,
                        backend=self.backend)
                results[key] = img
                results['img_shape'] = img.shape[:2]
                results['scale_factor'] = (w_scale, h_scale)
                results['keep_ratio'] = self.keep_ratio
