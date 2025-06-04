""" 
-*- coding: utf-8 -*-
    @Time    : 2025/4/11  10:53
    @Author  : AresDrw
    @File    : load_multimodal_image_from_file.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
from numpy import random
from mmcv.transforms.builder import TRANSFORMS
from mmseg.datasets import PhotoMetricDistortion


@TRANSFORMS.register_module()
class AppliedPhotoMetricDistortion(PhotoMetricDistortion):
    def __init__(self, apply_to, **kwargs):
        self.apply_to = apply_to
        super().__init__(**kwargs)

    def transform(self, results: dict) -> dict:
        """Transform function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """
        for key in self.apply_to:
            img = results[key]
            # random brightness
            img = self.brightness(img)

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                img = self.contrast(img)

            # random saturation
            img = self.saturation(img)

            # random hue
            img = self.hue(img)

            # random contrast
            if mode == 0:
                img = self.contrast(img)

            results[key] = img
        return results
