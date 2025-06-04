""" 
-*- coding: utf-8 -*-
    @Time    : 2025/4/11  10:53
    @Author  : AresDrw
    @File    : load_multimodal_image_from_file.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
from mmcv import LoadImageFromFile, Resize, LoadAnnotations
from mmcv.transforms.builder import TRANSFORMS
import numpy as np


@TRANSFORMS.register_module()
class LoadMultiModalImageFromFile(LoadImageFromFile):
    def __init__(self, keys, **kwargs):
        self.keys = keys
        super().__init__(**kwargs)

    def transform(self, results):
        for key in self.keys:
            if key == 'gt_seg_map': continue
            if key != 'intensity':
                load_results = super().transform({
                    'img_path': results[f'{key}_path'],
                    'img': results.get(key, None)
                })
                results[key] = load_results.pop('img')
            else:
                results[key] = np.load(results['intensity_path'])  # ndarray:[H, W]
        return results
