""" 
-*- coding: utf-8 -*-
    @Time    : 2025/4/11  10:53
    @Author  : AresDrw
    @File    : load_multimodal_image_from_file.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
from mmcv.transforms.utils import cache_randomness
from mmseg.datasets import RandomCrop
from mmcv.transforms.builder import TRANSFORMS
import numpy as np


@TRANSFORMS.register_module()
class RandomMultiModalCrop(RandomCrop):
    def __init__(self, keys, crop_size, cat_max_ratio, **kwargs):
        super().__init__(crop_size, cat_max_ratio, **kwargs)
        self.keys = keys

    def transform(self, results: dict) -> dict:
        """Transform function to randomly crop images, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        clear_img = results['clear_img']
        heavy_foggy_img = results['heavy_foggy_img']
        light_foggy_img = results['light_foggy_img']
        intensity = results['intensity']
        crop_bbox = self.crop_bbox(results=results)

        # crop the image
        clear_img = self.crop(clear_img, crop_bbox)
        heavy_foggy_img = self.crop(heavy_foggy_img, crop_bbox)
        light_foggy_img = self.crop(light_foggy_img, crop_bbox)
        intensity = self.crop(intensity, crop_bbox)

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        results['clear_img'] = clear_img
        results['heavy_foggy_img'] = heavy_foggy_img
        results['light_foggy_img'] = light_foggy_img
        results['intensity'] = intensity
        results['img_shape'] = clear_img.shape[:2]
        return results

    @cache_randomness
    def crop_bbox(self, results: dict) -> tuple:
        """get a crop bounding box.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            tuple: Coordinates of the cropped image.
        """

        def generate_crop_bbox(img: np.ndarray) -> tuple:
            """Randomly get a crop bounding box.

            Args:
                img (np.ndarray): Original input image.

            Returns:
                tuple: Coordinates of the cropped image.
            """

            margin_h = max(img.shape[0] - self.crop_size[0], 0)
            margin_w = max(img.shape[1] - self.crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

            return crop_y1, crop_y2, crop_x1, crop_x2

        img = results['clear_img']
        crop_bbox = generate_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results['gt_seg_map'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = generate_crop_bbox(img)

        return crop_bbox
