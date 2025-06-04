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
from mmcv import RandomFlip
from mmcv.transforms.builder import TRANSFORMS


@TRANSFORMS.register_module()
class RandomMultiModalFlip(RandomFlip):
    def __init__(self, keys, prob, **kwargs):
        super().__init__(prob, **kwargs)
        self.keys = keys

    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes, semantic segmentation map and
        keypoints."""
        # flip image
        for key in self.keys:
            results[key] = mmcv.imflip(
                results[key], direction=results['flip_direction'])
        img_shape = results[self.keys[0]].shape[:2]

        # flip bboxes
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'] = self._flip_bbox(results['gt_bboxes'],
                                                   img_shape,
                                                   results['flip_direction'])

        # flip keypoints
        if results.get('gt_keypoints', None) is not None:
            results['gt_keypoints'] = self._flip_keypoints(
                results['gt_keypoints'], img_shape, results['flip_direction'])

        # flip seg map
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = self._flip_seg_map(
                results['gt_seg_map'], direction=results['flip_direction'])
            results['swap_seg_labels'] = self.swap_seg_labels
