""" 
-*- coding: utf-8 -*-
    @Time    : 2025/4/11  10:53
    @Author  : AresDrw
    @File    : load_multimodal_image_from_file.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
from mmseg.datasets import PackSegInputs
import warnings
import numpy as np
from mmcv.transforms import to_tensor
from mmengine.structures import PixelData
from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample


@TRANSFORMS.register_module()
class PackMultiModalInputs(PackSegInputs):

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:
            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`SegDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'heavy_foggy_img' in results:
            img = results['heavy_foggy_img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results['inputs'] = dict(heavy_foggy_img=img)

        if 'light_foggy_img' in results:
            img = results['light_foggy_img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results['inputs'].update(dict(light_foggy_img=img))

        if 'clear_img' in results:
            img = results['clear_img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results['inputs'].update(dict(clear_img=img))

        if 'intensity' in results:
            packed_results['inputs'].update(dict(intensity=to_tensor(results['intensity'].copy()[None, ...])))

        data_sample = SegDataSample()
        if 'gt_seg_map' in results:
            if len(results['gt_seg_map'].shape) == 2:
                data = to_tensor(results['gt_seg_map'][None,
                                                       ...].astype(np.int64))
            else:
                warnings.warn('Please pay attention your ground truth '
                              'segmentation map, usually the segmentation '
                              'map is 2D, but got '
                              f'{results["gt_seg_map"].shape}')
                data = to_tensor(results['gt_seg_map'].astype(np.int64))
            gt_sem_seg_data = dict(data=data)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        # if 'intensity' in results:  # depth-related
        #     gt_intensity_data = dict(
        #         data=to_tensor(results['intensity'].copy()[None, ...]))
        #     data_sample.set_data(dict(gt_intensity_map=PixelData(**gt_intensity_data)))

        if 'gt_edge_map' in results:
            gt_edge_data = dict(
                data=to_tensor(results['gt_edge_map'][None,
                                                      ...].astype(np.int64)))
            data_sample.set_data(dict(gt_edge_map=PixelData(**gt_edge_data)))

        if 'gt_depth_map' in results:
            gt_depth_data = dict(
                data=to_tensor(results['gt_depth_map'][None, ...]))
            data_sample.set_data(dict(gt_depth_map=PixelData(**gt_depth_data)))

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results
