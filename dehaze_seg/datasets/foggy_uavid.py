""" 
-*- coding: utf-8 -*-
    @Time    : 2025/4/11  10:16
    @Author  : AresDrw
    @File    : foggy_uavid.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
from abc import ABC

import mmengine
from mmseg.datasets.uavid import UAVidDataset
from mmseg.registry import DATASETS
import os.path as osp


@DATASETS.register_module()
class FoggyUAVidDataset(UAVidDataset, ABC):
    def __init__(self,
                 foggy_img_suffix='_foggy_uav_image.png',
                 clear_img_suffix='_clear_uav_image.png',
                 intensity_suffix='_fog_intensity.npy',
                 **kwargs):
        # 扩展后缀参数
        self.foggy_img_suffix = foggy_img_suffix
        self.clear_img_suffix = clear_img_suffix
        self.intensity_suffix = intensity_suffix
        super().__init__(**kwargs)  # 要放在父类构造函数的前面，后面的函数是父类的

    def load_data_list(self):
        data_list = []
        clear_img_dir = self.data_prefix.get('clear_img_path', None)
        heavy_foggy_img_dir = self.data_prefix.get('heavy_foggy_img_path', None)
        light_foggy_img_dir = self.data_prefix.get('light_foggy_img_path', None)
        intensity_dir = self.data_prefix.get('fog_intensity', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        if not osp.isdir(self.ann_file) and self.ann_file:
            assert osp.isfile(self.ann_file), f'Failed to load `ann_file` {self.ann_file}'
            files = mmengine.list_from_file(self.ann_file, backend_args=self.backend_args)  # 一个file中的多行形式
            for file in files:
                img_name = file.strip()
                data_info = dict(
                    clear_img_path=osp.join(clear_img_dir, img_name + self.clear_img_suffix),
                    heavy_foggy_img_path=osp.join(heavy_foggy_img_dir, img_name + self.foggy_img_suffix),
                    light_foggy_img_path=osp.join(light_foggy_img_dir, img_name + self.foggy_img_suffix),
                    intensity_path=osp.join(intensity_dir, img_name + self.intensity_suffix),
                )
                if ann_dir is not None:
                    seg_map = img_name + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else:
            _suffix_len = len(self.img_suffix)
            files = mmengine.list_dir_or_file(clear_img_dir, backend_args=self.backend_args)
            for file in files:
                img_name = file.replace(self.clear_img_suffix, '')
                data_info = dict(
                    clear_img_path=osp.join(clear_img_dir, img_name + self.clear_img_suffix),
                    heavy_foggy_img_path=osp.join(heavy_foggy_img_dir, img_name + self.foggy_img_suffix),
                    light_foggy_img_path=osp.join(light_foggy_img_dir, img_name + self.foggy_img_suffix),
                    intensity_path=osp.join(intensity_dir, img_name + self.intensity_suffix),
                )
                if ann_dir is not None:
                    seg_map = img_name + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
            data_list = sorted(data_list, key=lambda x: x['clear_img_path'])
        return data_list
