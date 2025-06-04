""" 
-*- coding: utf-8 -*-
    @Time    : 2025/5/27  16:32
    @Author  : AresDrw
    @File    : udd.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC

from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset


@DATASETS.register_module()
class UDDDataset(BaseSegDataset, ABC):
    """ UDD dataset."""
    METAINFO = dict(
        classes=('Vetetation', 'Facade', 'Road', 'Vehicle', 'Roof', 'Other'),  # 根据实际类别修改
        palette=[[107, 142, 35], [102, 102, 156], [128, 64, 128], [0, 0, 142],     # Vegetation
                 [70, 70, 70], [0, 0, 0]])

    def __init__(self,
                 img_suffix='.JPG',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
