""" 
-*- coding: utf-8 -*-
    @Time    : 2025/5/4  13:35
    @Author  : AresDrw
    @File    : __init__.py.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
import numpy as np
aero_color_map = np.array([
    (0, 0, 0),  # Background
    (192, 128, 128),  # Person
    (0, 128, 0),  # Bike
    (128, 128, 128),  # Car
    (128, 0, 0),  # Drone
    (0, 0, 128),  # Boat
    (192, 0, 128),  # Animal
    (192, 0, 0),  # Obstacle
    (192, 128, 0),  # Construction
    (0, 64, 0),  # Vegetation
    (128, 128, 0),  # Road
    (0, 128, 128)  # Sky
], dtype=np.uint8)

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

potsdam_color_map = np.array([
    [255, 255, 255],
    [0, 0, 255],
    [0, 255, 255],
    [0, 255, 0],
    [255, 255, 0],
    [255, 0, 0]], dtype=np.uint8)

cityscapes_color_map = np.array([
    [128, 64, 128],  # road
    [244, 35, 232],  # sidewalk
    [70, 70, 70],  # building
    [102, 102, 156],  # wall
    [190, 153, 153],  # fence
    [153, 153, 153],  # pole
    [250, 170, 30],  # traffic light
    [220, 220, 0],  # traffic sign
    [107, 142, 35],  # vegetation
    [152, 251, 152],  # terrain
    [70, 130, 180],  # sky
    [220, 20, 60],  # person
    [255, 0, 0],  # rider
    [0, 0, 142],  # car
    [0, 0, 70],  # truck
    [0, 60, 100],  # bus
    [140, 230, 20],  # train
    [0, 0, 230],  # motorcycle
    [119, 11, 32]  # bicycle
], dtype=np.uint8)