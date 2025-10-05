""" 
-*- coding: utf-8 -*-
    @Time    : 2025/4/10  9:02
    @Author  : AresDrw
    @File    : fog_simulator_depth.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
import os

import cv2
import numpy as np
from scipy.stats import truncnorm
from PIL import Image
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt


def truncated_normal(mean, std, low, high):
    """生成截断正态分布采样值"""
    a = (low - mean) / std
    b = (high - mean) / std
    return truncnorm(a, b, loc=mean, scale=std).rvs()


def add_synthetic_fog(img, depth_map):
    """
    基于大气散射模型添加合成雾
    参数：
        img: 原始BGR图像 (0-255 uint8)
        depth_map: 深度图 (二维float数组，单位：米)
    返回：
        带合成雾的BGR图像
    """
    # 参数采样
    A = truncated_normal(mean=0.8, std=0.05, low=0.7, high=0.9)
    beta = truncated_normal(mean=0.025, std=0.02, low=0.02, high=0.16)

    # 归一化处理
    img = img.astype(np.float32) / 255.0
    depth_map = depth_map.astype(np.float32)

    # 计算透射率 (加入数值稳定性处理)
    t = np.exp(-beta * depth_map + 1e-7)
    t = np.clip(t, 0.05, 1.0)  # 物理约束

    # 扩展维度用于广播运算
    t = t[:, :, np.newaxis]

    # 大气散射模型
    foggy_img = img * t + A * (1 - t)

    # 转换为0-255范围
    return (np.clip(foggy_img, 0, 1) * 255).astype(np.uint8), t


def normalize_depth(depth_map):
    """将深度图归一化到0-1范围"""
    min_val = np.min(depth_map)
    max_val = np.max(depth_map)

    # 防止除零（当所有深度值相同时）
    if max_val - min_val < 1e-6:
        return np.zeros_like(depth_map)

    normalized = (depth_map - min_val) / (max_val - min_val + 1e-7)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


# make foggy_uavid dataset
if __name__ == "__main__":
    # 加载图像和深度图（示例路径）
    subset = 'train'
    density = 0.025
    depth_dir = f'/hy-tmp/datasets/foggy_uavid_for_train/{subset}/depth/npy'
    src_img_dir = f'/hy-tmp/datasets/uavid_for_train/{subset}/images'
    # tgt_clear_dir = f'/hy-tmp/datasets/foggy_uavid_for_train/{subset}/images/clear'
    tgt_foggy_dir = f'/hy-tmp/datasets/foggy_uavid_for_train/{subset}/images/foggy_{density}'
    tgt_fog_intensity_dir = f'/hy-tmp/datasets/foggy_uavid_for_train/{subset}/images/intensity'

    os.makedirs(tgt_fog_intensity_dir, exist_ok=True)
    # os.makedirs(tgt_clear_dir, exist_ok=True)
    os.makedirs(tgt_foggy_dir, exist_ok=True)
    os.makedirs(os.path.join(tgt_fog_intensity_dir, 'value_inv_tx'), exist_ok=True)
    os.makedirs(os.path.join(tgt_fog_intensity_dir, 'visual_inv_tx'), exist_ok=True)
    os.makedirs(os.path.join(tgt_fog_intensity_dir, 'value_tx'), exist_ok=True)
    os.makedirs(os.path.join(tgt_fog_intensity_dir, 'visual_tx'), exist_ok=True)

    for depth_file in tqdm(os.listdir(depth_dir)):
        d_path = os.path.join(depth_dir, depth_file)
        img_base_name = depth_file[:-4][depth_file[:-4].find('seq'):].replace('_uav_image', '')
        img_path = os.path.join(src_img_dir, f'{img_base_name}_uav_image.png')
        # shutil.copy(img_path, os.path.join(tgt_clear_dir, f'{img_base_name}_uav_image.png'))

        original_img = np.array(Image.open(img_path).convert('RGB'))
        depth_map = np.load(d_path)  # 深度图保持绝对尺度

        scale = int(400 / np.max(depth_map))
        depth_map = depth_map * scale

        # 生成带雾图像
        foggy_img, t = add_synthetic_fog(original_img, depth_map)
        # np.save(os.path.join(tgt_fog_intensity_dir, 'value_tx',
        #                      f'{img_base_name}_fog_intensity.npy'), t[:, :, 0])
        # np.save(os.path.join(tgt_fog_intensity_dir, 'value_inv_tx',
        #                      f'{img_base_name}_fog_intensity.npy'), 1 - t[:, :, 0])
        # plt.imsave(os.path.join(tgt_fog_intensity_dir, 'visual_tx',
        #                         f'{img_base_name}_fog_intensity.png'), t[:, :, 0], cmap='gray')
        # plt.imsave(os.path.join(tgt_fog_intensity_dir, 'visual_inv_tx',
        #                         f'{img_base_name}_fog_intensity.png'), 1 - t[:, :, 0], cmap='gray')

        cv2.imwrite(os.path.join(tgt_foggy_dir, f'{img_base_name}_200_{density}_foggy_uav_image.png'), foggy_img)

