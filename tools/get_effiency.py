""" 
-*- coding: utf-8 -*-
    @Time    : 2024/11/15  13:22
    @Author  : AresDrw
    @File    : get_effiency.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
# tools/get_flops.py
import argparse
import time
import os.path as osp
from mmengine import Config
from mmengine.registry import DATASETS
import mmcv
import numpy as np
import torch
from mmseg.models import build_segmentor
from mmseg.utils import register_all_modules
from thop import profile
import dehaze_seg
import inspect

import torchvision.models.segmentation as models

# dehaze methods
from tools.compare_dehaze_methods.DehazeFormer.models.dehazeformer import *
from tools.compare_dehaze_methods.DehazeNet.DehazeNet_pytorch import DehazeNet
from tools.compare_dehaze_methods.SwinIR.network_swinir import SwinIR
from tools.compare_dehaze_methods.TransWeather.transweather_model import Transweather
from tools.compare_dehaze_methods.FFANet.models.FFA import FFA
from tools.compare_dehaze_methods.DEANet.model.backbone import Backbone

register_all_modules()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Get the FLOPs and fps of a segmentor')
    parser.add_argument('--config', help='train config file path',
                        default='/hy-tmp/Rein/configs/compare/unetformer/unetformer_r50-d8_4xb2-40k_uavid-512x512.py')
    parser.add_argument('--shape', type=int, nargs='+', default=[512, 512], help='input image size')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--log-interval', type=int, default=50, help='interval of logging')
    parser.add_argument('--work-dir',
                        help='if specified, the results will be dumped into the directory as json',
                        default='/hy-tmp/Rein/work_dirs/aeroscapes_results/classic/hrnet/')
    parser.add_argument('--repeat-times', type=int, default=1)
    args = parser.parse_args()
    return args


def get_params_flops():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.pop('train_cfg')  # to avoid the conda
    cfg.model.pop('test_cfg')

    model = build_segmentor(cfg.model).cuda()
    # model = models.deeplabv3_resnet50(pretrained=False).cuda()
    # model = DehazeNet().cuda()
    # model = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
    #                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
    #                mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv').cuda()
    # model = dehazeformer_w().cuda()
    # model = Backbone().cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    elif hasattr(model, '_forward'):
        model.forward = model._forward
    elif len(inspect.signature(model.forward).parameters) == 1:
        pass
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.format(model.__class__.__name__))

    input_tensor = torch.randn(1, 3, args.shape[0], args.shape[1]).to('cuda:0')

    with torch.no_grad():
        flops, params = profile(model, inputs=(input_tensor,))

    gflops = flops / 1e9
    print(f"GFlops: {gflops:.2f}")

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total parameters: {total_params:.2f}M")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Trainable parameters: {trainable_params:.2f}M")


def get_fps():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.pop('train_cfg')  # to avoid the conda
    cfg.model.pop('test_cfg')

    model = build_segmentor(cfg.model).cuda()
    # model = models.deeplabv3_resnet50(pretrained=False).cuda()
    # model = DehazeNet().cuda()
    # model = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
    #                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
    #                mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv').cuda()
    # model = dehazeformer_w().cuda()
    # model = Transweather().cuda()
    # model = Backbone().cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    elif hasattr(model, '_forward'):
        model.forward = model._forward
    elif len(inspect.signature(model.forward).parameters) == 1:
        pass
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.format(model.__class__.__name__))

    input_tensor = torch.randn(1, 3, args.shape[0], args.shape[1]).to('cuda:0')

    # 设置测试次数和预热次数
    num_test_iterations = 100
    num_warmup_iterations = 10

    # 预热
    with torch.no_grad():
        for _ in range(num_warmup_iterations):
            model(input_tensor)

    # 测量时间
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_test_iterations):
            model(input_tensor)
    end_time = time.time()

    # 计算平均时间和 FPS
    total_time = end_time - start_time
    average_time = total_time / num_test_iterations
    fps = 1 / average_time  # 修正:  考虑 batch size

    # 打印 FPS
    print(f"\n--- FPS Test ---")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average time per frame(s): {average_time:.4f} seconds")
    print(f"Frames Per Second (FPS): {fps:.2f}")


if __name__ == '__main__':
    get_params_flops()
    get_fps()
