""" 
-*- coding: utf-8 -*-
    @Time    : 2025/4/29  21:02
    @Author  : AresDrw
    @File    : test_dehaze_effects.py
    @Software: PyCharm
    @Describe: test all the dehaze methods according the unified API
-*- encoding:utf-8 -*-
"""

import os
import argparse


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from mmengine import Config
from mmseg.utils import register_all_modules
from mmseg.models import build_segmentor
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict
from torch.utils.data import Dataset
import cv2

from dehaze_seg.core.dehaze_metric import calculate_psnr
from dehaze_seg.models.backbones.dehazeformer import *
import dehaze_seg
from dehaze_seg.utils.transforms import denorm
from tqdm import tqdm
import os.path as osp

os.chdir(osp.abspath(osp.dirname(osp.dirname(__file__))))
import sys

sys.path.append(os.curdir)

register_all_modules()

from torch.nn.functional import pad
from typing import Tuple, Optional
class BlendingSlidingWindowPredictor:
    def __init__(self,
                 model: nn.Module,
                 window_size: Tuple[int, int] = (512, 512),
                 stride: Optional[Tuple[int, int]] = None,
                 padding_mode: str = 'reflect',
                 blend_type: str = 'cosine'):
        """
        带混合权重的滑动窗口预测器

        参数:
            model: 去雾网络模型
            window_size: 滑动窗口大小 (H, W)
            stride: 滑动步长，默认为窗口大小的50%
            padding_mode: 边界填充方式 ('constant', 'reflect', 'replicate')
            blend_type: 混合类型 ('cosine'|'linear'|'gaussian')
        """
        self.model = model
        self.window_size = window_size
        self.stride = stride or (window_size[0] // 2, window_size[1] // 2)
        self.padding_mode = padding_mode
        self.blend_type = blend_type

        # 预计算混合权重（缓存提升性能）
        self.blend_mask = self._create_blend_mask(window_size, blend_type).to(next(model.parameters()).device)

    def _create_blend_mask(self,
                           window_size: Tuple[int, int],
                           blend_type: str) -> torch.Tensor:
        """生成混合权重矩阵"""
        h, w = window_size

        if blend_type == 'cosine':
            # 余弦渐变权重（中心权重高，边缘低）
            y = torch.cos(torch.linspace(math.pi / 2, 0, h, dtype=torch.float32))
            x = torch.cos(torch.linspace(math.pi / 2, 0, w, dtype=torch.float32))
            mask = torch.outer(y, x)

        elif blend_type == 'linear':
            # 线性渐变权重
            y = torch.linspace(1, 0, h, dtype=torch.float32)
            x = torch.linspace(1, 0, w, dtype=torch.float32)
            mask = torch.minimum.outer(y, x)

        elif blend_type == 'gaussian':
            # 二维高斯权重
            center_y, center_x = h // 2, w // 2
            y = torch.exp(-(torch.arange(h, dtype=torch.float32) - center_y) ** 2 / (2 * (h / 4) ** 2))
            x = torch.exp(-(torch.arange(w, dtype=torch.float32) - center_x) ** 2 / (2 * (w / 4) ** 2))
            mask = torch.outer(y, x)
            mask = mask / mask.max()  # 归一化到[0,1]

        else:
            raise ValueError(f"未知混合类型: {blend_type}")

        return mask.unsqueeze(0).unsqueeze(0)  # 形状: (1,1,H,W)

    def _compute_padding(self, h: int, w: int) -> Tuple[int, int, int, int]:
        """计算需要填充的边界大小 (pad_top, pad_bottom, pad_left, pad_right)"""
        pad_h = (self.window_size[0] - h % self.stride[0]) % self.stride[0]
        pad_w = (self.window_size[1] - w % self.stride[1]) % self.stride[1]
        return 0, pad_h, 0, pad_w

    def _get_windows(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """将输入图像划分为滑动窗口"""
        assert x.dim() == 4, "输入应为4D张量 (B,C,H,W)"
        b, c, h, w = x.shape

        # 计算填充
        pad_top, pad_bottom, pad_left, pad_right = self._compute_padding(h, w)
        x_padded = pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode=self.padding_mode)

        # 生成窗口坐标
        windows = []
        new_h, new_w = x_padded.shape[-2:]
        coords = []

        for i in range(0, new_h - self.window_size[0] + 1, self.stride[0]):
            for j in range(0, new_w - self.window_size[1] + 1, self.stride[1]):
                window = x_padded[..., i:i + self.window_size[0], j:j + self.window_size[1]]
                windows.append(window)
                coords.append((i, j))

        return torch.stack(windows, dim=0), (new_h, new_w)

    def _merge_windows(self,
                       windows: torch.Tensor,
                       target_shape: Tuple[int, int]) -> torch.Tensor:
        """使用混合权重合并窗口"""
        h, w = target_shape
        b, c, win_h, win_w = windows.shape[-4:]

        # 初始化输出和权重累加器
        output = torch.zeros((b, c, h, w), device=windows.device)
        weight = torch.zeros((b, 1, h, w), device=windows.device)

        # 应用混合权重
        idx = 0
        for i in range(0, h - win_h + 1, self.stride[0]):
            for j in range(0, w - win_w + 1, self.stride[1]):
                # 提取当前窗口区域
                output_slice = output[..., i:i + win_h, j:j + win_w]
                weight_slice = weight[..., i:i + win_h, j:j + win_w]

                # 加权累加
                output_slice += windows[idx] * self.blend_mask
                weight_slice += self.blend_mask

                idx += 1

        # 归一化处理
        weight = torch.where(weight > 1e-6, weight, torch.ones_like(weight))
        return output / weight

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """执行带混合的滑动窗口预测"""
        self.model.eval()

        # 1. 划分窗口
        windows, padded_shape = self._get_windows(x)
        orig_h, orig_w = x.shape[-2:]

        # 2. 批量预测（支持分批处理避免OOM）
        pred_windows = []
        for win in windows.split(4, dim=0):  # 每批4个窗口
            pred = self.model(win.squeeze(dim=1))  # [4, 1, 3, 512, 512] -> [4, 3, 512, 512]
            pred_windows.append(pred.unsqueeze(dim=1))  # [4, 3, 512, 512] -> [4, 1, 3, 512, 512]
        pred_windows = torch.cat(pred_windows, dim=0)

        # 3. 混合合并
        output = self._merge_windows(pred_windows, padded_shape)

        # 4. 裁剪回原始尺寸
        return output[..., :orig_h, :orig_w]


class SingleInferenceLoader(Dataset):
    def __init__(self, root_dir, mean, std, size=512, edge_decay=0, only_h_flip=False):
        self.size = size
        self.edge_decay = edge_decay
        self.only_h_flip = only_h_flip

        self.root_dir = root_dir
        self.img_names = sorted(os.listdir(os.path.join(self.root_dir)))
        self.img_num = len(self.img_names)
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        # read image, and scale [0, 1] to [-1, 1]
        img_name = self.img_names[idx]
        ori_size, source_img = read_img(os.path.join(self.root_dir, img_name), self.mean, self.std)
        return {'source': hwc_to_chw(source_img), 'filename': img_name, 'ori_size': ori_size}


def read_img(filename, means, std):
    img = cv2.imread(filename)
    img = img[:, :, ::-1].astype('float32')

    # 将列表转换为numpy数组
    means = np.array(means, dtype='float32')
    std = np.array(std, dtype='float32')

    # 归一化
    return img.shape[0:2], (img - means) / std


def write_img(filename, img):
    img = np.round((img[:, :, ::-1].copy() * 255.0)).astype('uint8')
    cv2.imwrite(filename, img)


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1]).copy()


def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0]).copy()


def single(save_dir):
    state_dict = torch.load(save_dir)['state_dict']
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    return new_state_dict


def pad_img(x, patch_size):
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x


def inference(test_loader, network, result_dir, use_window_test=False):
    if hasattr(network, 'forward_dehaze'):
        network.forward = network.forward_dehaze
    torch.cuda.empty_cache()

    mean_tensor = torch.from_numpy(np.array([123.675, 116.28, 103.53],
                                            dtype=np.float32).reshape(1, 3, 1, 1)).to('cuda:0')
    std_tensor = torch.from_numpy(np.array([58.395, 57.12, 57.375],
                                           dtype=np.float32).reshape(1, 3, 1, 1)).to('cuda:0')

    network.eval()

    os.makedirs(os.path.join(result_dir), exist_ok=True)

    if use_window_test:
        predictor = BlendingSlidingWindowPredictor(model=network, window_size=(1024, 1024), stride=(512, 512))

    for idx, batch in tqdm(enumerate(test_loader)):
        filename = batch['filename'][0]
        H, W = batch['source'].shape[2:]
        if use_window_test:
            with torch.no_grad():
                output = predictor.predict(batch['source'].cuda())
                output = torch.clamp(denorm(output, mean_tensor, std_tensor)[..., :H, :W], 0, 1)
        else:
            new_input = pad_img(batch['source'], 32)
            with torch.no_grad():
                output = model(new_input.cuda())
                output = torch.clamp(denorm(output, mean_tensor, std_tensor)[..., :H, :W], 0, 1)

        out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
        write_img(os.path.join(result_dir, filename[:-4] + '.jpg'), out_img)


"""
/hy-tmp/datasets/03-Final_Foggy_Driving_for_train/FDD/img
/hy-tmp/datasets/ACDC/rgb_anon/fog/train
/hy-tmp/paper_results/images
/hy-tmp/datasets/HazyDet_DeHaze/val/hazy_images
/hy-tmp/datasets/foggy_uavid_for_train/val/images/heavy_foggy
/hy-tmp/datasets/foggy_udd/val/images/heavy_foggy
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default='/hy-tmp/Rein/work_dirs/dehaze_seg/ablation/dehazeformer_s/alter_new/'
                                'dehaze_seg_dehazeformer-S_mask2former_HazyUavid_alter-512x512.py',
                        type=str,
                        help='model name')
    parser.add_argument('--checkpoint',
                        default='/hy-tmp/Rein/work_dirs/dehaze_seg/ablation/dehazeformer_s/alter_new/iter_60000.pth',
                        type=str, help='path to models saving')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
    parser.add_argument('--data_dir', default='/hy-tmp/datasets/foggy_udd/val/images/light_foggy',
                        type=str, help='path to dataset')

    parser.add_argument('--result_dir',
                        default='/hy-tmp/dehaze_results/04-Train-HazyUDD/Dehaze-Seg/dehazeformer-S-light/',
                        type=str, help='path to results saving')
    parser.add_argument('--dataset', default='HazyUAVid', type=str, help='dataset name')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.model.pop('train_cfg')  # to avoid the conda
    cfg.model.pop('test_cfg')

    print(f'Now is processing {args.dataset}')

    model = dehaze_seg.init_model(cfg, args.checkpoint, device='cuda:0')
    if hasattr(model, 'backbone'):
        model = model.backbone
    elif hasattr(model, 'dehaze_backbone'):
        model = model.dehaze_backbone

    dataset_dir = args.data_dir
    test_dataset = SingleInferenceLoader(dataset_dir,
                                         mean=[123.675, 116.28, 103.53],
                                         std=[58.395, 57.12, 57.375])
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             num_workers=args.num_workers,
                             pin_memory=True)

    result_dir = args.result_dir
    inference(test_loader, model, result_dir, use_window_test=True)
