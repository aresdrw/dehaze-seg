""" 
-*- coding: utf-8 -*-
    @Time    : 2025/4/23  12:25
    @Author  : AresDrw
    @File    : vgg_preputal_loss.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
import torch
import torch.nn as nn
from torchvision import models, transforms


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer_ids=[20], use_normalize=True):
        super().__init__()
        # 加载预训练VGG19并冻结参数
        vgg = models.vgg19(pretrained=True).features.eval()
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:max(layer_ids) + 1])

        # 参数冻结
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.layer_ids = layer_ids
        self.use_normalize = use_normalize
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, gen_img, target_img):
        # 输入标准化（ImageNet统计量）
        if self.use_normalize:
            gen_img = (gen_img - self.mean) / self.std
            target_img = (target_img - self.mean) / self.std

        # 特征提取
        features_gen = []
        features_target = []
        x_gen, x_target = gen_img, target_img
        for i, layer in enumerate(self.feature_extractor):
            x_gen = layer(x_gen)
            x_target = layer(x_target)
            if i in self.layer_ids:
                features_gen.append(x_gen)
                features_target.append(x_target.detach())  # 阻止梯度流向目标图像

        # 多层级损失计算
        loss = 0
        for f_gen, f_target in zip(features_gen, features_target):
            loss += torch.nn.functional.mse_loss(f_gen, f_target)

        return loss / len(self.layer_ids)
