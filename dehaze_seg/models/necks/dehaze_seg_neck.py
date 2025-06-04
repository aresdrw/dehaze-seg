""" 
-*- coding: utf-8 -*-
    @Time    : 2025/4/28  10:12
    @Author  : AresDrw
    @File    : dehaze_seg_neck.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
import math
from typing import List

from mmseg.models.builder import NECKS
import torch.nn as nn
import torch
import torch.nn.functional as F
import pywt
import torch_dct as dct


# ----------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 双路径特征聚合（参考网页6的CAM设计）
        avg_out = self.fc(self.avg_pool(x).squeeze(dim=[2, 3]))
        max_out = self.fc(self.max_pool(x).squeeze(dim=[2, 3]))
        weights = self.sigmoid(avg_out + max_out).unsqueeze(2).unsqueeze(3)
        return x * weights


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 空间特征融合（参考网页12的SAM实现）
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        weights = self.sigmoid(self.conv(combined))
        return x * weights


class CBAM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.ca = ChannelAttention(channel)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)  # 通道注意力
        x = self.sa(x)  # 空间注意力
        return x


class WindowSelfAttention(nn.Module):
    def __init__(self, dim, window_size=16, num_heads=8, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # QKV投影矩阵[5](@ref)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # 相对位置编码表[6,7](@ref)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        self._init_relative_position_index()

        # 缩放因子[2](@ref)
        self.scale = self.head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)

    def _init_relative_position_index(self):
        # 生成窗口内相对位置索引[7](@ref)
        coords = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid(coords, coords, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 输入特征图
        Returns:
            (B, C, H, W) 增强后的特征图
        """
        B, C, H, W = x.shape

        # 窗口划分[6](@ref)
        x = x.view(B, C, H // self.window_size, self.window_size,
                   W // self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # [B, nH, nW, ws, ws, C]
        x = x.view(-1, self.window_size * self.window_size, C)  # [B*nH*nW, ws^2, C]

        # 生成QKV[5](@ref)
        qkv = self.qkv(x).reshape(-1, self.window_size ** 2, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # [B*nH*nW, heads, ws^2, head_dim]

        # 计算注意力矩阵[2](@ref)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 添加相对位置偏置[7](@ref)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size ** 2, self.window_size ** 2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn += relative_position_bias.unsqueeze(0)

        # 注意力权重归一化[5](@ref)
        attn = self.softmax(attn)

        # 特征聚合[2](@ref)
        x = (attn @ v).transpose(1, 2).reshape(-1, self.window_size, self.window_size, C)

        # 窗口合并[6](@ref)
        x = x.view(B, H // self.window_size, W // self.window_size,
                   self.window_size, self.window_size, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, H, W, C)

        return self.proj(x).view(B, C, H, W)


class SpatialGate(nn.Module):
    def __init__(self, dim, reduction_ratio=16, kernel_size=7):
        super().__init__()
        hidden_dim = dim // reduction_ratio

        # 通道压缩（Triplet Attention的双路径压缩思想）
        self.channel_avg = nn.AdaptiveAvgPool2d(1)
        self.channel_max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

        # 空间卷积（深度可分离卷积优化计算量）
        self.dw_conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, groups=1)

        # 动态尺度因子（参考SGFN的残差连接）
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape

        # 通道压缩（双路径聚合）
        avg_out = self.mlp(self.channel_avg(x).squeeze(-1).squeeze(-1))
        max_out = self.mlp(self.channel_max(x).squeeze(-1).squeeze(-1))
        channel_weights = torch.sigmoid(avg_out + max_out).view(B, C, 1, 1)

        # 空间注意力（深度卷积增强局部上下文）
        spatial_avg = torch.mean(x, dim=1, keepdim=True)
        spatial_max, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([spatial_avg, spatial_max], dim=1)
        spatial_weights = torch.sigmoid(self.dw_conv(spatial))

        # 动态融合（引入残差连接避免梯度消失）
        return x * (channel_weights + spatial_weights) + self.gamma * x


class SemanticEnhancer(nn.Module):
    def __init__(self, in_channels):
        """
        :param in_channels: [24, 48, 96, 48, 24]
        out: [768 = (96 + 96) * 4] cause 128->32, 16
        """
        super().__init__()
        # 金字塔池化（网页5的多尺度特征增强）
        self.hidden_dim = in_channels[2] * 4
        self.out_dim = in_channels[2] * 8
        self.pyramid_pool = nn.Sequential(
            nn.Conv2d(in_channels[2], in_channels[2], (3, 3), stride=(2, 2), padding=1),
            nn.Conv2d(in_channels[2], self.hidden_dim, (3, 3), stride=(2, 2), padding=1),
        )
        # 多分支特征精炼（网页4的MSDCL改进）
        self.refine_blocks = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, (3, 3), dilation=(2, 2), padding=2),
            nn.Conv2d(self.hidden_dim, self.out_dim, (3, 3), dilation=(2, 2), padding=2)
        )

    def forward(self, l3):
        p16 = self.pyramid_pool(l3)  # 生成1/16特征
        p32 = F.max_pool2d(p16, 2)  # 生成1/32特征（网页8的步幅卷积优化）
        return self.refine_blocks(p16), self.refine_blocks(p32)


class InteractiveFusion(nn.Module):
    def __init__(self, in_channels):
        """
            in_channels=[48, 48]
            out: [384 = (48 + 48) * 4] cause 256->64
        """
        super().__init__()
        self.out_channels = in_channels[1] * 8
        self.expand = nn.Conv2d(in_channels[1] * 2, in_channels[1] * 4, (1, 1))
        self.spatial_attn = SpatialGate(in_channels[1] * 4)
        # 多尺度下采样（网页4的扩张卷积思想）
        self.down = nn.Sequential(
            nn.Conv2d(in_channels[1] * 4, self.out_channels, (3, 3), stride=(4, 4), padding=1),
            CBAM(self.out_channels)  # 网页2的注意力增强
        )

    def forward(self, l2, l4):
        # 特征拼接与通道对齐（网页7的降采样单元改进）
        concat_feat = torch.cat([l2, l4], dim=1)  # [B, 96, 256, 256]
        expanded = self.expand(concat_feat)
        return self.down(self.spatial_attn(expanded))  # 空间注意力引导（网页3的特征校正模块）


class DynamicFusion(nn.Module):
    def __init__(self, in_channels, hidden_dim=96):
        super().__init__()
        """
            in_channels=[24, 48, 96, 48, 24]
            out: [192 = (24 + 24) * 4] cause 512->128
        """
        self.out_channels = in_channels[0] * 8
        self.channel_expand = nn.Sequential(
            nn.Conv2d(in_channels[0], hidden_dim, (1, 1)),  # [96]
            nn.GELU()
        )
        # 跨层注意力（网页1窗口自注意力改进）
        self.cross_attn = WindowSelfAttention(hidden_dim)
        # 双路径特征对齐（网页6的级联融合思想）
        self.downsample = nn.Sequential(
            nn.Conv2d(hidden_dim, self.out_channels, (3, 3), stride=(4, 4), padding=1),
            nn.GroupNorm(4, self.out_channels)
        )

    def forward(self, l1, l5):
        fused = self.channel_expand(l1 + l5)  # 通道扩展与特征融合 [B, 96, 512, 512]
        attn_weight = self.cross_attn(fused)  # 动态注意力增强 [B, 96, 512, 512]
        return self.downsample(fused * attn_weight)  # 空间降采样(步幅卷积策略) [B, 192, 128, 128] # 48*4


@NECKS.register_module()
class DehazeFusionNeck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fusion_1_4 = DynamicFusion(in_channels)  # 1/4尺度融合(layer1 + layer5)
        self.fusion_1_8 = InteractiveFusion(in_channels)  # 1/8尺度融合(layer2 + layer4)
        self.semantic_enhance = SemanticEnhancer(in_channels)  # 1/16 & 1/32 生成(layer3)

        # 位置嵌入增强（参考网页3的全局上下文机制）
        self.pos_embed = nn.ParameterDict({
            '1_4': nn.Parameter(torch.randn(1, out_channels[0], 128, 128) * 0.02),
            '1_8': nn.Parameter(torch.randn(1, out_channels[1], 64, 64) * 0.02),
            '1_16': nn.Parameter(torch.randn(1, out_channels[2], 32, 32) * 0.02),
            '1_32': nn.Parameter(torch.randn(1, out_channels[3], 16, 16) * 0.02)
        })

    def forward(self, features):
        """
            features: 包含五层特征的列表
            [layer1, layer2, layer3, layer4, layer5]
            各层尺寸：
            layer1: [B,24,512,512]
            layer2: [B,48,256,256]
            layer3: [B,96,128,128]
            layer4: [B,48,256,256]
            layer5: [B,24,512,512]

            最终输出：
                [24*8, 48*8, 96*8, 96*8]
        """
        # 阶段1：多尺度特征融合
        f1_4 = self.fusion_1_4(features[0], features[4])  # [B, 192, 128, 128]
        f1_8 = self.fusion_1_8(features[1], features[3])  # [B, 384, 64, 64]
        f1_16, f1_32 = self.semantic_enhance(features[2])  # [B, 768, 32, 32], [B, 768, 16, 16]

        pyramid = {
            '1_4': f1_4 + self.pos_embed['1_4'],
            '1_8': f1_8 + self.pos_embed['1_8'],
            '1_16': f1_16 + self.pos_embed['1_16'],
            '1_32': f1_32 + self.pos_embed['1_32']
        }

        # 阶段3：多尺度特征归一化（参考网页4的RMSNorm改进）
        return [
            F.layer_norm(pyramid['1_4'], [self.out_channels[0], 128, 128]),
            F.layer_norm(pyramid['1_8'], [self.out_channels[1], 64, 64]),
            F.layer_norm(pyramid['1_16'], [self.out_channels[2], 32, 32]),
            F.layer_norm(pyramid['1_32'], [self.out_channels[3], 16, 16])
        ]


# ----------------Freq version -------------------
class WaveletTransform(nn.Module):
    def __init__(self, wavelet='haar', in_channels=3):
        super().__init__()
        # 生成小波滤波器（参考网页3的滤波器生成方法）
        if not hasattr(self, 'dec_lo'):
            dec_lo, dec_hi, _, _ = pywt.Wavelet(wavelet).filter_bank

            # 注册为buffer实现参数持久化（网页4的参数处理策略）
            self.register_buffer('dec_lo', torch.Tensor(dec_lo)[None, None, :].repeat(in_channels, 1, 1))
            self.register_buffer('dec_hi', torch.Tensor(dec_hi)[None, None, :].repeat(in_channels, 1, 1))

    def forward(self, x):
        # 低频分量卷积
        B, C, H, W = x.shape
        ll = F.conv2d(x, self.dec_lo.unsqueeze(-1), stride=2, padding=(1, 0), groups=C)
        # 高频分量卷积（网页6的分解逻辑）
        lh = F.conv2d(x, self.dec_hi.unsqueeze(-1), stride=2, padding=(1, 0), groups=C)
        hl = F.conv2d(x, self.dec_lo.unsqueeze(-1).transpose(2, 3), stride=2, padding=(0, 1), groups=C)
        hh = F.conv2d(x, self.dec_hi.unsqueeze(-1).transpose(2, 3), stride=2, padding=(0, 1), groups=C)
        return torch.cat([ll, lh, hl, hh], dim=1)


class InverseWaveletTransform(nn.Module):
    def __init__(self, wavelet='haar', in_channels=3):
        super().__init__()
        # 生成重构滤波器
        _, _, rec_lo, rec_hi = pywt.Wavelet(wavelet).filter_bank
        self.rec_lo = torch.Tensor(rec_lo)[None, None, :].repeat(in_channels, 1, 1)
        self.rec_hi = torch.Tensor(rec_hi)[None, None, :].repeat(in_channels, 1, 1)

        self.register_buffer('rec_lo', self.rec_lo)
        self.register_buffer('rec_hi', self.rec_hi)

    def forward(self, x):
        # 通道拆分（网页3的特征重组策略）
        ll, lh, hl, hh = torch.chunk(x, 4, dim=1)

        # 逆变换卷积（网页5的边界处理方案）
        pad = (self.rec_lo.shape[-1] - 1) // 2
        x_ll = F.conv_transpose2d(ll, self.rec_lo.unsqueeze(-1), stride=2,
                                  padding=pad, groups=x.size(1))
        x_lh = F.conv_transpose2d(lh, self.rec_hi.unsqueeze(-1), stride=2,
                                  padding=pad, groups=x.size(1))
        x_hl = F.conv_transpose2d(hl, self.rec_lo.unsqueeze(-1).transpose(2, 3),
                                  stride=2, padding=pad, groups=x.size(1))
        x_hh = F.conv_transpose2d(hh, self.rec_hi.unsqueeze(-1).transpose(2, 3),
                                  stride=2, padding=pad, groups=x.size(1))
        return (x_ll + x_lh + x_hl + x_hh) / 2.0


class DCT2D_Processor:
    def __init__(self, block_size=8):
        self.block_size = block_size  # 分块大小（网页9的硬件优化策略）

    def __call__(self, x):
        # 分块处理（网页2的DCT分块思想）
        B, C, H, W = x.shape
        x = x.view(B * C, 1, H, W)

        # 执行2D DCT（网页8的库函数调用）
        dct_coeff = dct.dct_2d(x, norm='ortho')

        # 分块处理（网页10的工程实践）
        blocks = []
        for i in range(0, H, self.block_size):
            for j in range(0, W, self.block_size):
                block = dct_coeff[..., i:i + self.block_size, j:j + self.block_size]
                blocks.append(block)
        return torch.stack(blocks, dim=2)


class CustomDCT2D(nn.Module):
    def __init__(self, size=8):
        super().__init__()
        # 预计算DCT基（网页9的矩阵预计算策略）
        self.dct_matrix = self._create_dct_matrix(size).cuda()

    def _create_dct_matrix(self, N):
        matrix = torch.zeros(N, N)
        for k in range(N):
            for n in range(N):
                matrix[k, n] = torch.cos((math.pi / N) * (n + 0.5) * k)
        matrix[0, :] *= 1 / math.sqrt(2)
        return matrix.T

    def forward(self, x):
        # 分块DCT变换（网页10的分块处理优化）
        B, C, H, W = x.shape
        x = x.view(B * C, H, W)
        x = torch.matmul(self.dct_matrix, x)
        x = torch.matmul(x, self.dct_matrix.T)
        return x.view(B, C, H, W)


class FreqDynamicFusion(nn.Module):
    def __init__(self, in_channels, hidden_dim=96):
        super().__init__()
        """
            in_channels=[24, 48, 96, 48, 24]
            out: [192 = (24 + 24) * 4] cause 512->128
        """
        self.out_channels = in_channels[0] * 8
        self.channel_expand = nn.Sequential(
            nn.Conv2d(in_channels[0], hidden_dim, (1, 1)),  # [96]
            nn.GELU()
        )

        self.wavelet_att = WaveletTransform(wavelet='db2', in_channels=hidden_dim)
        self.freq_gate = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim, (1, 1)),
            nn.Sigmoid()
        )

        # 跨层注意力（网页1窗口自注意力改进）
        self.cross_attn = WindowSelfAttention(hidden_dim)
        # 双路径特征对齐（网页6的级联融合思想）
        self.downsample = nn.Sequential(
            nn.Conv2d(hidden_dim, self.out_channels, (3, 3), stride=(4, 4), padding=1),
            nn.GroupNorm(4, self.out_channels)
        )

    def forward(self, l1, l5):
        fused = self.channel_expand(l1 + l5)  # 通道扩展与特征融合 [B, 96, 512, 512]
        attn_weight_spatial = self.cross_attn(fused)  # 动态注意力增强 [B, 96, 512, 512]
        wavelet_feat = self.wavelet_att(fused)  # TODO: aviod OOM
        attn_weight_freq = self.freq_gate(wavelet_feat)
        attn_weight_freq = F.interpolate(attn_weight_freq, (fused.shape[2], fused.shape[3]))
        return self.downsample(fused * (attn_weight_freq + attn_weight_spatial))
        # 空间降采样(步幅卷积策略) [B, 192, 128, 128] # 48*4


class FreqSemanticEnhancer(SemanticEnhancer):
    def __init__(self, in_channels):
        super().__init__(in_channels)
        # 添加全局DCT注意力（网页8的频域处理）
        self.dct_processor = DCT2D_Processor(block_size=16)
        self.dct_att = nn.Sequential(
            nn.Conv2d(384, 96, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(96, 384, (1, 1)),
            nn.Sigmoid()
        )

    def forward(self, l3):
        p16 = self.pyramid_pool(l3)  # [1, 384, 32, 32]
        B, C, H, W = p16.shape
        # DCT全局注意力（网页10的频域特征增强）
        dct_feat = self.dct_processor(p16).view(B, C, H, W).contiguous()  # [384, 1, 4, 16, 16]
        p16 = p16 * self.dct_att(dct_feat)
        p32 = F.max_pool2d(p16, 2)
        return self.refine_blocks(p16), self.refine_blocks(p32)


@NECKS.register_module()
class DehazeFreqFusionNeck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fusion_1_4 = FreqDynamicFusion(in_channels)  # 1/4尺度融合(layer1 + layer5)
        self.fusion_1_8 = InteractiveFusion(in_channels)  # 1/8尺度融合(layer2 + layer4)
        self.semantic_enhance = FreqSemanticEnhancer(in_channels)  # 1/16 & 1/32 生成(layer3)

        # 位置嵌入增强（参考网页3的全局上下文机制）
        self.pos_embed = nn.ParameterDict({
            '1_4': nn.Parameter(torch.randn(1, out_channels[0], 128, 128) * 0.02),
            '1_8': nn.Parameter(torch.randn(1, out_channels[1], 64, 64) * 0.02),
            '1_16': nn.Parameter(torch.randn(1, out_channels[2], 32, 32) * 0.02),
            '1_32': nn.Parameter(torch.randn(1, out_channels[3], 16, 16) * 0.02)
        })

    def forward(self, features):
        """
            backbone部分：
                输入图像尺寸[1,3,512,512]
                输出内容：
                    [x, layer1, layer2, layer3, layer4, layer5]
                    其中：x为去雾后的图像，[1,3,512,512]
                    layer1 到 layer5是去雾模型各个关键层得到的特征，类似U-Net结构，dehazeformer是最典型的
                        layer1: [B,24,512,512]
                        layer2: [B,48,256,256]
                        layer3: [B,96,128,128]
                        layer4: [B,48,256,256]
                        layer5: [B,24,512,512]
                    [24, 48, 96, 48, 24]是通道数，可以自己指定

            features: 包含五层特征的列表
            各层尺寸：
            layer1: [B,24,512,512]
            layer2: [B,48,256,256]
            layer3: [B,96,128,128]
            layer4: [B,48,256,256]
            layer5: [B,24,512,512]

            最终输出：
                out1: [B, 192, 128, 128]
                out2: [B, 384, 64, 64]
                out3: [B, 768, 32, 32]
                out4: [B, 768, 16, 16]
                通道数变化规律为：[24*8, 48*8, 96*8, 96*8]
        """
        # 阶段1：多尺度特征融合
        f1_4 = self.fusion_1_4(features[0], features[4])  # [B, 192, 128, 128]
        f1_8 = self.fusion_1_8(features[1], features[3])  # [B, 384, 64, 64]
        f1_16, f1_32 = self.semantic_enhance(features[2])  # [B, 768, 32, 32], [B, 768, 16, 16]

        pyramid = {
            '1_4': f1_4 + self.pos_embed['1_4'],
            '1_8': f1_8 + self.pos_embed['1_8'],
            '1_16': f1_16 + self.pos_embed['1_16'],
            '1_32': f1_32 + self.pos_embed['1_32']
        }

        # 阶段3：多尺度特征归一化（参考网页4的RMSNorm改进）
        return [
            F.layer_norm(pyramid['1_4'], [self.out_channels[0], 128, 128]),
            F.layer_norm(pyramid['1_8'], [self.out_channels[1], 64, 64]),
            F.layer_norm(pyramid['1_16'], [self.out_channels[2], 32, 32]),
            F.layer_norm(pyramid['1_32'], [self.out_channels[3], 16, 16])
        ]


if __name__ == "__main__":
    # 输入特征示例
    layer1 = torch.randn(1, 24, 512, 512).cuda()
    layer2 = torch.randn(1, 48, 256, 256).cuda()
    layer3 = torch.randn(1, 96, 128, 128).cuda()
    layer4 = torch.randn(1, 48, 256, 256).cuda()
    layer5 = torch.randn(1, 24, 512, 512).cuda()

    neck = DehazeFreqFusionNeck(in_channels=[24, 48, 96, 48, 24],
                                out_channels=[192, 384, 768, 768]).cuda()
    features = neck([layer1, layer2, layer3, layer4, layer5])

    # 输出特征金字塔
    print([f.shape for f in features])  # [192, 384, 768, 768]
    # [ (2,256,128,128), (2,256,64,64),
    #   (2,256,32,32), (2,256,16,16) ]
