""" 
-*- coding: utf-8 -*-
    @Time    : 2025/4/17  16:44
    @Author  : AresDrw
    @File    : dehaze_seg_encoder_decoder.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
import os
import pickle
from typing import List

from mmseg.models.segmentors import EncoderDecoder
from mmseg.models.utils import resize

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import matplotlib.pyplot as plt

from dehaze_seg.models.losses.ssim_loss import SSIMLoss
from dehaze_seg.models.losses.vgg_preputal_loss import VGGPerceptualLoss
from dehaze_seg.utils.transforms import denorm
from dehaze_seg.utils.visualization import calc_entropy, subplotimg
from dehaze_seg.core.dehaze_metric import calculate_psnr


def froze_module(module):
    for param in module.parameters():
        param.requires_grad = False


@MODELS.register_module()
class DehazeSegEncoderDecoder(EncoderDecoder):
    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 decode_head: ConfigType,
                 use_multi_density: List,
                 use_alter=False,
                 task_selection=None,
                 **kwargs):
        super().__init__(backbone=backbone, decode_head=decode_head, neck=neck, **kwargs)
        if task_selection is None:
            task_selection = ['dehaze', 'seg']
        self.use_multi_density = use_multi_density
        self.backbone = MODELS.build(backbone)  # 分割主干网络-处理雾图）
        self.fusion_neck = MODELS.build(neck)  # 特征融合模块
        self.decode_head = MODELS.build(decode_head)  # 分割解码器
        self.means = self.data_preprocessor.mean.cuda()
        self.stds = self.data_preprocessor.std.cuda()
        self.ssim = SSIMLoss()
        self.vgg_loss = VGGPerceptualLoss()
        self.local_iter = 0
        self.debug_visual = 1000
        self.tasks = task_selection
        self.use_alter = use_alter
        with open("/hy-tmp/Rein/dummy_datasample.pkl", "rb") as f:
            self.dummy_data_samples = pickle.load(f)
        if self.tasks == ['dehaze']:
            print('Only Dehaze, the neck and seg_head has been frozen!')
            froze_module(self.decode_head)
            froze_module(self.neck)

    def extract_feat(self, inputs: dict, return_dehazed=True):
        """
        :param inputs:
        :param return_dehazed:
            single:  return fused features
            multi: return fseg_features, cseg_features, dehaze_features, pred_intensity
        :return:
        """
        """多特征提取流程"""
        dehazed, l1, l2, l3, l4, l5 = self.backbone(inputs)
        if return_dehazed:
            return dehazed, self.fusion_neck(features=[l1, l2, l3, l4, l5])
        else:
            return self.fusion_neck(features=[l1, l2, l3, l4, l5])

    def forward_dummy(self, x):
        """
            for test the effects
        :param x:
        :return:
        """
        dehaze_features = self.backbone(x)
        decode_features = self.neck(dehaze_features[1:])
        return self.get_decode_logits(decode_features, self.dummy_data_samples)

    def predict(self,
                inputs: dict,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        mode == 'predict'

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [dict(
                ori_shape=inputs['clear_img'].shape[2:],
                img_shape=inputs['clear_img'].shape[2:],
                pad_shape=inputs['clear_img'].shape[2:],
                padding_size=[0, 0, 0, 0])
                              ] * inputs['clear_img'].shape[0]

        seg_logits = self.inference(inputs['foggy_img'], batch_img_metas)

        return self.postprocess_result(seg_logits, data_samples)

    def encode_decode(self, inputs: dict,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input.
        this will be used in 'inference(x)', so it should be only 1 output
        """
        x = self.extract_feat(inputs, return_dehazed=False)
        seg_logits = self.decode_head.predict(x, batch_img_metas, self.test_cfg)
        return seg_logits

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList, **kwargs) -> dict:
        losses = dict()
        mode = kwargs['density']
        loss_decode = self.decode_head.loss(inputs, data_samples, self.train_cfg)
        losses.update(add_prefix(loss_decode, f'seg_decode_{mode}'))
        return losses

    def get_decode_logits(self, feat_pyr, data_samples):
        batch_gt_instances, batch_img_metas = self.decode_head._seg_data_to_instance_data(data_samples)
        all_cls_scores, all_mask_preds = self.decode_head.forward(feat_pyr, batch_img_metas)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        if 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape']
        else:
            size = batch_img_metas[0]['img_shape']
        # upsample mask
        mask_pred_results = F.interpolate(mask_pred_results, size=size, mode='bilinear', align_corners=False)
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        return torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)

    def loss_alter(self, inputs: dict, data_samples: SampleList) -> dict:
        # 特征提取
        losses = dict()
        batch_size = inputs['clear_img'].size()[0]
        results = dict(recon_clear=None, feat_pyr_clear=None,
                       recon_light=None, feat_pyr_light=None,
                       recon_heavy=None, feat_pyr_heavy=None)

        if self.local_iter % 3 == 0:
            recon_clear, feat_pyr_clear = self.extract_feat(inputs['clear_img'], return_dehazed=True)
            loss_decode_clear = self._decode_head_forward_train(feat_pyr_clear, data_samples, density='clear')
            losses.update(loss_decode_clear)

            # reconstruction loss
            loss_char_light = torch.sqrt(F.mse_loss(recon_clear, inputs['clear_img']) + 1e-6)
            loss_vgg_light = self.vgg_loss(recon_clear, inputs['clear_img'])
            losses.update(add_prefix(dict(loss=loss_char_light), 'dehaze_char_clear'))
            losses.update(add_prefix(dict(loss=loss_vgg_light * 0.5), 'dehaze_vgg_clear'))

            results['recon_clear'] = recon_clear
            results['feat_pyr_clear'] = feat_pyr_clear

        if self.local_iter % 3 == 1:
            # seg_loss
            recon_light, feat_pyr_light = self.extract_feat(inputs['light_foggy_img'], return_dehazed=True)
            loss_decode_foggy_l = self._decode_head_forward_train(feat_pyr_light, data_samples, density='light')
            losses.update(loss_decode_foggy_l)

            # reconstruction loss
            loss_char_light = torch.sqrt(F.mse_loss(recon_light, inputs['clear_img']) + 1e-6)
            loss_vgg_light = self.vgg_loss(recon_light, inputs['clear_img'])
            losses.update(add_prefix(dict(loss=loss_char_light), 'dehaze_char_light'))
            losses.update(add_prefix(dict(loss=loss_vgg_light * 0.5), 'dehaze_vgg_light'))

            results['recon_light'] = recon_light
            results['feat_pyr_light'] = feat_pyr_light

        if self.local_iter % 3 == 2:
            # seg_loss
            recon_heavy, feat_pyr_heavy = self.extract_feat(inputs['heavy_foggy_img'], return_dehazed=True)
            loss_decode_foggy_h = self._decode_head_forward_train(feat_pyr_heavy, data_samples, density='heavy')
            losses.update(loss_decode_foggy_h)

            # reconstruction loss heavy
            loss_char_heavy = torch.sqrt(F.mse_loss(recon_heavy, inputs['clear_img']) + 1e-6)
            loss_vgg_heavy = self.vgg_loss(recon_heavy, inputs['clear_img'])
            losses.update(add_prefix(dict(loss=loss_char_heavy), 'dehaze_char_heavy'))
            losses.update(add_prefix(dict(loss=loss_vgg_heavy * 0.5), 'dehaze_vgg_heavy'))

            results['recon_heavy'] = recon_heavy
            results['feat_pyr_heavy'] = feat_pyr_heavy

        if self.local_iter % self.debug_visual == 0:
            with torch.no_grad():
                timestr = self.train_cfg['time']
                out_dir = os.path.join(self.train_cfg['work_dir'], f'visualization_{timestr}')
                os.makedirs(out_dir, exist_ok=True)

                vis_light_foggy_img = torch.clamp(denorm(inputs['light_foggy_img'], self.means, self.stds), 0, 1)
                vis_heavy_foggy_img = torch.clamp(denorm(inputs['heavy_foggy_img'], self.means, self.stds), 0, 1)
                vis_clear_img = torch.clamp(denorm(inputs['clear_img'], self.means, self.stds), 0, 1)

                # clear
                if results['recon_clear'] is not None:
                    clear_foggy_logits = self.get_decode_logits(feat_pyr_clear, data_samples)
                    clear_foggy_logits = torch.softmax(clear_foggy_logits, dim=1)
                    clear_ent = calc_entropy(clear_foggy_logits)
                    _, pred_clear = torch.max(clear_foggy_logits, dim=1)
                    vis_dehaze_clear = torch.clamp(denorm(recon_clear, self.means, self.stds), 0, 1)

                # light
                if results['recon_light'] is not None:
                    light_foggy_logits = self.get_decode_logits(feat_pyr_light, data_samples)
                    light_foggy_softmax_prob = torch.softmax(light_foggy_logits, dim=1)
                    light_foggy_ent = calc_entropy(light_foggy_softmax_prob)
                    _, pred_light_foggy = torch.max(light_foggy_softmax_prob, dim=1)
                    vis_dehaze_light = torch.clamp(denorm(recon_light, self.means, self.stds), 0, 1)

                # heavy
                if results['recon_heavy'] is not None:
                    heavy_foggy_logits = self.get_decode_logits(feat_pyr_heavy, data_samples)
                    heavy_foggy_softmax_prob = torch.softmax(heavy_foggy_logits, dim=1)
                    heavy_foggy_ent = calc_entropy(heavy_foggy_softmax_prob)
                    _, pred_heavy_foggy = torch.max(heavy_foggy_softmax_prob, dim=1)
                    vis_dehaze_heavy = torch.clamp(denorm(recon_heavy, self.means, self.stds), 0, 1)

                for j in range(batch_size):
                    rows, cols = 4, 5
                    fig, axs = plt.subplots(
                        rows,
                        cols,
                        figsize=(3 * cols, 3 * rows),
                        gridspec_kw={
                            'hspace': 0.1,
                            'wspace': 0,
                            'top': 0.95,
                            'bottom': 0,
                            'right': 1,
                            'left': 0
                        },
                    )
                    subplotimg(axs[0][0], vis_clear_img[j], 'Clear Image')
                    subplotimg(axs[1][0], vis_light_foggy_img[j], 'Light Foggy Image')
                    subplotimg(axs[2][0], vis_heavy_foggy_img[j], 'Heavy Foggy Image')
                    # subplotimg(axs[3][0], inputs['intensity'][j], 'GT intensity', cmap='gray')
                    subplotimg(axs[3][0], data_samples[j].gt_sem_seg.data[0], 'GT Seg', cmap='udd')

                    # subplotimg(axs[0][1], pred_clear[j], 'Clear Pred', cmap='uavid')
                    if results['recon_clear'] is not None:
                        subplotimg(axs[0][1], pred_clear[j], 'Clear Pred', cmap='udd')
                        subplotimg(axs[0][2], clear_ent[j], 'Clear Entropy', cmap='viridis')
                        psnr_clear = calculate_psnr(pred=vis_dehaze_clear[j], target=vis_clear_img[j])
                        subplotimg(axs[0][3], vis_dehaze_clear[j], f'Dehaze Clear(PSNR={psnr_clear})')
                        subplotimg(axs[0][4], torch.abs(vis_dehaze_clear[j] - vis_clear_img[j]), 'Dehaze Error C')

                    if results['recon_light'] is not None:
                        subplotimg(axs[1][1], pred_light_foggy[j], 'Light Foggy Pred', cmap='udd')
                        subplotimg(axs[1][2], light_foggy_ent[j], 'Light Foggy Entropy', cmap='viridis')
                        psnr_light = calculate_psnr(pred=vis_dehaze_light[j], target=vis_clear_img[j])
                        subplotimg(axs[1][3], vis_dehaze_light[j], f'Dehaze Light(PSNR={psnr_light})')
                        subplotimg(axs[1][4], torch.abs(vis_dehaze_light[j] - vis_clear_img[j]), 'Dehaze Error L')

                    if results['recon_heavy'] is not None:
                        subplotimg(axs[2][1], pred_heavy_foggy[j], 'Heavy Foggy Pred', cmap='udd')
                        subplotimg(axs[2][2], heavy_foggy_ent[j], 'Heavy Foggy Entropy', cmap='viridis')
                        psnr_heavy = calculate_psnr(pred=vis_dehaze_heavy[j], target=vis_clear_img[j])
                        subplotimg(axs[2][3], vis_dehaze_heavy[j], f'Dehaze Heavy(PSNR={psnr_heavy})')
                        subplotimg(axs[2][4], torch.abs(vis_dehaze_heavy[j] - vis_clear_img[j]), 'Dehaze Error H')

                    # subplotimg(axs[0][2], clear_ent[j], 'Clear Entropy', cmap='viridis')

                    for ax in axs.flat:
                        ax.axis('off')
                    plt.savefig(os.path.join(out_dir, f'{(self.local_iter + 1):06d}_{j}.png'))
                    plt.close()

        self.local_iter += 1
        return losses

    def loss_single(self, inputs: dict, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        mode == 'loss'

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # 特征提取
        losses = dict()
        batch_size = inputs['clear_img'].size()[0]

        # seg-loss
        # loss_decode_clear = self._decode_head_forward_train(feat_pyr_clear, data_samples, density='clear')
        # losses.update(loss_decode_clear)

        if 'light' in self.use_multi_density:
            recon_light, feat_pyr_light = self.extract_feat(inputs['light_foggy_img'], return_dehazed=True)
            if 'seg' in self.tasks:
                loss_decode_foggy_l = self._decode_head_forward_train(feat_pyr_light, data_samples, density='light')
                losses.update(loss_decode_foggy_l)

            if 'dehaze' in self.tasks:
                # reconstruction loss light
                loss_char_light = torch.sqrt(F.mse_loss(recon_light, inputs['clear_img']) + 1e-6)
                loss_vgg_light = self.vgg_loss(recon_light, inputs['clear_img'])
                losses.update(add_prefix(dict(loss=loss_char_light), 'dehaze_char_light'))
                losses.update(add_prefix(dict(loss=loss_vgg_light * 0.05), 'dehaze_vgg_light'))

        if 'heavy' in self.use_multi_density:
            # seg_loss
            recon_heavy, feat_pyr_heavy = self.extract_feat(inputs['heavy_foggy_img'], return_dehazed=True)
            if 'seg' in self.tasks:
                loss_decode_foggy_h = self._decode_head_forward_train(feat_pyr_heavy, data_samples, density='heavy')
                losses.update(loss_decode_foggy_h)

            # reconstruction loss heavy
            if 'dehaze' in self.tasks:
                loss_char_heavy = torch.sqrt(F.mse_loss(recon_heavy, inputs['clear_img']) + 1e-6)
                loss_vgg_heavy = self.vgg_loss(recon_heavy, inputs['clear_img'])
                losses.update(add_prefix(dict(loss=loss_char_heavy), 'dehaze_char_heavy'))
                losses.update(add_prefix(dict(loss=loss_vgg_heavy * 0.5), 'dehaze_vgg_heavy'))

        if self.local_iter % self.debug_visual == 0:
            with torch.no_grad():
                timestr = self.train_cfg['time']
                out_dir = os.path.join(self.train_cfg['work_dir'], f'visualization_{timestr}')
                os.makedirs(out_dir, exist_ok=True)

                vis_light_foggy_img = torch.clamp(denorm(inputs['light_foggy_img'], self.means, self.stds), 0, 1)
                vis_heavy_foggy_img = torch.clamp(denorm(inputs['heavy_foggy_img'], self.means, self.stds), 0, 1)
                vis_clear_img = torch.clamp(denorm(inputs['clear_img'], self.means, self.stds), 0, 1)

                # clear
                # clear_logits = self.decode_head.predict(feat_pyr_clear)
                # clear_softmax_prob = torch.softmax(clear_logits, dim=1)
                # clear_ent = calc_entropy(clear_softmax_prob)
                # _, pred_clear = torch.max(clear_softmax_prob, dim=1)

                # light
                if 'light' in self.use_multi_density:
                    light_foggy_logits = self.get_decode_logits(feat_pyr_light, data_samples)
                    light_foggy_softmax_prob = torch.softmax(light_foggy_logits, dim=1)
                    light_foggy_ent = calc_entropy(light_foggy_softmax_prob)
                    _, pred_light_foggy = torch.max(light_foggy_softmax_prob, dim=1)
                    vis_dehaze_light = torch.clamp(denorm(recon_light, self.means, self.stds), 0, 1)

                # heavy
                if 'heavy' in self.use_multi_density:
                    heavy_foggy_logits = self.get_decode_logits(feat_pyr_heavy, data_samples)
                    heavy_foggy_softmax_prob = torch.softmax(heavy_foggy_logits, dim=1)
                    heavy_foggy_ent = calc_entropy(heavy_foggy_softmax_prob)
                    _, pred_heavy_foggy = torch.max(heavy_foggy_softmax_prob, dim=1)
                    vis_dehaze_heavy = torch.clamp(denorm(recon_heavy, self.means, self.stds), 0, 1)

                for j in range(batch_size):
                    rows, cols = 5, 4
                    fig, axs = plt.subplots(
                        rows,
                        cols,
                        figsize=(3 * cols, 3 * rows),
                        gridspec_kw={
                            'hspace': 0.1,
                            'wspace': 0,
                            'top': 0.95,
                            'bottom': 0,
                            'right': 1,
                            'left': 0
                        },
                    )
                    subplotimg(axs[0][0], vis_clear_img[j], 'Clear Image')
                    subplotimg(axs[1][0], vis_light_foggy_img[j], 'Light Foggy Image')
                    subplotimg(axs[2][0], vis_heavy_foggy_img[j], 'Heavy Foggy Image')
                    # subplotimg(axs[3][0], inputs['intensity'][j], 'GT intensity', cmap='gray')
                    subplotimg(axs[3][0], data_samples[j].gt_sem_seg.data[0], 'GT Seg', cmap='uavid')

                    # subplotimg(axs[0][1], pred_clear[j], 'Clear Pred', cmap='uavid')
                    if 'light' in self.use_multi_density:
                        subplotimg(axs[1][1], pred_light_foggy[j], 'Light Foggy Pred', cmap='uavid')
                        subplotimg(axs[1][2], light_foggy_ent[j], 'Light Foggy Entropy', cmap='viridis')
                        psnr_light = calculate_psnr(pred=vis_dehaze_light[j], target=vis_clear_img[j])
                        subplotimg(axs[1][3], vis_dehaze_light[j], f'Dehaze Light(PSNR={psnr_light})')
                        subplotimg(axs[2][3], torch.abs(vis_dehaze_light[j] - vis_clear_img[j]),
                                   'Dehaze Error dehazeformer-W')

                    if 'heavy' in self.use_multi_density:
                        subplotimg(axs[2][1], pred_heavy_foggy[j], 'Heavy Foggy Pred', cmap='uavid')
                        subplotimg(axs[2][2], heavy_foggy_ent[j], 'Heavy Foggy Entropy', cmap='viridis')
                        psnr_heavy = calculate_psnr(pred=vis_dehaze_heavy[j], target=vis_clear_img[j])
                        subplotimg(axs[3][3], vis_dehaze_heavy[j], f'Dehaze Heavy(PSNR={psnr_heavy})')
                        subplotimg(axs[4][3], torch.abs(vis_dehaze_heavy[j] - vis_clear_img[j]), 'Dehaze Error H')

                    # subplotimg(axs[0][2], clear_ent[j], 'Clear Entropy', cmap='viridis')

                    for ax in axs.flat:
                        ax.axis('off')
                    plt.savefig(os.path.join(out_dir, f'{(self.local_iter + 1):06d}_{j}.png'))
                    plt.close()

        self.local_iter += 1
        return losses

    def loss(self, inputs: dict, data_samples: SampleList) -> dict:
        if self.use_alter:
            return self.loss_alter(inputs=inputs, data_samples=data_samples)
        else:
            return self.loss_single(inputs=inputs, data_samples=data_samples)
