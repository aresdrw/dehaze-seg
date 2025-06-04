""" 
-*- coding: utf-8 -*-
    @Time    : 2025/4/17  22:33
    @Author  : AresDrw
    @File    : dehaze_seg_dehazeformer_s_mask2former_uavid-512x512.py
    @Software: PyCharm
    @Describe: 
-*- encoding:utf-8 -*-
"""
_base_ = [
    '/hy-tmp/Dehaze_Seg/configs/_base_/models/dehaze_seg/dehaze_dehazeformer_S_FreqNeck_mask2former.py',
    '/hy-tmp/Dehaze_Seg/configs/_base_/datasets/foggy_uavid_512x512.py',
    '/hy-tmp/Dehaze_Seg/configs/_base_/default_runtime.py',
    '/hy-tmp/Dehaze_Seg/configs/_base_/schedules/schedule_40k.py'
]
crop_size = (512, 512)
n_aero_classes = 12
n_uavid_classes = 8
model = dict(data_preprocessor=dict(type='MultiModalSegDataPreProcessor',
                                    size=crop_size,
                                    mean=[123.675, 116.28, 103.53],
                                    std=[58.395, 57.12, 57.375],
                                    pad_val=0),
             use_multi_density=['light', 'heavy'],
             task_selection=['dehaze', 'seg'],
             use_alter=True,
             decode_head=dict(num_classes=n_uavid_classes))

train_dataloader = dict(batch_size=1)

# 优化器设置
optim_wrapper = dict(type='OptimWrapper',
                     optimizer=dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01))
param_scheduler = [
    dict(type="PolyLR", eta_min=0, power=0.9, begin=0, end=60000, by_epoch=False)
]

# training schedule for 160k
train_cfg = dict(type="IterBasedTrainLoop", max_iters=60000, val_interval=2000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook", by_epoch=False, interval=4000, max_keep_ckpts=1
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)
