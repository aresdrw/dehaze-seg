uavid_type = "UAVidDataset"
uavid_root = "/hy-tmp/datasets/uavid_for_train/"
foggy_uavid_type = "FoggyUAVidDataset"
foggy_uavid_root = "/hy-tmp/datasets/foggy_uavid_for_train/"
uavid_crop_size = (512, 512)

train_pipeline = [
    # 多模态数据加载
    dict(type='LoadMultiModalImageFromFile',  # 新增的加载器
         keys=['heavy_foggy_img', 'light_foggy_img', 'clear_img', 'intensity']),
    dict(type='LoadAnnotations'),
    dict(type='ResizeMultiModal', scale=(1024, 512),
         keys=['heavy_foggy_img', 'light_foggy_img', 'clear_img', 'intensity']),
    dict(type='RandomMultiModalCrop',
         crop_size=uavid_crop_size,
         cat_max_ratio=0.75,
         keys=['heavy_foggy_img', 'light_foggy_img', 'clear_img', 'intensity', 'gt_seg_map']),
    # dict(type='RandomMultiModalFlip',
    #      prob=0.5,
    #      keys=['foggy_img', 'clear_img', 'intensity', 'gt_seg_map']),
    # 单独增强策略
    dict(type='AppliedPhotoMetricDistortion', apply_to=['heavy_foggy_img', 'light_foggy_img']),  # 仅对雾化图像增强
    # 数据包装
    dict(type='PackMultiModalInputs')
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1024, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]

train_dataset = dict(
    type=foggy_uavid_type,
    data_root=foggy_uavid_root,
    data_prefix=dict(
        clear_img_path="train/images/clear",
        light_foggy_img_path="train/images/light_foggy",
        heavy_foggy_img_path="train/images/heavy_foggy",
        fog_intensity="train/images/intensity/value_inv_tx",
        seg_map_path="train/labelTrainIds",
    ),
    foggy_img_suffix="_foggy_uav_image.png",
    clear_img_suffix="_uav_image.png",
    intensity_suffix="_fog_intensity.npy",
    seg_map_suffix="_labelTrainId.png",
    pipeline=train_pipeline,
)

val_dataset = dict(
    type=uavid_type,
    data_root=foggy_uavid_root,
    data_prefix=dict(
        img_path="val/images/heavy_foggy",
        seg_map_path="val/labelTrainIds",
    ),
    img_suffix="_foggy_uav_image.png",
    seg_map_suffix="_labelTrainId.png",
    pipeline=test_pipeline,
)

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=train_dataset,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=val_dataset,
)

val_evaluator = dict(
    type="IoUMetric",
    iou_metrics=["mIoU", "mFscore"],
)
test_dataloader = val_dataloader
test_evaluator = val_evaluator
