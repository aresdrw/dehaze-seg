aero_type = "UDDDataset"
aero_root = "/hy-tmp/datasets/foggy_udd/"
aero_crop_size = (512, 512)
aero_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", scale=(1280, 720)),
    dict(type="RandomCrop", crop_size=aero_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
aero_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1280, 720), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_dataset = dict(
    type=aero_type,
    data_root=aero_root,
    data_prefix=dict(
        img_path="train/images/clear",
        seg_map_path="train/gt/labelTrainIds",
    ),
    img_suffix=".JPG",
    seg_map_suffix=".png",
    pipeline=aero_train_pipeline,
)

val_aero = dict(
    type=aero_type,
    data_root=aero_root,
    data_prefix=dict(
        img_path="val/images/clear",
        seg_map_path="val/gt/labelTrainIds",
    ),
    img_suffix=".JPG",
    seg_map_suffix=".png",
    pipeline=aero_test_pipeline,
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
    dataset=val_aero,
)

val_evaluator = dict(
    type="IoUMetric",
    iou_metrics=["mIoU", "mFscore"],
)
test_dataloader = val_dataloader
test_evaluator = val_evaluator
