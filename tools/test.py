# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

os.chdir(osp.abspath(osp.dirname(osp.dirname(__file__))))
import sys

sys.path.append(os.curdir)

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
import rein
import rele


# TODO: support fuse_conv_bn, visualization, and format_only
def parse_args():
    parser = argparse.ArgumentParser(description="MMSeg test (and eval) a model")
    parser.add_argument("--config", help="train config file path")
    parser.add_argument("--checkpoint", help="rein and head checkpoint file")
    parser.add_argument("--density", help="['light', 'heavy']")
    parser.add_argument("--backbone", help="backbone checkpoint file", default="")
    parser.add_argument(
        "--work-dir",
        help=(
            "if specified, the evaluation metric results will be dumped"
            "into the directory as json"
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        help="The directory to save output prediction for offline evaluation",
    )
    parser.add_argument("--show", action="store_true", help="show prediction results")
    parser.add_argument(
        "--show-dir",
        help="directory where painted images will be saved. "
        "If specified, it will be automatically saved "
        "to the work_dir/timestamp/show_dir",
    )
    parser.add_argument(
        "--wait-time", type=float, default=2, help="the interval of show (s)"
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--tta", action="store_true", help="Test time augmentation")
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if "visualization" in default_hooks:
        visualization_hook = default_hooks["visualization"]
        visualization_hook["interval"] = 1
        # Turn on visualization
        # visualization_hook['draw'] = True
        # if args.show:
        #     visualization_hook['show'] = True
        #     visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visulizer = cfg.visualizer
            visulizer["save_dir"] = args.show_dir
    else:
        raise RuntimeError(
            "VisualizationHook must be included in default_hooks."
            "refer to usage "
            "\"visualization=dict(type='VisualizationHook')\""
        )

    return cfg


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )
    cfg.work_dir = cfg.work_dir + "_test"
    cfg.load_from = args.checkpoint
    if args.backbone:
        custom_hooks = getattr(cfg, "custom_hooks", [])
        custom_hooks.append(
            dict(type="LoadBackboneHook", checkpoint_path=args.backbone)
        )
        setattr(cfg, "custom_hooks", custom_hooks)

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    # add output_dir in metric
    if args.out is not None:
        cfg.test_evaluator["output_dir"] = args.out
        cfg.test_evaluator["keep_results"] = True

    # Myself modify the test_dataloader
    cfg.test_dataloader.dataset.data_prefix.img_path = f'/hy-tmp/images/{args.density}_foggy'
    cfg.test_dataloader.dataset.data_root = '/hy-tmp/datasets/foggy_uavid_for_train/'
    cfg.test_dataloader.dataset.img_suffix = '_foggy_uav_image.png'
    cfg.test_evaluator.iou_metrics = ['mIoU', 'mFscore']

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start testing
    runner.test()


if __name__ == "__main__":
    main()
