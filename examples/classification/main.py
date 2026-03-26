import argparse
import os

import __init__
import numpy as np
import yaml
from torch import multiprocessing as mp

from examples.classification.pretrain import main as pretrain
from examples.classification.train import main as train
from openpoints.utils import (
    EasyConfig,
    Wandb,
    dist_utils,
    find_free_port,
    generate_exp_directory,
    resume_exp_directory,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("S3DIS scene segmentation training")
    parser.add_argument("--cfg", type=str, required=True, help="config file")
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="set to True to profile speed",
    )
    parser.add_argument(
        "--balance-train",
        action="store_true",
        default=False,
        help="use logit adjustment during training to handle class imbalance",
    )
    parser.add_argument(
        "--focal-loss",
        action="store_true",
        default=False,
        help="use focal loss during training to down-weight easy examples",
    )
    parser.add_argument(
        "--cb-loss",
        action="store_true",
        default=False,
        help="use class-balanced loss based on effective number of samples (CVPR 2019)",
    )
    parser.add_argument(
        "--cb-beta",
        type=float,
        default=0.9999,
        help="beta parameter for CB loss (default: 0.9999)",
    )
    parser.add_argument(
        "--ldam-loss",
        action="store_true",
        default=False,
        help="use LDAM loss with label-distribution-aware margins (NeurIPS 2019)",
    )
    parser.add_argument(
        "--ldam-c",
        type=float,
        default=0.5,
        help="margin scaling constant C for LDAM loss (default: 0.5)",
    )
    parser.add_argument(
        "--drw",
        action="store_true",
        default=False,
        help="enable deferred re-weighting (DRW) for LDAM loss",
    )
    parser.add_argument(
        "--drw-start-epoch",
        type=int,
        default=None,
        help="epoch to start DRW (default: 80%% of total epochs)",
    )
    parser.add_argument(
        "--balanced-sampling",
        action="store_true",
        default=False,
        help="use class-balanced sampling during training (ICLR 2020)",
    )
    parser.add_argument(
        "--sampling-alpha",
        type=float,
        default=0.5,
        help="sampling exponent: 0=class-balanced, 0.5=sqrt, 1=original (default: 0.5)",
    )
    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)
    cfg.balance_train = cfg.get("balance_train", False) or args.balance_train
    cfg.focal_loss = cfg.get("focal_loss", False) or args.focal_loss
    cfg.cb_loss = cfg.get("cb_loss", False) or args.cb_loss
    cfg.cb_beta = args.cb_beta if args.cb_beta != 0.9999 else cfg.get("cb_beta", 0.9999)
    cfg.ldam_loss = cfg.get("ldam_loss", False) or args.ldam_loss
    cfg.ldam_c = args.ldam_c if args.ldam_c != 0.5 else cfg.get("ldam_c", 0.5)
    cfg.drw = cfg.get("drw", False) or args.drw
    cfg.drw_start_epoch = (
        args.drw_start_epoch
        if args.drw_start_epoch is not None
        else cfg.get("drw_start_epoch", None)
    )
    cfg.balanced_sampling = (
        cfg.get("balanced_sampling", False) or args.balanced_sampling
    )
    cfg.sampling_alpha = (
        args.sampling_alpha
        if args.sampling_alpha != 0.5
        else cfg.get("sampling_alpha", 0.5)
    )
    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)

    # Check mutual exclusivity of loss functions
    active_losses = sum([cfg.focal_loss, cfg.cb_loss, cfg.ldam_loss])
    if active_losses > 1:
        raise ValueError(
            "Cannot use multiple loss functions simultaneously. "
            "Choose one of: --focal-loss, --cb-loss, --ldam-loss"
        )

    if cfg.focal_loss:
        focal_cfg = EasyConfig()
        focal_cfg.NAME = "FocalLoss"
        focal_cfg.gamma = cfg.get("focal_gamma", 2.0)
        focal_alpha = cfg.get("focal_alpha", None)
        if focal_alpha is not None:
            focal_cfg.alpha = focal_alpha
        cfg.criterion_args = focal_cfg
        cfg.model.criterion_args = focal_cfg
        print(
            f"Using focal loss (gamma={focal_cfg.gamma}, alpha={getattr(focal_cfg, 'alpha', None)})"
        )

    if cfg.cb_loss:
        cb_cfg = EasyConfig()
        cb_cfg.NAME = "CBCrossEntropyLoss"
        cb_cfg.beta = cfg.cb_beta
        cfg.criterion_args = cb_cfg
        cfg.model.criterion_args = cb_cfg
        print(f"Using CB loss (beta={cfg.cb_beta})")

    if cfg.ldam_loss:
        ldam_cfg = EasyConfig()
        ldam_cfg.NAME = "LDAMLoss"
        ldam_cfg.C = cfg.ldam_c
        cfg.criterion_args = ldam_cfg
        cfg.model.criterion_args = ldam_cfg
        # Compute default DRW start epoch if not specified
        if cfg.drw and cfg.drw_start_epoch is None:
            cfg.drw_start_epoch = int(cfg.epochs * 0.8)
        drw_info = f", DRW from epoch {cfg.drw_start_epoch}" if cfg.drw else ""
        print(f"Using LDAM loss (C={cfg.ldam_c}{drw_info})")

    # Check mutual exclusivity: balanced sampling vs loss-based reweighting
    if cfg.balanced_sampling and (cfg.cb_loss or cfg.ldam_loss or cfg.focal_loss):
        raise ValueError(
            "Cannot use --balanced-sampling with loss-based reweighting methods. "
            "Choose either data-level (--balanced-sampling) or loss-level reweighting."
        )

    if cfg.balanced_sampling:
        alpha_desc = (
            "original"
            if cfg.sampling_alpha == 1.0
            else "sqrt"
            if cfg.sampling_alpha == 0.5
            else "class-balanced"
            if cfg.sampling_alpha == 0.0
            else "custom"
        )
        print(f"Using balanced sampling (alpha={cfg.sampling_alpha}, {alpha_desc})")

    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1

    # init log dir
    cfg.task_name = args.cfg.split(".")[-2].split("/")[-2]
    cfg.exp_name = args.cfg.split(".")[-2].split("/")[-1]
    tags = [
        cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode,
        cfg.exp_name,  # cfg file name
        f"ngpus{cfg.world_size}",
        f"seed{cfg.seed}",
    ]
    opt_list = []  # for checking experiment configs from logging file
    for i, opt in enumerate(opts):
        if (
            "rank" not in opt
            and "dir" not in opt
            and "root" not in opt
            and "pretrain" not in opt
            and "path" not in opt
            and "wandb" not in opt
            and "/" not in opt
        ):
            opt_list.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)
    cfg.opts = "-".join(opt_list)

    if cfg.mode in ["resume", "val", "test"]:
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
        cfg.wandb.tags = [cfg.mode]
    else:  # resume from the existing ckpt and reuse the folder.
        generate_exp_directory(
            cfg, tags, additional_id=os.environ.get("MASTER_PORT", None)
        )
        cfg.wandb.tags = tags
    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, indent=2)
        os.system("cp %s %s" % (args.cfg, cfg.run_dir))
    cfg.cfg_path = cfg_path
    cfg.wandb.name = cfg.run_name

    if cfg.mode == "pretrain":
        main = pretrain
    else:
        main = train

    if cfg.mp:
        port = find_free_port()
        cfg.dist_url = f"tcp://localhost:{port}"
        print("using mp spawn for distributed training")
        mp.spawn(main, nprocs=cfg.world_size, args=(cfg, args.profile))
    else:
        main(0, cfg, profile=args.profile)
