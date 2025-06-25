"""
PVRCNN-IASSD 两阶段检测器训练脚本

第一阶段：PVRCNN的体素化检测流程生成proposals
第二阶段：IASSD backbone进行点云特征提取 + PVRCNN ROI Head进行refinement

支持KITTI和Waymo数据集的训练
作者: MiniMax Agent
日期: 2025-06-22
"""

import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model


def parse_config():
    """解析命令行参数和配置文件"""
    parser = argparse.ArgumentParser(description='训练IASSD-PVRCNN模型的参数')
    parser.add_argument('--cfg_file', type=str, default=None, 
                        help='指定配置文件')
    parser.add_argument('--batch_size', type=int, default=None, required=False,
                        help='每GPU的batch size')
    parser.add_argument('--epochs', type=int, default=None, required=False,
                        help='训练轮数')
    parser.add_argument('--workers', type=int, default=4, 
                        help='用于数据加载的worker数量')
    parser.add_argument('--extra_tag', type=str, default='default', 
                        help='实验标签')
    parser.add_argument('--ckpt', type=str, default=None, 
                        help='加载的checkpoint路径')
    parser.add_argument('--pretrained_model', type=str, default=None, 
                        help='预训练模型路径')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, 
                        help='用于分布式训练的tcp端口')
    parser.add_argument('--sync_bn', action='store_true', default=False, 
                        help='是否使用sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, 
                        help='是否固定随机种子')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, 
                        help='checkpoint保存间隔')
    parser.add_argument('--local_rank', type=int, default=0, 
                        help='用于分布式训练的local rank')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, 
                        help='最大保存的checkpoint数量')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, 
                        help='是否将所有iteration合并到一个epoch')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='设置额外的配置键值对')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # 去除'cfgs'和yaml文件名

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    """主训练函数"""
    args, cfg = parse_config()
    
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should be matched with GPUS'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    # 创建输出目录
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 创建日志文件
    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # 记录git信息和配置
    logger.info('**********************开始记录**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('总批次大小: %d' % (total_gpus * args.batch_size))
    log_config_to_file(cfg, logger=logger)

    # 创建TensorBoard记录器
    if cfg.LOCAL_RANK == 0:
        tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # 构建数据集
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs
    )

    # 构建模型
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    model.cuda()

    # 构建优化器
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # 加载checkpoint
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, 
                                                          optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            it, start_epoch = model.load_params_with_optimizer(
                ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
            )
            last_epoch = start_epoch + 1

    # 构建学习率调度器
    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # 分布式训练设置
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(model)

    logger.info('**********************开始训练 %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    
    # 开始训练
    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch
    )

    logger.info('**********************训练结束 %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()
