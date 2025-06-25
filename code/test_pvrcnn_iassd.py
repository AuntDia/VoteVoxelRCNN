"""
IASSD-PVRCNN 两阶段检测器推理和评估脚本

支持KITTI和Waymo数据集的测试和评估
作者: MiniMax Agent
日期: 2025-06-21
"""

import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path
import pickle

import numpy as np
import torch
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from eval_utils import eval_utils


def parse_config():
    """解析命令行参数和配置文件"""
    parser = argparse.ArgumentParser(description='IASSD-PVRCNN模型测试参数')
    parser.add_argument('--cfg_file', type=str, default=None, 
                        help='指定配置文件')
    parser.add_argument('--data_path', type=str, default=None, 
                        help='指定数据路径')
    parser.add_argument('--batch_size', type=int, default=None, required=False,
                        help='每GPU的batch size')
    parser.add_argument('--workers', type=int, default=4, 
                        help='用于数据加载的worker数量')
    parser.add_argument('--extra_tag', type=str, default='default', 
                        help='实验标签')
    parser.add_argument('--ckpt', type=str, default=None, 
                        help='加载的checkpoint路径')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, 
                        help='用于分布式训练的tcp端口')
    parser.add_argument('--local_rank', type=int, default=0, 
                        help='用于分布式训练的local rank')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='设置额外的配置键值对')
    parser.add_argument('--eval_tag', type=str, default='default', 
                        help='评估标签')
    parser.add_argument('--eval_all', action='store_true', default=False, 
                        help='是否评估所有checkpoint')
    parser.add_argument('--ckpt_dir', type=str, default=None, 
                        help='checkpoint目录')
    parser.add_argument('--save_to_file', action='store_true', default=False, 
                        help='是否保存结果到文件')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):
    """评估单个checkpoint"""
    # 加载checkpoint
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    model.cuda()
    model.eval()

    # 生成预测结果
    logger.info('*************** 开始评估 %s ***************' % args.ckpt)
    if args.save_to_file:
        final_output_dir = eval_output_dir / 'final_result' / 'data'
        if args.extra_tag != 'default':
            final_output_dir = final_output_dir / args.extra_tag

        final_output_dir.mkdir(parents=True, exist_ok=True)
        args.infer_time = False
    else:
        final_output_dir = None

    metric_result, result_str = eval_utils.eval_one_epoch(
        cfg, args, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=final_output_dir, save_to_file=args.save_to_file
    )

    logger.info(result_str)
    logger.info('****************评估完成****************')
    
    return metric_result


def get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args):
    """获取尚未评估的checkpoint列表"""
    ckpt_list = glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    evaluated_ckpt_list = [float(x.strip()) for x in open(ckpt_record_file, 'r').readlines()]

    for cur_ckpt in ckpt_list:
        num_list = re.findall('checkpoint_epoch_(.*).pth', cur_ckpt)
        if num_list.__len__() == 0:
            continue

        epoch_id = num_list[-1]
        if 'optim' in epoch_id:
            continue
        if float(epoch_id) not in evaluated_ckpt_list and int(float(epoch_id)) >= args.start_epoch:
            return epoch_id, cur_ckpt
    return -1, None


def repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=False):
    """重复评估多个checkpoint"""
    # 记录已评估的checkpoint
    ckpt_record_file = eval_output_dir / ('eval_list_%s.txt' % cfg.DATA_CONFIG.DATA_SPLIT['test'])
    with open(ckpt_record_file, 'a'):
        pass

    # 创建时间戳，避免重复评估
    cur_time = time.time()
    first_eval = True

    while True:
        # 检查是否有新的checkpoint
        epoch_id, cur_ckpt = get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args)
        if epoch_id == -1 or int(float(epoch_id)) < args.start_epoch:
            wait_second = 30
            if cfg.LOCAL_RANK == 0 and first_eval:
                print('等待checkpoint生成... (%s)' % str(datetime.datetime.now()))
                first_eval = False
            time.sleep(wait_second)
            continue

        # 评估checkpoint
        logger.info('==> 开始评估 Epoch %s ==> %s' % (epoch_id, cur_ckpt))
        tb_dict = eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test)

        # 记录评估结果
        if cfg.LOCAL_RANK == 0:
            with open(ckpt_record_file, 'a') as f:
                print('%s' % epoch_id, file=f)


def main():
    """主测试函数"""
    args, cfg = parse_config()

    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should be matched with GPUS'
        args.batch_size = args.batch_size // total_gpus

    # 创建输出目录
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    if args.eval_tag != 'default':
        output_dir = output_dir / ('eval_%s' % args.eval_tag)

    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    if not args.eval_all:
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    else:
        eval_output_dir = eval_output_dir / 'eval_all_default'

    if args.save_to_file:
        eval_output_dir.mkdir(parents=True, exist_ok=True)

    # 创建日志文件
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # 记录配置
    logger.info('**********************开始记录**********************')
    log_config_to_file(cfg, logger=logger)

    # 构建测试数据集
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    # 构建模型
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.cuda()

    # 分布式测试设置
    if dist_test:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])

    # 开始评估
    if args.eval_all:
        repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir=args.ckpt_dir, 
                        dist_test=dist_test)
    else:
        eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id=0, dist_test=dist_test)


if __name__ == '__main__':
    main()
