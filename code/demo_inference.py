"""
PVRCNN-IASSD 模型推理演示脚本

快速演示如何使用训练好的模型进行3D目标检测推理
第一阶段：PVRCNN体素化检测  第二阶段：IASSD特征提取 + PVRCNN ROI Head

作者: ITSRC
"""

import argparse
import numpy as np
import torch
from pathlib import Path
import pickle

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.datasets import DatasetTemplate
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    """演示数据集类，用于加载单个点云文件"""
    
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, 
            root_path=root_path, logger=logger
        )
        
    def __len__(self):
        return 1
        
    def __getitem__(self, index):
        # 这是一个简化的实现，实际使用时需要根据具体数据格式调整
        info = {}
        pc_info = {'num_features': 4, 'lidar_idx': '000000'}
        info['point_cloud'] = pc_info
        info['frame_id'] = '000000'
        
        return self.prepare_data(data_dict=info)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='IASSD-PVRCNN推理演示')
    parser.add_argument('--cfg_file', type=str, required=True,
                        help='模型配置文件路径')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='模型checkpoint路径')
    parser.add_argument('--data_path', type=str, required=True,
                        help='点云数据文件路径 (.bin或.pcd)')
    parser.add_argument('--output_path', type=str, default='./demo_output',
                        help='输出结果保存路径')
    parser.add_argument('--score_thresh', type=float, default=0.3,
                        help='检测置信度阈值')
    
    return parser.parse_args()


def load_point_cloud(data_path):
    """加载点云数据"""
    data_path = Path(data_path)
    
    if data_path.suffix == '.bin':
        # KITTI格式的二进制文件
        points = np.fromfile(str(data_path), dtype=np.float32).reshape(-1, 4)
    elif data_path.suffix == '.pcd':
        # PCD格式文件 (需要额外的处理库)
        raise NotImplementedError("PCD格式支持待实现")
    else:
        raise ValueError(f"不支持的文件格式: {data_path.suffix}")
    
    return points


def preprocess_points(points, point_cloud_range):
    """预处理点云数据"""
    # 过滤超出范围的点
    mask = (points[:, 0] >= point_cloud_range[0]) & \
           (points[:, 0] <= point_cloud_range[3]) & \
           (points[:, 1] >= point_cloud_range[1]) & \
           (points[:, 1] <= point_cloud_range[4]) & \
           (points[:, 2] >= point_cloud_range[2]) & \
           (points[:, 2] <= point_cloud_range[5])
    
    points = points[mask]
    
    # 添加batch索引
    batch_idx = np.zeros((points.shape[0], 1), dtype=np.float32)
    points = np.concatenate([batch_idx, points], axis=1)  # [bs_idx, x, y, z, intensity]
    
    return points


def visualize_results(points, pred_dicts, output_path, score_thresh=0.3):
    """可视化检测结果"""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("检测结果:")
    print("-" * 50)
    
    for i, pred_dict in enumerate(pred_dicts):
        pred_scores = pred_dict['pred_scores'].cpu().numpy()
        pred_boxes = pred_dict['pred_boxes'].cpu().numpy()
        pred_labels = pred_dict['pred_labels'].cpu().numpy()
        
        # 根据置信度阈值过滤
        valid_mask = pred_scores >= score_thresh
        valid_boxes = pred_boxes[valid_mask]
        valid_scores = pred_scores[valid_mask]
        valid_labels = pred_labels[valid_mask]
        
        print(f"帧 {i}: 检测到 {len(valid_boxes)} 个目标")
        
        for j, (box, score, label) in enumerate(zip(valid_boxes, valid_scores, valid_labels)):
            print(f"  目标 {j+1}: 类别={label}, 置信度={score:.3f}")
            print(f"    位置: x={box[0]:.2f}, y={box[1]:.2f}, z={box[2]:.2f}")
            print(f"    尺寸: l={box[3]:.2f}, w={box[4]:.2f}, h={box[5]:.2f}")
            print(f"    旋转: {box[6]:.3f} rad")
        
        # 保存结果到文件
        result_file = output_path / f'frame_{i:06d}_results.pkl'
        with open(result_file, 'wb') as f:
            pickle.dump({
                'pred_boxes': valid_boxes,
                'pred_scores': valid_scores,
                'pred_labels': valid_labels,
                'points': points[1:, 1:4]  # 移除batch索引，只保留xyz
            }, f)
        
        print(f"结果已保存到: {result_file}")


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    cfg_from_yaml_file(args.cfg_file, cfg)
    logger = common_utils.create_logger()
    
    # 创建数据集和模型
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, 
        class_names=cfg.CLASS_NAMES, 
        training=False,
        logger=logger
    )
    
    # 构建模型
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger)
    model.cuda()
    model.eval()
    
    print(f"模型加载完成: {args.ckpt}")
    print(f"支持的类别: {cfg.CLASS_NAMES}")
    
    # 加载点云数据
    print(f"加载点云数据: {args.data_path}")
    raw_points = load_point_cloud(args.data_path)
    print(f"原始点云: {raw_points.shape[0]} 个点")
    
    # 预处理
    points = preprocess_points(raw_points, cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
    print(f"预处理后: {points.shape[0]} 个点")
    
    # 准备输入数据
    input_dict = {
        'points': torch.from_numpy(points).float().cuda(),
        'frame_id': '000000',
        'batch_size': 1
    }
    
    # 推理
    print("开始推理...")
    with torch.no_grad():
        batch_dict = model(input_dict)
        pred_dicts = model.post_processing(batch_dict)
    
    print("推理完成!")
    
    # 可视化结果
    visualize_results(points, pred_dicts, args.output_path, args.score_thresh)
    
    print(f"\n演示完成! 结果保存在: {args.output_path}")


if __name__ == '__main__':
    main()
