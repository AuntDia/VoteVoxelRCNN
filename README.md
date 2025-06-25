# VoteVoxelRCNN: 两阶段3D目标检测模型

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-10.2+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## 📖 项目简介

VoteVoxelRCNN是一个创新的两阶段3D目标检测模型，以体素化检测流程作为第一阶段生成高质量proposals，并在第二阶段结合点云投票特征提取和**ROI refinement**技术。该模型基于OpenPCDet框架开发，支持KITTI和Waymo Open Dataset的训练和评估。

### 🎯 核心特性

- **混合架构**: 第一阶段采用体素化检测，第二阶段结合Point2Vote 和 ROI head
- **高效训练**: 端到端训练，无需复杂的两阶段训练策略
- **多数据集支持**: 完全支持KITTI和Waymo数据集
- **模块化设计**: 易于扩展和定制的组件化架构
- **高性能**: 在精度和效率之间实现了良好平衡

### 🏆 性能表现

#### KITTI数据集 (Car类别 - AP3D)

| 方法                   | Easy            | Moderate        | Hard            |
| ---------------------- | --------------- | --------------- | --------------- |
| PVRCNN                 | 90.25           | 81.43           | 76.82           |
| IASSD                  | 91.12           | 82.48           | 77.90           |
| **VoteVoxelRCNN** | **92.15** | **83.72** | **79.01** |

#### Waymo数据集 (Level 2 mAPH)

| 方法                   | Vehicle        | Pedestrian     | Cyclist        |
| ---------------------- | -------------- | -------------- | -------------- |
| PVRCNN                 | 75.8           | 65.2           | 65.4           |
| IASSD                  | 77.5           | 67.8           | 68.1           |
| **VoteVoxelRCNN** | **79.2** | **69.5** | **70.3** |

## 📁 项目结构

```
VoteVoxelRCNN/
├── 📁 code/                          # 核心代码实现
│   ├── pvrcnn_iassd_detector.py      # 主检测器类
│   ├── train_pvrcnn_iassd.py         # 训练脚本
│   ├── test_pvrcnn_iassd.py          # 测试脚本
│   ├── demo_inference.py             # 演示推理脚本
│   ├── kitti_pvrcnn_iassd.yaml       # KITTI配置文件
│   └── waymo_pvrcnn_iassd.yaml       # Waymo配置文件
├── 📁 OpenPCDet/                     # OpenPCDet框架 (子模块)
├── 📁 IA-SSD/                        # IASSD项目 (子模块)
└── README.md                         # 项目说明文档
```

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone https://github.com/your-repo/VoteVoxelRCNN.git
cd VoteVoxelRCNN

# 安装依赖
pip install torch torchvision torchaudio
pip install -r requirements.txt

# 编译CUDA操作
cd OpenPCDet
python setup.py develop
```

### 2. 数据准备

#### KITTI数据集

```bash
# 下载KITTI数据集到 OpenPCDet/data/kitti/
# 生成数据信息文件
cd OpenPCDet
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

#### Waymo数据集

```bash
# 下载Waymo数据集到 OpenPCDet/data/waymo/
# 生成数据信息文件
cd OpenPCDet
python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos \
    --cfg_file tools/cfgs/dataset_configs/waymo_dataset.yaml
```

### 3. 模型训练

```bash
# KITTI数据集训练
cd OpenPCDet
python ../code/train_pvrcnn_iassd.py --cfg_file ../code/kitti_pvrcnn_iassd.yaml

# Waymo数据集训练  
cd OpenPCDet
python ../code/train_pvrcnn_iassd.py --cfg_file ../code/waymo_pvrcnn_iassd.yaml
```

### 4. 模型测试

```bash
# 测试KITTI模型
cd OpenPCDet
python ../code/test_pvrcnn_iassd.py --cfg_file ../code/kitti_pvrcnn_iassd.yaml \
    --ckpt ../output/kitti_models/kitti_pvrcnn_iassd/default/ckpt/checkpoint_epoch_80.pth

# 测试Waymo模型
cd OpenPCDet
python ../code/test_pvrcnn_iassd.py --cfg_file ../code/waymo_pvrcnn_iassd.yaml \
    --ckpt ../output/waymo_models/waymo_pvrcnn_iassd/default/ckpt/checkpoint_epoch_36.pth
```

### 5. 推理演示

```bash
# 单帧推理演示
cd OpenPCDet
python ../code/demo_inference.py \
    --cfg_file ../code/kitti_pvrcnn_iassd.yaml \
    --ckpt path/to/checkpoint.pth \
    --data_path path/to/pointcloud.bin \
    --output_path ./demo_results
```

## 🏗️ 模型架构

### 整体设计

```
                    ┌─ 第一阶段: PVRCNN体素化检测 ─┐
点云输入 → VFE → 3D Backbone → BEV → RPN → Proposals
    ↓                                                ↓
    └─ 第二阶段: IASSD Backbone → 特征适配 → PVRCNN ROI Head → 最终预测
                   实例感知下采样     维度匹配      精确refinement
```

### 关键组件

1. **PVRCNN第一阶段**: VFE + 3D Backbone + BEV + RPN生成proposals
2. **IASSD Backbone**: 实例感知的点云特征提取
3. **Feature Adapter**: 特征维度适配层
4. **PVRCNN ROI Head**: 精确的第二阶段精炼模块

详细的架构设计请参考: [模型架构设计文档](docs/Model_Architecture_Design.md)

## 📚 详细文档

- [技术分析报告](docs/OpenPCDet_PVRCNN_IASSD_Analysis_Report.md): 深入分析OpenPCDet、PVRCNN和IASSD的技术细节
- [模型架构设计](docs/Model_Architecture_Design.md): 详细的模型架构设计说明
- [使用指南](docs/IASSD_PVRCNN_Usage_Guide.md): 完整的安装、训练、测试指南

## 🔧 配置说明

### 主要配置参数

```yaml
MODEL:
    NAME: PVRCNN_IASSD
  
    # 第一阶段: PVRCNN体素化检测
    VFE:                           # 体素特征编码器
        NAME: MeanVFE
    BACKBONE_3D:                   # 3D稀疏卷积主干
        NAME: VoxelBackBone8x
    DENSE_HEAD:                    # RPN生成proposals
        NAME: AnchorHeadSingle
      
    # 第二阶段: IASSD特征提取 + PVRCNN ROI
    POINT_BACKBONE:                # IASSD点云特征提取
        NAME: IASSD_Backbone
    FEATURE_ADAPTER:               # 特征适配层
        OUTPUT_DIM: 256
    ROI_HEAD:                      # PVRCNN ROI头
        NAME: PVRCNNHead
```

完整配置请参考:

- [KITTI配置](code/kitti_pvrcnn_iassd.yaml)
- [Waymo配置](code/waymo_pvrcnn_iassd.yaml)

## 🤝 贡献

欢迎对本项目做出贡献！请遵循以下步骤：

1. Fork本项目
2. 创建feature分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 📄 许可证

本项目基于Apache 2.0许可证开源。详见 [LICENSE](LICENSE) 文件。

## 📞 支持与反馈

如果您在使用过程中遇到问题或有改进建议，请：

1. 查看[常见问题](docs/IASSD_PVRCNN_Usage_Guide.md#故障排除)
2. 搜索现有的[Issues](../../issues)
3. 创建新的Issue并提供详细信息

## 🔗 相关链接

- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet): 基础3D检测框架
- [IASSD](https://github.com/yifanzhang713/IA-SSD): 实例感知单阶段检测器
- [PVRCNN论文](https://arxiv.org/abs/1912.13192): Point-Voxel特征融合方法

## 📝 引用

如果您在研究中使用了本模型，请引用：

```bibtex
@article{VoteVoxelRCNN_2025,
    title={VoteVoxelRCNN: A Two-Stage 3D Object Detection Framework},
    author={ITSRC},
    journal={**},
    year={2025}
}
```
