"""
PVRCNN-IASSD 两阶段目标检测器
第一阶段：PVRCNN的体素化检测流程生成proposals
第二阶段：IASSD backbone进行点云特征提取 + PVRCNN ROI Head进行refinement

作者: MiniMax Agent
日期: 2025-06-22
"""

import torch
import torch.nn as nn
from pcdet.models.detectors import detector3d_template
from pcdet.models.backbones_3d import VoxelBackBone8x
from pcdet.models.backbones_2d import BaseBEVBackbone  
from pcdet.models.dense_heads import AnchorHeadSingle
from pcdet.models.roi_heads import PVRCNNHead


class PVRCNN_IASSD(detector3d_template.Detector3DTemplate):
    """
    PVRCNN-IASSD 两阶段检测器
    
    第一阶段: PVRCNN的完整体素化检测流程，生成高质量proposals
    第二阶段: IASSD backbone处理原始点云 + PVRCNN ROI Head进行refinement
    """
    
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        
    def build_networks(self):
        """构建网络模块"""
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size,
            'depth_downsample_factor': self.dataset.depth_downsample_factor
        }
        
        # ========== 第一阶段: PVRCNN的体素化检测流程 ==========
        
        # VFE: 体素特征编码器
        if self.model_cfg.get('VFE', None) is not None:
            from pcdet.models.backbones_3d.vfe import build_vfe
            self.vfe = build_vfe(
                model_cfg=self.model_cfg.VFE,
                num_point_features=model_info_dict['num_rawpoint_features'],
                point_cloud_range=model_info_dict['point_cloud_range'],
                voxel_size=model_info_dict['voxel_size'],
                grid_size=model_info_dict['grid_size'],
                depth_downsample_factor=model_info_dict['depth_downsample_factor']
            )
            model_info_dict['module_list'].append(self.vfe)
            model_info_dict['num_point_features'] = self.vfe.get_output_feature_dim()
        
        # 3D Backbone: 稀疏卷积主干网络
        if self.model_cfg.get('BACKBONE_3D', None) is not None:
            self.backbone_3d = VoxelBackBone8x(
                model_cfg=self.model_cfg.BACKBONE_3D,
                input_channels=model_info_dict['num_point_features'],
                grid_size=model_info_dict['grid_size']
            )
            model_info_dict['module_list'].append(self.backbone_3d)
            model_info_dict['backbone_channels'] = self.backbone_3d.num_point_features
        
        # Map-to-BEV: 3D特征映射到BEV
        if self.model_cfg.get('MAP_TO_BEV', None) is not None:
            from pcdet.models.backbones_2d.map_to_bev import HeightCompression
            self.map_to_bev_module = HeightCompression(
                model_cfg=self.model_cfg.MAP_TO_BEV,
                grid_size=model_info_dict['grid_size']
            )
            model_info_dict['module_list'].append(self.map_to_bev_module)
            model_info_dict['num_bev_features'] = self.map_to_bev_module.num_bev_features
        
        # 2D BEV Backbone
        if self.model_cfg.get('BACKBONE_2D', None) is not None:
            self.backbone_2d = BaseBEVBackbone(
                model_cfg=self.model_cfg.BACKBONE_2D,
                input_channels=model_info_dict.get('num_bev_features', None)
            )
            model_info_dict['module_list'].append(self.backbone_2d)
            model_info_dict['num_bev_features'] = self.backbone_2d.num_bev_features
        
        # Dense Head: RPN生成proposals
        if self.model_cfg.get('DENSE_HEAD', None) is not None:
            self.dense_head = AnchorHeadSingle(
                model_cfg=self.model_cfg.DENSE_HEAD,
                input_channels=model_info_dict['num_bev_features'],
                num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
                class_names=self.class_names,
                grid_size=model_info_dict['grid_size'],
                point_cloud_range=model_info_dict['point_cloud_range'],
                predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False)
            )
            model_info_dict['module_list'].append(self.dense_head)
        
        # ========== 第二阶段: IASSD Backbone + PVRCNN ROI Head ==========
        
        # IASSD Backbone: 实例感知点云特征提取
        if self.model_cfg.get('POINT_BACKBONE', None) is not None:
            from pcdet.models.backbones_3d.IASSD_backbone import IASSD_Backbone
            self.point_backbone = IASSD_Backbone(
                model_cfg=self.model_cfg.POINT_BACKBONE,
                input_channels=model_info_dict['num_rawpoint_features']
            )
            model_info_dict['module_list'].append(self.point_backbone)
            model_info_dict['num_point_features'] = self.point_backbone.num_point_features
        
        # 特征适配层: 对齐IASSD输出与PVRCNN ROI Head输入
        if hasattr(self.model_cfg, 'FEATURE_ADAPTER'):
            self.feature_adapter = nn.Conv1d(
                in_channels=model_info_dict['num_point_features'],
                out_channels=self.model_cfg.FEATURE_ADAPTER.get('OUTPUT_DIM', 256),
                kernel_size=1
            )
            model_info_dict['num_point_features'] = self.model_cfg.FEATURE_ADAPTER.get('OUTPUT_DIM', 256)
        
        # PVRCNN ROI Head: 基于proposals和点特征进行refinement
        if self.model_cfg.get('ROI_HEAD', None) is not None:
            self.roi_head = PVRCNNHead(
                model_cfg=self.model_cfg.ROI_HEAD,
                input_channels=model_info_dict['num_point_features'],
                backbone_channels=model_info_dict.get('backbone_channels', {}),
                point_cloud_range=model_info_dict['point_cloud_range'],
                voxel_size=model_info_dict['voxel_size'],
                num_class=self.num_class
            )
            model_info_dict['module_list'].append(self.roi_head)
            
        return model_info_dict['module_list']
    
    def forward(self, batch_dict):
        """前向传播"""
        # ========== 第一阶段: PVRCNN体素化检测流程 ==========
        
        # VFE: 体素特征提取
        if hasattr(self, 'vfe'):
            batch_dict = self.vfe(batch_dict)
        
        # 3D Backbone: 稀疏卷积特征提取
        if hasattr(self, 'backbone_3d'):
            batch_dict = self.backbone_3d(batch_dict)
        
        # Map-to-BEV: 转换为鸟瞰图
        if hasattr(self, 'map_to_bev_module'):
            batch_dict = self.map_to_bev_module(batch_dict)
        
        # 2D Backbone: BEV特征提取
        if hasattr(self, 'backbone_2d'):
            batch_dict = self.backbone_2d(batch_dict)
        
        # Dense Head: 生成proposals
        if hasattr(self, 'dense_head'):
            if self.training:
                targets_dict = self.assign_targets(batch_dict)
                batch_dict['targets'] = targets_dict
            
            batch_dict = self.dense_head(batch_dict)
        
        # ========== 第二阶段: IASSD Backbone + PVRCNN ROI Head ==========
        
        # IASSD Backbone: 处理原始点云
        if hasattr(self, 'point_backbone'):
            # 为IASSD backbone准备点云数据
            original_batch_dict = self.prepare_points_for_iassd(batch_dict)
            point_batch_dict = self.point_backbone(original_batch_dict)
            
            # 将点云特征结果合并到主batch_dict
            batch_dict = self.merge_point_features(batch_dict, point_batch_dict)
        
        # 特征适配
        if hasattr(self, 'feature_adapter'):
            point_features = batch_dict.get('point_features')
            if point_features is not None:
                # 调整维度: (N, C) -> (1, C, N) -> (1, C_out, N) -> (N, C_out)
                point_features_adapted = self.feature_adapter(
                    point_features.transpose(1, 0).unsqueeze(0)
                ).squeeze(0).transpose(1, 0)
                batch_dict['point_features'] = point_features_adapted
        
        # PVRCNN ROI Head: 基于proposals和点特征进行refinement
        if hasattr(self, 'roi_head') and not self.training or \
           (self.training and batch_dict.get('rois') is not None):
            
            # 从dense head结果生成ROIs (训练时)
            if self.training and 'rois' not in batch_dict:
                batch_dict = self.generate_rois_from_dense_head(batch_dict)
            
            # 确保点云特征格式正确
            self.prepare_features_for_roi_head(batch_dict)
            
            # ROI head处理
            batch_dict = self.roi_head(batch_dict)
            
        return batch_dict
    
    def prepare_points_for_iassd(self, batch_dict):
        """为IASSD backbone准备点云数据"""
        # 创建IASSD专用的batch_dict，包含原始点云
        iassd_batch_dict = {
            'points': batch_dict['points'],  # 原始点云
            'batch_size': batch_dict['batch_size'],
            'frame_id': batch_dict.get('frame_id', None)
        }
        
        # 如果训练时，传递ground truth信息
        if self.training and 'gt_boxes' in batch_dict:
            iassd_batch_dict['gt_boxes'] = batch_dict['gt_boxes']
            
        return iassd_batch_dict
    
    def merge_point_features(self, main_batch_dict, point_batch_dict):
        """将IASSD backbone的输出合并到主batch_dict"""
        # 合并点云特征
        if 'point_features' in point_batch_dict:
            main_batch_dict['point_features'] = point_batch_dict['point_features']
        
        # 合并点云坐标 (如果IASSD改变了点的索引)
        if 'point_coords' in point_batch_dict:
            main_batch_dict['point_coords'] = point_batch_dict['point_coords']
        elif 'points' in point_batch_dict:
            # 使用IASSD处理后的点云坐标
            points = point_batch_dict['points']
            main_batch_dict['point_coords'] = points[:, :4]  # [bs_idx, x, y, z]
        
        # 保留其他IASSD特有的输出
        for key in ['sa_ins_preds', 'sa_ins_labels', 'ctr_offsets']:
            if key in point_batch_dict:
                main_batch_dict[key] = point_batch_dict[key]
                
        return main_batch_dict
    
    def generate_rois_from_dense_head(self, batch_dict):
        """从dense head的预测结果生成ROIs用于训练ROI head"""
        if 'batch_cls_preds' in batch_dict and 'batch_box_preds' in batch_dict:
            batch_size = batch_dict['batch_size']
            rois_list = []
            
            for bs_idx in range(batch_size):
                cls_preds = batch_dict['batch_cls_preds'][bs_idx]  # (N, num_class)
                box_preds = batch_dict['batch_box_preds'][bs_idx]  # (N, 7)
                
                # 选择置信度最高的proposals
                max_cls_scores, _ = torch.max(torch.sigmoid(cls_preds), dim=-1)
                top_k = min(self.model_cfg.ROI_HEAD.get('NMS_CONFIG', {}).get('TRAIN_PRE_NMS_TOP_N', 128), 
                           len(max_cls_scores))
                _, top_indices = torch.topk(max_cls_scores, top_k)
                
                # 构造ROI格式: [bs_idx, x, y, z, dx, dy, dz, ry]
                batch_indices = torch.full((top_k, 1), bs_idx, 
                                         dtype=torch.float32, device=box_preds.device)
                rois = torch.cat([batch_indices, box_preds[top_indices]], dim=-1)
                rois_list.append(rois)
                
            batch_dict['rois'] = torch.cat(rois_list, dim=0)
            
        return batch_dict
    
    def prepare_features_for_roi_head(self, batch_dict):
        """为ROI head准备特征数据"""
        # 确保point_coords格式正确: (N, 4) [bs_idx, x, y, z]
        if 'point_coords' not in batch_dict and 'points' in batch_dict:
            points = batch_dict['points']
            batch_dict['point_coords'] = points[:, :4]
        
        # 确保具有必要的体素特征 (来自第一阶段的3D backbone)
        if hasattr(self, 'backbone_3d') and hasattr(self.backbone_3d, 'conv_out'):
            # 从第一阶段获取多尺度体素特征
            for i, conv_out in enumerate(self.backbone_3d.conv_out):
                batch_dict[f'multi_scale_3d_features.x_conv{i+1}'] = conv_out
    
    def get_training_loss(self):
        """计算训练损失"""
        disp_dict = {}
        
        # 第一阶段损失 (Dense Head)
        if hasattr(self, 'dense_head'):
            loss_dense, tb_dict = self.dense_head.get_loss()
            disp_dict.update(tb_dict)
            total_loss = loss_dense
        else:
            total_loss = 0
        
        # 第二阶段损失 (ROI Head)
        if hasattr(self, 'roi_head'):
            loss_roi, tb_dict_roi = self.roi_head.get_loss()
            total_loss += loss_roi
            
            # 更新显示字典，避免键冲突
            for key, val in tb_dict_roi.items():
                disp_dict[f'roi_{key}'] = val
        
        # IASSD的辅助损失 (如果有)
        if hasattr(self, 'point_backbone') and hasattr(self.point_backbone, 'get_loss'):
            loss_point, tb_dict_point = self.point_backbone.get_loss()
            total_loss += loss_point
            
            for key, val in tb_dict_point.items():
                disp_dict[f'point_{key}'] = val
                
        return total_loss, disp_dict
    
    def post_processing(self, batch_dict):
        """后处理"""
        # 如果有ROI head且已生成predictions，使用ROI head结果
        if hasattr(self, 'roi_head') and 'batch_cls_preds' in batch_dict and 'batch_box_preds' in batch_dict:
            # 检查是否是ROI head的输出 (通常会有更精确的预测)
            if 'rcnn_cls' in batch_dict or 'rcnn_reg' in batch_dict:
                batch_dict = self.roi_head.post_processing(batch_dict)
            else:
                # 使用dense head的后处理
                batch_dict = self.dense_head.post_processing(batch_dict)
        elif hasattr(self, 'dense_head'):
            # 只有dense head的结果
            batch_dict = self.dense_head.post_processing(batch_dict)
            
        return batch_dict
