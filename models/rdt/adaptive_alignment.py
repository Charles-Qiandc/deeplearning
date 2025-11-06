# models/rdt/adaptive_alignment.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from models.rdt.routing import TemporalExpertRouter


class GlobalAlignmentChannel(nn.Module):
    """
    全局语义对齐通道
    使用DINOv2特征进行高层语义对齐
    """
    
    def __init__(self, 
                 action_dim: int = 2048,
                 vision_dim: int = 1024,
                 hidden_dim: int = 1536,
                 dropout: float = 0.1):
        """
        初始化全局对齐通道
        
        Args:
            action_dim: 动作token维度
            vision_dim: 视觉特征维度（DINOv2）
            hidden_dim: 中间层维度
            dropout: Dropout率
        """
        super().__init__()
        
        self.action_dim = action_dim
        self.vision_dim = vision_dim
        
        # 动作到视觉空间的投影器
        self.action_projector = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vision_dim)
        )
        
        # 可学习的温度参数（用于对比学习）
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
        # 特征归一化层
        self.action_norm = nn.LayerNorm(vision_dim)
        self.vision_norm = nn.LayerNorm(vision_dim)
        
    def forward(self, action_features: torch.Tensor, vision_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算全局语义对齐
        
        Args:
            action_features: (B, T, action_dim) 动作特征
            vision_features: (B, 1, vision_dim) 全局视觉特征（CLS token）
            
        Returns:
            alignment_dict: 包含对齐分数和中间结果
        """
        B, T, _ = action_features.shape
        
        # 投影动作特征到视觉空间
        projected_actions = self.action_projector(action_features)  # (B, T, vision_dim)
        
        # 归一化
        projected_actions = self.action_norm(projected_actions)
        vision_features = self.vision_norm(vision_features)
        
        # 时间维度聚合：使用注意力权重
        # 计算每个时间步对全局表示的重要性
        attention_scores = torch.matmul(
            projected_actions, 
            vision_features.transpose(-2, -1)
        ) / (self.vision_dim ** 0.5)  # (B, T, 1)
        
        attention_weights = F.softmax(attention_scores, dim=1)  # (B, T, 1)
        
        # 加权聚合动作特征
        aggregated_actions = torch.sum(
            projected_actions * attention_weights, 
            dim=1, 
            keepdim=True
        )  # (B, 1, vision_dim)
        
        # L2归一化用于余弦相似度
        aggregated_actions_norm = F.normalize(aggregated_actions, p=2, dim=-1)
        vision_features_norm = F.normalize(vision_features, p=2, dim=-1)
        
        # 计算相似度
        similarity = torch.sum(
            aggregated_actions_norm * vision_features_norm, 
            dim=-1
        )  # (B, 1)
        
        # 温度缩放
        scaled_similarity = similarity / torch.clamp(self.temperature, min=0.01)
        
        return {
            'similarity': similarity,
            'scaled_similarity': scaled_similarity,
            'attention_weights': attention_weights,
            'projected_actions': projected_actions,
            'aggregated_actions': aggregated_actions
        }


class DepthAlignmentChannel(nn.Module):
    """
    深度几何对齐通道
    使用DepthAnythingV2特征进行几何结构对齐
    """
    
    def __init__(self,
                 action_dim: int = 2048,
                 depth_dim: int = 1024,
                 hidden_dim: int = 1536,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        初始化深度对齐通道
        
        Args:
            action_dim: 动作token维度
            depth_dim: 深度特征维度
            hidden_dim: 中间层维度
            num_heads: 多头注意力头数
            dropout: Dropout率
        """
        super().__init__()
        
        self.action_dim = action_dim
        self.depth_dim = depth_dim
        self.num_heads = num_heads
        
        # 动作到深度空间的投影器
        self.action_projector = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, depth_dim)
        )
        
        # 多头交叉注意力（动作查询深度特征）
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=depth_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 特征细化网络
        self.refinement = nn.Sequential(
            nn.Linear(depth_dim * 2, depth_dim),
            nn.LayerNorm(depth_dim),
            nn.GELU(),
            nn.Linear(depth_dim, depth_dim)
        )
        
        # 几何一致性评分器
        self.consistency_scorer = nn.Sequential(
            nn.Linear(depth_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, 
                action_features: torch.Tensor, 
                depth_features: torch.Tensor,
                depth_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算深度几何对齐
        
        Args:
            action_features: (B, T, action_dim) 动作特征
            depth_features: (B, N_patches, depth_dim) 深度特征patches
            depth_mask: (B, N_patches) 可选的掩码
            
        Returns:
            alignment_dict: 包含对齐分数和中间结果
        """
        B, T, _ = action_features.shape
        
        # 投影动作特征到深度空间
        projected_actions = self.action_projector(action_features)  # (B, T, depth_dim)
        
        # 交叉注意力：动作查询深度特征
        attended_features, attention_weights = self.cross_attention(
            query=projected_actions,
            key=depth_features,
            value=depth_features,
            key_padding_mask=depth_mask
        )  # (B, T, depth_dim), (B, T, N_patches)
        
        # 特征细化：融合原始投影和注意力结果
        refined_features = self.refinement(
            torch.cat([projected_actions, attended_features], dim=-1)
        )  # (B, T, depth_dim)
        
        # 计算几何一致性分数
        # 对每个时间步，评估与深度特征的几何一致性
        depth_global = depth_features.mean(dim=1, keepdim=True)  # (B, 1, depth_dim)
        depth_global_expanded = depth_global.expand(-1, T, -1)  # (B, T, depth_dim)
        
        consistency_input = torch.cat([refined_features, depth_global_expanded], dim=-1)
        consistency_scores = self.consistency_scorer(consistency_input)  # (B, T, 1)
        
        # 聚合时间维度的一致性分数
        avg_consistency = consistency_scores.mean(dim=1)  # (B, 1)
        
        # 计算特征相似度作为补充指标
        refined_norm = F.normalize(refined_features, p=2, dim=-1)
        depth_norm = F.normalize(depth_global, p=2, dim=-1)
        
        similarity = torch.matmul(
            refined_norm, 
            depth_norm.transpose(-2, -1)
        ).mean(dim=1)  # (B, 1)
        
        return {
            'consistency_scores': consistency_scores,
            'avg_consistency': avg_consistency,
            'similarity': similarity,
            'attention_weights': attention_weights,
            'refined_features': refined_features
        }


class DualChannelAlignmentModule(nn.Module):
    """
    双通道对齐模块：整合全局语义和深度几何对齐
    """
    
    def __init__(self,
                 action_dim: int = 2048,
                 vision_dim: int = 1024,
                 depth_dim: int = 1024,
                 router_config: Optional[Dict] = None):
        """
        初始化双通道对齐模块
        
        Args:
            action_dim: 动作维度
            vision_dim: 视觉特征维度（DINOv2）
            depth_dim: 深度特征维度（DepthAnythingV2）
            router_config: 路由网络配置
        """
        super().__init__()
        
        # 两个对齐通道
        self.global_alignment = GlobalAlignmentChannel(
            action_dim=action_dim,
            vision_dim=vision_dim
        )
        
        self.depth_alignment = DepthAlignmentChannel(
            action_dim=action_dim,
            depth_dim=depth_dim
        )
        
        # 路由网络
        if router_config is None:
            router_config = {
                'input_dim': action_dim,
                'num_experts': 2,
                'hidden_dim': 512,
                'temperature_init': 1.0,
                'use_gumbel': True
            }
        
        self.router = TemporalExpertRouter(**router_config)
        
        # 最终对齐分数融合层
        self.alignment_fusion = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self,
                action_tokens: torch.Tensor,
                global_features: torch.Tensor,
                depth_features: torch.Tensor,
                routing_weights: Optional[torch.Tensor] = None,
                critical_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        双通道自适应对齐
        
        Args:
            action_tokens: (B, T, D) 动作token序列
            global_features: (B, 1, vision_dim) DINOv2全局特征
            depth_features: (B, N_patches, depth_dim) DepthAnythingV2特征
            routing_weights: (B, T, 2) 可选的外部路由权重
            critical_labels: (B, T) 可选的监督标签
            
        Returns:
            对齐结果字典
        """
        B, T, D = action_tokens.shape
        
        # 1. 路由决策
        if routing_weights is None:
            routing_results = self.router(
                action_tokens, 
                critical_labels=critical_labels,
                return_analysis=False
            )
            routing_weights = routing_results['routing_weights']  # (B, T, 2)
            routing_loss = routing_results['routing_loss']
        else:
            routing_loss = torch.tensor(0.0, device=action_tokens.device)
            
        # 2. 全局语义对齐
        global_align_results = self.global_alignment(action_tokens, global_features)
        
        # 3. 深度几何对齐
        depth_align_results = self.depth_alignment(action_tokens, depth_features)
        
        # 4. 根据路由权重加权组合对齐损失
        # 提取每个通道的对齐分数
        global_similarity = global_align_results['similarity']  # (B, 1)
        depth_consistency = depth_align_results['avg_consistency']  # (B, 1)
        
        # 扩展到时间维度
        global_similarity_expanded = global_similarity.unsqueeze(1).expand(-1, T, -1)  # (B, T, 1)
        depth_consistency_expanded = depth_consistency.unsqueeze(1).expand(-1, T, -1)  # (B, T, 1)
        
        # 应用路由权重
        weighted_global = global_similarity_expanded * routing_weights[:, :, 0:1]  # (B, T, 1)
        weighted_depth = depth_consistency_expanded * routing_weights[:, :, 1:2]  # (B, T, 1)
        
        # 组合对齐分数
        combined_alignment = weighted_global + weighted_depth  # (B, T, 1)
        
        # 5. 计算对齐损失
        # 目标是最大化对齐分数（相似度）
        global_loss = 1.0 - global_similarity.mean()
        depth_loss = 1.0 - torch.sigmoid(depth_consistency).mean()
        
        # 加权损失
        avg_global_weight = routing_weights[:, :, 0].mean()
        avg_depth_weight = routing_weights[:, :, 1].mean()
        
        alignment_loss = (
            avg_global_weight * global_loss + 
            avg_depth_weight * depth_loss
        )
        
        # 6. 总损失
        total_loss = alignment_loss + routing_loss
        
        return {
            'total_loss': total_loss,
            'alignment_loss': alignment_loss,
            'routing_loss': routing_loss,
            'global_loss': global_loss,
            'depth_loss': depth_loss,
            'routing_weights': routing_weights,
            'combined_alignment': combined_alignment,
            'global_similarity': global_similarity,
            'depth_consistency': depth_consistency,
            'global_align_details': global_align_results,
            'depth_align_details': depth_align_results
        }
    
    def get_routing_stats(self) -> Dict:
        """获取路由统计信息"""
        return self.router.get_routing_stats() if hasattr(self.router, 'get_routing_stats') else {}


class AdaptiveREPALoss(nn.Module):
    """
    自适应REPA损失：整合双通道对齐的完整损失函数
    """
    
    def __init__(self,
                 alignment_module: DualChannelAlignmentModule,
                 base_weight: float = 0.2,
                 warmup_steps: int = 1000):
        """
        初始化自适应REPA损失
        
        Args:
            alignment_module: 双通道对齐模块
            base_weight: 基础损失权重
            warmup_steps: 预热步数
        """
        super().__init__()
        
        self.alignment_module = alignment_module
        self.base_weight = base_weight
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
    def forward(self,
                action_tokens: torch.Tensor,
                global_features: torch.Tensor,
                depth_features: torch.Tensor,
                critical_labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        计算自适应REPA损失
        
        Returns:
            loss: 标量损失值
            metrics: 详细指标字典
        """
        # 获取对齐结果
        align_results = self.alignment_module(
            action_tokens,
            global_features,
            depth_features,
            critical_labels=critical_labels
        )
        
        # 计算损失权重（预热策略）
        if self.current_step < self.warmup_steps:
            weight = self.base_weight * (self.current_step / self.warmup_steps)
        else:
            weight = self.base_weight
            
        # 应用权重
        weighted_loss = weight * align_results['total_loss']
        
        # 准备指标
        metrics = {
            'repa_loss': weighted_loss.item(),
            'alignment_loss': align_results['alignment_loss'].item(),
            'routing_loss': align_results['routing_loss'].item(),
            'global_loss': align_results['global_loss'].item(),
            'depth_loss': align_results['depth_loss'].item(),
            'loss_weight': weight
        }
        
        # 添加路由权重统计
        routing_weights = align_results['routing_weights']
        metrics['global_expert_usage'] = routing_weights[:, :, 0].mean().item()
        metrics['depth_expert_usage'] = routing_weights[:, :, 1].mean().item()
        
        self.current_step += 1
        
        return weighted_loss, metrics


# 测试代码
if __name__ == "__main__":
    # 设置维度
    B, T = 2, 64
    action_dim = 2048
    vision_dim = 1024
    depth_dim = 1024
    n_patches = 256
    
    # 创建测试数据
    action_tokens = torch.randn(B, T, action_dim)
    global_features = torch.randn(B, 1, vision_dim)
    depth_features = torch.randn(B, n_patches, depth_dim)
    critical_labels = torch.randint(0, 2, (B, T))
    
    # 创建对齐模块
    alignment_module = DualChannelAlignmentModule(
        action_dim=action_dim,
        vision_dim=vision_dim,
        depth_dim=depth_dim
    )
    
    # 测试前向传播
    results = alignment_module(
        action_tokens,
        global_features,
        depth_features,
        critical_labels=critical_labels
    )
    
    print("双通道对齐测试结果:")
    print(f"  总损失: {results['total_loss'].item():.4f}")
    print(f"  对齐损失: {results['alignment_loss'].item():.4f}")
    print(f"  路由损失: {results['routing_loss'].item():.4f}")
    print(f"  全局损失: {results['global_loss'].item():.4f}")
    print(f"  深度损失: {results['depth_loss'].item():.4f}")
    
    routing_weights = results['routing_weights']
    print(f"\n路由权重统计:")
    print(f"  全局专家平均权重: {routing_weights[:, :, 0].mean().item():.3f}")
    print(f"  深度专家平均权重: {routing_weights[:, :, 1].mean().item():.3f}")
    
    # 测试自适应损失
    print("\n测试自适应REPA损失:")
    repa_loss = AdaptiveREPALoss(alignment_module, base_weight=0.2)
    
    for step in range(3):
        loss, metrics = repa_loss(
            action_tokens,
            global_features,
            depth_features,
            critical_labels
        )
        print(f"  Step {step}: Loss={loss.item():.4f}, Weight={metrics['loss_weight']:.3f}")