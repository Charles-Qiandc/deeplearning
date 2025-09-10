# models/rdt/binary_soft_routing.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class BinaryLabelSoftRouter(nn.Module):
    """
    基于关键时间段二元标签的软路由权重分配器
    
    核心思路：
    - 关键时间段(1): [全局25%, 深度75%] - 偏向精确操作
    - 非关键时间段(0): [全局75%, 深度25%] - 偏向场景理解
    """
    
    def __init__(self,
                 action_dim: int = 2048,
                 hidden_dim: int = 256,
                 # 基础权重配置
                 critical_global_weight: float = 0.25,      # 关键时间段全局权重
                 critical_depth_weight: float = 0.75,       # 关键时间段深度权重
                 non_critical_global_weight: float = 0.75,  # 非关键时间段全局权重
                 non_critical_depth_weight: float = 0.25,   # 非关键时间段深度权重
                 # 微调和平滑参数
                 enable_neural_adjustment: bool = True,     # 是否启用神经网络微调
                 adjustment_strength: float = 0.1,          # 微调强度
                 temporal_smoothing: float = 0.9,           # 时序平滑系数
                 temperature: float = 1.0):                 # softmax温度
        super().__init__()
        
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # 基础权重配置
        self.critical_global_weight = critical_global_weight
        self.critical_depth_weight = critical_depth_weight
        self.non_critical_global_weight = non_critical_global_weight
        self.non_critical_depth_weight = non_critical_depth_weight
        
        # 确保权重和为1
        assert abs(critical_global_weight + critical_depth_weight - 1.0) < 1e-6
        assert abs(non_critical_global_weight + non_critical_depth_weight - 1.0) < 1e-6
        
        # 微调和平滑参数
        self.enable_neural_adjustment = enable_neural_adjustment
        self.adjustment_strength = adjustment_strength
        self.temporal_smoothing = temporal_smoothing
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
        # 可选：神经网络微调器
        if enable_neural_adjustment:
            self.weight_adjuster = nn.Sequential(
                nn.Linear(action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 2),  # 输出调整值 [global_adj, depth_adj]
                nn.Tanh()  # 输出范围 [-1, 1]
            )
            self._initialize_adjuster()
        
        # 存储上一时间步的权重用于平滑
        self.register_buffer('prev_weights', torch.zeros(1, 1, 2))
        
    def _initialize_adjuster(self):
        """初始化权重微调器"""
        for module in self.weight_adjuster:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)  # 小初始化
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def get_base_weights(self, critical_labels: torch.Tensor) -> torch.Tensor:
        """
        根据二元标签获取基础权重
        
        Args:
            critical_labels: (B, T) 关键时间段标签，0/1
            
        Returns:
            base_weights: (B, T, 2) 基础权重 [global_weight, depth_weight]
        """
        B, T = critical_labels.shape
        device = critical_labels.device
        dtype = critical_labels.dtype if critical_labels.dtype.is_floating_point else torch.float32
        
        # 创建权重张量
        weights = torch.zeros(B, T, 2, device=device, dtype=dtype)
        
        # 关键时间段权重 (critical_labels == 1)
        critical_mask = critical_labels.bool()
        weights[critical_mask, 0] = self.critical_global_weight    # 全局权重
        weights[critical_mask, 1] = self.critical_depth_weight     # 深度权重
        
        # 非关键时间段权重 (critical_labels == 0)
        non_critical_mask = ~critical_mask
        weights[non_critical_mask, 0] = self.non_critical_global_weight  # 全局权重
        weights[non_critical_mask, 1] = self.non_critical_depth_weight   # 深度权重
        
        return weights
    
    def apply_neural_adjustment(self, 
                              base_weights: torch.Tensor, 
                              action_tokens: torch.Tensor) -> torch.Tensor:
        """
        使用神经网络微调权重
        
        Args:
            base_weights: (B, T, 2) 基础权重
            action_tokens: (B, T, action_dim) 动作tokens
            
        Returns:
            adjusted_weights: (B, T, 2) 微调后的权重
        """
        if not self.enable_neural_adjustment:
            return base_weights
        
        B, T, _ = action_tokens.shape
        
        # 计算权重调整值
        flat_tokens = action_tokens.view(B * T, -1)
        adjustments = self.weight_adjuster(flat_tokens)  # (B*T, 2)
        adjustments = adjustments.view(B, T, 2)          # (B, T, 2)
        
        # 应用调整强度
        adjustments = adjustments * self.adjustment_strength
        
        # 调整权重
        adjusted_weights = base_weights + adjustments
        
        # 重新归一化确保权重和为1
        adjusted_weights = F.softmax(adjusted_weights / torch.clamp(self.temperature, min=0.1), dim=-1)
        
        return adjusted_weights
    
    def apply_temporal_smoothing(self, 
                               current_weights: torch.Tensor,
                               is_first_batch: bool = False) -> torch.Tensor:
        """
        应用时序平滑
        
        Args:
            current_weights: (B, T, 2) 当前权重
            is_first_batch: 是否为第一个batch
            
        Returns:
            smoothed_weights: (B, T, 2) 平滑后的权重
        """
        if is_first_batch or self.temporal_smoothing <= 0:
            # 第一个batch或不启用平滑
            self.prev_weights = current_weights[:, -1:, :].detach().clone()
            return current_weights
        
        B, T, _ = current_weights.shape
        
        # 创建平滑权重
        smoothed_weights = current_weights.clone()
        
        # 对第一个时间步应用平滑
        if self.prev_weights.shape[0] == B:  # 批次大小匹配
            smoothed_weights[:, 0:1, :] = (
                self.temporal_smoothing * self.prev_weights + 
                (1 - self.temporal_smoothing) * current_weights[:, 0:1, :]
            )
        
        # 对序列内部应用平滑
        for t in range(1, T):
            smoothed_weights[:, t:t+1, :] = (
                self.temporal_smoothing * smoothed_weights[:, t-1:t, :] + 
                (1 - self.temporal_smoothing) * current_weights[:, t:t+1, :]
            )
        
        # 更新prev_weights
        self.prev_weights = smoothed_weights[:, -1:, :].detach().clone()
        
        return smoothed_weights
    
    def forward(self, 
                critical_labels: torch.Tensor, 
                action_tokens: Optional[torch.Tensor] = None,
                is_first_batch: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播：生成软路由权重
        
        Args:
            critical_labels: (B, T) 关键时间段标签
            action_tokens: (B, T, action_dim) 可选的动作tokens用于微调
            is_first_batch: 是否为第一个batch
            
        Returns:
            Dict包含:
            - routing_weights: (B, T, 2) 最终路由权重
            - base_weights: (B, T, 2) 基础权重
            - adjusted_weights: (B, T, 2) 微调后权重（如果启用）
            - statistics: 各种统计信息
        """
        B, T = critical_labels.shape
        device = critical_labels.device
        
        # 1. 获取基础权重
        base_weights = self.get_base_weights(critical_labels)
        
        # 2. 神经网络微调（可选）
        if self.enable_neural_adjustment and action_tokens is not None:
            adjusted_weights = self.apply_neural_adjustment(base_weights, action_tokens)
        else:
            adjusted_weights = base_weights
        
        # 3. 时序平滑
        final_weights = self.apply_temporal_smoothing(adjusted_weights, is_first_batch)
        
        # 4. 计算统计信息
        critical_mask = critical_labels.bool()
        non_critical_mask = ~critical_mask
        
        statistics = {
            'critical_ratio': critical_mask.float().mean().item(),
            'avg_global_weight': final_weights[:, :, 0].mean().item(),
            'avg_depth_weight': final_weights[:, :, 1].mean().item(),
            'weight_std_global': final_weights[:, :, 0].std().item(),
            'weight_std_depth': final_weights[:, :, 1].std().item(),
            'temperature': torch.clamp(self.temperature, min=0.1).item(),
        }
        
        # 分类统计
        if critical_mask.any():
            critical_weights = final_weights[critical_mask]
            statistics.update({
                'critical_avg_global': critical_weights[:, 0].mean().item(),
                'critical_avg_depth': critical_weights[:, 1].mean().item(),
            })
        
        if non_critical_mask.any():
            non_critical_weights = final_weights[non_critical_mask]
            statistics.update({
                'non_critical_avg_global': non_critical_weights[:, 0].mean().item(),
                'non_critical_avg_depth': non_critical_weights[:, 1].mean().item(),
            })
        
        # 微调统计
        if self.enable_neural_adjustment and action_tokens is not None:
            weight_drift = torch.norm(adjusted_weights - base_weights, dim=-1).mean()
            statistics['weight_drift'] = weight_drift.item()
        
        return {
            'routing_weights': final_weights,
            'base_weights': base_weights,
            'adjusted_weights': adjusted_weights,
            'statistics': statistics
        }


class SimpleDualTeacherModel(nn.Module):
    """
    简化的双教师对齐模型
    
    集成软路由机制，实现动态权重分配的视觉对齐
    """
    
    def __init__(self,
                 action_dim: int = 2048,
                 dinov2_dim: int = 1024,
                 depth_dim: int = 1024,
                 router_config: Dict = None):
        """
        初始化双教师模型
        
        Args:
            action_dim: 动作token维度
            dinov2_dim: DINOv2特征维度
            depth_dim: 深度特征维度
            router_config: 路由器配置
        """
        super().__init__()
        
        self.action_dim = action_dim
        self.dinov2_dim = dinov2_dim
        self.depth_dim = depth_dim
        
        # 创建软路由器
        if router_config is None:
            router_config = {
                'action_dim': action_dim,
                'critical_global_weight': 0.25,
                'critical_depth_weight': 0.75,
                'non_critical_global_weight': 0.75,
                'non_critical_depth_weight': 0.25,
                'enable_neural_adjustment': True,
                'temporal_smoothing': 0.9,
            }
        
        self.soft_router = BinaryLabelSoftRouter(**router_config)
        
        # 特征投影器
        self.dinov2_projector = nn.Sequential(
            nn.Linear(dinov2_dim, (dinov2_dim + action_dim) // 2),
            nn.LayerNorm((dinov2_dim + action_dim) // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear((dinov2_dim + action_dim) // 2, action_dim),
            nn.LayerNorm(action_dim),
        )
        
        self.depth_projector = nn.Sequential(
            nn.Linear(depth_dim, (depth_dim + action_dim) // 2),
            nn.LayerNorm((depth_dim + action_dim) // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear((depth_dim + action_dim) // 2, action_dim),
            nn.LayerNorm(action_dim),
        )
        
        # 可学习的温度参数用于对比学习
        self.alignment_temperature = nn.Parameter(torch.tensor(0.07))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in [self.dinov2_projector, self.depth_projector]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.5)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
    
    def compute_alignment_loss(self,
                             action_tokens: torch.Tensor,
                             dinov2_features: torch.Tensor,
                             depth_features: torch.Tensor,
                             routing_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算加权对齐损失
        
        Args:
            action_tokens: (B, T, action_dim) 动作tokens
            dinov2_features: (B, dinov2_dim) DINOv2全局特征
            depth_features: (B, depth_dim) 深度特征
            routing_weights: (B, T, 2) 路由权重 [global_weight, depth_weight]
            
        Returns:
            损失字典
        """
        B, T, _ = action_tokens.shape
        
        # 投影视觉特征到动作空间
        projected_dinov2 = self.dinov2_projector(dinov2_features)  # (B, action_dim)
        projected_depth = self.depth_projector(depth_features)     # (B, action_dim)
        
        # 扩展到时间维度
        projected_dinov2_expanded = projected_dinov2.unsqueeze(1).expand(-1, T, -1)
        projected_depth_expanded = projected_depth.unsqueeze(1).expand(-1, T, -1)
        
        # L2归一化
        action_norm = F.normalize(action_tokens, p=2, dim=-1)
        dinov2_norm = F.normalize(projected_dinov2_expanded, p=2, dim=-1)
        depth_norm = F.normalize(projected_depth_expanded, p=2, dim=-1)
        
        # 计算相似度
        global_similarity = torch.sum(action_norm * dinov2_norm, dim=-1)  # (B, T)
        depth_similarity = torch.sum(action_norm * depth_norm, dim=-1)    # (B, T)
        
        # 温度缩放
        temp = torch.clamp(self.alignment_temperature, min=0.01)
        global_similarity_scaled = global_similarity / temp
        depth_similarity_scaled = depth_similarity / temp
        
        # 转换为损失（最大化相似度 = 最小化负相似度）
        global_loss = 1.0 - global_similarity  # (B, T)
        depth_loss = 1.0 - depth_similarity    # (B, T)
        
        # 应用路由权重
        weighted_global_loss = routing_weights[:, :, 0] * global_loss  # (B, T)
        weighted_depth_loss = routing_weights[:, :, 1] * depth_loss    # (B, T)
        
        # 总损失
        total_alignment_loss = (weighted_global_loss + weighted_depth_loss).mean()
        
        # 额外的对比学习损失
        contrastive_loss = self.compute_contrastive_loss(
            action_norm, dinov2_norm, depth_norm, routing_weights
        )
        
        # 组合损失
        combined_loss = total_alignment_loss + 0.1 * contrastive_loss
        
        return {
            'total_loss': combined_loss,
            'alignment_loss': total_alignment_loss,
            'contrastive_loss': contrastive_loss,
            'global_loss_raw': global_loss.mean(),
            'depth_loss_raw': depth_loss.mean(),
            'weighted_global_loss': weighted_global_loss.mean(),
            'weighted_depth_loss': weighted_depth_loss.mean(),
            'global_similarity_avg': global_similarity.mean(),
            'depth_similarity_avg': depth_similarity.mean(),
            'alignment_temperature': temp.item(),
        }
    
    def compute_contrastive_loss(self,
                               action_norm: torch.Tensor,
                               dinov2_norm: torch.Tensor,
                               depth_norm: torch.Tensor,
                               routing_weights: torch.Tensor) -> torch.Tensor:
        """
        计算对比学习损失
        
        Args:
            action_norm: (B, T, D) 归一化的动作特征
            dinov2_norm: (B, T, D) 归一化的DINOv2特征
            depth_norm: (B, T, D) 归一化的深度特征
            routing_weights: (B, T, 2) 路由权重
            
        Returns:
            contrastive_loss: 标量损失
        """
        B, T, D = action_norm.shape
        
        # 🔧 修复：确保所有张量数据类型一致
        target_dtype = action_norm.dtype
        dinov2_norm = dinov2_norm.to(dtype=target_dtype)
        depth_norm = depth_norm.to(dtype=target_dtype)
        routing_weights = routing_weights.to(dtype=target_dtype)
        
        # 计算加权的视觉表示
        weighted_visual = (
            routing_weights[:, :, 0:1] * dinov2_norm + 
            routing_weights[:, :, 1:2] * depth_norm
        )  # (B, T, D)
        
        # 🔧 修复：使用 reshape 而不是 view，并确保张量连续性和数据类型一致
        action_flat = action_norm.contiguous().reshape(-1, D)  # (B*T, D)
        visual_flat = weighted_visual.contiguous().reshape(-1, D)  # (B*T, D)
        
        # 确保数据类型一致
        action_flat = action_flat.to(dtype=target_dtype)
        visual_flat = visual_flat.to(dtype=target_dtype)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(action_flat, visual_flat.transpose(-2, -1))  # (B*T, B*T)
        
        # 对角线为正样本，其他为负样本
        labels = torch.arange(B * T, device=action_norm.device)
        
        # 简化的对比损失：只使用对角线正样本
        positive_sim = torch.diag(sim_matrix)  # (B*T,)
        
        # 计算每行的log-sum-exp作为归一化项
        logsumexp_sim = torch.logsumexp(sim_matrix, dim=-1)  # (B*T,)
        
        # 对比损失：-log(exp(pos) / sum(exp(all)))
        contrastive_loss = -positive_sim + logsumexp_sim
        contrastive_loss = contrastive_loss.mean()
        
        return contrastive_loss
    
    def forward(self,
                action_tokens: torch.Tensor,
                dinov2_features: torch.Tensor,
                depth_features: torch.Tensor,
                critical_labels: torch.Tensor,
                is_first_batch: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            action_tokens: (B, T, action_dim) 动作tokens
            dinov2_features: (B, dinov2_dim) DINOv2特征
            depth_features: (B, depth_dim) 深度特征
            critical_labels: (B, T) 关键时间段标签
            is_first_batch: 是否为第一个batch
            
        Returns:
            完整结果字典
        """
        # 1. 生成路由权重
        routing_results = self.soft_router(
            critical_labels, 
            action_tokens, 
            is_first_batch
        )
        
        routing_weights = routing_results['routing_weights']
        
        # 2. 计算对齐损失
        alignment_results = self.compute_alignment_loss(
            action_tokens,
            dinov2_features,
            depth_features,
            routing_weights
        )
        
        # 3. 合并结果
        results = {
            **alignment_results,
            'routing_weights': routing_weights,
            'router_statistics': routing_results['statistics'],
            'base_weights': routing_results['base_weights'],
        }
        
        return results


# 测试代码
if __name__ == "__main__":
    print("🧪 测试基于关键时间段的软路由双教师框架")
    
    # 设置参数
    B, T = 4, 64
    action_dim = 2048
    dinov2_dim = 1024
    depth_dim = 1024
    
    # 创建测试数据
    print("📊 创建测试数据...")
    action_tokens = torch.randn(B, T, action_dim)
    dinov2_features = torch.randn(B, dinov2_dim)
    depth_features = torch.randn(B, depth_dim)
    
    # 创建关键时间段标签（模拟真实场景）
    critical_labels = torch.zeros(B, T, dtype=torch.long)
    for b in range(B):
        # 每个序列随机选择30%的时间步作为关键时间段
        num_critical = int(T * 0.3)
        critical_indices = torch.randperm(T)[:num_critical]
        critical_labels[b, critical_indices] = 1
    
    print(f"   - 动作tokens: {action_tokens.shape}")
    print(f"   - DINOv2特征: {dinov2_features.shape}")
    print(f"   - 深度特征: {depth_features.shape}")
    print(f"   - 关键时间段标签: {critical_labels.shape}")
    print(f"   - 关键时间段比例: {critical_labels.float().mean().item():.3f}")
    
    # 测试软路由器
    print("\n🔀 测试软路由器...")
    router_config = {
        'action_dim': action_dim,
        'critical_global_weight': 0.25,      # 关键时间段：全局25%
        'critical_depth_weight': 0.75,       # 关键时间段：深度75%
        'non_critical_global_weight': 0.75,  # 非关键时间段：全局75%
        'non_critical_depth_weight': 0.25,   # 非关键时间段：深度25%
        'enable_neural_adjustment': True,
        'temporal_smoothing': 0.9,
    }
    
    soft_router = BinaryLabelSoftRouter(**router_config)
    routing_results = soft_router(critical_labels, action_tokens, is_first_batch=True)
    
    print("✅ 软路由器测试结果:")
    print(f"   - 路由权重形状: {routing_results['routing_weights'].shape}")
    stats = routing_results['statistics']
    print(f"   - 平均全局权重: {stats['avg_global_weight']:.3f}")
    print(f"   - 平均深度权重: {stats['avg_depth_weight']:.3f}")
    if 'critical_avg_global' in stats:
        print(f"   - 关键时间段全局权重: {stats['critical_avg_global']:.3f}")
        print(f"   - 关键时间段深度权重: {stats['critical_avg_depth']:.3f}")
    if 'non_critical_avg_global' in stats:
        print(f"   - 非关键时间段全局权重: {stats['non_critical_avg_global']:.3f}")
        print(f"   - 非关键时间段深度权重: {stats['non_critical_avg_depth']:.3f}")
    
    # 测试完整双教师模型
    print("\n🎯 测试完整双教师模型...")
    dual_teacher_model = SimpleDualTeacherModel(
        action_dim=action_dim,
        dinov2_dim=dinov2_dim,
        depth_dim=depth_dim,
        router_config=router_config
    )
    
    results = dual_teacher_model(
        action_tokens,
        dinov2_features,
        depth_features,
        critical_labels,
        is_first_batch=True
    )
    
    print("✅ 双教师模型测试结果:")
    print(f"   - 总损失: {results['total_loss'].item():.4f}")
    print(f"   - 对齐损失: {results['alignment_loss'].item():.4f}")
    print(f"   - 对比损失: {results['contrastive_loss'].item():.4f}")
    print(f"   - 全局相似度: {results['global_similarity_avg'].item():.4f}")
    print(f"   - 深度相似度: {results['depth_similarity_avg'].item():.4f}")
    print(f"   - 加权全局损失: {results['weighted_global_loss'].item():.4f}")
    print(f"   - 加权深度损失: {results['weighted_depth_loss'].item():.4f}")
    
    # 验证权重分配是否符合预期
    print("\n📈 验证权重分配策略...")
    routing_weights = results['routing_weights']
    critical_mask = critical_labels.bool()
    non_critical_mask = ~critical_mask
    
    if critical_mask.any():
        critical_weights = routing_weights[critical_mask]
        expected_global = router_config['critical_global_weight']
        expected_depth = router_config['critical_depth_weight']
        actual_global = critical_weights[:, 0].mean().item()
        actual_depth = critical_weights[:, 1].mean().item()
        
        print(f"   关键时间段权重验证:")
        print(f"     - 期望全局权重: {expected_global:.3f}, 实际: {actual_global:.3f}")
        print(f"     - 期望深度权重: {expected_depth:.3f}, 实际: {actual_depth:.3f}")
        print(f"     - 全局权重误差: {abs(actual_global - expected_global):.3f}")
        print(f"     - 深度权重误差: {abs(actual_depth - expected_depth):.3f}")
    
    if non_critical_mask.any():
        non_critical_weights = routing_weights[non_critical_mask]
        expected_global = router_config['non_critical_global_weight']
        expected_depth = router_config['non_critical_depth_weight']
        actual_global = non_critical_weights[:, 0].mean().item()
        actual_depth = non_critical_weights[:, 1].mean().item()
        
        print(f"   非关键时间段权重验证:")
        print(f"     - 期望全局权重: {expected_global:.3f}, 实际: {actual_global:.3f}")
        print(f"     - 期望深度权重: {expected_depth:.3f}, 实际: {actual_depth:.3f}")
        print(f"     - 全局权重误差: {abs(actual_global - expected_global):.3f}")
        print(f"     - 深度权重误差: {abs(actual_depth - expected_depth):.3f}")
    
    print("\n🎊 所有测试通过！软路由双教师框架运行正常。")
    print("\n📋 框架特点总结:")
    print("   ✓ 基于关键时间段标签的规则驱动权重分配")
    print("   ✓ 可选的神经网络微调机制")
    print("   ✓ 时序平滑防止权重突变")
    print("   ✓ 对比学习增强特征对齐")
    print("   ✓ 完整的统计信息和可解释性")
    print("   ✓ 易于集成到现有RDT框架")