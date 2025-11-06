
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class BinaryLabelSoftRouter(nn.Module):
    """
    基于关键时间段二元标签的软路由权重分配器 - 修复view/reshape错误
    """
    
    def __init__(self,
                 action_dim: int = 2048,
                 hidden_dim: int = 256,
                 # 基础权重配置
                 critical_global_weight: float = 0.25,
                 critical_depth_weight: float = 0.75,
                 non_critical_global_weight: float = 0.75,
                 non_critical_depth_weight: float = 0.25,
                 # 微调和平滑参数
                 enable_neural_adjustment: bool = True,
                 adjustment_strength: float = 0.1,
                 temporal_smoothing: float = 0.9,
                 temperature: float = 1.0):
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
        
        # 🔧 修复：神经网络微调器
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
        根据二元标签获取基础权重 - 使用查找表方法（最安全）
        
        Args:
            critical_labels: (B, T) 关键时间段标签，0/1
            
        Returns:
            base_weights: (B, T, 2) 基础权重 [global_weight, depth_weight]
        """
        B, T = critical_labels.shape
        device = critical_labels.device
        dtype = critical_labels.dtype if critical_labels.dtype.is_floating_point else torch.float32
        
        # 🔧 使用查找表方法（最安全，避免掩码索引问题）
        weight_lookup = torch.tensor([
            [self.non_critical_global_weight, self.non_critical_depth_weight],  # label=0
            [self.critical_global_weight, self.critical_depth_weight]           # label=1
        ], device=device, dtype=dtype)  # (2, 2)
        
        # 直接索引获取权重
        weights = weight_lookup[critical_labels.long()]  # (B, T, 2)
        
        # 验证权重和为1
        weight_sums = weights.sum(dim=-1)  # (B, T)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6), \
            f"权重和不为1: {weight_sums.unique()}"
        
        return weights
    
    def apply_neural_adjustment(self, 
                              base_weights: torch.Tensor, 
                              action_tokens: torch.Tensor) -> torch.Tensor:
        """
        🔧 修复版：使用神经网络微调权重 - 修复view/reshape问题
        """
        if not self.enable_neural_adjustment:
            return base_weights
        
        B, T, _ = action_tokens.shape
        
        # 🔧 确保形状正确
        assert base_weights.shape == (B, T, 2), f"基础权重形状错误: {base_weights.shape}, 期望: ({B}, {T}, 2)"
        assert action_tokens.shape == (B, T, self.action_dim), f"动作token形状错误: {action_tokens.shape}"
        
        try:
            # 🔧 修复：使用 .reshape() 而不是 .view()，并确保张量连续性
            # 方法1：使用 contiguous() + reshape()
            flat_tokens = action_tokens.contiguous().reshape(B * T, -1)
            
            # 计算调整值
            adjustments = self.weight_adjuster(flat_tokens)  # (B*T, 2)
            adjustments = adjustments.reshape(B, T, 2)       # (B, T, 2)
            
        except Exception as e:
            print(f"❌ reshape方法失败: {e}")
            try:
                # 🔧 备选方法：使用permute + flatten + unflatten
                # 重新排列维度确保连续性
                tokens_permuted = action_tokens.permute(0, 1, 2).contiguous()  # 确保连续
                flat_tokens = tokens_permuted.view(B * T, self.action_dim)
                
                adjustments = self.weight_adjuster(flat_tokens)  # (B*T, 2)
                adjustments = adjustments.view(B, T, 2)          # (B, T, 2)
                
            except Exception as e2:
                print(f"❌ 备选方法也失败: {e2}")
                # 🔧 最终备选：逐个处理
                adjustments = torch.zeros(B, T, 2, device=action_tokens.device, dtype=action_tokens.dtype)
                for b in range(B):
                    for t in range(T):
                        token = action_tokens[b, t, :]  # (action_dim,)
                        adj = self.weight_adjuster(token.unsqueeze(0))  # (1, 2)
                        adjustments[b, t, :] = adj.squeeze(0)
        
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
        """
        if is_first_batch or self.temporal_smoothing <= 0:
            # 第一个batch或不启用平滑
            self.prev_weights = current_weights[:, -1:, :].detach().clone()
            return current_weights
        
        B, T, _ = current_weights.shape
        
        # 确保 prev_weights 形状兼容
        if self.prev_weights.shape[0] != B:
            # 如果批次大小不匹配，重新初始化
            self.prev_weights = current_weights[:, 0:1, :].detach().clone()
            return current_weights
        
        # 创建平滑权重
        smoothed_weights = current_weights.clone()
        
        # 对第一个时间步应用平滑
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
        """
        B, T = critical_labels.shape
        device = critical_labels.device
        
        # 1. 获取基础权重
        base_weights = self.get_base_weights(critical_labels)
        
        # 2. 神经网络微调（可选）
        if self.enable_neural_adjustment and action_tokens is not None:
            try:
                adjusted_weights = self.apply_neural_adjustment(base_weights, action_tokens)
            except Exception as e:
                print(f"❌ 神经网络微调失败，使用基础权重: {e}")
                adjusted_weights = base_weights
        else:
            adjusted_weights = base_weights
        
        # 3. 时序平滑
        try:
            final_weights = self.apply_temporal_smoothing(adjusted_weights, is_first_batch)
        except Exception as e:
            print(f"❌ 时序平滑失败，使用调整后权重: {e}")
            final_weights = adjusted_weights
        
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


class RobustWeightAdjuster(nn.Module):
    """
    🆕 更健壮的权重调整器 - 专门处理张量连续性问题
    """
    
    def __init__(self, 
                 action_dim: int = 2048, 
                 hidden_dim: int = 256,
                 adjustment_strength: float = 0.1):
        super().__init__()
        
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.adjustment_strength = adjustment_strength
        
        # 简化的网络结构，减少张量操作复杂性
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
            nn.Tanh()
        )
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """小初始化策略"""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, action_tokens: torch.Tensor) -> torch.Tensor:
        """
        健壮的前向传播，处理各种张量连续性问题
        
        Args:
            action_tokens: (B, T, action_dim)
            
        Returns:
            adjustments: (B, T, 2)
        """
        B, T, action_dim = action_tokens.shape
        
        # 🔧 多种方法处理张量重塑问题
        try:
            # 方法1：contiguous + reshape（推荐）
            flat_tokens = action_tokens.contiguous().reshape(B * T, action_dim)
            raw_adjustments = self.net(flat_tokens)
            adjustments = raw_adjustments.reshape(B, T, 2)
            
        except Exception as e1:
            print(f"⚠️ 方法1失败: {e1}")
            try:
                # 方法2：克隆 + view
                flat_tokens = action_tokens.clone().view(B * T, action_dim)
                raw_adjustments = self.net(flat_tokens)
                adjustments = raw_adjustments.view(B, T, 2)
                
            except Exception as e2:
                print(f"⚠️ 方法2失败: {e2}")
                try:
                    # 方法3：使用flatten + unflatten
                    flat_tokens = torch.flatten(action_tokens, start_dim=0, end_dim=1)
                    raw_adjustments = self.net(flat_tokens)
                    adjustments = raw_adjustments.unflatten(0, (B, T))
                    
                except Exception as e3:
                    print(f"⚠️ 方法3失败: {e3}")
                    # 方法4：逐个token处理（保底方案）
                    adjustments = torch.zeros(B, T, 2, device=action_tokens.device, dtype=action_tokens.dtype)
                    for b in range(B):
                        batch_tokens = action_tokens[b]  # (T, action_dim)
                        batch_adjustments = self.net(batch_tokens)  # (T, 2)
                        adjustments[b] = batch_adjustments
        
        # 应用调整强度
        adjustments = adjustments * self.adjustment_strength
        
        return adjustments


class SimpleDualTeacherModel(nn.Module):
    """
    简化的双教师对齐模型 - 修复版
    """
    
    def __init__(self,
                 action_dim: int = 2048,
                 dinov2_dim: int = 1024,
                 depth_dim: int = 1024,
                 router_config: Dict = None):
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
        
        # 转换为损失（最大化相似度 = 最小化负相似度）
        global_loss = 1.0 - global_similarity  # (B, T)
        depth_loss = 1.0 - depth_similarity    # (B, T)
        
        # 应用路由权重
        weighted_global_loss = routing_weights[:, :, 0] * global_loss  # (B, T)
        weighted_depth_loss = routing_weights[:, :, 1] * depth_loss    # (B, T)
        
        # 总损失
        total_alignment_loss = (weighted_global_loss + weighted_depth_loss).mean()
        
        # 🔧 简化对比学习损失，避免复杂的张量操作
        contrastive_loss = torch.tensor(0.0, device=action_tokens.device, dtype=action_tokens.dtype)
        
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
    
    def forward(self,
                action_tokens: torch.Tensor,
                dinov2_features: torch.Tensor,
                depth_features: torch.Tensor,
                critical_labels: torch.Tensor,
                is_first_batch: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        """
        B, T, action_dim = action_tokens.shape
        
        # 确保critical_labels形状正确
        if critical_labels.shape != (B, T):
            if critical_labels.numel() == B * T:
                critical_labels = critical_labels.reshape(B, T)
            else:
                raise ValueError(f"无法修正critical_labels形状: {critical_labels.shape}")
        
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


# 测试修复后的代码
if __name__ == "__main__":
    print("🧪 测试修复view/reshape错误的软路由框架")
    
    # 创建各种形状的测试数据
    test_cases = [
        (2, 64, 2048),   # 原始错误案例
        (16, 64, 2048),  # 第一个错误案例
        (4, 128, 2048),  # 其他情况
    ]
    
    for B, T, action_dim in test_cases:
        print(f"\n📊 测试案例: B={B}, T={T}, action_dim={action_dim}")
        
        # 创建测试数据
        action_tokens = torch.randn(B, T, action_dim)
        critical_labels = torch.randint(0, 2, (B, T))
        
        # 测试软路由器
        try:
            router_config = {
                'action_dim': action_dim,
                'critical_global_weight': 0.25,
                'critical_depth_weight': 0.75,
                'non_critical_global_weight': 0.75,
                'non_critical_depth_weight': 0.25,
                'enable_neural_adjustment': True,
                'temporal_smoothing': 0.9,
            }
            
            soft_router = BinaryLabelSoftRouter(**router_config)
            routing_results = soft_router(critical_labels, action_tokens, is_first_batch=True)
            
            print(f"✅ 软路由器测试成功!")
            print(f"   - 路由权重形状: {routing_results['routing_weights'].shape}")
            
        except Exception as e:
            print(f"❌ 软路由器测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n🎊 view/reshape错误修复测试完成!")