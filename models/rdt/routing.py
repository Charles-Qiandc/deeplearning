# models/rdt/routing.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict


class TemporalExpertRouter(nn.Module):
    """
    时序感知的专家路由网络
    为每个时间步的动作token动态选择合适的视觉对齐专家
    """
    
    def __init__(self,
                 input_dim: int = 2048,
                 num_experts: int = 2,
                 hidden_dim: int = 512,
                 temperature_init: float = 1.0,
                 temperature_min: float = 0.1,
                 use_gumbel: bool = True,
                 load_balance_weight: float = 0.01):
        """
        初始化时序路由网络
        
        Args:
            input_dim: 输入动作token维度
            num_experts: 专家数量（2：全局语义专家 + 深度几何专家）
            hidden_dim: 路由网络隐藏层维度
            temperature_init: Gumbel Softmax初始温度
            temperature_min: 最小温度
            use_gumbel: 是否使用Gumbel Softmax（否则使用标准softmax）
            load_balance_weight: 负载均衡损失权重
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.use_gumbel = use_gumbel
        self.load_balance_weight = load_balance_weight
        
        # 温度参数（用于Gumbel Softmax）
        self.temperature = nn.Parameter(torch.tensor(temperature_init))
        self.temperature_min = temperature_min
        
        # 时序感知的路由网络
        self.router = nn.Sequential(
            # 第一层：特征提取
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            # 第二层：时序建模
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            
            # 输出层：专家选择logits
            nn.Linear(hidden_dim, num_experts)
        )
        
        # 时序上下文编码器（可选：增强时序感知）
        self.temporal_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # 融合时序特征和原始特征
        self.fusion_layer = nn.Linear(input_dim + hidden_dim, input_dim)
        
        # 专家特化的投影层（可选：让每个专家有自己的特征空间）
        self.expert_projections = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_experts)
        ])
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
                        
    def gumbel_softmax(self, logits: torch.Tensor, temperature: float, hard: bool = False) -> torch.Tensor:
        """
        Gumbel Softmax采样
        
        Args:
            logits: (B, T, num_experts) 未归一化的logits
            temperature: 温度参数
            hard: 是否使用hard采样（离散化）
            
        Returns:
            samples: (B, T, num_experts) 采样结果
        """
        if self.training:
            # 添加Gumbel噪声
            U = torch.rand_like(logits)
            gumbel_noise = -torch.log(-torch.log(U + 1e-10) + 1e-10)
            noisy_logits = (logits + gumbel_noise) / temperature
        else:
            # 推理时不添加噪声
            noisy_logits = logits / temperature
            
        # Softmax
        y_soft = F.softmax(noisy_logits, dim=-1)
        
        if hard:
            # Straight-through estimator
            index = y_soft.max(-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
            
        return ret
    
    def compute_load_balance_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """
        计算负载均衡损失，确保专家使用均衡
        
        Args:
            routing_weights: (B, T, num_experts) 路由权重
            
        Returns:
            loss: 标量损失值
        """
        # 计算每个专家的平均负载
        expert_loads = routing_weights.mean(dim=[0, 1])  # (num_experts,)
        
        # 目标是均匀分布
        target_load = 1.0 / self.num_experts
        
        # L2损失：惩罚偏离均匀分布
        load_balance_loss = torch.sum((expert_loads - target_load) ** 2)
        
        # 额外：最大负载惩罚（防止某个专家过载）
        max_load_penalty = torch.max(expert_loads) - target_load * 2
        max_load_penalty = F.relu(max_load_penalty)  # 只在超过2倍平均负载时惩罚
        
        total_loss = load_balance_loss + max_load_penalty
        
        return total_loss
    
    def forward(self, 
                action_tokens: torch.Tensor,
                critical_labels: Optional[torch.Tensor] = None,
                return_analysis: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播：为动作序列生成路由决策
        
        Args:
            action_tokens: (B, T, D) 动作token序列
            critical_labels: (B, T) 可选的监督标签（0/1表示使用哪个专家）
            return_analysis: 是否返回详细分析信息
            
        Returns:
            Dictionary containing:
                - routing_weights: (B, T, num_experts) 软路由权重
                - hard_decisions: (B, T) 硬路由决策
                - routing_loss: 路由相关损失
                - (optional) analysis: 详细分析信息
        """
        B, T, D = action_tokens.shape
        
        # 1. 时序特征增强（可选）
        temporal_features, _ = self.temporal_encoder(action_tokens)
        
        # 2. 特征融合
        enhanced_tokens = self.fusion_layer(
            torch.cat([action_tokens, temporal_features], dim=-1)
        )
        
        # 3. 计算路由logits
        routing_logits = self.router(enhanced_tokens)  # (B, T, num_experts)
        
        # 4. 应用温度调节的softmax或Gumbel softmax
        current_temp = torch.clamp(self.temperature, min=self.temperature_min)
        
        if self.use_gumbel:
            routing_weights = self.gumbel_softmax(
                routing_logits, 
                temperature=current_temp,
                hard=False  # 训练时使用软决策
            )
        else:
            routing_weights = F.softmax(routing_logits / current_temp, dim=-1)
            
        # 5. 生成硬决策（用于分析）
        hard_decisions = torch.argmax(routing_weights, dim=-1)  # (B, T)
        
        # 6. 计算损失
        total_loss = torch.tensor(0.0, device=action_tokens.device)
        
        # 6.1 监督损失（如果有标签）
        supervised_loss = torch.tensor(0.0, device=action_tokens.device)
        if critical_labels is not None:
            # 将标签转换为one-hot
            labels_onehot = F.one_hot(critical_labels.long(), num_classes=self.num_experts).float()
            
            # 交叉熵损失
            supervised_loss = F.binary_cross_entropy(
                routing_weights,
                labels_onehot,
                reduction='mean'
            )
            total_loss = total_loss + supervised_loss
            
        # 6.2 负载均衡损失
        load_balance_loss = self.compute_load_balance_loss(routing_weights)
        total_loss = total_loss + self.load_balance_weight * load_balance_loss
        
        # 6.3 熵正则化（鼓励决策的确定性）
        entropy = -torch.sum(routing_weights * torch.log(routing_weights + 1e-10), dim=-1)
        entropy_loss = entropy.mean() * 0.01  # 小权重
        total_loss = total_loss + entropy_loss
        
        # 7. 准备返回结果
        results = {
            'routing_weights': routing_weights,
            'hard_decisions': hard_decisions,
            'routing_loss': total_loss,
            'supervised_loss': supervised_loss,
            'load_balance_loss': load_balance_loss,
            'entropy_loss': entropy_loss
        }
        
        # 8. 可选：详细分析信息
        if return_analysis:
            with torch.no_grad():
                analysis = self._analyze_routing(routing_weights, hard_decisions, critical_labels)
                results['analysis'] = analysis
                
        return results
    
    def _analyze_routing(self, 
                        routing_weights: torch.Tensor,
                        hard_decisions: torch.Tensor,
                        critical_labels: Optional[torch.Tensor] = None) -> Dict:
        """
        分析路由决策的统计信息
        
        Returns:
            analysis: 包含各种统计信息的字典
        """
        analysis = {}
        
        # 专家使用率
        expert_usage = routing_weights.mean(dim=[0, 1])  # (num_experts,)
        analysis['expert_usage'] = expert_usage.cpu().numpy().tolist()
        
        # 决策确定性（熵的反向指标）
        entropy = -torch.sum(routing_weights * torch.log(routing_weights + 1e-10), dim=-1)
        analysis['avg_entropy'] = entropy.mean().item()
        analysis['decision_confidence'] = (1 - entropy / np.log(self.num_experts)).mean().item()
        
        # 时序模式分析
        # 检测切换频率
        switches = (hard_decisions[:, 1:] != hard_decisions[:, :-1]).float()
        analysis['switch_rate'] = switches.mean().item()
        
        # 如果有监督标签，计算准确率
        if critical_labels is not None:
            accuracy = (hard_decisions == critical_labels).float().mean()
            analysis['routing_accuracy'] = accuracy.item()
            
            # 每个专家的准确率
            for expert_id in range(self.num_experts):
                mask = critical_labels == expert_id
                if mask.any():
                    expert_acc = (hard_decisions[mask] == expert_id).float().mean()
                    analysis[f'expert_{expert_id}_accuracy'] = expert_acc.item()
                    
        return analysis
    
    def update_temperature(self, decay_rate: float = 0.99):
        """
        更新温度参数（退火策略）
        
        Args:
            decay_rate: 温度衰减率
        """
        with torch.no_grad():
            self.temperature.mul_(decay_rate)
            self.temperature.clamp_(min=self.temperature_min)
            
    def get_expert_features(self, action_tokens: torch.Tensor, expert_id: int) -> torch.Tensor:
        """
        获取特定专家的特征表示
        
        Args:
            action_tokens: (B, T, D) 动作tokens
            expert_id: 专家ID
            
        Returns:
            expert_features: (B, T, D) 专家特化的特征
        """
        return self.expert_projections[expert_id](action_tokens)


class AdaptiveRoutingModule(nn.Module):
    """
    自适应路由模块：集成路由网络和温度调度
    """
    
    def __init__(self, router_config: Dict):
        super().__init__()
        
        self.router = TemporalExpertRouter(**router_config)
        
        # 温度调度参数
        self.warmup_steps = router_config.get('warmup_steps', 1000)
        self.current_step = 0
        
    def forward(self, action_tokens: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """前向传播"""
        return self.router(action_tokens, **kwargs)
    
    def step(self):
        """训练步骤更新"""
        self.current_step += 1
        
        # 温度退火
        if self.current_step > self.warmup_steps:
            self.router.update_temperature(decay_rate=0.999)
            
    def get_routing_stats(self) -> Dict:
        """获取路由统计信息"""
        return {
            'temperature': self.router.temperature.item(),
            'current_step': self.current_step
        }


# 测试代码
if __name__ == "__main__":
    # 创建路由器
    router = TemporalExpertRouter(
        input_dim=2048,
        num_experts=2,
        hidden_dim=512,
        temperature_init=1.0,
        use_gumbel=True
    )
    
    # 测试输入
    B, T, D = 4, 64, 2048
    action_tokens = torch.randn(B, T, D)
    
    # 可选：监督标签（0表示使用全局专家，1表示使用深度专家）
    critical_labels = torch.randint(0, 2, (B, T))
    
    # 前向传播
    results = router(action_tokens, critical_labels, return_analysis=True)
    
    print("路由网络测试结果:")
    print(f"  输入形状: {action_tokens.shape}")
    print(f"  路由权重形状: {results['routing_weights'].shape}")
    print(f"  硬决策形状: {results['hard_decisions'].shape}")
    print(f"  总损失: {results['routing_loss'].item():.4f}")
    print(f"  监督损失: {results['supervised_loss'].item():.4f}")
    print(f"  负载均衡损失: {results['load_balance_loss'].item():.4f}")
    
    if 'analysis' in results:
        print("\n路由分析:")
        for key, value in results['analysis'].items():
            print(f"  {key}: {value}")
            
    # 测试温度更新
    print(f"\n温度更新前: {router.temperature.item():.4f}")
    router.update_temperature(0.95)
    print(f"温度更新后: {router.temperature.item():.4f}")