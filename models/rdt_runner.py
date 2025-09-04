import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from models.hub_mixin import CompatiblePyTorchModelHubMixin
from models.rdt.model import RDT


class ConstrainedAdaptiveWeightLearner(nn.Module):
    """
    基于关键时间段约束的自适应权重学习器
    关键时间段：强制深度权重占主导，但具体比例可学习
    非关键时间段：强制全局权重占主导，但具体比例可学习
    """
    
    def __init__(self, 
                 action_dim: int = 2048,
                 hidden_dim: int = 512,
                 temperature_init: float = 1.0,
                 critical_depth_min: float = 0.6,      # 关键时间段深度权重最小值
                 critical_depth_max: float = 0.9,      # 关键时间段深度权重最大值
                 non_critical_global_min: float = 0.6, # 非关键时间段全局权重最小值
                 non_critical_global_max: float = 0.9): # 非关键时间段全局权重最大值
        super().__init__()
        
        # 权重预测网络：输出单个logit，用于在约束范围内调节权重
        self.weight_predictor = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),  # 只输出1个值用于权重调节
        )
        
        # 可学习参数
        self.temperature = nn.Parameter(torch.tensor(temperature_init))
        
        # 约束范围
        self.critical_depth_min = critical_depth_min
        self.critical_depth_max = critical_depth_max
        self.non_critical_global_min = non_critical_global_min
        self.non_critical_global_max = non_critical_global_max
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.weight_predictor:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)  # 小的初始化
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, action_tokens: torch.Tensor, routing_weights: torch.Tensor):
        """
        Args:
            action_tokens: (B, T, action_dim)
            routing_weights: (B, T, 2) 路由网络输出的软权重 [全局, 深度]
            
        Returns:
            dict: 包含约束后的自适应权重
        """
        B, T, D = action_tokens.shape
        device = action_tokens.device
        dtype = action_tokens.dtype  # 使用输入张量的数据类型
        
        # 1. 预测权重调节参数 - 修复内存布局问题
        flat_tokens = action_tokens.contiguous().view(B * T, D)  # 确保张量连续
        raw_adjustment = self.weight_predictor(flat_tokens)  # (B*T, 1)
        raw_adjustment = raw_adjustment.view(B, T)  # (B, T)
        
        # 2. 应用温度缩放的sigmoid，映射到 [0, 1]
        temp = torch.clamp(self.temperature, min=0.1, max=5.0)
        adjustment_factor = torch.sigmoid(raw_adjustment / temp)  # (B, T) ∈ [0, 1]
        
        # 3. 根据路由权重判断时间段类型
        # routing_weights[:,:,1] > routing_weights[:,:,0] 表示倾向于深度专家（关键时间段）
        depth_preference = routing_weights[:, :, 1] > routing_weights[:, :, 0]  # (B, T)
        
        # 4. 根据路由偏好应用不同的权重约束
        global_weights = torch.zeros_like(adjustment_factor, dtype=dtype, device=device)  # (B, T)
        depth_weights = torch.zeros_like(adjustment_factor, dtype=dtype, device=device)   # (B, T)
        
        # 深度偏好时间段：深度权重在约束范围内可学习
        if depth_preference.any():
            depth_pref_adjustment = adjustment_factor[depth_preference]
            
            # 将调节因子映射到深度权重范围 - 确保数据类型一致
            constrained_depth_weights = (
                torch.tensor(self.critical_depth_min, dtype=dtype, device=device) + 
                depth_pref_adjustment * (self.critical_depth_max - self.critical_depth_min)
            )
            constrained_global_weights = torch.tensor(1.0, dtype=dtype, device=device) - constrained_depth_weights
            
            depth_weights[depth_preference] = constrained_depth_weights
            global_weights[depth_preference] = constrained_global_weights
        
        # 全局偏好时间段：全局权重在约束范围内可学习
        global_preference = ~depth_preference
        if global_preference.any():
            global_pref_adjustment = adjustment_factor[global_preference]
            
            # 将调节因子映射到全局权重范围 - 确保数据类型一致
            constrained_global_weights = (
                torch.tensor(self.non_critical_global_min, dtype=dtype, device=device) + 
                global_pref_adjustment * (self.non_critical_global_max - self.non_critical_global_min)
            )
            constrained_depth_weights = torch.tensor(1.0, dtype=dtype, device=device) - constrained_global_weights
            
            global_weights[global_preference] = constrained_global_weights
            depth_weights[global_preference] = constrained_depth_weights
        
        # 5. 组装最终权重
        adaptive_weights = torch.stack([global_weights, depth_weights], dim=-1)  # (B, T, 2)
        
        # 6. 计算多样性损失：鼓励权重在约束范围内有所变化
        diversity_loss = self._compute_diversity_loss(adjustment_factor, depth_preference, dtype=dtype)
        
        return {
            'adaptive_weights': adaptive_weights,           # (B, T, 2)
            'adjustment_factor': adjustment_factor,         # (B, T)
            'diversity_loss': diversity_loss,              # 标量
            'temperature': temp.item(),                    # 当前温度值
            'depth_preference_ratio': depth_preference.float().mean().item(),
        }
    
    def _compute_diversity_loss(self, adjustment_factor: torch.Tensor, depth_preference: torch.Tensor, dtype=None):
        """计算多样性损失：鼓励权重在约束范围内的多样性"""
        if dtype is None:
            dtype = adjustment_factor.dtype
        device = adjustment_factor.device
        
        diversity_loss = torch.tensor(0.0, dtype=dtype, device=device)
        
        if depth_preference.any():
            depth_pref_adjustments = adjustment_factor[depth_preference]
            if len(depth_pref_adjustments) > 1:
                depth_var = torch.var(depth_pref_adjustments)
                target_var = torch.tensor(0.1, dtype=dtype, device=device)
                diversity_loss += F.mse_loss(depth_var, target_var)
        
        global_preference = ~depth_preference
        if global_preference.any():
            global_pref_adjustments = adjustment_factor[global_preference]
            if len(global_pref_adjustments) > 1:
                global_var = torch.var(global_pref_adjustments)
                target_var = torch.tensor(0.1, dtype=dtype, device=device)
                diversity_loss += F.mse_loss(global_var, target_var)
        
        return diversity_loss


class RDTRunner(nn.Module, CompatiblePyTorchModelHubMixin, 
               repo_url="https://huggingface.co/robotics-diffusion-transformer/rdt-1b"):
    """
    集成双教师REPA对齐损失和约束自适应权重学习的RDT运行器
    """
    def __init__(self, *, action_dim, pred_horizon, config, 
                 lang_token_dim, img_token_dim, state_token_dim, 
                 max_lang_cond_len, img_cond_len, lang_pos_embed_config=None, 
                 img_pos_embed_config=None, dtype=torch.bfloat16,
                 # REPA相关参数
                 enable_repa_loss=True, repa_loss_weight=0.2,
                 # 双教师相关参数
                 use_dual_teachers=False, routing_loss_weight=0.1,
                 # 🆕 约束自适应权重参数
                 enable_constrained_weights=True,
                 constrained_weight_config=None):
        super(RDTRunner, self).__init__()
        
        # REPA损失配置
        self.enable_repa_loss = enable_repa_loss
        self.repa_loss_weight = repa_loss_weight
        self.dtype = dtype
        
        # 双教师配置
        self.use_dual_teachers = use_dual_teachers
        self.routing_loss_weight = routing_loss_weight
        
        # 🆕 约束自适应权重配置
        self.enable_constrained_weights = enable_constrained_weights
        
        print(f"🔧 RDTRunner初始化:")
        print(f"   - REPA损失启用: {enable_repa_loss}")
        print(f"   - REPA损失权重: {repa_loss_weight}")
        print(f"   - 双教师模式: {use_dual_teachers}")
        print(f"   - 路由损失权重: {routing_loss_weight}")
        print(f"   - 约束自适应权重: {enable_constrained_weights}")
        print(f"   - 数据类型: {dtype}")
        
        # 创建扩散模型
        hidden_size = config['rdt']['hidden_size']
        self.model = RDT(
            output_dim=action_dim,
            horizon=pred_horizon,
            hidden_size=hidden_size,
            depth=config['rdt']['depth'], 
            num_heads=config['rdt']['num_heads'],
            max_lang_cond_len=max_lang_cond_len,
            img_cond_len=img_cond_len,
            lang_pos_embed_config=lang_pos_embed_config,
            img_pos_embed_config=img_pos_embed_config,
            dtype=dtype,
            enable_repa_loss=enable_repa_loss,
            use_dual_teachers=use_dual_teachers,
            dinov2_feature_dim=1024,
            depth_feature_dim=1024,
        )

        # 适配器创建
        self.lang_adaptor = self.build_condition_adapter(
            config['lang_adaptor'], 
            in_features=lang_token_dim, 
            out_features=hidden_size
        )
        self.img_adaptor = self.build_condition_adapter(
            config['img_adaptor'], 
            in_features=img_token_dim, 
            out_features=hidden_size
        )
        self.state_adaptor = self.build_condition_adapter(
            config['state_adaptor'], 
            in_features=state_token_dim * 2,    
            out_features=hidden_size
        )
        
        # 转换适配器到正确的数据类型
        self.lang_adaptor = self.lang_adaptor.to(dtype)
        self.img_adaptor = self.img_adaptor.to(dtype)
        self.state_adaptor = self.state_adaptor.to(dtype)
        
        # 🆕 约束权重学习器初始化
        if self.enable_constrained_weights and enable_repa_loss:
            default_config = {
                'action_dim': hidden_size,
                'hidden_dim': 512,
                'temperature_init': 1.0,
                'critical_depth_min': 0.6,
                'critical_depth_max': 0.9,
                'non_critical_global_min': 0.6,
                'non_critical_global_max': 0.9,
            }
            if constrained_weight_config:
                default_config.update(constrained_weight_config)
            
            self.constrained_weight_config = default_config
            self.constrained_weight_learner = None  # 延迟初始化
            
            print(f"🎯 约束权重学习器配置: {default_config}")
        
        # 噪声调度器创建
        noise_scheduler_config = config['noise_scheduler']
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            beta_schedule=noise_scheduler_config['beta_schedule'],
            prediction_type=noise_scheduler_config['prediction_type'],
            clip_sample=noise_scheduler_config['clip_sample'],
        )
        self.noise_scheduler_sample = DPMSolverMultistepScheduler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            beta_schedule=noise_scheduler_config['beta_schedule'],
            prediction_type=noise_scheduler_config['prediction_type'],
        )

        self.num_train_timesteps = noise_scheduler_config['num_train_timesteps']
        self.num_inference_timesteps = noise_scheduler_config['num_inference_timesteps']
        self.prediction_type = noise_scheduler_config['prediction_type']

        self.pred_horizon = pred_horizon
        self.action_dim = action_dim

        print("Diffusion params: %e" % sum(
            [p.numel() for p in self.model.parameters()] + 
            [p.numel() for p in self.lang_adaptor.parameters()] + 
            [p.numel() for p in self.img_adaptor.parameters()] + 
            [p.numel() for p in self.state_adaptor.parameters()]))

    def compute_constrained_dual_teacher_alignment_loss(self, action_tokens, dinov2_cls_token, 
                                                  depth_cls_token, routing_weights, critical_labels=None):
        """
        基于路由网络和约束权重学习的双教师对齐损失
        
        Args:
            action_tokens: (B, T, hidden_size) 动作tokens
            dinov2_cls_token: (B, 1, 1024) DINOv2全局特征
            depth_cls_token: (B, 1024) 深度特征
            routing_weights: (B, T, 2) 路由网络输出权重
            critical_labels: (B, T) 关键时间段标签（可选，用于分析）
        """
        B, T, hidden_size = action_tokens.shape
        device = action_tokens.device
        dtype = action_tokens.dtype  # 获取正确的数据类型
        
        # 1. 延迟初始化约束权重学习器，确保数据类型一致
        if self.constrained_weight_learner is None:
            self.constrained_weight_learner = ConstrainedAdaptiveWeightLearner(
                **self.constrained_weight_config
            ).to(device, dtype=dtype)  # 确保权重学习器使用正确的数据类型
            print(f"约束权重学习器已初始化，数据类型: {dtype}")
        
        # 2. 投影视觉特征到动作空间
        dinov2_cls_squeezed = dinov2_cls_token.squeeze(1) if dinov2_cls_token.dim() == 3 else dinov2_cls_token
        depth_cls_squeezed = depth_cls_token.squeeze(1) if depth_cls_token.dim() == 3 else depth_cls_token
        
        projected_dinov2 = self.model.dinov2_to_action_projector(dinov2_cls_squeezed)  # (B, 2048)
        projected_depth = self.model.depth_to_action_projector(depth_cls_squeezed)      # (B, 2048)
        
        # 扩展到时间维度
        projected_dinov2_expanded = projected_dinov2.unsqueeze(1).expand(-1, T, -1)  # (B, T, 2048)
        projected_depth_expanded = projected_depth.unsqueeze(1).expand(-1, T, -1)    # (B, T, 2048)
        
        # 3. 计算基础对齐损失
        action_norm = F.normalize(action_tokens, p=2, dim=-1)
        dinov2_norm = F.normalize(projected_dinov2_expanded, p=2, dim=-1)
        depth_norm = F.normalize(projected_depth_expanded, p=2, dim=-1)
        
        # 余弦相似度
        global_similarity = torch.sum(action_norm * dinov2_norm, dim=-1)  # (B, T)
        depth_similarity = torch.sum(action_norm * depth_norm, dim=-1)    # (B, T)
        
        # 转换为损失
        global_loss = torch.tensor(1.0, dtype=dtype, device=device) - global_similarity  # (B, T)
        depth_loss = torch.tensor(1.0, dtype=dtype, device=device) - depth_similarity    # (B, T)
        
        # 4. 获取约束后的自适应权重
        weight_results = self.constrained_weight_learner(action_tokens, routing_weights)
        adaptive_weights = weight_results['adaptive_weights']  # (B, T, 2)
        diversity_loss = weight_results['diversity_loss']
        
        # 5. 应用约束权重加权损失
        weighted_global_loss = adaptive_weights[:, :, 0] * global_loss  # (B, T)
        weighted_depth_loss = adaptive_weights[:, :, 1] * depth_loss    # (B, T)
        
        # 主对齐损失
        alignment_loss = (weighted_global_loss + weighted_depth_loss).mean()
        
        # 6. 时序平滑性正则化
        smoothness_loss = torch.tensor(0.0, device=device, dtype=dtype)
        if T > 1:
            weight_diff = torch.diff(adaptive_weights, dim=1)  # (B, T-1, 2)
            smoothness_loss = torch.mean(weight_diff ** 2)
        
        # 7. 总损失组合
        total_loss = (
            alignment_loss + 
            torch.tensor(0.05, dtype=dtype, device=device) * diversity_loss +      # 鼓励权重多样性
            torch.tensor(0.02, dtype=dtype, device=device) * smoothness_loss       # 时序平滑性
        )
        
        # 8. 构建详细指标
        metrics_dict = {
            # 主要损失组件
            'alignment_loss': alignment_loss.item(),
            'diversity_loss': diversity_loss.item(),
            'smoothness_loss': smoothness_loss.item(),
            'total_constrained_loss': total_loss.item(),
            
            # 基础损失（未加权）
            'raw_global_loss': global_loss.mean().item(),
            'raw_depth_loss': depth_loss.mean().item(),
            
            # 权重学习器状态
            'weight_temperature': weight_results['temperature'],
            'depth_preference_ratio': weight_results['depth_preference_ratio'],
            
            # 权重统计
            'constrained_avg_global_weight': adaptive_weights[:, :, 0].mean().item(),
            'constrained_avg_depth_weight': adaptive_weights[:, :, 1].mean().item(),
            'constrained_global_weight_std': adaptive_weights[:, :, 0].std().item(),
            'constrained_depth_weight_std': adaptive_weights[:, :, 1].std().item(),
            
            # 路由权重统计（用于对比）
            'routing_avg_global_weight': routing_weights[:, :, 0].mean().item(),
            'routing_avg_depth_weight': routing_weights[:, :, 1].mean().item(),
            
            # 存储权重用于后续分析
            'adaptive_weights_tensor': adaptive_weights.detach(),
            'routing_weights_tensor': routing_weights.detach(),
        }
        
        # 9. 如果有关键时间段标签，进行更详细的分析
        if critical_labels is not None:
            print(f"🔍 开始分类统计分析")
            critical_mask = critical_labels.bool()
            
            print(f"🔍 critical_mask shape: {critical_mask.shape}")
            print(f"🔍 critical_mask sum: {critical_mask.sum().item()}")
            print(f"🔍 adaptive_weights shape: {adaptive_weights.shape}")
            
            if critical_mask.any():
                critical_adaptive_weights = adaptive_weights[critical_mask]
                critical_routing_weights = routing_weights[critical_mask]
                
                print(f"🔍 critical_adaptive_weights shape: {critical_adaptive_weights.shape}")
                print(f"🔍 关键时间段权重 - 全局: {critical_adaptive_weights[:, 0].mean().item():.3f}")
                print(f"🔍 关键时间段权重 - 深度: {critical_adaptive_weights[:, 1].mean().item():.3f}")
                
                metrics_dict.update({
                    'critical_constrained_global': critical_adaptive_weights[:, 0].mean().item(),
                    'critical_constrained_depth': critical_adaptive_weights[:, 1].mean().item(),
                    'critical_routing_global': critical_routing_weights[:, 0].mean().item(),
                    'critical_routing_depth': critical_routing_weights[:, 1].mean().item(),
                })
                print(f"🔍 已添加关键时间段指标到metrics_dict")
            else:
                print(f"⚠️ 没有关键时间段")
            
            non_critical_mask = ~critical_mask
            print(f"🔍 non_critical_mask sum: {non_critical_mask.sum().item()}")
            
            if non_critical_mask.any():
                non_critical_adaptive_weights = adaptive_weights[non_critical_mask]
                non_critical_routing_weights = routing_weights[non_critical_mask]
                
                print(f"🔍 非关键时间段权重 - 全局: {non_critical_adaptive_weights[:, 0].mean().item():.3f}")
                print(f"🔍 非关键时间段权重 - 深度: {non_critical_adaptive_weights[:, 1].mean().item():.3f}")
                
                metrics_dict.update({
                    'non_critical_constrained_global': non_critical_adaptive_weights[:, 0].mean().item(),
                    'non_critical_constrained_depth': non_critical_adaptive_weights[:, 1].mean().item(),
                    'non_critical_routing_global': non_critical_routing_weights[:, 0].mean().item(),
                    'non_critical_routing_depth': non_critical_routing_weights[:, 1].mean().item(),
                })
                print(f"🔍 已添加非关键时间段指标到metrics_dict")
                
                # 计算路由网络与关键时间段标签的一致性
                critical_routing_correct = (critical_routing_weights[:, 1] > critical_routing_weights[:, 0]).float().mean()
                non_critical_routing_correct = (non_critical_routing_weights[:, 0] > non_critical_routing_weights[:, 1]).float().mean()
                
                metrics_dict.update({
                    'routing_critical_accuracy': critical_routing_correct.item(),
                    'routing_non_critical_accuracy': non_critical_routing_correct.item(),
                    'routing_overall_accuracy': (critical_routing_correct + non_critical_routing_correct).item() / 2,
                })
                print(f"🔍 已添加路由准确率指标")
            else:
                print(f"⚠️ 没有非关键时间段")
            
            print(f"🔍 metrics_dict keys: {list(metrics_dict.keys())}")
                
        return total_loss, metrics_dict

    def compute_loss(self, lang_tokens, lang_attn_mask, img_tokens, 
                     state_tokens, action_gt, action_mask, ctrl_freqs,
                     cls_token=None, depth_features=None, critical_labels=None):
        """
        计算总损失，包括扩散损失、路由损失和约束权重对齐损失
        """
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device

        # 确保所有输入都转换为正确的数据类型
        lang_tokens = lang_tokens.to(self.dtype)
        img_tokens = img_tokens.to(self.dtype)
        state_tokens = state_tokens.to(self.dtype)
        action_gt = action_gt.to(self.dtype)
        action_mask = action_mask.to(self.dtype)
        
        # 扩散损失计算
        noise = torch.randn(action_gt.shape, dtype=action_gt.dtype, device=device)
        timesteps = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device).long()
        noisy_action = self.noise_scheduler.add_noise(action_gt, noise, timesteps)
        
        state_action_traj = torch.cat([state_tokens, noisy_action], dim=1)
        action_mask = action_mask.expand(-1, state_action_traj.shape[1], -1)
        state_action_traj = torch.cat([state_action_traj, action_mask], dim=2)
        
        lang_cond, img_cond, state_action_traj = self.adapt_conditions(
            lang_tokens, img_tokens, state_action_traj)
        
        # 获取模型预测和中间激活
        pred, intermediate_activations = self.model(
            state_action_traj, ctrl_freqs, timesteps, lang_cond, img_cond, 
            lang_mask=lang_attn_mask
        )

        # 计算扩散损失
        pred_type = self.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = action_gt
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        diffusion_loss = F.mse_loss(pred, target)
        
        # 🆕 计算约束权重双教师REPA对齐损失
        repa_loss = torch.tensor(0.0, device=device, dtype=diffusion_loss.dtype)
        routing_loss = torch.tensor(0.0, device=device, dtype=diffusion_loss.dtype)
        detailed_metrics = {}
        
        if self.enable_repa_loss and 'action_tokens_for_repa' in intermediate_activations:
            action_tokens = intermediate_activations['action_tokens_for_repa']
            
            if self.use_dual_teachers and cls_token is not None and depth_features is not None:
                cls_token = cls_token.to(self.dtype)
                depth_features = depth_features.to(self.dtype)
                
                # 提取深度CLS token
                if depth_features.shape[1] >= 1:
                    depth_cls_token = depth_features[:, 0, :]  # (B, 1024)
                else:
                    depth_cls_token = cls_token.squeeze(1)  # Fallback
                
                # 获取路由权重
                routing_weights = intermediate_activations.get('routing_weights', None)
                
                if routing_weights is not None:
                    if self.enable_constrained_weights:
                        # 🆕 使用约束权重模式
                        repa_loss, detailed_metrics = self.compute_constrained_dual_teacher_alignment_loss(
                            action_tokens, 
                            cls_token,       
                            depth_cls_token, 
                            routing_weights,
                            critical_labels
                        )
                    else:
                        # 原始双教师模式
                        repa_loss, routing_loss, loss_dict = self.compute_dual_teacher_alignment_loss(
                            action_tokens, 
                            cls_token,       
                            depth_cls_token, 
                            routing_weights,
                            critical_labels
                        )
                        detailed_metrics = loss_dict
                    
                    # 路由损失（监督路由网络学习）
                    if critical_labels is not None:
                        target_routing = torch.zeros_like(routing_weights)
                        target_routing[:, :, 0] = 1 - critical_labels.float()  # 全局专家权重
                        target_routing[:, :, 1] = critical_labels.float()      # 深度专家权重
                        
                        routing_loss = F.binary_cross_entropy(routing_weights, target_routing, reduction='mean')
                        detailed_metrics['routing_supervision_loss'] = routing_loss.item()
        
        # 总损失
        total_loss = (
            diffusion_loss + 
            self.repa_loss_weight * repa_loss + 
            self.routing_loss_weight * routing_loss
        )
        
        return total_loss, diffusion_loss, repa_loss, routing_loss, detailed_metrics

    # 其他方法保持不变...
    def conditional_sample(self, lang_cond, lang_attn_mask, img_cond, state_traj, action_mask, ctrl_freqs):
        """推理时的条件采样，不涉及对齐机制"""
        device = state_traj.device
        dtype = state_traj.dtype
        noisy_action = torch.randn(
            size=(state_traj.shape[0], self.pred_horizon, self.action_dim), 
            dtype=dtype, device=device)
        action_mask = action_mask.expand(-1, self.pred_horizon, -1)
    
        self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)
        
        for t in self.noise_scheduler_sample.timesteps:
            action_traj = torch.cat([noisy_action, action_mask], dim=2)
            action_traj = self.state_adaptor(action_traj)
            state_action_traj = torch.cat([state_traj, action_traj], dim=1)
            
            model_output, _ = self.model(state_action_traj, ctrl_freqs,
                                        t.unsqueeze(-1).to(device),
                                        lang_cond, img_cond, lang_mask=lang_attn_mask)
            
            noisy_action = self.noise_scheduler_sample.step(
                model_output, t, noisy_action).prev_sample
            noisy_action = noisy_action.to(state_traj.dtype)
        
        noisy_action = noisy_action * action_mask
        return noisy_action

    def build_condition_adapter(self, projector_type, in_features, out_features):
        """构建条件适配器"""
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)
        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')
        return projector

    def adapt_conditions(self, lang_tokens, img_tokens, state_tokens):
        """适配条件输入"""
        adapted_lang = self.lang_adaptor(lang_tokens)
        adapted_img = self.img_adaptor(img_tokens)
        adapted_state = self.state_adaptor(state_tokens)
        return adapted_lang, adapted_img, adapted_state

    def predict_action(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens,
                    action_mask, ctrl_freqs, vision_features=None):
        """预测动作，推理时不需要视觉对齐特征"""
        lang_tokens = lang_tokens.to(self.dtype)
        img_tokens = img_tokens.to(self.dtype)
        state_tokens = state_tokens.to(self.dtype)
        action_mask = action_mask.to(self.dtype)
        
        state_tokens = torch.cat([state_tokens, action_mask], dim=2)
        lang_cond, img_cond, state_traj = self.adapt_conditions(
            lang_tokens, img_tokens, state_tokens)
        
        action_pred = self.conditional_sample(
            lang_cond, lang_attn_mask, img_cond, 
            state_traj, action_mask, ctrl_freqs,
        )
        
        return action_pred

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """保持兼容性，只返回总损失"""
        result = self.compute_loss(*args, **kwargs)
        if isinstance(result, tuple) and len(result) >= 1:
            return result[0]  # 只返回total_loss
        return result