# import re
# from pathlib import Path

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
# from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

# from models.hub_mixin import CompatiblePyTorchModelHubMixin
# from models.rdt.model import RDT
# from models.rdt.binary_soft_routing import SimpleDualTeacherModel


# class RDTRunner(nn.Module, CompatiblePyTorchModelHubMixin, 
#                repo_url="https://huggingface.co/robotics-diffusion-transformer/rdt-1b"):
#     """
#     集成软路由双教师REPA对齐损失的RDT运行器
    
#     核心创新：
#     1. 基于关键时间段二元标签的软路由权重分配
#     2. 规则驱动的权重映射：关键时间段偏向深度，非关键时间段偏向全局
#     3. 可选的神经网络微调和时序平滑
#     4. 完整的对比学习和统计分析
#     """
#     def __init__(self, *, action_dim, pred_horizon, config, 
#                  lang_token_dim, img_token_dim, state_token_dim, 
#                  max_lang_cond_len, img_cond_len, lang_pos_embed_config=None, 
#                  img_pos_embed_config=None, dtype=torch.bfloat16,
#                  # 软路由双教师REPA参数
#                  enable_soft_routing_repa=True, 
#                  soft_routing_repa_weight=0.2,
#                  dinov2_feature_dim=1024,
#                  depth_feature_dim=1024,
#                  # 软路由配置
#                  soft_routing_config=None):
#         super(RDTRunner, self).__init__()
        
#         # 软路由双教师REPA配置
#         self.enable_soft_routing_repa = enable_soft_routing_repa
#         self.soft_routing_repa_weight = soft_routing_repa_weight
#         self.dtype = dtype
#         self.dinov2_feature_dim = dinov2_feature_dim
#         self.depth_feature_dim = depth_feature_dim
        
#         print(f"🔧 软路由双教师RDTRunner初始化:")
#         print(f"   - 软路由REPA损失启用: {enable_soft_routing_repa}")
#         print(f"   - 软路由REPA损失权重: {soft_routing_repa_weight}")
#         print(f"   - DINOv2特征维度: {dinov2_feature_dim}")
#         print(f"   - 深度特征维度: {depth_feature_dim}")
#         print(f"   - 数据类型: {dtype}")
        
#         # 创建扩散模型
#         hidden_size = config['rdt']['hidden_size']
#         self.model = RDT(
#             output_dim=action_dim,
#             horizon=pred_horizon,
#             hidden_size=hidden_size,
#             depth=config['rdt']['depth'], 
#             num_heads=config['rdt']['num_heads'],
#             max_lang_cond_len=max_lang_cond_len,
#             img_cond_len=img_cond_len,
#             lang_pos_embed_config=lang_pos_embed_config,
#             img_pos_embed_config=img_pos_embed_config,
#             dtype=dtype,
#             enable_repa_loss=enable_soft_routing_repa,
#             repa_activation_layer=21,  # 在第21层提取动作tokens
#             dinov2_feature_dim=dinov2_feature_dim,
#             depth_feature_dim=depth_feature_dim,
#         )

#         # 适配器创建
#         self.lang_adaptor = self.build_condition_adapter(
#             config['lang_adaptor'], 
#             in_features=lang_token_dim, 
#             out_features=hidden_size
#         )
#         self.img_adaptor = self.build_condition_adapter(
#             config['img_adaptor'], 
#             in_features=img_token_dim, 
#             out_features=hidden_size
#         )
#         self.state_adaptor = self.build_condition_adapter(
#             config['state_adaptor'], 
#             in_features=state_token_dim * 2,    
#             out_features=hidden_size
#         )
        
#         # 转换适配器到正确的数据类型
#         self.lang_adaptor = self.lang_adaptor.to(dtype)
#         self.img_adaptor = self.img_adaptor.to(dtype)
#         self.state_adaptor = self.state_adaptor.to(dtype)
        
#         # 🆕 创建软路由双教师对齐模型
#         if self.enable_soft_routing_repa:
#             # 设置默认软路由配置
#             default_soft_routing_config = {
#                 'action_dim': hidden_size,
#                 'dinov2_dim': dinov2_feature_dim,
#                 'depth_dim': depth_feature_dim,
#                 'router_config': {
#                     'action_dim': hidden_size,
#                     'critical_global_weight': 0.25,      # 关键时间段：全局25%，深度75%
#                     'critical_depth_weight': 0.75,
#                     'non_critical_global_weight': 0.75,  # 非关键时间段：全局75%，深度25%
#                     'non_critical_depth_weight': 0.25,
#                     'enable_neural_adjustment': True,    # 启用神经网络微调
#                     'adjustment_strength': 0.1,          # 微调强度
#                     'temporal_smoothing': 0.9,           # 时序平滑系数
#                     'temperature': 1.0,                  # softmax温度
#                 }
#             }
            
#             # 合并用户配置
#             if soft_routing_config:
#                 default_soft_routing_config.update(soft_routing_config)
#                 if 'router_config' in soft_routing_config:
#                     default_soft_routing_config['router_config'].update(soft_routing_config['router_config'])
            
#             self.soft_routing_config = default_soft_routing_config
#             self.dual_teacher_model = SimpleDualTeacherModel(**default_soft_routing_config)
#             self.dual_teacher_model.to(dtype)
            
#             print(f"🎯 软路由双教师对齐模型配置:")
#             router_config = default_soft_routing_config['router_config']
#             print(f"   - 关键时间段权重: 全局{router_config['critical_global_weight']:.2f}, 深度{router_config['critical_depth_weight']:.2f}")
#             print(f"   - 非关键时间段权重: 全局{router_config['non_critical_global_weight']:.2f}, 深度{router_config['non_critical_depth_weight']:.2f}")
#             print(f"   - 神经网络微调: {router_config['enable_neural_adjustment']}")
#             print(f"   - 时序平滑系数: {router_config['temporal_smoothing']}")
#             print(f"   - 微调强度: {router_config['adjustment_strength']}")
        
#         # 噪声调度器创建
#         noise_scheduler_config = config['noise_scheduler']
#         self.noise_scheduler = DDPMScheduler(
#             num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
#             beta_schedule=noise_scheduler_config['beta_schedule'],
#             prediction_type=noise_scheduler_config['prediction_type'],
#             clip_sample=noise_scheduler_config['clip_sample'],
#         )
#         self.noise_scheduler_sample = DPMSolverMultistepScheduler(
#             num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
#             beta_schedule=noise_scheduler_config['beta_schedule'],
#             prediction_type=noise_scheduler_config['prediction_type'],
#         )

#         self.num_train_timesteps = noise_scheduler_config['num_train_timesteps']
#         self.num_inference_timesteps = noise_scheduler_config['num_inference_timesteps']
#         self.prediction_type = noise_scheduler_config['prediction_type']

#         self.pred_horizon = pred_horizon
#         self.action_dim = action_dim
        
#         # 用于追踪训练状态
#         self.training_step = 0
#         self.batch_count = 0

#         print("Diffusion params: %e" % sum(
#             [p.numel() for p in self.model.parameters()] + 
#             [p.numel() for p in self.lang_adaptor.parameters()] + 
#             [p.numel() for p in self.img_adaptor.parameters()] + 
#             [p.numel() for p in self.state_adaptor.parameters()]))
        
#         if self.enable_soft_routing_repa:
#             dual_teacher_params = sum(p.numel() for p in self.dual_teacher_model.parameters())
#             print(f"Soft routing dual teacher params: {dual_teacher_params:e}")

#     def compute_soft_routing_dual_teacher_repa_loss(self, action_tokens, dinov2_cls_token, 
#                                                    depth_cls_token, critical_labels):
#         """
#         计算基于软路由的双教师REPA对齐损失
        
#         Args:
#             action_tokens: (B, T, hidden_size) 动作tokens
#             dinov2_cls_token: (B, 1, dinov2_dim) 或 (B, dinov2_dim) DINOv2全局特征
#             depth_cls_token: (B, depth_dim) 深度特征
#             critical_labels: (B, T) 关键时间段标签（0/1）
        
#         Returns:
#             total_loss: 总损失
#             detailed_metrics: 详细指标字典
#         """
#         B, T, hidden_size = action_tokens.shape
#         device = action_tokens.device
#         dtype = action_tokens.dtype
        
#         # 确保特征维度正确
#         if dinov2_cls_token.dim() == 3:
#             dinov2_cls_squeezed = dinov2_cls_token.squeeze(1)  # (B, dinov2_dim)
#         else:
#             dinov2_cls_squeezed = dinov2_cls_token
        
#         if depth_cls_token.dim() == 3:
#             depth_cls_squeezed = depth_cls_token.squeeze(1)  # (B, depth_dim)
#         else:
#             depth_cls_squeezed = depth_cls_token
        
#         # 确保关键时间段标签类型正确
#         if critical_labels.dtype != torch.long:
#             critical_labels = critical_labels.long()
        
#         # 判断是否为第一个batch（用于时序平滑）
#         is_first_batch = (self.batch_count == 0)
        
#         try:
#             # 使用软路由双教师模型计算对齐损失
#             results = self.dual_teacher_model(
#                 action_tokens=action_tokens,
#                 dinov2_features=dinov2_cls_squeezed,
#                 depth_features=depth_cls_squeezed,
#                 critical_labels=critical_labels,
#                 is_first_batch=is_first_batch
#             )
            
#             # 提取损失和统计信息
#             total_loss = results['total_loss']
            
#             # 构建详细指标
#             detailed_metrics = {
#                 # 主要损失组件
#                 'soft_routing_total_loss': total_loss.item(),
#                 'soft_routing_alignment_loss': results['alignment_loss'].item(),
#                 'soft_routing_contrastive_loss': results['contrastive_loss'].item(),
                
#                 # 原始损失（未加权）
#                 'global_loss_raw': results['global_loss_raw'].item(),
#                 'depth_loss_raw': results['depth_loss_raw'].item(),
                
#                 # 加权损失
#                 'weighted_global_loss': results['weighted_global_loss'].item(),
#                 'weighted_depth_loss': results['weighted_depth_loss'].item(),
                
#                 # 相似度指标
#                 'global_similarity_avg': results['global_similarity_avg'].item(),
#                 'depth_similarity_avg': results['depth_similarity_avg'].item(),
                
#                 # 温度参数
#                 'alignment_temperature': results['alignment_temperature'],
#             }
            
#             # 路由统计信息
#             router_stats = results['router_statistics']
#             detailed_metrics.update({
#                 'critical_ratio': router_stats['critical_ratio'],
#                 'avg_global_weight': router_stats['avg_global_weight'],
#                 'avg_depth_weight': router_stats['avg_depth_weight'],
#                 'weight_std_global': router_stats['weight_std_global'],
#                 'weight_std_depth': router_stats['weight_std_depth'],
#                 'routing_temperature': router_stats['temperature'],
#             })
            
#             # 分类统计
#             if 'critical_avg_global' in router_stats:
#                 detailed_metrics.update({
#                     'critical_avg_global_weight': router_stats['critical_avg_global'],
#                     'critical_avg_depth_weight': router_stats['critical_avg_depth'],
#                 })
            
#             if 'non_critical_avg_global' in router_stats:
#                 detailed_metrics.update({
#                     'non_critical_avg_global_weight': router_stats['non_critical_avg_global'],
#                     'non_critical_avg_depth_weight': router_stats['non_critical_avg_depth'],
#                 })
            
#             # 微调统计
#             if 'weight_drift' in router_stats:
#                 detailed_metrics['weight_drift'] = router_stats['weight_drift']
            
#             # 存储路由权重用于分析
#             detailed_metrics['routing_weights_tensor'] = results['routing_weights'].detach()
#             detailed_metrics['base_weights_tensor'] = results['base_weights'].detach()
            
#             return total_loss, detailed_metrics
            
#         except Exception as e:
#             print(f"⚠️ 软路由双教师REPA损失计算失败: {e}")
#             import traceback
#             traceback.print_exc()
            
#             # 返回零损失和基础指标
#             zero_loss = torch.tensor(0.0, device=device, dtype=dtype)
#             fallback_metrics = {
#                 'soft_routing_total_loss': 0.0,
#                 'soft_routing_alignment_loss': 0.0,
#                 'error': str(e),
#             }
#             return zero_loss, fallback_metrics

#     def compute_loss(self, lang_tokens, lang_attn_mask, img_tokens, 
#                      state_tokens, action_gt, action_mask, ctrl_freqs,
#                      cls_token=None, depth_features=None, critical_labels=None):
#         """
#         计算总损失，包括扩散损失和软路由双教师REPA损失
#         """
#         batch_size = lang_tokens.shape[0]
#         device = lang_tokens.device

#         # 确保所有输入都转换为正确的数据类型
#         lang_tokens = lang_tokens.to(self.dtype)
#         img_tokens = img_tokens.to(self.dtype)
#         state_tokens = state_tokens.to(self.dtype)
#         action_gt = action_gt.to(self.dtype)
#         action_mask = action_mask.to(self.dtype)
        
#         # 扩散损失计算
#         noise = torch.randn(action_gt.shape, dtype=action_gt.dtype, device=device)
#         timesteps = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device).long()
#         noisy_action = self.noise_scheduler.add_noise(action_gt, noise, timesteps)
        
#         state_action_traj = torch.cat([state_tokens, noisy_action], dim=1)
#         action_mask = action_mask.expand(-1, state_action_traj.shape[1], -1)
#         state_action_traj = torch.cat([state_action_traj, action_mask], dim=2)
        
#         lang_cond, img_cond, state_action_traj = self.adapt_conditions(
#             lang_tokens, img_tokens, state_action_traj)
        
#         # 获取模型预测和中间激活
#         pred, intermediate_activations = self.model(
#             state_action_traj, ctrl_freqs, timesteps, lang_cond, img_cond, 
#             lang_mask=lang_attn_mask
#         )

#         # 计算扩散损失
#         pred_type = self.prediction_type 
#         if pred_type == 'epsilon':
#             target = noise
#         elif pred_type == 'sample':
#             target = action_gt
#         else:
#             raise ValueError(f"Unsupported prediction type {pred_type}")

#         diffusion_loss = F.mse_loss(pred, target)
        
#         # 🆕 计算软路由双教师REPA对齐损失
#         repa_loss = torch.tensor(0.0, device=device, dtype=diffusion_loss.dtype)
#         detailed_metrics = {}
        
#         if (self.enable_soft_routing_repa and 
#             'action_tokens_for_repa' in intermediate_activations and
#             cls_token is not None and 
#             depth_features is not None and 
#             critical_labels is not None):
            
#             action_tokens = intermediate_activations['action_tokens_for_repa']
#             cls_token = cls_token.to(self.dtype)
#             depth_features = depth_features.to(self.dtype)
            
#             # 提取深度CLS token
#             if depth_features.shape[1] >= 1:
#                 depth_cls_token = depth_features[:, 0, :]  # (B, depth_dim)
#             else:
#                 depth_cls_token = cls_token.squeeze(1) if cls_token.dim() == 3 else cls_token
            
#             # 计算软路由双教师对齐损失
#             repa_loss, detailed_metrics = self.compute_soft_routing_dual_teacher_repa_loss(
#                 action_tokens=action_tokens,
#                 dinov2_cls_token=cls_token,
#                 depth_cls_token=depth_cls_token,
#                 critical_labels=critical_labels
#             )
        
#         # 总损失
#         total_loss = diffusion_loss + self.soft_routing_repa_weight * repa_loss
        
#         # 更新训练状态
#         self.training_step += 1
#         self.batch_count += 1
        
#         return total_loss, diffusion_loss, repa_loss, detailed_metrics

#     def conditional_sample(self, lang_cond, lang_attn_mask, img_cond, state_traj, action_mask, ctrl_freqs):
#         """推理时的条件采样，不涉及对齐机制"""
#         device = state_traj.device
#         dtype = state_traj.dtype
#         noisy_action = torch.randn(
#             size=(state_traj.shape[0], self.pred_horizon, self.action_dim), 
#             dtype=dtype, device=device)
#         action_mask = action_mask.expand(-1, self.pred_horizon, -1)
    
#         self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)
        
#         for t in self.noise_scheduler_sample.timesteps:
#             action_traj = torch.cat([noisy_action, action_mask], dim=2)
#             action_traj = self.state_adaptor(action_traj)
#             state_action_traj = torch.cat([state_traj, action_traj], dim=1)
            
#             model_output, _ = self.model(state_action_traj, ctrl_freqs,
#                                         t.unsqueeze(-1).to(device),
#                                         lang_cond, img_cond, lang_mask=lang_attn_mask)
            
#             noisy_action = self.noise_scheduler_sample.step(
#                 model_output, t, noisy_action).prev_sample
#             noisy_action = noisy_action.to(state_traj.dtype)
        
#         noisy_action = noisy_action * action_mask
#         return noisy_action

#     def build_condition_adapter(self, projector_type, in_features, out_features):
#         """构建条件适配器"""
#         projector = None
#         if projector_type == 'linear':
#             projector = nn.Linear(in_features, out_features)
#         else:
#             mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu', projector_type)
#             if mlp_gelu_match:
#                 mlp_depth = int(mlp_gelu_match.group(1))
#                 modules = [nn.Linear(in_features, out_features)]
#                 for _ in range(1, mlp_depth):
#                     modules.append(nn.GELU(approximate="tanh"))
#                     modules.append(nn.Linear(out_features, out_features))
#                 projector = nn.Sequential(*modules)
#         if projector is None:
#             raise ValueError(f'Unknown projector type: {projector_type}')
#         return projector

#     def adapt_conditions(self, lang_tokens, img_tokens, state_tokens):
#         """适配条件输入"""
#         adapted_lang = self.lang_adaptor(lang_tokens)
#         adapted_img = self.img_adaptor(img_tokens)
#         adapted_state = self.state_adaptor(state_tokens)
#         return adapted_lang, adapted_img, adapted_state

#     def predict_action(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens,
#                     action_mask, ctrl_freqs, vision_features=None):
#         """预测动作，推理时不需要视觉对齐特征"""
#         lang_tokens = lang_tokens.to(self.dtype)
#         img_tokens = img_tokens.to(self.dtype)
#         state_tokens = state_tokens.to(self.dtype)
#         action_mask = action_mask.to(self.dtype)
        
#         state_tokens = torch.cat([state_tokens, action_mask], dim=2)
#         lang_cond, img_cond, state_traj = self.adapt_conditions(
#             lang_tokens, img_tokens, state_tokens)
        
#         action_pred = self.conditional_sample(
#             lang_cond, lang_attn_mask, img_cond, 
#             state_traj, action_mask, ctrl_freqs,
#         )
        
#         return action_pred

#     def reset_batch_count(self):
#         """重置batch计数器（用于新epoch）"""
#         self.batch_count = 0

#     def get_soft_routing_statistics(self):
#         """获取软路由统计信息"""
#         if not self.enable_soft_routing_repa:
#             return {}
        
#         stats = {
#             'training_step': self.training_step,
#             'batch_count': self.batch_count,
#             'soft_routing_config': self.soft_routing_config,
#         }
        
#         # 如果有路由器，获取其统计信息
#         if hasattr(self.dual_teacher_model, 'soft_router'):
#             router = self.dual_teacher_model.soft_router
#             stats.update({
#                 'routing_temperature': router.temperature.item(),
#                 'enable_neural_adjustment': router.enable_neural_adjustment,
#                 'temporal_smoothing': router.temporal_smoothing,
#                 'adjustment_strength': router.adjustment_strength,
#             })
        
#         return stats

#     def forward(self, *args, **kwargs) -> torch.Tensor:
#         """保持兼容性，只返回总损失"""
#         result = self.compute_loss(*args, **kwargs)
#         if isinstance(result, tuple) and len(result) >= 1:
#             return result[0]  # 只返回total_loss
#         return result
    
    
    
#输入测融合版  
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from models.hub_mixin import CompatiblePyTorchModelHubMixin
from models.rdt.model import RDT
from models.rdt.binary_soft_routing import SimpleDualTeacherModel
from models.multimodal_encoder.vision_fusion_module import create_vision_fusion_module

class RDTRunner(nn.Module, CompatiblePyTorchModelHubMixin):
    """
    RDT运行器 - 集成视觉特征融合模块
    """
    def __init__(
        self, 
        *,
        action_dim, 
        pred_horizon, 
        config, 
        lang_token_dim, 
        img_token_dim, 
        state_token_dim, 
        max_lang_cond_len, 
        img_cond_len, 
        lang_pos_embed_config=None, 
        img_pos_embed_config=None, 
        dtype=torch.bfloat16,
        # 🆕 视觉融合配置
        enable_vision_fusion=True,
        vision_fusion_type="cross_attention",
        dinov2_feature_dim=1024,
        depth_feature_dim=1024,
        fusion_num_heads=8,
        fusion_dropout=0.1,
        img_history_size=2,  # 🆕 从config传入
        num_cameras=3,       # 🆕 从config传入
    ):
        super(RDTRunner, self).__init__()
        
        self.dtype = dtype
        self.enable_vision_fusion = enable_vision_fusion
        self.dinov2_feature_dim = dinov2_feature_dim
        self.depth_feature_dim = depth_feature_dim
        self.img_history_size = img_history_size
        self.num_cameras = num_cameras
        
        print(f"🔧 RDTRunner初始化 (视觉融合模式):")
        print(f"   - 视觉融合启用: {enable_vision_fusion}")
        print(f"   - 融合类型: {vision_fusion_type}")
        print(f"   - 图像历史: {img_history_size}帧")
        print(f"   - 相机数量: {num_cameras}个")
        print(f"   - SigLIP总tokens: {img_history_size * num_cameras * 729}")
        print(f"   - DINOv2特征维度: {dinov2_feature_dim}")
        print(f"   - 深度特征维度: {depth_feature_dim}")
        
        # 创建原始RDT模型（不含REPA）
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
            # 🔴 关闭REPA
            enable_repa_loss=False,
        )

        # 适配器
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
        
        self.lang_adaptor = self.lang_adaptor.to(dtype)
        self.img_adaptor = self.img_adaptor.to(dtype)
        self.state_adaptor = self.state_adaptor.to(dtype)
        
        # 🆕 创建视觉特征融合模块
        if self.enable_vision_fusion:
            self.vision_fusion_module = create_vision_fusion_module(
                fusion_type=vision_fusion_type,
                siglip_dim=img_token_dim,
                dinov2_dim=dinov2_feature_dim,
                depth_dim=depth_feature_dim,
                img_history_size=img_history_size,
                num_cameras=num_cameras,
                num_heads=fusion_num_heads,
                dropout=fusion_dropout
            )
            self.vision_fusion_module.to(dtype)
            
            fusion_params = sum(p.numel() for p in self.vision_fusion_module.parameters())
            print(f"   - 视觉融合模块参数量: {fusion_params:,}")
        else:
            self.vision_fusion_module = None

        # 噪声调度器
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
        
    def compute_soft_routing_dual_teacher_repa_loss(self, action_tokens, dinov2_cls_token, 
                                                   depth_cls_token, critical_labels):
        """
        计算基于软路由的双教师REPA对齐损失
        
        Args:
            action_tokens: (B, T, hidden_size) 动作tokens
            dinov2_cls_token: (B, 1, dinov2_dim) 或 (B, dinov2_dim) DINOv2全局特征
            depth_cls_token: (B, depth_dim) 深度特征
            critical_labels: (B, T) 关键时间段标签（0/1）
        
        Returns:
            total_loss: 总损失
            detailed_metrics: 详细指标字典
        """
        B, T, hidden_size = action_tokens.shape
        device = action_tokens.device
        dtype = action_tokens.dtype
        
        # 确保特征维度正确
        if dinov2_cls_token.dim() == 3:
            dinov2_cls_squeezed = dinov2_cls_token.squeeze(1)  # (B, dinov2_dim)
        else:
            dinov2_cls_squeezed = dinov2_cls_token
        
        if depth_cls_token.dim() == 3:
            depth_cls_squeezed = depth_cls_token.squeeze(1)  # (B, depth_dim)
        else:
            depth_cls_squeezed = depth_cls_token
        
        # 确保关键时间段标签类型正确
        if critical_labels.dtype != torch.long:
            critical_labels = critical_labels.long()
        
        # 判断是否为第一个batch（用于时序平滑）
        is_first_batch = (self.batch_count == 0)
        
        try:
            # 使用软路由双教师模型计算对齐损失
            results = self.dual_teacher_model(
                action_tokens=action_tokens,
                dinov2_features=dinov2_cls_squeezed,
                depth_features=depth_cls_squeezed,
                critical_labels=critical_labels,
                is_first_batch=is_first_batch
            )
            
            # 提取损失和统计信息
            total_loss = results['total_loss']
            
            # 构建详细指标
            detailed_metrics = {
                # 主要损失组件
                'soft_routing_total_loss': total_loss.item(),
                'soft_routing_alignment_loss': results['alignment_loss'].item(),
                'soft_routing_contrastive_loss': results['contrastive_loss'].item(),
                
                # 原始损失（未加权）
                'global_loss_raw': results['global_loss_raw'].item(),
                'depth_loss_raw': results['depth_loss_raw'].item(),
                
                # 加权损失
                'weighted_global_loss': results['weighted_global_loss'].item(),
                'weighted_depth_loss': results['weighted_depth_loss'].item(),
                
                # 相似度指标
                'global_similarity_avg': results['global_similarity_avg'].item(),
                'depth_similarity_avg': results['depth_similarity_avg'].item(),
                
                # 温度参数
                'alignment_temperature': results['alignment_temperature'],
            }
            
            # 路由统计信息
            router_stats = results['router_statistics']
            detailed_metrics.update({
                'critical_ratio': router_stats['critical_ratio'],
                'avg_global_weight': router_stats['avg_global_weight'],
                'avg_depth_weight': router_stats['avg_depth_weight'],
                'weight_std_global': router_stats['weight_std_global'],
                'weight_std_depth': router_stats['weight_std_depth'],
                'routing_temperature': router_stats['temperature'],
            })
            
            # 分类统计
            if 'critical_avg_global' in router_stats:
                detailed_metrics.update({
                    'critical_avg_global_weight': router_stats['critical_avg_global'],
                    'critical_avg_depth_weight': router_stats['critical_avg_depth'],
                })
            
            if 'non_critical_avg_global' in router_stats:
                detailed_metrics.update({
                    'non_critical_avg_global_weight': router_stats['non_critical_avg_global'],
                    'non_critical_avg_depth_weight': router_stats['non_critical_avg_depth'],
                })
            
            # 微调统计
            if 'weight_drift' in router_stats:
                detailed_metrics['weight_drift'] = router_stats['weight_drift']
            
            # 存储路由权重用于分析
            detailed_metrics['routing_weights_tensor'] = results['routing_weights'].detach()
            detailed_metrics['base_weights_tensor'] = results['base_weights'].detach()
            
            return total_loss, detailed_metrics
            
        except Exception as e:
            print(f"⚠️ 软路由双教师REPA损失计算失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 返回零损失和基础指标
            zero_loss = torch.tensor(0.0, device=device, dtype=dtype)
            fallback_metrics = {
                'soft_routing_total_loss': 0.0,
                'soft_routing_alignment_loss': 0.0,
                'error': str(e),
            }
            return zero_loss, fallback_metrics

    def compute_loss(
        self, 
        lang_tokens, 
        lang_attn_mask, 
        img_tokens,
        state_tokens, 
        action_gt, 
        action_mask, 
        ctrl_freqs,
        # 🆕 新增参数
        dinov2_features=None,
        depth_features=None,
    ):
        """计算损失（仅扩散损失）"""
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device

        # 确保数据类型正确
        lang_tokens = lang_tokens.to(self.dtype)
        img_tokens = img_tokens.to(self.dtype)
        state_tokens = state_tokens.to(self.dtype)
        action_gt = action_gt.to(self.dtype)
        action_mask = action_mask.to(self.dtype)
        
        # 🆕 视觉特征融合
        if self.enable_vision_fusion and dinov2_features is not None and depth_features is not None:
            dinov2_features = dinov2_features.to(self.dtype)
            depth_features = depth_features.to(self.dtype)
            
            # 去除CLS token，只保留patch tokens
            if dinov2_features.shape[1] == 1370:
                dinov2_patch_tokens = dinov2_features[:, 1:, :]
            else:
                dinov2_patch_tokens = dinov2_features
            
            if depth_features.shape[1] == 1370:
                depth_patch_tokens = depth_features[:, 1:, :]
            else:
                depth_patch_tokens = depth_features
            
            # 融合特征（融合到最后一帧，即当前观测）
            img_tokens = self.vision_fusion_module(
                siglip_tokens=img_tokens,
                dinov2_tokens=dinov2_patch_tokens,
                depth_tokens=depth_patch_tokens,
                current_frame_idx=-1  # 融合最后一帧
            )[0]
        
        # 扩散损失计算
        noise = torch.randn(action_gt.shape, dtype=action_gt.dtype, device=device)
        timesteps = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device).long()
        noisy_action = self.noise_scheduler.add_noise(action_gt, noise, timesteps)
        
        state_action_traj = torch.cat([state_tokens, noisy_action], dim=1)
        action_mask = action_mask.expand(-1, state_action_traj.shape[1], -1)
        state_action_traj = torch.cat([state_action_traj, action_mask], dim=2)
        lang_cond, img_cond, state_action_traj = self.adapt_conditions(
            lang_tokens, img_tokens, state_action_traj
        )
        
        # RDT前向传播
        pred, _ = self.model(
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
        
        return diffusion_loss

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

    def predict_action(
        self, 
        lang_tokens, 
        lang_attn_mask, 
        img_tokens,
        state_tokens,
        action_mask, 
        ctrl_freqs, 
        # 🆕 新增参数
        dinov2_features=None,
        depth_features=None
        ):
        """预测动作"""
        lang_tokens = lang_tokens.to(self.dtype)
        img_tokens = img_tokens.to(self.dtype)
        state_tokens = state_tokens.to(self.dtype)
        action_mask = action_mask.to(self.dtype)
        
        # 🆕 视觉特征融合（推理时）
        if self.enable_vision_fusion and dinov2_features is not None and depth_features is not None:
            dinov2_features = dinov2_features.to(self.dtype)
            depth_features = depth_features.to(self.dtype)
            
            # 去除CLS token
            if dinov2_features.shape[1] == 1370:
                dinov2_patch_tokens = dinov2_features[:, 1:, :]
            else:
                dinov2_patch_tokens = dinov2_features
            
            if depth_features.shape[1] == 1370:
                depth_patch_tokens = depth_features[:, 1:, :]
            else:
                depth_patch_tokens = depth_features
            
            # 融合
            img_tokens = self.vision_fusion_module(
                siglip_tokens=img_tokens,
                dinov2_tokens=dinov2_patch_tokens,
                depth_tokens=depth_patch_tokens,
                current_frame_idx=-1
            )[0]
        
        state_tokens = torch.cat([state_tokens, action_mask], dim=2)
        lang_cond, img_cond, state_traj = self.adapt_conditions(
            lang_tokens, img_tokens, state_tokens
        )
        
        action_pred = self.conditional_sample(
            lang_cond, lang_attn_mask, img_cond, 
            state_traj, action_mask, ctrl_freqs,
        )
        
        return action_pred

    def reset_batch_count(self):
        """重置batch计数器（用于新epoch）"""
        self.batch_count = 0

    def get_soft_routing_statistics(self):
        """获取软路由统计信息"""
        if not self.enable_soft_routing_repa:
            return {}
        
        stats = {
            'training_step': self.training_step,
            'batch_count': self.batch_count,
            'soft_routing_config': self.soft_routing_config,
        }
        
        # 如果有路由器，获取其统计信息
        if hasattr(self.dual_teacher_model, 'soft_router'):
            router = self.dual_teacher_model.soft_router
            stats.update({
                'routing_temperature': router.temperature.item(),
                'enable_neural_adjustment': router.enable_neural_adjustment,
                'temporal_smoothing': router.temporal_smoothing,
                'adjustment_strength': router.adjustment_strength,
            })
        
        return stats

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """保持兼容性，只返回总损失"""
        result = self.compute_loss(*args, **kwargs)
        if isinstance(result, tuple) and len(result) >= 1:
            return result[0]  # 只返回total_loss
        return result