import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from models.hub_mixin import CompatiblePyTorchModelHubMixin
from models.rdt.model import RDT


class RDTRunner(nn.Module, CompatiblePyTorchModelHubMixin, 
               repo_url="https://huggingface.co/robotics-diffusion-transformer/rdt-1b"):
    """
    集成双教师REPA对齐损失的RDT运行器
    每个视觉教师有独立的投影器，将视觉特征投影到动作空间
    """
    def __init__(self, *, action_dim, pred_horizon, config, 
                 lang_token_dim, img_token_dim, state_token_dim, 
                 max_lang_cond_len, img_cond_len, lang_pos_embed_config=None, 
                 img_pos_embed_config=None, dtype=torch.bfloat16,
                 # REPA相关参数
                 enable_repa_loss=True, repa_loss_weight=0.2,
                 # 双教师相关参数
                 use_dual_teachers=False, routing_loss_weight=0.1):
        super(RDTRunner, self).__init__()
        
        # REPA损失配置
        self.enable_repa_loss = enable_repa_loss
        self.repa_loss_weight = repa_loss_weight
        self.dtype = dtype
        
        # 双教师配置
        self.use_dual_teachers = use_dual_teachers
        self.routing_loss_weight = routing_loss_weight
        
        print(f"🔧 RDTRunner初始化:")
        print(f"   - REPA损失启用: {enable_repa_loss}")
        print(f"   - REPA损失权重: {repa_loss_weight}")
        print(f"   - 双教师模式: {use_dual_teachers}")
        print(f"   - 路由损失权重: {routing_loss_weight}")
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
            dinov2_feature_dim=1024,  # DINOv2特征维度
            depth_feature_dim=1024,   # DepthAnythingV2特征维度
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

    def compute_dual_teacher_alignment_loss(self, action_tokens, dinov2_cls_token, depth_cls_token, 
                                           routing_weights=None, critical_labels=None):
        """
        🔄 修改：使用独立投影器的双教师对齐
        将视觉特征投影到动作空间，而不是相反
        
        Args:
            action_tokens: (B, T, 2048) 动作tokens
            dinov2_cls_token: (B, 1, 1024) DINOv2 CLS token
            depth_cls_token: (B, 1024) Depth CLS token 
            routing_weights: (B, T, 2) 路由权重 [全局专家权重, 深度专家权重]
            critical_labels: (B, T) 关键时间段标签，仅用于监督路由网络
            
        Returns:
            alignment_loss: 对齐损失
            routing_loss: 路由监督损失
        """
        B, T, hidden_size = action_tokens.shape
        device = action_tokens.device
        dtype = action_tokens.dtype
        
        # 🔄 关键修改：将视觉特征投影到动作空间
        # 1. 投影DINOv2全局特征到动作空间
        dinov2_cls_squeezed = dinov2_cls_token.squeeze(1)  # (B, 1024)
        projected_dinov2 = self.model.dinov2_to_action_projector(dinov2_cls_squeezed)  # (B, 2048)
        
        # 2. 投影Depth特征到动作空间
        if depth_cls_token.dim() == 3:
            depth_cls_squeezed = depth_cls_token.squeeze(1)  # (B, 1024)
        else:
            depth_cls_squeezed = depth_cls_token  # Already (B, 1024)
        projected_depth = self.model.depth_to_action_projector(depth_cls_squeezed)  # (B, 2048)
        
        # 3. 扩展投影后的视觉特征到时间维度
        projected_dinov2_expanded = projected_dinov2.unsqueeze(1).expand(-1, T, -1)  # (B, T, 2048)
        projected_depth_expanded = projected_depth.unsqueeze(1).expand(-1, T, -1)  # (B, T, 2048)
        
        # 4. 归一化用于余弦相似度计算
        action_tokens_norm = F.normalize(action_tokens, p=2, dim=-1)  # (B, T, 2048)
        dinov2_norm = F.normalize(projected_dinov2_expanded, p=2, dim=-1)  # (B, T, 2048)
        depth_norm = F.normalize(projected_depth_expanded, p=2, dim=-1)  # (B, T, 2048)
        
        # 5. 计算余弦相似度
        global_similarity = torch.sum(action_tokens_norm * dinov2_norm, dim=-1)  # (B, T)
        depth_similarity = torch.sum(action_tokens_norm * depth_norm, dim=-1)  # (B, T)
        
        # 6. 转换为损失（1 - similarity）
        global_losses = 1.0 - global_similarity  # (B, T)
        depth_losses = 1.0 - depth_similarity  # (B, T)
        
        # 7. 使用路由权重进行软组合
        if routing_weights is not None:
            # routing_weights: (B, T, 2) - [全局权重, 深度权重]
            weighted_losses = (routing_weights[:, :, 0] * global_losses + 
                             routing_weights[:, :, 1] * depth_losses)  # (B, T)
            alignment_loss = weighted_losses.mean()
            
            # 额外记录：每个专家的平均损失（用于监控）
            global_expert_loss = (routing_weights[:, :, 0] * global_losses).sum() / (routing_weights[:, :, 0].sum() + 1e-6)
            depth_expert_loss = (routing_weights[:, :, 1] * depth_losses).sum() / (routing_weights[:, :, 1].sum() + 1e-6)
        else:
            # 如果没有路由权重，使用均等权重
            alignment_loss = 0.5 * (global_losses.mean() + depth_losses.mean())
            global_expert_loss = global_losses.mean()
            depth_expert_loss = depth_losses.mean()
        
        # 8. 计算路由监督损失（如果有标签）
        routing_loss = torch.tensor(0.0, device=device, dtype=dtype)
        if routing_weights is not None and critical_labels is not None:
            # 将标签转换为目标路由权重
            target_routing = torch.zeros_like(routing_weights)
            target_routing[:, :, 0] = 1 - critical_labels.float()  # 全局专家权重
            target_routing[:, :, 1] = critical_labels.float()      # 深度专家权重
            
            # 交叉熵损失
            routing_loss = F.binary_cross_entropy(routing_weights, target_routing, reduction='mean')
        
        # 返回详细的损失信息
        loss_dict = {
            'alignment_loss': alignment_loss,
            'routing_loss': routing_loss,
            'global_expert_loss': global_expert_loss.detach(),
            'depth_expert_loss': depth_expert_loss.detach(),
        }
        
        return alignment_loss, routing_loss, loss_dict

    def compute_loss(self, lang_tokens, lang_attn_mask, img_tokens, 
                     state_tokens, action_gt, action_mask, ctrl_freqs,
                     cls_token=None, depth_features=None, critical_labels=None):
        """
        计算总损失，包括扩散损失和双教师对齐损失
        
        Returns:
            tuple: (total_loss, diffusion_loss, repa_loss, routing_loss)
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
        
        # 计算双教师REPA对齐损失
        repa_loss = torch.tensor(0.0, device=device, dtype=diffusion_loss.dtype)
        routing_loss = torch.tensor(0.0, device=device, dtype=diffusion_loss.dtype)
        
        if self.enable_repa_loss and 'action_tokens_for_repa' in intermediate_activations:
            action_tokens = intermediate_activations['action_tokens_for_repa']
            
            if self.use_dual_teachers:
                # 双教师模式：使用独立投影器
                if cls_token is not None and depth_features is not None:
                    cls_token = cls_token.to(self.dtype)
                    depth_features = depth_features.to(self.dtype)
                    
                    # 提取深度CLS token
                    if depth_features.shape[1] >= 1:
                        depth_cls_token = depth_features[:, 0, :]  # (B, 1024)
                    else:
                        depth_cls_token = cls_token.squeeze(1)  # Fallback
                    
                    # 获取路由权重
                    routing_weights = intermediate_activations.get('routing_weights', None)
                    
                    # 计算双教师对齐损失
                    repa_loss, routing_loss, loss_dict = self.compute_dual_teacher_alignment_loss(
                        action_tokens, 
                        cls_token,       # DINOv2 CLS token
                        depth_cls_token, # Depth CLS token
                        routing_weights,
                        critical_labels
                    )
            else:
                # 单教师模式（向后兼容，但也改为投影视觉到动作空间）
                if cls_token is not None:
                    cls_token = cls_token.to(self.dtype)
                    # 简化版：直接计算平均动作token与投影后视觉特征的对齐
                    action_mean = action_tokens.mean(dim=1)  # (B, 2048)
                    cls_squeezed = cls_token.squeeze(1)  # (B, 1024)
                    
                    # 使用DINOv2投影器
                    projected_cls = self.model.dinov2_to_action_projector(cls_squeezed)  # (B, 2048)
                    
                    # 归一化
                    action_norm = F.normalize(action_mean, p=2, dim=-1)
                    cls_norm = F.normalize(projected_cls, p=2, dim=-1)
                    
                    # 余弦相似度
                    similarity = F.cosine_similarity(action_norm, cls_norm, dim=-1)
                    repa_loss = 1.0 - similarity.mean()
        
        # 总损失
        total_loss = (diffusion_loss + 
                     self.repa_loss_weight * repa_loss + 
                     self.routing_loss_weight * routing_loss)
        
        return total_loss, diffusion_loss, repa_loss, routing_loss

    # 推理相关方法保持不变
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
            
            # 推理时不返回中间激活，只要预测结果
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
       total_loss, _, _, _ = self.compute_loss(*args, **kwargs)
        return total_loss