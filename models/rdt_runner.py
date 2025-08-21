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
    🔄 集成双教师REPA对齐损失的RDT运行器
    支持DINOv2（全局语义）和DepthAnythingV2（深度几何）的选择性对齐
    """
    def __init__(self, *, action_dim, pred_horizon, config, 
                 lang_token_dim, img_token_dim, state_token_dim, 
                 max_lang_cond_len, img_cond_len, lang_pos_embed_config=None, 
                 img_pos_embed_config=None, dtype=torch.bfloat16,
                 # REPA相关参数（现有）
                 enable_repa_loss=True, repa_loss_weight=0.2,
                 # 🆕 双教师相关参数
                 use_dual_teachers=False, routing_loss_weight=0.1):
        super(RDTRunner, self).__init__()
        
        # REPA损失配置（现有）
        self.enable_repa_loss = enable_repa_loss
        self.repa_loss_weight = repa_loss_weight
        self.dtype = dtype
        
        # 🆕 双教师配置
        self.use_dual_teachers = use_dual_teachers
        self.routing_loss_weight = routing_loss_weight
        
        print(f"🔧 RDTRunner初始化:")
        print(f"   - REPA损失启用: {enable_repa_loss}")
        print(f"   - REPA损失权重: {repa_loss_weight}")
        print(f"   - 双教师模式: {use_dual_teachers}")
        print(f"   - 路由损失权重: {routing_loss_weight}")
        print(f"   - 数据类型: {dtype}")
        
        # 创建扩散模型（修改以支持双教师）
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
            use_dual_teachers=use_dual_teachers,  # 🆕
        )

        # 现有的适配器创建保持不变
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
        
        # 转换适配器到正确的数据类型（现有）
        self.lang_adaptor = self.lang_adaptor.to(dtype)
        self.img_adaptor = self.img_adaptor.to(dtype)
        self.state_adaptor = self.state_adaptor.to(dtype)
        
        # 现有的噪声调度器创建保持不变
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

    def compute_global_alignment_loss(self, action_tokens, cls_token):
        """
        计算与DINOv2全局特征的对齐损失
        
        Args:
            action_tokens: (B, T, hidden_size) 动作tokens
            cls_token: (B, 1, dinov2_dim) DINOv2 CLS token
            
        Returns:
            loss: 标量对齐损失
        """
        B, T, hidden_size = action_tokens.shape
        
        # 时间平均得到整体动作表示
        action_mean = action_tokens.mean(dim=1)  # (B, hidden_size)
        
        # 投影到视觉特征空间
        projected_action = self.model.action_to_vision_projector(action_mean)  # (B, dinov2_dim)
        
        # 处理视觉特征
        cls_token_squeezed = cls_token.squeeze(1)  # (B, dinov2_dim)
        
        # L2归一化
        projected_action = F.normalize(projected_action, p=2, dim=-1)
        cls_token_norm = F.normalize(cls_token_squeezed, p=2, dim=-1)
        
        # 计算余弦相似度
        cosine_similarity = F.cosine_similarity(projected_action, cls_token_norm, dim=-1)
        mean_similarity = cosine_similarity.mean()
        
        # 转换为损失（最大化相似度 = 最小化损失）
        loss = 1.0 - mean_similarity
        
        return loss

    def compute_selective_alignment_loss(self, action_tokens, cls_token, depth_features, critical_labels):
        """
        🆕 根据关键时间段选择性计算对齐损失
        
        Args:
            action_tokens: (B, T, hidden_size) 动作tokens
            cls_token: (B, 1, dinov2_dim) DINOv2 CLS token (全局语义特征)
            depth_features: (B, N_patches, depth_dim) 深度特征，其中第0个是CLS token
            critical_labels: (B, T) 关键时间段标签 (0=非关键, 1=关键)
            
        Returns:
            loss: 标量对齐损失
        """
        if critical_labels is None:
            # 如果没有标签，回退到全局对齐
            return self.compute_global_alignment_loss(action_tokens, cls_token)
        
        B, T, hidden_size = action_tokens.shape
        total_loss = torch.tensor(0.0, device=action_tokens.device, dtype=action_tokens.dtype)
        
        # 投影所有动作tokens到视觉特征空间
        projected_actions = self.model.action_to_vision_projector(
            action_tokens.reshape(B * T, hidden_size)
        ).reshape(B, T, -1)  # (B, T, 1024)
        
        # 🆕 提取深度CLS token
        if depth_features.shape[1] == 1370:  # 包含CLS token
            depth_cls_token = depth_features[:, 0, :]  # (B, depth_dim) - 第0个token是CLS
        else:
            # 如果没有CLS token，报错或回退
            raise ValueError("深度特征中没有找到CLS token，请检查DepthAnythingV2的输出格式")
        
        # 准备目标特征
        # 全局语义特征：DINOv2 CLS token
        global_cls_token = cls_token.squeeze(1)  # (B, dinov2_dim)
        global_targets = global_cls_token.unsqueeze(1).expand(-1, T, -1)  # (B, T, dinov2_dim)
        
        # 深度几何特征：DepthAnythingV2 CLS token
        depth_targets = depth_cls_token.unsqueeze(1).expand(-1, T, -1)  # (B, T, depth_dim)
        
        # 分别处理关键和非关键时间段
        critical_mask = critical_labels.bool()  # (B, T)
        non_critical_mask = ~critical_mask      # (B, T)
        
        valid_loss_computed = False
        
        # 处理非关键时间段：动作token与DINOv2 CLS token对齐
        if non_critical_mask.any():
            # 获取非关键时间段的tokens和目标
            non_critical_actions = projected_actions[non_critical_mask]  # (N_non_critical, 1024)
            non_critical_targets = global_targets[non_critical_mask]     # (N_non_critical, 1024)
            
            if non_critical_actions.numel() > 0:
                # L2归一化
                non_critical_actions_norm = F.normalize(non_critical_actions, p=2, dim=-1)
                non_critical_targets_norm = F.normalize(non_critical_targets, p=2, dim=-1)
                
                # 余弦相似度
                non_critical_similarity = F.cosine_similarity(
                    non_critical_actions_norm, non_critical_targets_norm, dim=-1
                )
                non_critical_loss = 1.0 - non_critical_similarity.mean()
                total_loss = total_loss + non_critical_loss
                valid_loss_computed = True
        
        # 处理关键时间段：动作token与DepthAnythingV2 CLS token对齐
        if critical_mask.any():
            # 获取关键时间段的tokens和目标
            critical_actions = projected_actions[critical_mask]  # (N_critical, 1024)
            critical_targets = depth_targets[critical_mask]      # (N_critical, 1024)
            
            if critical_actions.numel() > 0:
                # L2归一化
                critical_actions_norm = F.normalize(critical_actions, p=2, dim=-1)
                critical_targets_norm = F.normalize(critical_targets, p=2, dim=-1)
                
                # 余弦相似度
                critical_similarity = F.cosine_similarity(
                    critical_actions_norm, critical_targets_norm, dim=-1
                )
                critical_loss = 1.0 - critical_similarity.mean()
                total_loss = total_loss + critical_loss
                valid_loss_computed = True
        
        # 如果没有计算任何有效损失，返回零损失
        if not valid_loss_computed:
            total_loss = torch.tensor(0.0, device=action_tokens.device, dtype=action_tokens.dtype)
        
        return total_loss

    def compute_routing_loss(self, routing_weights, critical_labels):
        """
        🆕 计算路由监督损失
        
        Args:
            routing_weights: (B, T, 2) 路由权重 [全局专家, 深度专家]
            critical_labels: (B, T) 关键时间段标签 (0=非关键, 1=关键)
            
        Returns:
            loss: 标量路由损失
        """
        if critical_labels is None:
            return torch.tensor(0.0, device=routing_weights.device, dtype=routing_weights.dtype)
        
        # 将关键标签转换为one-hot格式
        # 0 -> [1, 0] (使用全局专家)
        # 1 -> [0, 1] (使用深度专家)
        target_weights = torch.zeros_like(routing_weights)
        target_weights[:, :, 0] = 1 - critical_labels.float()  # 全局专家权重
        target_weights[:, :, 1] = critical_labels.float()      # 深度专家权重
        
        # 交叉熵损失
        loss = F.binary_cross_entropy(routing_weights, target_weights, reduction='mean')
        
        return loss

    def compute_loss(self, lang_tokens, lang_attn_mask, img_tokens, 
                     state_tokens, action_gt, action_mask, ctrl_freqs,
                     cls_token=None, depth_features=None, critical_labels=None):
        """
        🔄 修改：计算总损失，包括扩散损失、REPA对齐损失和路由损失
        
        Returns:
            tuple: (total_loss, diffusion_loss, repa_loss, routing_loss)
        """
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device

        # 确保所有输入都转换为正确的数据类型（现有代码保持不变）
        lang_tokens = lang_tokens.to(self.dtype)
        img_tokens = img_tokens.to(self.dtype)
        state_tokens = state_tokens.to(self.dtype)
        action_gt = action_gt.to(self.dtype)
        action_mask = action_mask.to(self.dtype)
        
        # 原有的扩散损失计算逻辑（现有代码保持不变）
        noise = torch.randn(action_gt.shape, dtype=action_gt.dtype, device=device)
        timesteps = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device).long()
        noisy_action = self.noise_scheduler.add_noise(action_gt, noise, timesteps)
        
        state_action_traj = torch.cat([state_tokens, noisy_action], dim=1)
        action_mask = action_mask.expand(-1, state_action_traj.shape[1], -1)
        state_action_traj = torch.cat([state_action_traj, action_mask], dim=2)
        
        lang_cond, img_cond, state_action_traj = self.adapt_conditions(
            lang_tokens, img_tokens, state_action_traj)
        
        # 获取模型预测和中间激活（现有代码保持不变）
        pred, intermediate_activations = self.model(
            state_action_traj, ctrl_freqs, timesteps, lang_cond, img_cond, 
            lang_mask=lang_attn_mask
        )

        # 计算扩散损失（现有代码保持不变）
        pred_type = self.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = action_gt
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        diffusion_loss = F.mse_loss(pred, target)
        
        # 🔄 修改：计算双教师REPA对齐损失和路由损失
        repa_loss = torch.tensor(0.0, device=device, dtype=diffusion_loss.dtype)
        routing_loss = torch.tensor(0.0, device=device, dtype=diffusion_loss.dtype)
        
        if self.enable_repa_loss and 'action_tokens_for_repa' in intermediate_activations:
            action_tokens = intermediate_activations['action_tokens_for_repa']
            
            if self.use_dual_teachers:
                # 🆕 双教师模式：根据关键时间段选择CLS token对齐
                if cls_token is not None and depth_features is not None:
                    cls_token = cls_token.to(self.dtype)
                    depth_features = depth_features.to(self.dtype)
                    
                    # 使用深度CLS token的选择性对齐
                    repa_loss = self.compute_selective_alignment_loss(
                        action_tokens, cls_token, depth_features, critical_labels
                    )
                
                # 计算路由监督损失（如果有路由权重）
                if 'routing_weights' in intermediate_activations and critical_labels is not None:
                    routing_weights = intermediate_activations['routing_weights']
                    routing_loss = self.compute_routing_loss(routing_weights, critical_labels)
                    
            else:
                # 保持原有的单教师模式
                if cls_token is not None:
                    cls_token = cls_token.to(self.dtype)
                    repa_loss = self.compute_global_alignment_loss(action_tokens, cls_token)
        
        # 总损失
        total_loss = (diffusion_loss + 
                     self.repa_loss_weight * repa_loss + 
                     self.routing_loss_weight * routing_loss)
        
        return total_loss, diffusion_loss, repa_loss, routing_loss

    # 现有的其他方法保持不变
    def conditional_sample(self, lang_cond, lang_attn_mask, img_cond, 
                           state_traj, action_mask, ctrl_freqs):
        """现有代码保持不变"""
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
        """现有代码保持不变"""
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu, projector_type)
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
        """现有代码保持不变"""
        adapted_lang = self.lang_adaptor(lang_tokens)
        adapted_img = self.img_adaptor(img_tokens)
        adapted_state = self.state_adaptor(state_tokens)
        return adapted_lang, adapted_img, adapted_state

    def predict_action(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens,
                       action_mask, ctrl_freqs, vision_features=None):
        """现有代码保持不变"""
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
        """🔄 修改：保持兼容性，只返回总损失"""
        total_loss, _, _, _ = self.compute_loss(*args, **kwargs)
        return total_loss