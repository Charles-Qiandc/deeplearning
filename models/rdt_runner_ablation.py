import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from models.hub_mixin import CompatiblePyTorchModelHubMixin
from models.rdt.model_ablation import RDTAblation


class RDTRunnerAblation(nn.Module, CompatiblePyTorchModelHubMixin, 
                       repo_url="https://huggingface.co/robotics-diffusion-transformer/rdt-1b"):
    """
    消融实验版RDT运行器：移除REPA对齐，将多模态视觉特征注入DIT输入侧
    
    核心改动：
    1. 移除所有REPA对齐相关组件
    2. 将DINOv2和DepthAnything特征在输入侧与SigLIP特征融合
    3. 保持原有的扩散训练流程
    """
    
    def __init__(self, *, action_dim, pred_horizon, config, 
                 lang_token_dim, img_token_dim, state_token_dim, 
                 max_lang_cond_len, img_cond_len, lang_pos_embed_config=None, 
                 img_pos_embed_config=None, dtype=torch.bfloat16,
                 # 🆕 多模态视觉特征参数
                 use_dinov2_features=False,
                 dinov2_feature_dim=1024,
                 use_depth_features=False,
                 depth_feature_dim=1024,
                 visual_fusion_strategy="concat"):
        super(RDTRunnerAblation, self).__init__()
        
        self.dtype = dtype
        self.use_dinov2_features = use_dinov2_features
        self.use_depth_features = use_depth_features
        self.dinov2_feature_dim = dinov2_feature_dim
        self.depth_feature_dim = depth_feature_dim
        self.visual_fusion_strategy = visual_fusion_strategy
        
        print(f"🔧 消融实验RDTRunner初始化:")
        print(f"   - 移除REPA对齐机制")
        print(f"   - DINOv2特征注入: {use_dinov2_features}")
        print(f"   - 深度特征注入: {use_depth_features}")
        print(f"   - 视觉融合策略: {visual_fusion_strategy}")
        print(f"   - 数据类型: {dtype}")
        
        # 创建扩散模型（消融版本）
        hidden_size = config['rdt']['hidden_size']
        self.model = RDTAblation(
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
            # 🆕 多模态视觉特征配置
            use_dinov2_features=use_dinov2_features,
            dinov2_feature_dim=dinov2_feature_dim,
            use_depth_features=use_depth_features,
            depth_feature_dim=depth_feature_dim,
            visual_fusion_strategy=visual_fusion_strategy,
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

    def compute_loss(self, lang_tokens, lang_attn_mask, img_tokens, 
                     state_tokens, action_gt, action_mask, ctrl_freqs,
                     dinov2_features=None, depth_features=None):
        """
        计算扩散损失（消融版本，移除REPA损失）
        
        Args:
            dinov2_features: (B, 1, dinov2_dim) DINOv2 CLS token特征
            depth_features: (B, 1, depth_dim) DepthAnything CLS token特征
        """
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device

        # 确保所有输入都转换为正确的数据类型
        lang_tokens = lang_tokens.to(self.dtype)
        img_tokens = img_tokens.to(self.dtype)
        state_tokens = state_tokens.to(self.dtype)
        action_gt = action_gt.to(self.dtype)
        action_mask = action_mask.to(self.dtype)
        
        # 处理额外的视觉特征
        if dinov2_features is not None:
            dinov2_features = dinov2_features.to(self.dtype)
        if depth_features is not None:
            depth_features = depth_features.to(self.dtype)
        
        # 扩散损失计算
        noise = torch.randn(action_gt.shape, dtype=action_gt.dtype, device=device)
        timesteps = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device).long()
        noisy_action = self.noise_scheduler.add_noise(action_gt, noise, timesteps)
        
        state_action_traj = torch.cat([state_tokens, noisy_action], dim=1)
        action_mask = action_mask.expand(-1, state_action_traj.shape[1], -1)
        state_action_traj = torch.cat([state_action_traj, action_mask], dim=2)
        
        lang_cond, img_cond, state_action_traj = self.adapt_conditions(
            lang_tokens, img_tokens, state_action_traj)
        
        # 🆕 获取模型预测（传入额外的视觉特征）
        pred = self.model(
            state_action_traj, ctrl_freqs, timesteps, lang_cond, img_cond, 
            lang_mask=lang_attn_mask,
            dinov2_features=dinov2_features,
            depth_features=depth_features
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
        
        # 🔄 消融版本：只返回扩散损失
        total_loss = diffusion_loss
        
        # 简化的metrics
        metrics = {
            'diffusion_loss': diffusion_loss.item(),
            'total_loss': total_loss.item(),
            'ablation_mode': True,  # 标识消融模式
        }
        
        if self.use_dinov2_features:
            metrics['dinov2_features_used'] = True
        if self.use_depth_features:
            metrics['depth_features_used'] = True
            
        return total_loss, diffusion_loss, torch.tensor(0.0, device=device), metrics

    def conditional_sample(self, lang_cond, lang_attn_mask, img_cond, state_traj, action_mask, ctrl_freqs,
                          dinov2_features=None, depth_features=None):
        """推理时的条件采样（消融版本）"""
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
            
            # 🆕 传入额外的视觉特征
            model_output = self.model(
                state_action_traj, ctrl_freqs,
                t.unsqueeze(-1).to(device),
                lang_cond, img_cond, lang_mask=lang_attn_mask,
                dinov2_features=dinov2_features,
                depth_features=depth_features
            )
            
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
                    action_mask, ctrl_freqs, dinov2_features=None, depth_features=None):
        """预测动作（消融版本）"""
        lang_tokens = lang_tokens.to(self.dtype)
        img_tokens = img_tokens.to(self.dtype)
        state_tokens = state_tokens.to(self.dtype)
        action_mask = action_mask.to(self.dtype)
        
        # 处理额外的视觉特征
        if dinov2_features is not None:
            dinov2_features = dinov2_features.to(self.dtype)
        if depth_features is not None:
            depth_features = depth_features.to(self.dtype)
        
        state_tokens = torch.cat([state_tokens, action_mask], dim=2)
        lang_cond, img_cond, state_traj = self.adapt_conditions(
            lang_tokens, img_tokens, state_tokens)
        
        action_pred = self.conditional_sample(
            lang_cond, lang_attn_mask, img_cond, 
            state_traj, action_mask, ctrl_freqs,
            dinov2_features=dinov2_features,
            depth_features=depth_features
        )
        
        return action_pred

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """保持兼容性，只返回总损失"""
        result = self.compute_loss(*args, **kwargs)
        if isinstance(result, tuple) and len(result) >= 1:
            return result[0]  # 只返回total_loss
        return result