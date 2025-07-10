import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import \
    DPMSolverMultistepScheduler

from models.hub_mixin import CompatiblePyTorchModelHubMixin
from models.rdt.model import RDT


class RDTRunner(
        nn.Module, 
        CompatiblePyTorchModelHubMixin, 
        repo_url="https://huggingface.co/robotics-diffusion-transformer/rdt-1b"
    ):
    """
    🔄 修改：集成REPA对齐损失的RDT运行器
    """
    def __init__(self, *, action_dim, pred_horizon, config, 
                 lang_token_dim, img_token_dim, state_token_dim, 
                 max_lang_cond_len, img_cond_len, lang_pos_embed_config=None, 
                 img_pos_embed_config=None, dtype=torch.bfloat16,
                 # 🆕 REPA相关参数
                 enable_repa_loss=True, repa_loss_weight=0.2):
        super(RDTRunner, self).__init__()
        
        # 🆕 REPA损失配置
        self.enable_repa_loss = enable_repa_loss
        self.repa_loss_weight = repa_loss_weight
        self.dtype = dtype  # 保存数据类型
        
        print(f"🔧 RDTRunner初始化:")
        print(f"   - REPA损失启用: {enable_repa_loss}")
        print(f"   - REPA损失权重: {repa_loss_weight}")
        print(f"   - 数据类型: {dtype}")
        
        # 创建扩散模型
        hidden_size = config['rdt']['hidden_size']
        self.model = RDT(
            output_dim=action_dim,
            horizon=pred_horizon,
            hidden_size=hidden_size,
            depth=8,  # 🔄 修改：固定8层
            num_heads=config['rdt']['num_heads'],
            max_lang_cond_len=max_lang_cond_len,
            img_cond_len=img_cond_len,
            lang_pos_embed_config=lang_pos_embed_config,
            img_pos_embed_config=img_pos_embed_config,
            dtype=dtype,
            enable_repa_loss=enable_repa_loss,  # 🆕
        )

        # 创建各种条件输入的适配器
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
        # state包含状态和状态掩码
        self.state_adaptor = self.build_condition_adapter(
            config['state_adaptor'], 
            in_features=state_token_dim * 2,    
            out_features=hidden_size
        )
        
        # 🆕 将适配器转换为正确的数据类型
        self.lang_adaptor = self.lang_adaptor.to(dtype)
        self.img_adaptor = self.img_adaptor.to(dtype)
        self.state_adaptor = self.state_adaptor.to(dtype)
        
        # 创建噪声调度器
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

    def compute_repa_loss(self, action_tokens, vision_features):
        """
        🆕 核心方法：计算REPA风格的对齐损失
        
        Args:
            action_tokens: (B, horizon, hidden_size) 第4层的动作token
            vision_features: (B, N_patches, dinov2_dim) DINOv2视觉特征
        
        Returns:
            repa_loss: 标量损失值
        """
        if not self.enable_repa_loss or vision_features is None:
            return torch.tensor(0.0, device=action_tokens.device, dtype=action_tokens.dtype)
        
        B, horizon, hidden_size = action_tokens.shape
        B_v, N_patches, dinov2_dim = vision_features.shape
        
        # 验证输入维度
        assert B == B_v, f"批次大小不匹配: {B} vs {B_v}"
        
        print(f"🔍 REPA损失计算:")
        print(f"   - 动作token: {action_tokens.shape}")
        print(f"   - 视觉特征: {vision_features.shape}")
        
        # 确保视觉特征是正确的数据类型
        vision_features = vision_features.to(action_tokens.dtype)
        
        # Step 1: 投影动作token到视觉特征空间
        action_tokens_flat = action_tokens.reshape(-1, hidden_size)  # (B*horizon, hidden_size)
        projected_actions = self.model.action_to_vision_projector(action_tokens_flat)  # (B*horizon, dinov2_dim)
        projected_actions = projected_actions.reshape(B, horizon, dinov2_dim)  # (B, horizon, dinov2_dim)
        
        print(f"   - 投影后动作: {projected_actions.shape}")
        
        # Step 2: L2归一化特征
        projected_actions = F.normalize(projected_actions, dim=-1)  # (B, horizon, dinov2_dim)
        vision_features = F.normalize(vision_features, dim=-1)      # (B, N_patches, dinov2_dim)
        
        # Step 3: 计算对齐损失 (全局对齐策略)
        total_loss = 0.0
        similarities = []
        
        for b in range(B):
            # 每个batch单独计算余弦相似度
            action_feat = projected_actions[b]  # (horizon, dinov2_dim)
            vision_feat = vision_features[b]    # (N_patches, dinov2_dim)
            
            # 计算相似度矩阵: (horizon, N_patches)
            similarity_matrix = torch.mm(action_feat, vision_feat.t())
            
            # 对每个动作token，找到最相似的视觉patch
            max_similarity, best_patch_idx = similarity_matrix.max(dim=1)  # (horizon,)
            similarities.append(max_similarity.mean().item())
            
            # 负相似度作为损失 (鼓励高相似度)
            batch_loss = -max_similarity.mean()
            total_loss += batch_loss
        
        repa_loss = total_loss / B
        
        print(f"   - 平均相似度: {sum(similarities)/len(similarities):.4f}")
        print(f"   - REPA损失: {repa_loss.item():.4f}")
        
        return repa_loss

    def compute_loss(self, lang_tokens, lang_attn_mask, img_tokens, 
                     state_tokens, action_gt, action_mask, ctrl_freqs,
                     vision_features=None  # 🆕 DINOv2视觉特征
                    ):
        """
        🔄 修改：计算总损失，包括扩散损失和REPA对齐损失
        
        Args:
            vision_features: (B, N_patches, dinov2_dim) DINOv2提取的视觉特征
            
        Returns:
            tuple: (total_loss, diffusion_loss, repa_loss) 详细损失信息
        """
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device  

        # 确保所有输入都转换为正确的数据类型
        lang_tokens = lang_tokens.to(self.dtype)
        img_tokens = img_tokens.to(self.dtype)
        state_tokens = state_tokens.to(self.dtype)
        action_gt = action_gt.to(self.dtype)
        action_mask = action_mask.to(self.dtype)
        
        # 原有的扩散损失计算逻辑
        # 采样噪声
        noise = torch.randn(
            action_gt.shape, dtype=action_gt.dtype, device=device
        )
        # 采样随机扩散时间步
        timesteps = torch.randint(
            0, self.num_train_timesteps, 
            (batch_size,), device=device
        ).long()
        # 根据噪声大小在每个时间步添加噪声到干净动作
        noisy_action = self.noise_scheduler.add_noise(
            action_gt, noise, timesteps)
        
        # 拼接状态和动作token形成输入序列
        state_action_traj = torch.cat([state_tokens, noisy_action], dim=1)
        # 将动作掩码添加到输入序列
        action_mask = action_mask.expand(-1, state_action_traj.shape[1], -1)
        state_action_traj = torch.cat([state_action_traj, action_mask], dim=2)
        
        # 将维度对齐到隐藏大小
        lang_cond, img_cond, state_action_traj = self.adapt_conditions(
            lang_tokens, img_tokens, state_action_traj)
        
        # 🔄 修改：获取模型预测和中间激活
        pred, intermediate_activations = self.model(
            state_action_traj, ctrl_freqs, 
            timesteps, lang_cond, img_cond, 
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
        
        # 🆕 计算REPA对齐损失
        repa_loss = torch.tensor(0.0, device=device, dtype=diffusion_loss.dtype)
        if self.enable_repa_loss and 'action_tokens_for_repa' in intermediate_activations:
            action_tokens = intermediate_activations['action_tokens_for_repa']
            if vision_features is not None:
                vision_features = vision_features.to(self.dtype)
            repa_loss = self.compute_repa_loss(action_tokens, vision_features)
        
        # 总损失 = 扩散损失 + 加权对齐损失
        total_loss = diffusion_loss + self.repa_loss_weight * repa_loss
        
        print(f"💰 损失详情:")
        print(f"   - 扩散损失: {diffusion_loss.item():.4f}")
        print(f"   - REPA损失: {repa_loss.item():.4f}")
        print(f"   - 总损失: {total_loss.item():.4f}")
        
        return total_loss, diffusion_loss, repa_loss

    def conditional_sample(self, lang_cond, lang_attn_mask, img_cond, 
                           state_traj, action_mask, ctrl_freqs):
        """
        🔄 修改：条件采样（推理时使用）
        注意：推理时暂不使用REPA损失，只在训练时使用
        """
        device = state_traj.device
        dtype = state_traj.dtype
        noisy_action = torch.randn(
            size=(state_traj.shape[0], self.pred_horizon, self.action_dim), 
            dtype=dtype, device=device)
        action_mask = action_mask.expand(-1, self.pred_horizon, -1)
    
        # 设置采样时间步
        self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)
        
        for t in self.noise_scheduler_sample.timesteps:
            # 准备状态-动作轨迹
            action_traj = torch.cat([noisy_action, action_mask], dim=2)
            action_traj = self.state_adaptor(action_traj)
            state_action_traj = torch.cat([state_traj, action_traj], dim=1)
            
            # 🔄 修改：模型预测，忽略中间激活
            model_output, _ = self.model(state_action_traj, ctrl_freqs,
                                        t.unsqueeze(-1).to(device),
                                        lang_cond, img_cond, lang_mask=lang_attn_mask)
            
            # 计算前一步动作: x_t -> x_t-1
            noisy_action = self.noise_scheduler_sample.step(
                model_output, t, noisy_action).prev_sample
            noisy_action = noisy_action.to(state_traj.dtype)
        
        # 最后应用动作掩码
        noisy_action = noisy_action * action_mask

        return noisy_action
    
    def build_condition_adapter(
        self, projector_type, in_features, out_features):
        """构建条件适配器"""
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            # 修复：闭合字符串并修正正则表达式
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
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
        """适配条件输入的维度"""
        adapted_lang = self.lang_adaptor(lang_tokens)
        adapted_img = self.img_adaptor(img_tokens)
        adapted_state = self.state_adaptor(state_tokens)

        return adapted_lang, adapted_img, adapted_state

    def predict_action(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens,
                       action_mask, ctrl_freqs, vision_features=None):
        """
        🔄 修改：预测动作（推理接口）
        
        Args:
            vision_features: 推理时可选的视觉特征（暂不使用）
        """
        # 确保输入数据类型正确
        lang_tokens = lang_tokens.to(self.dtype)
        img_tokens = img_tokens.to(self.dtype)
        state_tokens = state_tokens.to(self.dtype)
        action_mask = action_mask.to(self.dtype)
        
        # 准备状态和条件
        state_tokens = torch.cat([state_tokens, action_mask], dim=2)
        lang_cond, img_cond, state_traj = self.adapt_conditions(
            lang_tokens, img_tokens, state_tokens)
        
        # 运行采样
        action_pred = self.conditional_sample(
            lang_cond, lang_attn_mask, img_cond, 
            state_traj, action_mask, ctrl_freqs,
        )
        
        return action_pred
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """保持兼容性，只返回总损失"""
        total_loss, _, _ = self.compute_loss(*args, **kwargs)
        return total_loss