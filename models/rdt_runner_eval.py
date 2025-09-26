# models/rdt_runner_eval.py - 专门用于评测的RDTRunner版本

import re
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from models.hub_mixin import CompatiblePyTorchModelHubMixin
from models.rdt.model import RDT


class RDTRunnerEval(nn.Module, CompatiblePyTorchModelHubMixin, 
                   repo_url="https://huggingface.co/robotics-diffusion-transformer/rdt-1b"):
    """
    专门用于评测的RDT运行器
    
    关键改动：
    1. 移除所有REPA对齐相关组件
    2. 移除双教师路由网络
    3. 移除深度特征处理
    4. 简化为纯推理模式
    """
    def __init__(self, *, action_dim, pred_horizon, config, 
                 lang_token_dim, img_token_dim, state_token_dim, 
                 max_lang_cond_len, img_cond_len, lang_pos_embed_config=None, 
                 img_pos_embed_config=None, dtype=torch.bfloat16):
        super(RDTRunnerEval, self).__init__()
        
        self.dtype = dtype
        
        print(f"🔧 评测专用RDTRunner初始化:")
        print(f"   - 移除REPA对齐组件")
        print(f"   - 移除双教师路由网络")
        print(f"   - 纯推理模式")
        print(f"   - 数据类型: {dtype}")
        
        # 创建扩散模型 - 禁用所有REPA相关功能
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
            # 🔧 关键：禁用所有REPA功能
            enable_repa_loss=False,
            use_dual_teachers=False,
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

    def load_pretrained_weights(self, checkpoint_path):
        """
        智能加载预训练权重，自动过滤不匹配的参数
        """
        print(f"🔧 开始加载权重: {checkpoint_path}")
        
        try:
            # 加载检查点
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 处理不同的检查点格式
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'module' in checkpoint:
                    state_dict = checkpoint['module']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            print(f"✅ Checkpoint加载成功，包含 {len(state_dict)} 个顶级键")
            
            # 获取当前模型的状态字典
            current_state_dict = self.state_dict()
            print(f"📋 当前模型包含 {len(current_state_dict)} 个参数")
            
            # 智能过滤权重
            filtered_state_dict = {}
            skipped_keys = []
            missing_keys = []
            shape_mismatched_keys = []
            
            # 检查checkpoint中的每个参数
            for key, value in state_dict.items():
                if key in current_state_dict:
                    current_shape = current_state_dict[key].shape
                    checkpoint_shape = value.shape
                    
                    if current_shape == checkpoint_shape:
                        filtered_state_dict[key] = value
                    else:
                        shape_mismatched_keys.append(f"{key} (形状不匹配: {checkpoint_shape} vs {current_shape})")
                else:
                    # 跳过不存在的参数（如REPA相关参数）
                    if any(keyword in key for keyword in [
                        'dual_teacher_model', 'soft_router', 'routing_network', 
                        'dinov2_to_action_projector', 'depth_to_action_projector',
                        'routing_temperature']):
                        skipped_keys.append(key)
                    else:
                        skipped_keys.append(key)
            
            # 检查模型中缺失的参数
            for key in current_state_dict.keys():
                if key not in state_dict:
                    missing_keys.append(key)
            
            # 加载过滤后的权重
            missing_keys_after_load, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)
            
            # 详细报告
            print(f"✅ 成功加载 {len(filtered_state_dict)} 个参数")
            
            if skipped_keys:
                print(f"⚠️  跳过 {len(skipped_keys)} 个其他参数")
                for key in skipped_keys[:5]:  # 只显示前5个
                    print(f"     - {key}")
                if len(skipped_keys) > 5:
                    print(f"     - ... 还有 {len(skipped_keys) - 5} 个参数")
            
            if shape_mismatched_keys:
                print(f"⚠️  形状不匹配的参数 ({len(shape_mismatched_keys)} 个):")
                for key in shape_mismatched_keys:
                    print(f"     - {key}")
            
            if missing_keys_after_load:
                print(f"⚠️  {len(missing_keys_after_load)} 个模型参数未找到对应权重（保持默认初始化）")
                # 只在调试时显示详细信息
                # for key in missing_keys_after_load[:3]:
                #     print(f"     - {key}")
            
            print(f"✅ 权重加载完成，模型可以正常评估")
            return True
            
        except Exception as e:
            print(f"❌ 权重加载失败: {e}")
            return False

    def conditional_sample(self, lang_cond, lang_attn_mask, img_cond, state_traj, action_mask, ctrl_freqs):
        """推理时的条件采样 - 纯净版本"""
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
            
            # 🔧 只获取动作预测，忽略中间激活
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
        """预测动作 - 纯推理版本，不使用任何对齐特征"""
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
        """简化的前向传播，只用于推理"""
        return self.predict_action(*args, **kwargs)


# 工厂函数：根据模式创建合适的RDTRunner
def create_rdt_runner(mode="eval", **kwargs):
    """
    创建RDTRunner的工厂函数
    
    Args:
        mode: "train" 或 "eval"
        **kwargs: RDTRunner的初始化参数
    
    Returns:
        RDTRunner实例
    """
    if mode == "eval":
        print("🔧 创建评测专用RDTRunner")
        return RDTRunnerEval(**kwargs)
    elif mode == "train":
        print("🔧 创建训练用RDTRunner")
        from models.rdt_runner import RDTRunner  # 导入原始训练版本
        return RDTRunner(**kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}, expected 'train' or 'eval'")