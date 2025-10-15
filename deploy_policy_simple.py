# policy/RDT_repa/deploy_policy_simple.py - 保持原文件位置的简化版

import os
import sys
import torch
import numpy as np
from pathlib import Path

# 🔧 添加当前目录到Python路径，这样可以导入同目录下的models
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 🎯 现在可以导入同目录下的models/rdt_runner_eval_simple.py
from models.rdt_runner_eval_simple import create_simple_eval_runner


class SimpleRDTRepaModel:
    """
    简化版RDT模型部署类
    """
    
    def __init__(self, config_path=None, checkpoint_path=None):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 SimpleRDTRepaModel初始化，设备: {self.device}")
    
    def load_model(self, usr_args):
        """加载模型"""
        print(f"🔧 开始加载简化版RDT模型...")
        
        # 🎯 构建checkpoint路径
        if self.checkpoint_path is None:
            # 从usr_args构建路径
            policy_name = usr_args.get("policy_name", "RDT_repa")
            ckpt_setting = usr_args.get("ckpt_setting", "default")
            checkpoint_id = usr_args.get("checkpoint_id", "latest")
            
            # 🔧 尝试多个可能的checkpoint路径
            possible_paths = [
                f"policy/{policy_name}/checkpoints/{ckpt_setting}/checkpoint-{checkpoint_id}/pytorch_model.bin",
                f"checkpoints/{ckpt_setting}/checkpoint-{checkpoint_id}/pytorch_model.bin",
                f"./policy/{policy_name}/checkpoints/{ckpt_setting}/checkpoint-{checkpoint_id}/pytorch_model.bin",
                f"/home/deng_xiang/qian_daichao/RoboTwin/policy/{policy_name}/checkpoints/{ckpt_setting}/checkpoint-{checkpoint_id}/pytorch_model.bin",
            ]
            
            checkpoint_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    checkpoint_path = path
                    break
            
            if checkpoint_path is None:
                print(f"⚠️ 在以下路径中都找不到checkpoint:")
                for path in possible_paths:
                    print(f"   - {path}")
                raise FileNotFoundError(f"找不到checkpoint文件")
        else:
            checkpoint_path = self.checkpoint_path
        
        print(f"📁 使用checkpoint路径: {checkpoint_path}")
        
        # 🎯 模型配置（标准RDT配置）
        model_config = {
            'action_dim': 128,  # 根据你的具体任务调整
            'pred_horizon': 64,
            'config': {
                'rdt': {
                    'hidden_size': 2048,
                    'depth': 28,
                    'num_heads': 32,
                },
                'lang_adaptor': 'linear',
                'img_adaptor': 'linear',
                'state_adaptor': 'linear',
                'noise_scheduler': {
                    'num_train_timesteps': 1000,
                    'beta_schedule': 'linear',
                    'prediction_type': 'epsilon',
                    'clip_sample': False,
                    'num_inference_timesteps': 50,
                }
            },
            'lang_token_dim': 4096,
            'img_token_dim': 1536,  # SigLIP特征维度
            'state_token_dim': usr_args.get("left_arm_dim", 7) + usr_args.get("right_arm_dim", 7),  # 机械臂维度
            'max_lang_cond_len': 1024,
            'img_cond_len': 4096,   # 🎯 标准SigLIP长度，不包含额外特征
            'dtype': torch.bfloat16,
        }
        
        # 🎯 创建简化评测Runner
        self.model = create_simple_eval_runner(
            checkpoint_path=checkpoint_path,
            **model_config
        )
        
        # 移动到指定设备
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ 简化版RDT模型加载完成")
        return self.model
    
    def predict(self, observation, model=None):
        """简化的预测接口"""
        if model is None:
            model = self.model
            
        if model is None:
            raise ValueError("模型尚未加载，请先调用load_model")
        
        # 🎯 提取标准输入（只需要基础特征）
        lang_tokens = observation.get('lang_tokens')
        lang_attn_mask = observation.get('lang_attn_mask')
        img_tokens = observation.get('img_tokens')
        state_tokens = observation.get('state_tokens')
        action_mask = observation.get('action_mask')
        ctrl_freqs = observation.get('ctrl_freqs')
        
        # 确保所有输入都在正确的设备上
        if lang_tokens is not None:
            lang_tokens = lang_tokens.to(self.device)
        if lang_attn_mask is not None:
            lang_attn_mask = lang_attn_mask.to(self.device)
        if img_tokens is not None:
            img_tokens = img_tokens.to(self.device)
        if state_tokens is not None:
            state_tokens = state_tokens.to(self.device)
        if action_mask is not None:
            action_mask = action_mask.to(self.device)
        if ctrl_freqs is not None:
            ctrl_freqs = ctrl_freqs.to(self.device)
        
        # 🎯 调用简化的预测接口
        with torch.no_grad():
            action_pred = model.predict_action(
                lang_tokens=lang_tokens,
                lang_attn_mask=lang_attn_mask,
                img_tokens=img_tokens,
                state_tokens=state_tokens,
                action_mask=action_mask,
                ctrl_freqs=ctrl_freqs,
            )
        
        return action_pred


# 🎯 全局模型实例
_global_model = None


def get_model_simple(usr_args):
    """简化版get_model函数"""
    global _global_model
    
    if _global_model is None:
        print(f"🔧 首次创建简化版RDT模型...")
        _global_model = SimpleRDTRepaModel()
        _global_model.load_model(usr_args)
    else:
        print(f"🔄 重用已创建的简化版RDT模型")
    
    return _global_model


def eval_simple(TASK_ENV, model, observation):
    """简化版评测函数"""
    try:
        # 🎯 使用简化的预测接口
        action_pred = model.predict(observation)
        
        # 将预测结果应用到环境
        if action_pred is not None:
            # 转换为numpy并应用到环境
            if torch.is_tensor(action_pred):
                action_np = action_pred.cpu().numpy()
            else:
                action_np = action_pred
            
            # 应用动作到环境
            TASK_ENV.apply_action(action_np)
        else:
            print("⚠️ 模型预测返回None")
            
    except Exception as e:
        print(f"❌ 评测过程中出错: {e}")
        import traceback
        traceback.print_exc()


def reset_model(model):
    """重置模型状态"""
    # 🎯 简化版本：只需要清理缓存
    if hasattr(model, 'model') and hasattr(model.model, 'eval'):
        model.model.eval()
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# 🎯 兼容性函数
def get_model(usr_args):
    """兼容性wrapper"""
    return get_model_simple(usr_args)


def eval(TASK_ENV, model, observation):
    """兼容性wrapper"""
    return eval_simple(TASK_ENV, model, observation)


if __name__ == "__main__":
    print("🧪 测试简化版RDT模型部署")
    
    test_usr_args = {
        'policy_name': 'RDT_repa',
        'ckpt_setting': 'lift_pot_repa_ablation', 
        'checkpoint_id': '10000',
        'left_arm_dim': 7,
        'right_arm_dim': 7,
    }
    
    try:
        model = get_model_simple(test_usr_args)
        print(f"✅ 模型加载成功: {type(model)}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
                # policy/RDT_repa/deploy_policy_simple.py - 基于你的源代码修改的简化版policy

import os
import torch
import numpy as np
from pathlib import Path

# 🎯 修改：导入简化版RDTRunner
from models.rdt_runner_eval_simple import create_simple_eval_runner


class SimpleRDTRepaModel:
    """
    简化版RDT模型部署类
    
    核心特点：
    1. 使用标准RDT结构（无REPA，无多模态融合）
    2. 智能加载消融实验checkpoint
    3. 简化的推理接口
    """
    
    def __init__(self, config_path=None, checkpoint_path=None):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 SimpleRDTRepaModel初始化，设备: {self.device}")
    
    def load_model(self, usr_args):
        """加载模型"""
        print(f"🔧 开始加载简化版RDT模型...")
        
        # 🎯 构建checkpoint路径
        if self.checkpoint_path is None:
            # 从usr_args构建路径
            policy_name = usr_args.get("policy_name", "RDT_repa")
            ckpt_setting = usr_args.get("ckpt_setting", "default")
            checkpoint_id = usr_args.get("checkpoint_id", "latest")
            
            checkpoint_path = f"policy/{policy_name}/checkpoints/{ckpt_setting}/checkpoint-{checkpoint_id}/pytorch_model.bin"
            
            # 检查路径是否存在
            if not os.path.exists(checkpoint_path):
                # 尝试其他可能的路径
                alternative_paths = [
                    f"checkpoints/{ckpt_setting}/checkpoint-{checkpoint_id}/pytorch_model.bin",
                    f"./policy/{policy_name}/checkpoints/{ckpt_setting}/checkpoint-{checkpoint_id}/pytorch_model.bin",
                ]
                
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        checkpoint_path = alt_path
                        break
                else:
                    raise FileNotFoundError(f"找不到checkpoint文件，尝试过的路径: {[checkpoint_path] + alternative_paths}")
        else:
            checkpoint_path = self.checkpoint_path
        
        print(f"📁 使用checkpoint路径: {checkpoint_path}")
        
        # 🎯 模型配置（标准RDT配置）
        model_config = {
            'action_dim': 128,  # 根据你的具体任务调整
            'pred_horizon': 64,
            'config': {
                'rdt': {
                    'hidden_size': 2048,
                    'depth': 28,
                    'num_heads': 32,
                },
                'lang_adaptor': 'linear',
                'img_adaptor': 'linear',
                'state_adaptor': 'linear',
                'noise_scheduler': {
                    'num_train_timesteps': 1000,
                    'beta_schedule': 'linear',
                    'prediction_type': 'epsilon',
                    'clip_sample': False,
                    'num_inference_timesteps': 50,
                }
            },
            'lang_token_dim': 4096,
            'img_token_dim': 1536,  # SigLIP特征维度
            'state_token_dim': usr_args.get("left_arm_dim", 7) + usr_args.get("right_arm_dim", 7),  # 机械臂维度
            'max_lang_cond_len': 1024,
            'img_cond_len': 4096,   # 🎯 标准SigLIP长度，不包含额外特征
            'dtype': torch.bfloat16,
        }
        
        # 🎯 创建简化评测Runner
        self.model = create_simple_eval_runner(
            checkpoint_path=checkpoint_path,
            **model_config
        )
        
        # 移动到指定设备
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ 简化版RDT模型加载完成")
        return self.model
    
    def predict(self, observation, model=None):
        """
        简化的预测接口
        
        Args:
            observation: 环境观测，包含lang_tokens, img_tokens等
            model: 模型实例（兼容性参数）
        
        Returns:
            action: 预测的动作
        """
        if model is None:
            model = self.model
            
        if model is None:
            raise ValueError("模型尚未加载，请先调用load_model")
        
        # 🎯 提取标准输入（只需要基础特征，不需要额外的视觉特征）
        lang_tokens = observation.get('lang_tokens')  # 语言特征
        lang_attn_mask = observation.get('lang_attn_mask')  # 语言注意力掩码
        img_tokens = observation.get('img_tokens')    # SigLIP图像特征
        state_tokens = observation.get('state_tokens')  # 状态特征
        action_mask = observation.get('action_mask')  # 动作掩码
        ctrl_freqs = observation.get('ctrl_freqs')    # 控制频率
        
        # 🎯 注意：不需要dinov2_features, depth_features等额外特征
        
        # 确保所有输入都在正确的设备上
        if lang_tokens is not None:
            lang_tokens = lang_tokens.to(self.device)
        if lang_attn_mask is not None:
            lang_attn_mask = lang_attn_mask.to(self.device)
        if img_tokens is not None:
            img_tokens = img_tokens.to(self.device)
        if state_tokens is not None:
            state_tokens = state_tokens.to(self.device)
        if action_mask is not None:
            action_mask = action_mask.to(self.device)
        if ctrl_freqs is not None:
            ctrl_freqs = ctrl_freqs.to(self.device)
        
        # 🎯 调用简化的预测接口
        with torch.no_grad():
            action_pred = model.predict_action(
                lang_tokens=lang_tokens,
                lang_attn_mask=lang_attn_mask,
                img_tokens=img_tokens,
                state_tokens=state_tokens,
                action_mask=action_mask,
                ctrl_freqs=ctrl_freqs,
                # 🎯 不传递额外的视觉特征参数
            )
        
        return action_pred


# 🎯 全局模型实例
_global_model = None


def get_model_simple(usr_args):
    """
    简化版get_model函数 - 替代原来的get_model
    
    Args:
        usr_args: 用户参数，包含checkpoint信息等
    
    Returns:
        SimpleRDTRepaModel实例
    """
    global _global_model
    
    if _global_model is None:
        print(f"🔧 首次创建简化版RDT模型...")
        _global_model = SimpleRDTRepaModel()
        _global_model.load_model(usr_args)
    else:
        print(f"🔄 重用已创建的简化版RDT模型")
    
    return _global_model


def eval_simple(TASK_ENV, model, observation):
    """
    简化版评测函数 - 替代原来的eval函数
    
    Args:
        TASK_ENV: 任务环境
        model: 模型实例
        observation: 环境观测
    """
    try:
        # 🎯 使用简化的预测接口
        action_pred = model.predict(observation)
        
        # 将预测结果应用到环境
        if action_pred is not None:
            # 转换为numpy并应用到环境
            if torch.is_tensor(action_pred):
                action_np = action_pred.cpu().numpy()
            else:
                action_np = action_pred
            
            # 应用动作到环境
            TASK_ENV.apply_action(action_np)
        else:
            print("⚠️ 模型预测返回None")
            
    except Exception as e:
        print(f"❌ 评测过程中出错: {e}")
        import traceback
        traceback.print_exc()


def reset_model(model):
    """
    重置模型状态 - 保持与原接口兼容
    
    Args:
        model: 模型实例
    """
    # 🎯 简化版本：只需要清理缓存
    if hasattr(model, 'model') and hasattr(model.model, 'eval'):
        model.model.eval()
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# 🎯 兼容性函数：如果有其他代码仍在使用原来的函数名
def get_model(usr_args):
    """兼容性wrapper"""
    return get_model_simple(usr_args)


def eval(TASK_ENV, model, observation):
    """兼容性wrapper"""
    return eval_simple(TASK_ENV, model, observation)


# 🎯 模块级别的配置
MODEL_CONFIG = {
    'type': 'simplified_rdt',
    'description': 'Simplified RDT model for evaluation, compatible with ablation experiment checkpoints',
    'features': [
        'Standard RDT structure',
        'No REPA alignment',
        'No multimodal fusion', 
        'Intelligent checkpoint loading',
        'Automatic dimension adaptation'
    ]
}


if __name__ == "__main__":
    # 测试代码
    print("🧪 测试简化版RDT模型部署")
    
    test_usr_args = {
        'policy_name': 'RDT_repa',
        'ckpt_setting': 'lift_pot_repa_ablation', 
        'checkpoint_id': '10000',
        'left_arm_dim': 7,
        'right_arm_dim': 7,
    }
    
    try:
        model = get_model_simple(test_usr_args)
        print(f"✅ 模型加载成功: {type(model)}")
        
        # 创建测试观测
        test_observation = {
            'lang_tokens': torch.randn(1, 256, 4096),
            'lang_attn_mask': torch.ones(1, 256),
            'img_tokens': torch.randn(1, 4096, 1536),
            'state_tokens': torch.randn(1, 1, 14),
            'action_mask': torch.ones(1, 64, 1),
            'ctrl_freqs': torch.tensor([10.0]),
        }
        
        # 测试预测
        prediction = model.predict(test_observation)
        print(f"✅ 预测成功，输出形状: {prediction.shape}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()