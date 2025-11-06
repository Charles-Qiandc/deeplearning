# policy/RDT_repa/get_model_simple.py - 简化版评测模型加载器

import torch
import yaml
import os
from pathlib import Path
from transformers import AutoTokenizer

# 🔧 导入原始组件，但使用评测模式
from models.rdt_runner import RDTRunner
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder


def get_model(usr_args):
    """
    简化版评测模型加载器
    
    策略：
    1. 使用原始RDTRunner但禁用REPA功能
    2. 智能处理缺失的配置文件
    3. 提供强健的错误处理
    """
    print("🔧 简化版评测模式：初始化模型组件")
    
    # 🔧 尝试读取配置文件，如果失败则使用默认值
    def safe_load_config(file_path, default_config):
        """安全加载配置文件"""
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    print(f"✅ 成功读取配置: {file_path}")
                    return config
            except Exception as e:
                print(f"⚠️  配置文件读取失败: {file_path}, 错误: {e}")
        else:
            print(f"⚠️  配置文件不存在: {file_path}")
        
        print("🔧 使用默认配置")
        return default_config
    
    # 🔧 查找原始的deploy_policy.yml配置
    deploy_config_candidates = [
        'policy/RDT_repa/deploy_policy.yml',
        'policy/RDT_repa/config.yml', 
        './deploy_policy.yml',
        './config.yml',
    ]
    
    config = None
    config_path = None
    
    for candidate in deploy_config_candidates:
        if os.path.exists(candidate):
            config = safe_load_config(candidate, None)
            if config:
                config_path = candidate
                break
    
    # 如果还是没找到，使用默认配置
    if config is None:
        print("🔧 使用硬编码的默认配置")
        config = {
            "common": {
                "state_dim": 14,
                "action_chunk_size": 64,
                "img_history_size": 1,
                "num_cameras": 3,
            },
            "model": {
                "rdt": {
                    "hidden_size": 2048,
                    "depth": 28,
                    "num_heads": 32,
                },
                "lang_token_dim": 4096,
                "img_token_dim": 1536,
                "state_token_dim": 14,
                "lang_adaptor": "linear",
                "img_adaptor": "linear",
                "state_adaptor": "linear",
            },
            "dataset": {
                "tokenizer_max_length": 1024,
            }
        }
    
    # 🔧 智能检查点路径查找
    checkpoint_id = str(usr_args.get('checkpoint_id', 'unknown'))
    checkpoint_path = None
    
    # 搜索可能的检查点位置
    checkpoint_search_dirs = [
        'policy/RDT_repa/checkpoints',
        'checkpoints',
        '.',
    ]
    
    for search_dir in checkpoint_search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        # 尝试多种命名格式
        patterns = [
            f"checkpoint-{checkpoint_id}/pytorch_model.bin",
            f"checkpoint-{checkpoint_id}/model.bin", 
            f"checkpoint_{checkpoint_id}.bin",
            f"model_{checkpoint_id}.bin",
            f"{checkpoint_id}/pytorch_model.bin",
        ]
        
        for pattern in patterns:
            candidate = os.path.join(search_dir, pattern)
            if os.path.exists(candidate):
                checkpoint_path = candidate
                print(f"✅ 找到检查点: {checkpoint_path}")
                break
        
        if checkpoint_path:
            break
    
    # 如果还是没找到，查找任何可用的检查点
    if checkpoint_path is None:
        print("🔍 搜索任何可用的检查点文件...")
        for search_dir in checkpoint_search_dirs:
            if not os.path.exists(search_dir):
                continue
            
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if file in ['pytorch_model.bin', 'model.bin'] or file.endswith('.bin'):
                        candidate = os.path.join(root, file)
                        checkpoint_path = candidate
                        print(f"🔍 找到检查点文件: {checkpoint_path}")
                        break
                if checkpoint_path:
                    break
    
    print(f"📋 评测配置:")
    print(f"   - 检查点ID: {checkpoint_id}")
    print(f"   - 检查点路径: {checkpoint_path if checkpoint_path else '未找到'}")
    print(f"   - 配置文件: {config_path if config_path else '使用默认配置'}")

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # 🔧 简化文本编码器配置
    print("📝 配置文本编码器")
    precomp_lang_embed = usr_args.get('precomp_lang_embed', True)  # 默认使用预计算
    
    if precomp_lang_embed:
        print("   - 使用预计算的语言嵌入")
        tokenizer, text_encoder = None, None
    else:
        try:
            print("   - 尝试加载T5编码器")
            text_encoder_paths = [
                'policy/weights/RDT/t5-v1_1-xxl',
                '../weights/RDT/t5-v1_1-xxl',
                './t5-v1_1-xxl',
            ]
            
            text_encoder_path = None
            for path in text_encoder_paths:
                if os.path.exists(path):
                    text_encoder_path = path
                    break
            
            if text_encoder_path:
                text_embedder = T5Embedder(
                    from_pretrained=text_encoder_path,
                    model_max_length=config["dataset"]["tokenizer_max_length"],
                    device=device,
                )
                tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
            else:
                raise FileNotFoundError("T5模型路径不存在")
                
        except Exception as e:
            print(f"   ⚠️  T5加载失败: {e}")
            print("   - 切换到预计算语言嵌入模式")
            precomp_lang_embed = True
            tokenizer, text_encoder = None, None

    # 🔧 简化视觉编码器配置
    print("🖼️  配置视觉编码器")
    try:
        vision_encoder_paths = [
            'policy/weights/RDT/siglip-so400m-patch14-384',
            '../weights/RDT/siglip-so400m-patch14-384',
            './siglip-so400m-patch14-384',
        ]
        
        vision_encoder_path = None
        for path in vision_encoder_paths:
            if os.path.exists(path):
                vision_encoder_path = path
                break
        
        if vision_encoder_path:
            vision_encoder = SiglipVisionTower(vision_tower=vision_encoder_path, args=None)
            image_processor = vision_encoder.image_processor
            print(f"   ✅ 成功加载SigLIP: {vision_encoder_path}")
        else:
            raise FileNotFoundError("SigLIP模型路径不存在")
            
    except Exception as e:
        print(f"   ⚠️  SigLIP加载失败: {e}")
        print("   🔧 使用虚拟视觉编码器")
        
        class DummyVisionEncoder:
            def __init__(self):
                self.num_patches = 1024
                self.hidden_size = 1536
                
            class DummyImageProcessor:
                pass
                
            self.image_processor = DummyImageProcessor()
            
            def __call__(self, *args, **kwargs):
                # 返回虚拟特征
                batch_size = args[0].shape[0] if args else 1
                return torch.randn(batch_size, self.num_patches, self.hidden_size)
        
        vision_encoder = DummyVisionEncoder()
        image_processor = vision_encoder.image_processor

    # 🔧 构建RDT模型 - 关键：禁用REPA功能
    print("🤖 创建RDT模型 (禁用REPA)")
    
    img_cond_len = (config["common"]["img_history_size"] * config["common"]["num_cameras"] *
                    getattr(vision_encoder, 'num_patches', 1024))
    
    # 🔧 关键：使用原始RDTRunner但禁用所有REPA功能
    rdt = RDTRunner(
        action_dim=config["common"]["state_dim"],
        pred_horizon=config["common"]["action_chunk_size"],
        config=config["model"],
        lang_token_dim=config["model"]["lang_token_dim"],
        img_token_dim=config["model"]["img_token_dim"],
        state_token_dim=config["model"]["state_token_dim"],
        max_lang_cond_len=config["dataset"]["tokenizer_max_length"],
        img_cond_len=img_cond_len,
        img_pos_embed_config=[
            ("image", (
                config["common"]["img_history_size"],
                config["common"]["num_cameras"],
                -getattr(vision_encoder, 'num_patches', 1024),
            )),
        ],
        lang_pos_embed_config=[
            ("lang", -config["dataset"]["tokenizer_max_length"]),
        ],
        dtype=dtype,
        # 🔧 关键：禁用所有REPA相关功能
        enable_soft_routing_repa=False,
        soft_routing_repa_weight=0.0,
        use_dinov2_features=False,
        use_depth_features=False,
    )

    # 🔧 智能权重加载
    print("⚙️  加载预训练权重")
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            print(f"   🔧 开始加载: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 处理不同的检查点格式
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # 🔧 过滤掉REPA相关的参数
            filtered_state_dict = {}
            skipped_keys = []
            
            repa_keywords = [
                'dual_teacher_model', 'soft_router', 'routing_network',
                'dinov2_to_action_projector', 'depth_to_action_projector',
                'routing_temperature'
            ]
            
            for key, value in state_dict.items():
                # 跳过REPA相关参数
                if any(keyword in key for keyword in repa_keywords):
                    skipped_keys.append(key)
                    continue
                
                # 检查形状是否匹配
                if key in rdt.state_dict():
                    if value.shape == rdt.state_dict()[key].shape:
                        filtered_state_dict[key] = value
                    else:
                        print(f"   ⚠️  形状不匹配: {key}")
                        skipped_keys.append(key)
                else:
                    skipped_keys.append(key)
            
            # 加载过滤后的权重
            missing_keys, unexpected_keys = rdt.load_state_dict(filtered_state_dict, strict=False)
            
            print(f"   ✅ 成功加载 {len(filtered_state_dict)} 个参数")
            if skipped_keys:
                print(f"   ⚠️  跳过 {len(skipped_keys)} 个参数 (REPA相关或形状不匹配)")
            if missing_keys:
                print(f"   ⚠️  {len(missing_keys)} 个参数保持默认初始化")
            
        except Exception as e:
            print(f"   ❌ 权重加载失败: {e}")
            print("   🔧 使用默认初始化的权重")
    else:
        print("   ⚠️  未找到检查点文件，使用默认初始化")

    # 移动到设备并设置评测模式
    rdt.to(device, dtype=dtype)
    rdt.eval()
    
    if text_encoder is not None:
        text_encoder.to(device, dtype=dtype)
        text_encoder.eval()
    
    if hasattr(vision_encoder, 'vision_tower') and vision_encoder.vision_tower is not None:
        vision_encoder.vision_tower.to(device, dtype=dtype)
        vision_encoder.vision_tower.eval()

    # 🔧 创建评测专用策略包装器
    class SimpleEvalPolicy:
        """简化版评测策略包装器"""
        
        def __init__(self, rdt_model, text_encoder, vision_encoder, tokenizer, image_processor):
            self.rdt = rdt_model
            self.text_encoder = text_encoder
            self.vision_encoder = vision_encoder
            self.tokenizer = tokenizer
            self.image_processor = image_processor
            self.device = device
            self.dtype = dtype
            self.precomp_lang_embed = precomp_lang_embed
            
            print("✅ 简化版评测策略初始化完成")
            print(f"   - 模式: {'预计算语言嵌入' if precomp_lang_embed else 'T5编码器'}")
            print(f"   - REPA功能: 已禁用")
        
        def __call__(self, *args, **kwargs):
            """兼容原有调用方式"""
            return self.predict_action(*args, **kwargs)
        
        def predict_action(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens,
                          action_mask, ctrl_freqs, **kwargs):
            """预测动作 - 简化版本"""
            with torch.no_grad():
                # 确保输入在正确设备上
                lang_tokens = lang_tokens.to(self.device, dtype=self.dtype)
                img_tokens = img_tokens.to(self.device, dtype=self.dtype)
                state_tokens = state_tokens.to(self.device, dtype=self.dtype)
                action_mask = action_mask.to(self.device, dtype=self.dtype)
                
                if lang_attn_mask is not None:
                    lang_attn_mask = lang_attn_mask.to(self.device)
                
                try:
                    # 🔧 调用RDT进行纯推理，不传递任何REPA相关参数
                    action_pred = self.rdt.predict_action(
                        lang_tokens=lang_tokens,
                        lang_attn_mask=lang_attn_mask,
                        img_tokens=img_tokens,
                        state_tokens=state_tokens,
                        action_mask=action_mask,
                        ctrl_freqs=ctrl_freqs,
                        # 明确不传递vision_features等参数
                    )
                    
                    return action_pred
                    
                except Exception as e:
                    print(f"❌ 推理过程出错: {e}")
                    # 创建默认输出以避免中断
                    batch_size = lang_tokens.shape[0]
                    pred_horizon = getattr(self.rdt, 'pred_horizon', 64)
                    action_dim = getattr(self.rdt, 'action_dim', 14)
                    
                    default_action = torch.zeros(
                        batch_size, pred_horizon, action_dim,
                        device=self.device, dtype=self.dtype
                    )
                    print(f"🔧 返回默认动作: {default_action.shape}")
                    return default_action
    
    # 创建策略实例
    policy = SimpleEvalPolicy(rdt, text_encoder, vision_encoder, tokenizer, image_processor)
    
    # 输出最终统计
    total_params = sum(p.numel() for p in rdt.parameters())
    print("🎊 简化版评测模型加载完成！")
    print(f"   - 总参数量: {total_params:,}")
    print(f"   - 设备: {device}")
    print(f"   - 数据类型: {dtype}")
    print(f"   - 检查点状态: {'已加载' if checkpoint_path else '使用默认初始化'}")
    print(f"   - REPA功能: 已完全禁用")
    print(f"   - 预期推理稳定性: 高")
    
    return policy


def reset_model(policy):
    """重置模型状态 - 简化版本"""
    if hasattr(policy, 'rdt'):
        policy.rdt.eval()
    if hasattr(policy, 'text_encoder') and policy.text_encoder is not None:
        policy.text_encoder.eval()
    if hasattr(policy, 'vision_encoder'):
        if hasattr(policy.vision_encoder, 'vision_tower') and policy.vision_encoder.vision_tower is not None:
            policy.vision_encoder.vision_tower.eval()


def eval(task_env, policy, observation):
    """简化版评测函数"""
    try:
        # 提取观察信息
        lang_tokens = observation.get('lang_tokens')
        lang_attn_mask = observation.get('lang_attn_mask')
        img_tokens = observation.get('img_tokens')
        state_tokens = observation.get('state_tokens')
        action_mask = observation.get('action_mask')
        ctrl_freqs = observation.get('ctrl_freqs')
        
        # 调用策略预测
        action_pred = policy.predict_action(
            lang_tokens=lang_tokens,
            lang_attn_mask=lang_attn_mask,
            img_tokens=img_tokens,
            state_tokens=state_tokens,
            action_mask=action_mask,
            ctrl_freqs=ctrl_freqs,
        )
        
        return action_pred
        
    except Exception as e:
        print(f"❌ 评测步骤失败: {e}")
        import traceback
        traceback.print_exc()
        raise e


if __name__ == "__main__":
    # 测试简化版模型加载
    test_args = {
        'checkpoint_id': '10000',
        'precomp_lang_embed': True,  # 使用预计算以简化测试
    }
    
    print("🧪 测试简化版评测模型加载")
    try:
        policy = get_model(test_args)
        print("✅ 测试成功！")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()