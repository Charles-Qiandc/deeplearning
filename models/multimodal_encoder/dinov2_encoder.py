import torch
import torch.nn as nn
from transformers import AutoConfig, AutoImageProcessor, AutoModel, Dinov2Model


class DinoV2VisionTower(nn.Module):
    def __init__(self, vision_tower="facebook/dinov2-large", args=None, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        
        # 支持args为None的情况，并设置默认值
        if args is not None:
            self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
            self.unfreeze_mm_vision_tower = getattr(args, 'unfreeze_mm_vision_tower', False)
        else:
            self.select_feature = 'patch'  # 默认使用patch tokens
            self.unfreeze_mm_vision_tower = False

        if not delay_load:
            self.load_model()
        elif self.unfreeze_mm_vision_tower:
            self.load_model()
        else:
            self.cfg_only = AutoConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        # 确保加载DINOv2的正确模型
        print(f"Loading DINOv2 model: {self.vision_tower_name}")
        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = AutoModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        
        # 验证模型配置
        print(f"DINOv2 config - hidden_size: {self.vision_tower.config.hidden_size}")
        print(f"DINOv2 config - image_size: {self.vision_tower.config.image_size}")
        print(f"DINOv2 config - patch_size: {self.vision_tower.config.patch_size}")
        
        # 冻结权重（训练时只使用特征提取）
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        """
        改进特征选择逻辑，确保输出维度正确
        """
        image_features = image_forward_outs.last_hidden_state
        
        if self.select_feature == 'patch':
            # 去除CLS token，只保留patch tokens
            image_features = image_features[:, 1:]  # (B, num_patches, hidden_size)
        elif self.select_feature == 'cls_patch':
            # 保留所有tokens (CLS + patches)
            image_features = image_features  # (B, num_patches+1, hidden_size)
        elif self.select_feature == 'cls_only':
            # 只使用CLS token
            image_features = image_features[:, :1]  # (B, 1, hidden_size)
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        
        return image_features

    @torch.no_grad()
    def forward(self, images):
        """
        改进前向传播，添加更好的错误处理和调试信息
        """
        if not self.is_loaded:
            raise RuntimeError("DINOv2 model not loaded. Call load_model() first.")
            
        if type(images) is list:
            image_features = []
            for image in images:
                # 确保图像尺寸正确
                if image.shape[-2:] != (518, 518):
                    print(f"⚠️ Warning: Image size {image.shape[-2:]} != (518, 518). Resizing...")
                    image = torch.nn.functional.interpolate(
                        image.unsqueeze(0), 
                        size=(518, 518), 
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0)
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            # 输入验证
            if images.dim() != 4:
                raise ValueError(f"Expected 4D tensor (B,C,H,W), got {images.dim()}D tensor")
            
            # 检查并调整图像尺寸
            if images.shape[-2:] != (518, 518):
                print(f"⚠️ Warning: Image size {images.shape[-2:]} != (518, 518). Resizing...")
                images = torch.nn.functional.interpolate(
                    images, 
                    size=(518, 518), 
                    mode='bilinear', 
                    align_corners=False
                )
            
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype)
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        if self.is_loaded:
            return self.vision_tower.dtype
        else:
            return torch.float32

    @property
    def device(self):
        if self.is_loaded:
            return self.vision_tower.device
        else:
            return torch.device('cpu')

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        """确保返回正确的DINOv2-L隐藏维度"""
        if self.is_loaded:
            return self.config.hidden_size
        else:
            # DINOv2-L的默认隐藏维度
            return 1024

    @property
    def num_patches_per_side(self):
        """计算每边的patch数量"""
        # DINOv2 使用 518x518 图像和 14x14 的 patch
        # 对于 DINOv2，这应该返回 37 (518/14 = 37)
        if self.is_loaded:
            # 从配置中获取实际值
            image_size = self.config.image_size
            patch_size = self.config.patch_size
            
            # 处理可能的元组形式
            if isinstance(image_size, (list, tuple)):
                image_size = image_size[0]
            if isinstance(patch_size, (list, tuple)):
                patch_size = patch_size[0]
                
            return image_size // patch_size
        else:
            # DINOv2 的默认值
            return 37

    @property
    def num_patches(self):
        """计算总patch数量"""
        patches_per_side = self.num_patches_per_side
        return patches_per_side * patches_per_side  # 14 * 14 = 196

    def print_model_info(self):
        """打印模型信息，用于验证"""
        if not self.is_loaded:
            print("❌ Model not loaded")
            return
            
        print("🔍 DINOv2 Model Information:")
        print(f"   - Model name: {self.vision_tower_name}")
        print(f"   - Hidden size: {self.hidden_size}")
        
        # 安全地获取配置值
        image_size = self.config.image_size
        patch_size = self.config.patch_size
        
        if isinstance(image_size, (list, tuple)):
            image_size = image_size[0]
        if isinstance(patch_size, (list, tuple)):
            patch_size = patch_size[0]
            
        print(f"   - Image size: {image_size}")
        print(f"   - Patch size: {patch_size}")
        print(f"   - Patches per side: {self.num_patches_per_side}")
        print(f"   - Total patches: {self.num_patches}")
        print(f"   - Select feature: {self.select_feature}")
        print(f"   - Device: {self.device}")
        print(f"   - Dtype: {self.dtype}")


# 便于测试的工厂函数
def create_dinov2_encoder(model_size="large", select_feature="patch"):
    """
    创建DINOv2编码器的工厂函数
    
    Args:
        model_size: 模型大小，支持 "large"
        select_feature: 特征选择方式，支持 "patch", "cls_patch", "cls_only"
    
    Returns:
        DinoV2VisionTower: 配置好的视觉编码器
    """
    model_mapping = {
        "large": "facebook/dinov2-large",
        "base": "facebook/dinov2-base",
        "small": "facebook/dinov2-small",
        "giant": "facebook/dinov2-giant"
    }
    
    if model_size not in model_mapping:
        raise ValueError(f"Unsupported model size: {model_size}. Choose from {list(model_mapping.keys())}")
    
    # 创建args对象
    class Args:
        def __init__(self):
            self.mm_vision_select_feature = select_feature
            self.unfreeze_mm_vision_tower = False
    
    args = Args()
    vision_tower = DinoV2VisionTower(
        vision_tower=model_mapping[model_size],
        args=args,
        delay_load=False
    )
    
    return vision_tower

