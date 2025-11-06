"""
SigLIP 深度特征提取器
作为深度教师的替代方案
"""

import torch
import torch.nn as nn
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower


class SiglipDepthFeatureExtractor(nn.Module):
    """
    SigLIP深度特征提取器
    使用SigLIP的patch tokens作为深度感知特征
    
    与DepthAnythingV2不同，SigLIP提供的是RGB特征而非真实深度信息，
    但同样可以用于几何对齐
    """
    
    def __init__(
        self, 
        vision_tower_name: str = "google/siglip-so400m-patch14-384",
        feature_dim: int = 1152,
        output_format: str = "patch_tokens",  # "patch_tokens" | "cls_patch"
        args=None
    ):
        """
        Args:
            vision_tower_name: SigLIP模型名称
            feature_dim: 输出特征维度（SigLIP-SO400M为1152）
            output_format: 输出格式
                - "patch_tokens": 只返回patch tokens (729个)
                - "cls_patch": 返回类CLS + patch tokens (730个)
            args: 额外参数
        """
        super().__init__()
        
        self.vision_tower_name = vision_tower_name
        self.feature_dim = feature_dim
        self.output_format = output_format
        self.is_loaded = False
        
        # 复用现有的SigLIP编码器
        self.siglip_encoder = SiglipVisionTower(
            vision_tower=vision_tower_name,
            args=args,
            delay_load=False
        )
        
        self.is_loaded = True
        
        print(f"✅ SigLIP深度特征提取器初始化完成")
        print(f"   - 模型: {vision_tower_name}")
        print(f"   - 输出格式: {output_format}")
        print(f"   - 特征维度: {feature_dim}")
    
    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> tuple:
        """
        提取深度感知特征（实际是RGB的patch特征）
        
        Args:
            images: (B, C, H, W) 输入图像
            
        Returns:
            depth_features: (B, N, D) 深度特征
                - N = 729 (patch_tokens) 或 730 (cls_patch)
                - D = 1152
            depth_map: (B, 1, H, W) 伪深度图（实际是特征可视化）
        """
        if not self.is_loaded:
            raise RuntimeError("SigLIP encoder未加载")
        
        B, C, H, W = images.shape
        
        # 获取patch-level特征
        patch_tokens = self.siglip_encoder(images)  # (B, 729, 1152)
        
        # 根据输出格式处理
        if self.output_format == "patch_tokens":
            depth_features = patch_tokens  # (B, 729, 1152)
        elif self.output_format == "cls_patch":
            # 添加一个伪CLS token（平均池化）
            cls_token = patch_tokens.mean(dim=1, keepdim=True)  # (B, 1, 1152)
            depth_features = torch.cat([cls_token, patch_tokens], dim=1)  # (B, 730, 1152)
        else:
            raise ValueError(f"未知的输出格式: {self.output_format}")
        
        # 生成伪深度图（用于可视化，实际不是真实深度）
        # 使用第一个通道的特征重塑为空间图
        spatial_dim = int(patch_tokens.shape[1] ** 0.5)  # 27 for 729 patches
        if spatial_dim * spatial_dim == patch_tokens.shape[1]:
            # 可以重塑为空间图
            feature_map = patch_tokens[:, :, 0].reshape(B, 1, spatial_dim, spatial_dim)
            # 上采样到原始尺寸
            depth_map = torch.nn.functional.interpolate(
                feature_map, 
                size=(H, W), 
                mode='bilinear', 
                align_corners=False
            )
            # 归一化到[0, 1]
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        else:
            # 无法重塑，返回零图
            depth_map = torch.zeros(B, 1, H, W, device=images.device, dtype=images.dtype)
        
        return depth_features, depth_map
    
    def get_cls_token(self, depth_features: torch.Tensor) -> torch.Tensor:
        """
        从深度特征中提取CLS token
        
        Args:
            depth_features: (B, N, D) 深度特征
            
        Returns:
            cls_token: (B, D) CLS token
        """
        if self.output_format == "cls_patch":
            # 第一个token是CLS
            return depth_features[:, 0, :]
        else:
            # 没有显式CLS，使用平均池化
            return depth_features.mean(dim=1)
    
    def get_patch_tokens(self, depth_features: torch.Tensor) -> torch.Tensor:
        """
        从深度特征中提取patch tokens
        
        Args:
            depth_features: (B, N, D) 深度特征
            
        Returns:
            patch_tokens: (B, N_patches, D) patch tokens
        """
        if self.output_format == "cls_patch":
            # 跳过第一个CLS token
            return depth_features[:, 1:, :]
        else:
            # 全部都是patch tokens
            return depth_features
    
    @property
    def dtype(self):
        return self.siglip_encoder.dtype
    
    @property
    def device(self):
        return self.siglip_encoder.device
    
    @property
    def hidden_size(self):
        return self.feature_dim
    
    @property
    def num_patches(self):
        """返回patch数量"""
        if self.output_format == "cls_patch":
            return 730  # CLS + 729 patches
        else:
            return 729  # 只有patches
    
    def print_model_info(self):
        """打印模型信息"""
        print("🔍 SigLIP深度特征提取器信息:")
        print(f"   - 模型名称: {self.vision_tower_name}")
        print(f"   - 输出格式: {self.output_format}")
        print(f"   - 特征维度: {self.feature_dim}")
        print(f"   - Token数量: {self.num_patches}")
        if self.output_format == "cls_patch":
            print(f"   - 输出格式: (B, 730, {self.feature_dim}) - CLS + patches")
        else:
            print(f"   - 输出格式: (B, 729, {self.feature_dim}) - patches only")
        print(f"   - 设备: {self.device}")
        print(f"   - 数据类型: {self.dtype}")
        print(f"   ⚠️  注意: 这是RGB特征，不是真实深度信息")


def create_siglip_depth_encoder(
    model_name: str = "google/siglip-so400m-patch14-384",
    feature_dim: int = 1152,
    output_format: str = "patch_tokens",
    device=None
):
    """
    工厂函数：创建SigLIP深度特征提取器
    
    Args:
        model_name: SigLIP模型名称
        feature_dim: 特征维度
        output_format: 输出格式
        device: 计算设备
        
    Returns:
        SiglipDepthFeatureExtractor实例
    """
    print(f"🔧 创建SigLIP深度特征提取器")
    print(f"   - 模型: {model_name}")
    print(f"   - 输出格式: {output_format}")
    
    encoder = SiglipDepthFeatureExtractor(
        vision_tower_name=model_name,
        feature_dim=feature_dim,
        output_format=output_format
    )
    
    if device is not None:
        encoder.to(device)
    
    return encoder


# 测试代码
if __name__ == "__main__":
    print("🧪 测试SigLIP深度特征提取器")
    
    # 创建编码器
    encoder = create_siglip_depth_encoder()
    
    # 测试前向传播
    print("\n🔬 测试前向传播:")
    test_input = torch.randn(2, 3, 384, 384)
    
    depth_features, depth_map = encoder(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"深度特征形状: {depth_features.shape}")
    print(f"深度图形状: {depth_map.shape}")
    
    # 测试CLS token和patch tokens提取
    cls_token = encoder.get_cls_token(depth_features)
    patch_tokens = encoder.get_patch_tokens(depth_features)
    print(f"CLS token形状: {cls_token.shape}")
    print(f"Patch tokens形状: {patch_tokens.shape}")
    
    print("\n✅ 所有测试通过！")