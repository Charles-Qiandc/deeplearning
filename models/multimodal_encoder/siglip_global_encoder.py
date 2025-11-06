"""
SigLIP 全局特征提取器
用于REPA对齐的CLS token提取
"""

import torch
import torch.nn as nn
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower


class SiglipGlobalFeatureExtractor(nn.Module):
    """
    SigLIP全局特征提取器
    从SigLIP的最后一层提取全局表示用于REPA对齐
    
    注意：SigLIP没有显式的CLS token，我们使用两种策略：
    1. 全局平均池化所有patch tokens
    2. 使用第一个token作为类CLS表示
    """
    
    def __init__(
        self, 
        vision_tower_name: str = "google/siglip-so400m-patch14-384",
        pooling_strategy: str = "mean",  # "mean" | "first_token" | "max"
        feature_dim: int = 1152,
        args=None
    ):
        """
        Args:
            vision_tower_name: SigLIP模型名称
            pooling_strategy: 池化策略
                - "mean": 平均池化所有patch tokens
                - "first_token": 使用第一个token
                - "max": 最大池化
            feature_dim: 输出特征维度（SigLIP-SO400M为1152）
            args: 额外参数
        """
        super().__init__()
        
        self.vision_tower_name = vision_tower_name
        self.pooling_strategy = pooling_strategy
        self.feature_dim = feature_dim
        self.is_loaded = False
        
        # 复用现有的SigLIP编码器
        self.siglip_encoder = SiglipVisionTower(
            vision_tower=vision_tower_name,
            args=args,
            delay_load=False
        )
        
        self.is_loaded = True
        
        print(f"✅ SigLIP全局特征提取器初始化完成")
        print(f"   - 模型: {vision_tower_name}")
        print(f"   - 池化策略: {pooling_strategy}")
        print(f"   - 特征维度: {feature_dim}")
    
    def _pool_features(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        对patch tokens进行池化得到全局特征
        
        Args:
            patch_tokens: (B, N, D) patch级别的特征
            
        Returns:
            global_features: (B, 1, D) 全局特征
        """
        if self.pooling_strategy == "mean":
            # 平均池化
            global_features = patch_tokens.mean(dim=1, keepdim=True)
        
        elif self.pooling_strategy == "first_token":
            # 使用第一个token
            global_features = patch_tokens[:, :1, :]
        
        elif self.pooling_strategy == "max":
            # 最大池化
            global_features = patch_tokens.max(dim=1, keepdim=True)[0]
        
        else:
            raise ValueError(f"未知的池化策略: {self.pooling_strategy}")
        
        return global_features
    
    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        提取全局特征（类CLS token）
        
        Args:
            images: (B, C, H, W) 输入图像
            
        Returns:
            global_features: (B, 1, D) 全局特征，维度与DINOv2 CLS token一致
        """
        if not self.is_loaded:
            raise RuntimeError("SigLIP encoder未加载")
        
        # 获取patch-level特征
        patch_tokens = self.siglip_encoder(images)  # (B, 729, 1152)
        
        # 池化得到全局特征
        global_features = self._pool_features(patch_tokens)  # (B, 1, 1152)
        
        return global_features
    
    @property
    def dtype(self):
        return self.siglip_encoder.dtype
    
    @property
    def device(self):
        return self.siglip_encoder.device
    
    @property
    def hidden_size(self):
        return self.feature_dim
    
    def print_model_info(self):
        """打印模型信息"""
        print("🔍 SigLIP全局特征提取器信息:")
        print(f"   - 模型名称: {self.vision_tower_name}")
        print(f"   - 池化策略: {self.pooling_strategy}")
        print(f"   - 输出维度: {self.feature_dim}")
        print(f"   - 输出格式: (B, 1, {self.feature_dim}) - 类CLS token")
        print(f"   - 设备: {self.device}")
        print(f"   - 数据类型: {self.dtype}")


def create_siglip_global_encoder(
    model_name: str = "google/siglip-so400m-patch14-384",
    pooling_strategy: str = "mean",
    feature_dim: int = 1152,
    device=None
):
    """
    工厂函数：创建SigLIP全局特征提取器
    
    Args:
        model_name: SigLIP模型名称
        pooling_strategy: 池化策略
        feature_dim: 特征维度
        device: 计算设备
        
    Returns:
        SiglipGlobalFeatureExtractor实例
    """
    print(f"🔧 创建SigLIP全局特征提取器")
    print(f"   - 模型: {model_name}")
    print(f"   - 池化策略: {pooling_strategy}")
    
    encoder = SiglipGlobalFeatureExtractor(
        vision_tower_name=model_name,
        pooling_strategy=pooling_strategy,
        feature_dim=feature_dim
    )
    
    if device is not None:
        encoder.to(device)
    
    return encoder


# 测试代码
if __name__ == "__main__":
    print("🧪 测试SigLIP全局特征提取器")
    
    # 创建编码器
    encoder = create_siglip_global_encoder()
    
    # 测试前向传播
    print("\n🔬 测试前向传播:")
    test_input = torch.randn(2, 3, 384, 384)
    
    global_features = encoder(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"全局特征形状: {global_features.shape}")
    print(f"期望形状: (2, 1, 1152)")
    
    assert global_features.shape == (2, 1, 1152), "输出形状不匹配！"
    
    print("\n✅ 所有测试通过！")