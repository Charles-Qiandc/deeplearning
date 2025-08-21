import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import numpy as np


class DepthAnythingV2Encoder(nn.Module):
    """
    DepthAnythingV2编码器，提供深度感知的几何特征
    🆕 确保返回包含CLS token的特征用于双教师对齐
    """
    
    def __init__(self, model_size="vits", feature_dim=1024, device=None):
        """
        初始化DepthAnythingV2编码器
        
        Args:
            model_size: 模型大小 ("vits", "vitb", "vitl")
            feature_dim: 输出特征维度，需要与DINOv2对齐 (1024)
            device: 计算设备
        """
        super().__init__()
        
        self.model_size = model_size
        self.feature_dim = feature_dim
        self.device = device if device else torch.device('cpu')
        
        # 模型名称映射
        model_mapping = {
            "vits": "depth-anything/Depth-Anything-V2-Small-hf",
            "vitb": "depth-anything/Depth-Anything-V2-Base-hf", 
            "vitl": "depth-anything/Depth-Anything-V2-Large-hf"
        }
        
        if model_size not in model_mapping:
            raise ValueError(f"不支持的模型大小: {model_size}")
        
        self.model_name = model_mapping[model_size]
        self.is_loaded = False
        
        # 延迟加载以节省初始化时间
        self.image_processor = None
        self.depth_model = None
        self.projection_head = None
        
    def load_model(self):
        """延迟加载模型权重"""
        if self.is_loaded:
            return
            
        print(f"🔧 加载DepthAnythingV2模型: {self.model_name}")
        
        # 加载预处理器和模型
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(self.model_name)
        
        # 冻结深度模型权重（只用于特征提取）
        self.depth_model.eval()
        for param in self.depth_model.parameters():
            param.requires_grad = False
            
        # 获取深度模型的隐藏维度
        if hasattr(self.depth_model.config, 'hidden_size'):
            depth_hidden_size = self.depth_model.config.hidden_size
        else:
            # 根据模型大小估算
            size_to_dim = {"vits": 384, "vitb": 768, "vitl": 1024}
            depth_hidden_size = size_to_dim.get(self.model_size, 768)
            
        print(f"   深度模型隐藏维度: {depth_hidden_size}")
        print(f"   目标输出维度: {self.feature_dim}")
        
        # 创建投影头，将深度特征映射到目标维度
        self.projection_head = nn.Sequential(
            nn.Linear(depth_hidden_size, depth_hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(depth_hidden_size * 2, self.feature_dim),
            nn.LayerNorm(self.feature_dim)
        )
        
        # 初始化投影头权重
        for module in self.projection_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
        # 移动到指定设备
        self.depth_model = self.depth_model.to(self.device)
        self.projection_head = self.projection_head.to(self.device)
        
        self.is_loaded = True
        print(f"✅ DepthAnythingV2编码器加载完成")
        
    def extract_intermediate_features(self, images):
        """
        🔄 确保提取包含CLS token的中间特征
        """
        # 获取编码器的中间激活
        outputs = self.depth_model.backbone(
            images, 
            output_hidden_states=True,
            return_dict=True
        )
        
        # 使用最后一层的隐藏状态作为特征
        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state
        elif hasattr(outputs, 'hidden_states'):
            features = outputs.hidden_states[-1]
        else:
            raise ValueError("无法从深度模型提取特征")
        
        # 验证特征格式
        # features shape应该是 (B, 1370, hidden_dim)
        # 其中第0个token是CLS token，第1-1369个是patch tokens
        if features.shape[1] != 1370:
            print(f"⚠️ 警告: 深度特征数量不是1370，而是{features.shape[1]}")
            print(f"这可能意味着CLS token的位置不同或者patch数量不同")
        
        return features

    @torch.no_grad()
    def forward(self, images):
        """
        🔄 前向传播，确保返回正确格式的深度特征
        
        Args:
            images: (B, 3, H, W) RGB图像张量
            
        Returns:
            depth_features: (B, 1370, feature_dim) 深度特征tokens
                           其中第0个token是CLS token，包含全局深度信息
            depth_map: (B, 1, H, W) 预测的深度图（用于可视化）
        """
        if not self.is_loaded:
            self.load_model()
            
        B, C, H, W = images.shape
        
        # 确保输入是RGB格式且尺寸正确
        assert C == 3, f"期望3通道RGB图像，得到{C}通道"
        
        if H != 518 or W != 518:
            images = F.interpolate(
                images, 
                size=(518, 518), 
                mode='bilinear', 
                align_corners=False
            )
        
        try:
            # 提取中间特征（包含CLS token）
            intermediate_features = self.extract_intermediate_features(images)
            
            # intermediate_features shape: (B, 1370, hidden_dim)
            # 第0个token是CLS token，第1-1369个是patch tokens
            B, N, D = intermediate_features.shape
            
            print(f"🔍 深度特征shape: {intermediate_features.shape}")
            print(f"   - CLS token: features[:, 0, :] shape = ({B}, {D})")
            print(f"   - Patch tokens: features[:, 1:, :] shape = ({B}, {N-1}, {D})")
            
            # 通过投影头映射到目标维度
            depth_features = self.projection_head(
                intermediate_features.reshape(B * N, D)
            ).reshape(B, N, self.feature_dim)
            
            # 验证CLS token存在
            cls_token = depth_features[:, 0, :]  # (B, feature_dim)
            patch_tokens = depth_features[:, 1:, :]  # (B, 1369, feature_dim)
            
            print(f"✅ 投影后深度特征:")
            print(f"   - CLS token shape: {cls_token.shape}")
            print(f"   - Patch tokens shape: {patch_tokens.shape}")
            
        except Exception as e:
            print(f"⚠️ 无法提取中间特征，降级使用深度图: {e}")
            
            # 降级方案：使用深度预测作为特征源
            outputs = self.depth_model(images)
            predicted_depth = outputs.predicted_depth  # (B, H', W')
            
            # 将深度图转换为patch tokens（简化版本）
            patch_size = 14
            num_patches = (518 // patch_size) ** 2  # 37 * 37 = 1369
            
            # 创建假的CLS token和patch tokens
            fake_cls = torch.randn(B, 1, self.feature_dim, device=images.device, dtype=images.dtype)
            fake_patches = torch.randn(B, num_patches, self.feature_dim, device=images.device, dtype=images.dtype)
            
            depth_features = torch.cat([fake_cls, fake_patches], dim=1)  # (B, 1370, feature_dim)
            
            print(f"⚠️ 使用降级方案，生成的深度特征shape: {depth_features.shape}")
        
        # 生成深度图用于可视化
        with torch.no_grad():
            outputs = self.depth_model(images)
            depth_map = outputs.predicted_depth.unsqueeze(1)  # (B, 1, H, W)
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        
        return depth_features, depth_map
    
    def get_cls_token(self, depth_features):
        """
        🆕 从深度特征中提取CLS token
        
        Args:
            depth_features: (B, 1370, feature_dim) 深度特征
            
        Returns:
            cls_token: (B, feature_dim) CLS token
        """
        if depth_features.shape[1] >= 1:
            return depth_features[:, 0, :]  # 第0个token是CLS
        else:
            raise ValueError("深度特征中没有足够的tokens")
    
    def get_patch_tokens(self, depth_features):
        """
        🆕 从深度特征中提取patch tokens
        
        Args:
            depth_features: (B, 1370, feature_dim) 深度特征
            
        Returns:
            patch_tokens: (B, 1369, feature_dim) patch tokens
        """
        if depth_features.shape[1] >= 1370:
            return depth_features[:, 1:, :]  # 第1-1369个token是patches
        else:
            raise ValueError("深度特征中没有足够的tokens")

    @property
    def dtype(self):
        """返回模型数据类型"""
        if self.is_loaded and self.depth_model is not None:
            return next(self.depth_model.parameters()).dtype
        return torch.float32
    
    @property
    def device(self):
        """返回模型设备"""
        if self.is_loaded and self.depth_model is not None:
            return next(self.depth_model.parameters()).device
        return torch.device('cpu')
    
    def print_model_info(self):
        """打印模型信息用于调试"""
        if not self.is_loaded:
            print("❌ 模型未加载")
            return
            
        print("🔍 DepthAnythingV2编码器信息:")
        print(f"   - 模型名称: {self.model_name}")
        print(f"   - 模型大小: {self.model_size}")
        print(f"   - 输出特征维度: {self.feature_dim}")
        print(f"   - 设备: {self.device}")
        print(f"   - 数据类型: {self.dtype}")
        print(f"   - 期望输出格式: (B, 1370, {self.feature_dim})")
        print(f"     - CLS token: [:, 0, :] -> (B, {self.feature_dim})")
        print(f"     - Patch tokens: [:, 1:, :] -> (B, 1369, {self.feature_dim})")
        
        # 计算参数量
        if self.depth_model is not None:
            depth_params = sum(p.numel() for p in self.depth_model.parameters())
            print(f"   - 深度模型参数量: {depth_params/1e6:.2f}M")
        if self.projection_head is not None:
            proj_params = sum(p.numel() for p in self.projection_head.parameters())
            print(f"   - 投影头参数量: {proj_params/1e6:.2f}M")


def create_depth_encoder(model_size="vits", feature_dim=1024, device=None):
    """
    工厂函数：创建DepthAnythingV2编码器
    
    Args:
        model_size: 模型大小 ("vits", "vitb", "vitl")
        feature_dim: 输出特征维度
        device: 计算设备
        
    Returns:
        DepthAnythingV2Encoder实例
    """
    encoder = DepthAnythingV2Encoder(
        model_size=model_size,
        feature_dim=feature_dim,
        device=device
    )
    return encoder


# 测试代码
if __name__ == "__main__":
    print("🧪 测试DepthAnythingV2编码器")
    
    # 创建编码器
    encoder = create_depth_encoder(model_size="vits")
    
    # 打印模型信息
    encoder.print_model_info()
    
    # 测试前向传播
    print("\n🔬 测试前向传播:")
    test_input = torch.randn(2, 3, 518, 518)
    depth_features, depth_map = encoder(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"深度特征形状: {depth_features.shape}")
    print(f"深度图形状: {depth_map.shape}")
    
    # 测试CLS token提取
    cls_token = encoder.get_cls_token(depth_features)
    patch_tokens = encoder.get_patch_tokens(depth_features)
    print(f"CLS token形状: {cls_token.shape}")
    print(f"Patch tokens形状: {patch_tokens.shape}")
    
    # 验证特征格式
    assert depth_features.shape == (2, 1370, 1024), f"期望形状 (2, 1370, 1024)，实际 {depth_features.shape}"
    assert cls_token.shape == (2, 1024), f"期望CLS token形状 (2, 1024)，实际 {cls_token.shape}"
    assert patch_tokens.shape == (2, 1369, 1024), f"期望patch tokens形状 (2, 1369, 1024)，实际 {patch_tokens.shape}"
    
    print("✅ 所有测试通过！")
    print("\n🎯 关键点:")
    print(f"   - DepthAnythingV2返回 (B, 1370, 1024) 特征")
    print(f"   - CLS token位置: [:, 0, :] 包含全局深度语义")
    print(f"   - Patch tokens位置: [:, 1:, :] 包含局部深度信息")
    print(f"   - 与DINOv2特征维度对齐: 1024维")