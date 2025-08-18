
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import numpy as np


class DepthAnythingV2Encoder(nn.Module):
    """
    DepthAnythingV2编码器，提供深度感知的几何特征
    用于RDT的混合专家路由对齐机制
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
        提取中间层特征而不是最终深度图
        这些特征包含更丰富的几何信息
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
            
        return features
    
    @torch.no_grad()
    def forward(self, images):
        """
        前向传播，提取深度感知的几何特征
        
        Args:
            images: (B, 3, H, W) RGB图像张量
            
        Returns:
            depth_features: (B, N_patches, feature_dim) 深度特征tokens
            depth_map: (B, 1, H, W) 预测的深度图（可选）
        """
        if not self.is_loaded:
            self.load_model()
            
        B, C, H, W = images.shape
        
        # 确保输入是RGB格式
        assert C == 3, f"期望3通道RGB图像，得到{C}通道"
        
        # 预处理图像
        # DepthAnything期望的输入尺寸通常是518x518
        if H != 518 or W != 518:
            images = F.interpolate(
                images, 
                size=(518, 518), 
                mode='bilinear', 
                align_corners=False
            )
        
        # 提取中间特征
        try:
            # 方案1：尝试直接提取中间特征
            intermediate_features = self.extract_intermediate_features(images)
            
            # intermediate_features shape: (B, num_patches, hidden_dim)
            B, N, D = intermediate_features.shape
            
            # 通过投影头映射到目标维度
            depth_features = self.projection_head(
                intermediate_features.reshape(B * N, D)
            ).reshape(B, N, self.feature_dim)
            
        except Exception as e:
            print(f"⚠️ 无法提取中间特征，降级使用深度图: {e}")
            
            # 方案2：使用深度预测作为特征源
            outputs = self.depth_model(images)
            predicted_depth = outputs.predicted_depth  # (B, H', W')
            
            # 将深度图转换为patch tokens
            # 假设使用14x14的patch size (类似ViT)
            patch_size = 14
            num_patches = (518 // patch_size) ** 2  # 37 * 37 = 1369
            
            # 重塑深度图为patches
            depth_patches = F.unfold(
                predicted_depth.unsqueeze(1),  # (B, 1, H, W)
                kernel_size=patch_size,
                stride=patch_size
            )  # (B, patch_size^2, num_patches)
            
            depth_patches = depth_patches.transpose(1, 2)  # (B, num_patches, patch_size^2)
            
            # 创建简单的patch嵌入
            patch_embed = nn.Linear(patch_size * patch_size, self.feature_dim).to(self.device)
            depth_features = patch_embed(depth_patches)
            
            # 应用投影头进行特征细化
            B, N, D = depth_features.shape
            depth_features = self.projection_head(
                depth_features.reshape(B * N, D)
            ).reshape(B, N, self.feature_dim)
        
        # 同时返回深度图用于可视化/调试
        with torch.no_grad():
            outputs = self.depth_model(images)
            depth_map = outputs.predicted_depth.unsqueeze(1)  # (B, 1, H, W)
            
            # 归一化深度图到[0, 1]
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        
        return depth_features, depth_map
    
    def get_cls_token(self, depth_features):
        """
        从深度特征中提取类似CLS token的全局表示
        
        Args:
            depth_features: (B, N_patches, feature_dim)
            
        Returns:
            cls_token: (B, 1, feature_dim)
        """
        # 使用平均池化作为全局表示
        cls_token = depth_features.mean(dim=1, keepdim=True)  # (B, 1, feature_dim)
        return cls_token
    
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
        
        # 计算参数量
        depth_params = sum(p.numel() for p in self.depth_model.parameters())
        proj_params = sum(p.numel() for p in self.projection_head.parameters())
        print(f"   - 深度模型参数量: {depth_params/1e6:.2f}M")
        print(f"   - 投影头参数量: {proj_params/1e6:.2f}M")


def create_depth_encoder(model_size="vits", feature_dim=1024, device=None):
    """
    工厂函数：创建DepthAnythingV2编码器
    
    Args:
        model_size: 模型大小
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
    # 创建编码器
    encoder = create_depth_encoder(model_size="vits")
    encoder.print_model_info()
    
    # 测试前向传播
    test_input = torch.randn(2, 3, 518, 518)
    depth_features, depth_map = encoder(test_input)
    
    print(f"\n测试结果:")
    print(f"  输入形状: {test_input.shape}")
    print(f"  深度特征形状: {depth_features.shape}")
    print(f"  深度图形状: {depth_map.shape}")
    
    # 测试CLS token提取
    cls_token = encoder.get_cls_token(depth_features)
    print(f"  CLS token形状: {cls_token.shape}")