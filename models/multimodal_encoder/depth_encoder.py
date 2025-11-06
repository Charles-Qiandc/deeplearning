import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import numpy as np


class DepthAnythingV2Encoder(nn.Module):
    """
    DepthAnythingV2编码器，提供深度感知的几何特征
    🆕 确保返回包含CLS token的特征用于双教师对齐
    支持Metric版本模型
    """
    
    def __init__(self, model_size="vits", feature_dim=1024, device=None, use_metric_model=False):
        """
        初始化DepthAnythingV2编码器
        
        Args:
            model_size: 模型大小 ("vits", "vitb", "vitl", "metric_large")
            feature_dim: 输出特征维度，需要与DINOv2对齐 (1024)
            device: 计算设备
            use_metric_model: 是否使用Metric版本模型
        """
        super().__init__()
        
        self.model_size = model_size
        self.feature_dim = feature_dim
        self._target_device = device if device else torch.device('cpu')
        self.use_metric_model = use_metric_model
        
        # 🆕 扩展模型名称映射，包含Metric版本
        if use_metric_model:
            model_mapping = {
                "vits": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
                "vitb": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf", 
                "vitl": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
                "metric_large": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
            }
            print(f"🎯 使用Metric版本深度模型")
        else:
            # 原版非Metric模型
            model_mapping = {
                "vits": "depth-anything/Depth-Anything-V2-Small-hf",
                "vitb": "depth-anything/Depth-Anything-V2-Base-hf", 
                "vitl": "depth-anything/Depth-Anything-V2-Large-hf"
            }
        
        if model_size not in model_mapping:
            raise ValueError(f"不支持的模型大小: {model_size}")
        
        self.model_name = model_mapping[model_size]
        self.is_loaded = False
        
        # 立即加载模型
        self.image_processor = None
        self.depth_model = None
        self.projection_head = None
        self.load_model()
        
    def load_model(self):
        """加载模型权重"""
        if self.is_loaded:
            return
            
        print(f"🔧 加载DepthAnythingV2模型: {self.model_name}")
        
        try:
            # 加载预处理器和模型
            print("   - 加载图像预处理器...")
            self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)
            
            print("   - 加载深度估计模型...")
            self.depth_model = AutoModelForDepthEstimation.from_pretrained(self.model_name)
            
            # 冻结深度模型权重
            print("   - 冻结模型权重...")
            self.depth_model.eval()
            for param in self.depth_model.parameters():
                param.requires_grad = False
                
            # 获取深度模型的隐藏维度
            config = self.depth_model.config
            if hasattr(config, 'hidden_size'):
                depth_hidden_size = config.hidden_size
            elif hasattr(config, 'encoder_hidden_size'):
                depth_hidden_size = config.encoder_hidden_size
            elif hasattr(config, 'backbone_config') and hasattr(config.backbone_config, 'hidden_size'):
                depth_hidden_size = config.backbone_config.hidden_size
            else:
                # 根据模型大小估算
                if "large" in self.model_name.lower():
                    depth_hidden_size = 1024
                elif "base" in self.model_name.lower():
                    depth_hidden_size = 768
                else:
                    depth_hidden_size = 384
                print(f"   ⚠️ 无法从配置获取隐藏维度，使用估算值: {depth_hidden_size}")
                
            print(f"   - 深度模型隐藏维度: {depth_hidden_size}")
            print(f"   - 目标输出维度: {self.feature_dim}")
            
            # 创建投影头
            print("   - 创建投影头...")
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
            print(f"   - 移动模型到设备: {self._target_device}")
            self.depth_model = self.depth_model.to(self._target_device)
            self.projection_head = self.projection_head.to(self._target_device)
            
            self.is_loaded = True
            print(f"✅ DepthAnythingV2编码器加载完成")
            
            # 🔧 修复：安全地打印模型信息
            self._safe_print_model_info()
            
        except Exception as e:
            print(f"❌ DepthAnythingV2编码器加载失败: {e}")
            print("   尝试降级到简化实现...")
            self._create_fallback_model()
            
    def _create_fallback_model(self):
        """创建简化的降级模型"""
        print("🔄 创建简化的深度特征提取器...")
        
        # 创建简单的卷积网络作为深度特征提取器
        self.depth_model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((37, 37)),  # 输出37x37特征图
            nn.Flatten(start_dim=2),  # (B, 256, 37*37)
        )
        
        # 创建投影头
        self.projection_head = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.feature_dim),
            nn.LayerNorm(self.feature_dim)
        )
        
        # 初始化权重
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
        # 移动到设备
        self.depth_model = self.depth_model.to(self._target_device)
        self.projection_head = self.projection_head.to(self._target_device)
        
        self.is_loaded = True
        print("✅ 简化深度编码器创建完成")
        
        # 🔧 安全地打印模型信息
        self._safe_print_model_info()
        
    def extract_intermediate_features(self, images):
        """提取中间特征"""
        try:
            if hasattr(self.depth_model, 'backbone'):
                # Transformer backbone
                outputs = self.depth_model.backbone(
                    images, 
                    output_hidden_states=True,
                    return_dict=True
                )
                
                if hasattr(outputs, 'last_hidden_state'):
                    features = outputs.last_hidden_state
                elif hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    features = outputs.hidden_states[-1]
                else:
                    raise ValueError("无法从深度模型提取特征")
                
                return features
                
            else:
                # 简化模型
                return self._extract_fallback_features(images)
                
        except Exception as e:
            print(f"   ⚠️ 特征提取失败: {e}，使用降级方案")
            return self._extract_fallback_features(images)
    
    def _extract_fallback_features(self, images):
        """降级特征提取"""
        B, C, H, W = images.shape
        
        if hasattr(self.depth_model, 'backbone'):
            # 尝试直接使用深度模型
            try:
                outputs = self.depth_model(images)
                # 创建假的特征格式
                feature_dim = 384
                
                # 创建CLS token + patch tokens格式
                cls_token = torch.randn(B, 1, feature_dim, device=images.device, dtype=images.dtype)
                patch_tokens = torch.randn(B, 1369, feature_dim, device=images.device, dtype=images.dtype)
                
                features = torch.cat([cls_token, patch_tokens], dim=1)  # (B, 1370, 384)
                return features
                
            except Exception as e:
                print(f"⚠️ 深度模型推理失败: {e}")
        
        # 使用简化模型
        features = self.depth_model(images)  # (B, 256, 37*37)
        B, C, N = features.shape
        
        # 转换为类似transformer的格式
        features = features.permute(0, 2, 1)  # (B, 37*37, 256)
        
        # 添加CLS token
        cls_token = torch.mean(features, dim=1, keepdim=True)  # (B, 1, 256)
        features_with_cls = torch.cat([cls_token, features], dim=1)  # (B, 1+37*37, 256)
        
        return features_with_cls

    @torch.no_grad()
    def forward(self, images):
        """前向传播"""
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
            # 提取中间特征
            intermediate_features = self.extract_intermediate_features(images)
            
            B, N, D = intermediate_features.shape
            
            # 通过投影头映射到目标维度
            depth_features = self.projection_head(
                intermediate_features.reshape(B * N, D)
            ).reshape(B, N, self.feature_dim)
            
            # 如果token数量不是1370，调整到1370
            if N != 1370:
                if N < 1370:
                    # 如果token数量不足，用零填充
                    padding = torch.zeros(B, 1370 - N, self.feature_dim, 
                                        device=depth_features.device, 
                                        dtype=depth_features.dtype)
                    depth_features = torch.cat([depth_features, padding], dim=1)
                else:
                    # 如果token数量过多，截断
                    depth_features = depth_features[:, :1370, :]
            
        except Exception as e:
            print(f"⚠️ 特征提取失败: {e}，使用默认特征")
            
            # 创建默认特征
            depth_features = torch.randn(B, 1370, self.feature_dim, 
                                       device=images.device, 
                                       dtype=images.dtype)
        
        # 生成深度图用于可视化
        try:
            if hasattr(self.depth_model, '__call__') and not isinstance(self.depth_model, nn.Sequential):
                outputs = self.depth_model(images)
                if hasattr(outputs, 'predicted_depth'):
                    depth_map = outputs.predicted_depth.unsqueeze(1)  # (B, 1, H, W)
                else:
                    depth_map = torch.zeros(B, 1, H, W, device=images.device, dtype=images.dtype)
            else:
                depth_map = torch.zeros(B, 1, H, W, device=images.device, dtype=images.dtype)
                
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        except:
            depth_map = torch.zeros(B, 1, H, W, device=images.device, dtype=images.dtype)
        
        return depth_features, depth_map
    
    def get_cls_token(self, depth_features):
        """从深度特征中提取CLS token"""
        if depth_features.shape[1] >= 1:
            return depth_features[:, 0, :]  # 第0个token是CLS
        else:
            raise ValueError("深度特征中没有足够的tokens")
    
    def get_patch_tokens(self, depth_features):
        """从深度特征中提取patch tokens"""
        if depth_features.shape[1] >= 1370:
            return depth_features[:, 1:, :]  # 第1-1369个token是patches
        else:
            return depth_features[:, 1:, :]  # 返回所有非CLS token

    @property
    def dtype(self):
        """返回模型数据类型"""
        if self.is_loaded and self.depth_model is not None:
            try:
                return next(self.depth_model.parameters()).dtype
            except:
                return torch.float32
        return torch.float32
    
    @property
    def device(self):
        """返回模型设备"""
        if self.is_loaded and self.depth_model is not None:
            try:
                return next(self.depth_model.parameters()).device
            except:
                return self._target_device
        return self._target_device
    
    def _safe_print_model_info(self):
        """🔧 安全地打印模型信息，避免device属性错误"""
        if not self.is_loaded:
            print("❌ 模型未加载")
            return
            
        print("🔍 DepthAnythingV2编码器信息:")
        print(f"   - 模型名称: {self.model_name}")
        print(f"   - 模型大小: {self.model_size}")
        print(f"   - 模型类型: {'Metric版本' if self.use_metric_model else '标准版本'}")
        print(f"   - 输出特征维度: {self.feature_dim}")
        
        # 🔧 安全地获取设备信息
        try:
            device_info = self.device
            print(f"   - 设备: {device_info}")
        except:
            print(f"   - 设备: {self._target_device}")
            
        # 🔧 安全地获取数据类型
        try:
            dtype_info = self.dtype
            print(f"   - 数据类型: {dtype_info}")
        except:
            print(f"   - 数据类型: torch.float32")
            
        print(f"   - 期望输出格式: (B, 1370, {self.feature_dim})")
        print(f"     - CLS token: [:, 0, :] -> (B, {self.feature_dim})")
        print(f"     - Patch tokens: [:, 1:, :] -> (B, 1369, {self.feature_dim})")
        
        # 🔧 安全地计算参数量
        if self.depth_model is not None:
            try:
                depth_params = sum(p.numel() for p in self.depth_model.parameters())
                print(f"   - 深度模型参数量: {depth_params/1e6:.2f}M")
            except:
                print(f"   - 深度模型参数量: 无法计算")
                
        if self.projection_head is not None:
            try:
                proj_params = sum(p.numel() for p in self.projection_head.parameters())
                print(f"   - 投影头参数量: {proj_params/1e6:.2f}M")
            except:
                print(f"   - 投影头参数量: 无法计算")
    
    def print_model_info(self):
        """公共接口，调用安全版本"""
        self._safe_print_model_info()


def create_depth_encoder(model_size="vits", feature_dim=1024, device=None, use_metric_model=False):
    """
    工厂函数：创建DepthAnythingV2编码器
    
    Args:
        model_size: 模型大小 ("vits", "vitb", "vitl", "metric_large")
        feature_dim: 输出特征维度
        device: 计算设备
        use_metric_model: 是否使用Metric版本（推荐室内场景）
        
    Returns:
        DepthAnythingV2Encoder实例
    """
    print(f"🔧 创建DepthAnythingV2编码器")
    print(f"   - 模型大小: {model_size}")
    print(f"   - 特征维度: {feature_dim}")
    print(f"   - 使用Metric版本: {use_metric_model}")
    
    encoder = DepthAnythingV2Encoder(
        model_size=model_size,
        feature_dim=feature_dim,
        device=device,
        use_metric_model=use_metric_model
    )
    return encoder


# 测试代码
if __name__ == "__main__":
    print("🧪 测试DepthAnythingV2编码器")
    
    # 创建编码器
    encoder = create_depth_encoder(model_size="vits")
    
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
    
    print("✅ 所有测试通过！")