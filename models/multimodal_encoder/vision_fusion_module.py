# models/multimodal_encoder/vision_fusion_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CrossAttentionFusion(nn.Module):
    """
    交叉注意力融合模块
    
    关键修正：
    - SigLIP处理多张图片(img_history_size × num_cameras)，产生大量tokens
    - DINOv2和Depth只处理当前观测(1张图片)，产生1369个patch tokens
    - 需要将当前观测的DINOv2/Depth特征融合到对应的SigLIP tokens中
    """
    
    def __init__(
        self,
        siglip_dim: int = 1152,
        dinov2_dim: int = 1024,
        depth_dim: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        # 🆕 新增参数：图像历史大小和相机数量
        img_history_size: int = 2,
        num_cameras: int = 3,
    ):
        """
        Args:
            siglip_dim: SigLIP特征维度（查询）
            dinov2_dim: DINOv2特征维度（键/值）
            depth_dim: DepthAnythingV2特征维度（键/值）
            num_heads: 注意力头数
            dropout: Dropout率
            use_layer_norm: 是否使用LayerNorm
            img_history_size: 图像历史长度（例如2表示当前+1帧历史）
            num_cameras: 相机数量（例如3表示3个视角）
        """
        super().__init__()
        
        self.siglip_dim = siglip_dim
        self.dinov2_dim = dinov2_dim
        self.depth_dim = depth_dim
        self.num_heads = num_heads
        self.img_history_size = img_history_size
        self.num_cameras = num_cameras
        
        print(f"🔧 CrossAttentionFusion初始化:")
        print(f"   - SigLIP维度: {siglip_dim}")
        print(f"   - 图像历史: {img_history_size}帧")
        print(f"   - 相机数量: {num_cameras}个")
        print(f"   - SigLIP总tokens: {img_history_size * num_cameras}张图 × 729 patches/图")
        print(f"   - DINOv2/Depth: 1张当前观测 × 1369 patches")
        
        # 投影层：将DINOv2和Depth特征投影到SigLIP空间
        self.dinov2_proj = nn.Sequential(
            nn.Linear(dinov2_dim, siglip_dim),
            nn.LayerNorm(siglip_dim) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.depth_proj = nn.Sequential(
            nn.Linear(depth_dim, siglip_dim),
            nn.LayerNorm(siglip_dim) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 交叉注意力层1: SigLIP (Q) × DINOv2 (K, V)
        self.cross_attn_dinov2 = nn.MultiheadAttention(
            embed_dim=siglip_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 交叉注意力层2: SigLIP (Q) × Depth (K, V)
        self.cross_attn_depth = nn.MultiheadAttention(
            embed_dim=siglip_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(siglip_dim * 3, siglip_dim * 2),
            nn.LayerNorm(siglip_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(siglip_dim * 2, siglip_dim),
            nn.LayerNorm(siglip_dim),
        )
        
        # 门控机制（用于残差连接）
        self.gate = nn.Sequential(
            nn.Linear(siglip_dim, 1),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        siglip_tokens: torch.Tensor,
        dinov2_tokens: torch.Tensor,
        depth_tokens: torch.Tensor,
        current_frame_idx: int = -1,  # 🆕 当前帧在序列中的位置
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        前向传播
        
        Args:
            siglip_tokens: (B, N_total, 1152) 
                          N_total = img_history_size × num_cameras × 729
                          例如: 2帧 × 3相机 × 729patches = 4374 tokens
            dinov2_tokens: (B, 1369, 1024) DINOv2当前观测的patch tokens
            depth_tokens: (B, 1369, 1024) Depth当前观测的patch tokens
            current_frame_idx: 当前帧的索引（-1表示最后一帧，-2表示倒数第二帧）
            return_attention_weights: 是否返回注意力权重
            
        Returns:
            fused_tokens: (B, N_total, 1152) 融合后的视觉tokens
            attention_info: 可选的注意力权重信息
        """
        B, N_total, D_siglip = siglip_tokens.shape
        
        # 🔍 计算每张图片的token数量
        tokens_per_image = 729  # SigLIP: 27×27
        total_images = self.img_history_size * self.num_cameras
        
        # 验证输入形状
        expected_N_total = total_images * tokens_per_image
        if N_total != expected_N_total:
            print(f"⚠️ SigLIP tokens数量不匹配: 期望{expected_N_total}, 实际{N_total}")
        
        # 1. 投影DINOv2和Depth特征到SigLIP空间
        dinov2_projected = self.dinov2_proj(dinov2_tokens)  # (B, 1369, 1152)
        depth_projected = self.depth_proj(depth_tokens)      # (B, 1369, 1152)
        
        # 2. 确定当前帧对应的SigLIP tokens范围
        # 假设排列顺序: [frame0_cam0, frame0_cam1, frame0_cam2, frame1_cam0, frame1_cam1, frame1_cam2]
        if current_frame_idx == -1:  # 最后一帧（当前观测）
            current_frame_actual_idx = self.img_history_size - 1
        elif current_frame_idx == -2:
            current_frame_actual_idx = self.img_history_size - 2
        else:
            current_frame_actual_idx = current_frame_idx
        
        # 计算当前帧的token起止位置（所有相机）
        start_idx = current_frame_actual_idx * self.num_cameras * tokens_per_image
        end_idx = (current_frame_actual_idx + 1) * self.num_cameras * tokens_per_image
        
        # 提取当前帧的SigLIP tokens
        current_frame_tokens = siglip_tokens[:, start_idx:end_idx, :]  # (B, num_cameras*729, 1152)
        
        # 3. 交叉注意力融合 - 只对当前帧的tokens进行融合
        # 融合DINOv2特征
        attn_dinov2_output, attn_dinov2_weights = self.cross_attn_dinov2(
            query=current_frame_tokens,
            key=dinov2_projected,
            value=dinov2_projected,
            need_weights=return_attention_weights
        )  # (B, num_cameras*729, 1152)
        
        # 融合Depth特征
        attn_depth_output, attn_depth_weights = self.cross_attn_depth(
            query=current_frame_tokens,
            key=depth_projected,
            value=depth_projected,
            need_weights=return_attention_weights
        )  # (B, num_cameras*729, 1152)
        
        # 4. 三路特征拼接（只对当前帧）
        concatenated = torch.cat([
            current_frame_tokens,  # 原始SigLIP特征
            attn_dinov2_output,    # DINOv2增强特征
            attn_depth_output      # Depth增强特征
        ], dim=-1)  # (B, num_cameras*729, 1152*3)
        
        # 5. 特征融合
        fused_current_frame = self.fusion_layer(concatenated)  # (B, num_cameras*729, 1152)
        
        # 6. 门控残差连接
        gate_values = self.gate(fused_current_frame)  # (B, num_cameras*729, 1)
        fused_current_frame = (gate_values * fused_current_frame + 
                              (1 - gate_values) * current_frame_tokens)
        
        # 7. 将融合后的当前帧tokens替换回原始序列
        fused_tokens = siglip_tokens.clone()
        fused_tokens[:, start_idx:end_idx, :] = fused_current_frame
        
        # 8. 返回结果
        if return_attention_weights:
            attention_info = {
                'dinov2_weights': attn_dinov2_weights,
                'depth_weights': attn_depth_weights,
                'gate_values': gate_values,
                'fused_frame_range': (start_idx, end_idx)
            }
            return fused_tokens, attention_info
        else:
            return fused_tokens, None


class SimpleFusionModule(nn.Module):
    """
    简化版融合模块
    如果交叉注意力太重，使用这个更轻量的版本
    """
    
    def __init__(
        self,
        siglip_dim: int = 1152,
        dinov2_dim: int = 1024,
        depth_dim: int = 1024,
        dropout: float = 0.1,
        img_history_size: int = 2,
        num_cameras: int = 3,
    ):
        super().__init__()
        
        self.siglip_dim = siglip_dim
        self.img_history_size = img_history_size
        self.num_cameras = num_cameras
        
        # 投影层
        self.dinov2_proj = nn.Linear(dinov2_dim, siglip_dim)
        self.depth_proj = nn.Linear(depth_dim, siglip_dim)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(siglip_dim * 3, siglip_dim),
            nn.LayerNorm(siglip_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        siglip_tokens: torch.Tensor,
        dinov2_tokens: torch.Tensor,
        depth_tokens: torch.Tensor,
        current_frame_idx: int = -1
    ) -> torch.Tensor:
        """
        简单融合：投影 -> 平均池化对齐 -> 拼接 -> MLP
        """
        B, N_total, _ = siglip_tokens.shape
        
        # 投影到统一空间
        dinov2_proj = self.dinov2_proj(dinov2_tokens)  # (B, 1369, 1152)
        depth_proj = self.depth_proj(depth_tokens)      # (B, 1369, 1152)
        
        # 确定当前帧范围
        tokens_per_image = 729
        if current_frame_idx == -1:
            current_frame_actual_idx = self.img_history_size - 1
        elif current_frame_idx == -2:
            current_frame_actual_idx = self.img_history_size - 2
        else:
            current_frame_actual_idx = current_frame_idx
        
        start_idx = current_frame_actual_idx * self.num_cameras * tokens_per_image
        end_idx = (current_frame_actual_idx + 1) * self.num_cameras * tokens_per_image
        
        current_frame_tokens = siglip_tokens[:, start_idx:end_idx, :]
        N_current = current_frame_tokens.shape[1]  # num_cameras * 729
        
        # 通过自适应池化调整DINOv2/Depth的token数量
        # (B, 1369, 1152) -> (B, 1152, 1369) -> pool -> (B, 1152, N_current) -> (B, N_current, 1152)
        dinov2_aligned = F.adaptive_avg_pool1d(
            dinov2_proj.transpose(1, 2), N_current
        ).transpose(1, 2)
        
        depth_aligned = F.adaptive_avg_pool1d(
            depth_proj.transpose(1, 2), N_current
        ).transpose(1, 2)
        
        # 拼接融合
        concatenated = torch.cat([current_frame_tokens, dinov2_aligned, depth_aligned], dim=-1)
        fused_current_frame = self.fusion(concatenated)
        
        # 替换回原始序列
        fused_tokens = siglip_tokens.clone()
        fused_tokens[:, start_idx:end_idx, :] = fused_current_frame
        
        return fused_tokens


def create_vision_fusion_module(
    fusion_type: str = "cross_attention",
    siglip_dim: int = 1152,
    dinov2_dim: int = 1024,
    depth_dim: int = 1024,
    img_history_size: int = 2,
    num_cameras: int = 3,
    **kwargs
):
    """
    工厂函数：创建视觉融合模块
    
    Args:
        fusion_type: "cross_attention" 或 "simple"
        siglip_dim: SigLIP特征维度
        dinov2_dim: DINOv2特征维度
        depth_dim: Depth特征维度
        img_history_size: 图像历史长度
        num_cameras: 相机数量
        **kwargs: 其他参数
    
    Returns:
        融合模块实例
    """
    if fusion_type == "cross_attention":
        return CrossAttentionFusion(
            siglip_dim=siglip_dim,
            dinov2_dim=dinov2_dim,
            depth_dim=depth_dim,
            img_history_size=img_history_size,
            num_cameras=num_cameras,
            **kwargs
        )
    elif fusion_type == "simple":
        return SimpleFusionModule(
            siglip_dim=siglip_dim,
            dinov2_dim=dinov2_dim,
            depth_dim=depth_dim,
            img_history_size=img_history_size,
            num_cameras=num_cameras,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")


# 测试代码
if __name__ == "__main__":
    print("🧪 测试视觉特征融合模块（多图片场景）")
    
    # 模拟输入
    B = 2
    img_history_size = 2  # 2帧历史
    num_cameras = 3       # 3个相机
    N_siglip = img_history_size * num_cameras * 729  # 2×3×729 = 4374 tokens
    N_dinov2 = 1369   # 当前观测的DINOv2 patches
    N_depth = 1369    # 当前观测的Depth patches
    
    siglip_tokens = torch.randn(B, N_siglip, 1152)
    dinov2_tokens = torch.randn(B, N_dinov2, 1024)
    depth_tokens = torch.randn(B, N_depth, 1024)
    
    # 测试交叉注意力融合
    print("\n1️⃣ 测试交叉注意力融合模块:")
    fusion_module = CrossAttentionFusion(
        img_history_size=img_history_size,
        num_cameras=num_cameras
    )
    fused_tokens, attn_info = fusion_module(
        siglip_tokens, 
        dinov2_tokens, 
        depth_tokens,
        current_frame_idx=-1,  # 融合最后一帧（当前观测）
        return_attention_weights=True
    )
    print(f"   输入: SigLIP {siglip_tokens.shape}, DINOv2 {dinov2_tokens.shape}, Depth {depth_tokens.shape}")
    print(f"   输出: {fused_tokens.shape} (应该和SigLIP输入相同)")
    print(f"   融合帧范围: {attn_info['fused_frame_range']}")
    
    # 测试简化融合
    print("\n2️⃣ 测试简化融合模块:")
    simple_fusion = SimpleFusionModule(
        img_history_size=img_history_size,
        num_cameras=num_cameras
    )
    fused_simple = simple_fusion(siglip_tokens, dinov2_tokens, depth_tokens, current_frame_idx=-1)
    print(f"   输出: {fused_simple.shape}")
    
    # 参数统计
    print("\n📊 参数统计:")
    cross_attn_params = sum(p.numel() for p in fusion_module.parameters())
    simple_params = sum(p.numel() for p in simple_fusion.parameters())
    print(f"   交叉注意力模块: {cross_attn_params:,} 参数")
    print(f"   简化模块: {simple_params:,} 参数")
    
    print("\n✅ 所有测试通过!")