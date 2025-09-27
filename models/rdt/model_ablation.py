# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
from collections import OrderedDict

import torch
import torch.nn as nn

from models.rdt.blocks import (FinalLayer, RDTBlock, TimestepEmbedder,
                               get_1d_sincos_pos_embed_from_grid,
                               get_multimodal_cond_pos_embed)


class RDTAblation(nn.Module):
    """
    消融实验版RDT模型：移除REPA对齐，将DINOv2和DepthAnything的特征注入DIT输入侧
    
    关键改动：
    1. 移除所有REPA相关组件
    2. 添加DINOv2/DepthAnything特征的输入投影器
    3. 在图像条件中融合多模态视觉特征
    4. 保持原有的transformer结构不变
    """

    def __init__(
        self,
        output_dim=128,
        horizon=64,
        hidden_size=2048,
        depth=28,
        num_heads=32,
        max_lang_cond_len=1024,
        img_cond_len=4096,
        lang_pos_embed_config=None,
        img_pos_embed_config=None,
        dtype=torch.bfloat16,
        # 🆕 多模态视觉特征参数
        use_dinov2_features=False,
        dinov2_feature_dim=1024,
        use_depth_features=False,
        depth_feature_dim=1024,
        # 🆕 特征融合策略
        visual_fusion_strategy="concat",  # "concat", "add", "gate"
    ):
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.max_lang_cond_len = max_lang_cond_len
        self.img_cond_len = img_cond_len
        self.dtype = dtype
        self.lang_pos_embed_config = lang_pos_embed_config
        self.img_pos_embed_config = img_pos_embed_config
        
        # 🆕 多模态视觉特征配置
        self.use_dinov2_features = use_dinov2_features
        self.dinov2_feature_dim = dinov2_feature_dim
        self.use_depth_features = use_depth_features
        self.depth_feature_dim = depth_feature_dim
        self.visual_fusion_strategy = visual_fusion_strategy
        
        print(f"🔧 RDT消融实验模型初始化:")
        print(f"   - DINOv2特征: {use_dinov2_features}")
        print(f"   - 深度特征: {use_depth_features}")
        print(f"   - 视觉融合策略: {visual_fusion_strategy}")

        # 嵌入器组件
        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        self.freq_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        
        # 🆕 计算实际的图像条件长度（包含额外的视觉特征）
        self.actual_img_cond_len = img_cond_len
        extra_tokens = 0
        if use_dinov2_features:
            extra_tokens += 1  # DINOv2 CLS token
        if use_depth_features:
            extra_tokens += 1  # DepthAnything CLS token
        self.actual_img_cond_len += extra_tokens
        
        print(f"   - 原始图像条件长度: {img_cond_len}")
        print(f"   - 额外视觉tokens: {extra_tokens}")
        print(f"   - 实际图像条件长度: {self.actual_img_cond_len}")
        
        # 位置编码参数
        self.x_pos_embed = nn.Parameter(torch.zeros(1, horizon+3, hidden_size))
        self.lang_cond_pos_embed = nn.Parameter(torch.zeros(1, max_lang_cond_len, hidden_size))
        self.img_cond_pos_embed = nn.Parameter(torch.zeros(1, self.actual_img_cond_len, hidden_size))

        # Transformer块
        self.blocks = nn.ModuleList([
            RDTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        
        # 🆕 多模态视觉特征投影器
        self.extra_visual_projectors = nn.ModuleDict()
        
        if use_dinov2_features:
            # DINOv2特征投影到hidden_size维度
            self.extra_visual_projectors['dinov2'] = nn.Sequential(
                nn.Linear(dinov2_feature_dim, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, hidden_size),
                nn.LayerNorm(hidden_size),
            )
            print(f"   - DINOv2投影器: {dinov2_feature_dim} -> {hidden_size}")
            
        if use_depth_features:
            # DepthAnything特征投影到hidden_size维度
            self.extra_visual_projectors['depth'] = nn.Sequential(
                nn.Linear(depth_feature_dim, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, hidden_size),
                nn.LayerNorm(hidden_size),
            )
            print(f"   - 深度特征投影器: {depth_feature_dim} -> {hidden_size}")
        
        # 🆕 视觉特征融合层（如果使用gate策略）
        if visual_fusion_strategy == "gate" and (use_dinov2_features or use_depth_features):
            fusion_input_dim = 1  # SigLIP默认
            if use_dinov2_features:
                fusion_input_dim += 1
            if use_depth_features:
                fusion_input_dim += 1
                
            self.visual_fusion_gate = nn.Sequential(
                nn.Linear(hidden_size * fusion_input_dim, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, fusion_input_dim),
                nn.Softmax(dim=-1)
            )
            print(f"   - 门控融合层: {fusion_input_dim}个视觉源")
        
        # 最终层
        self.final_layer = FinalLayer(hidden_size, output_dim)
        self.initialize_weights()

    def initialize_weights(self):
        """初始化权重"""
        # 基础权重初始化
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # 位置编码初始化
        x_pos_embed = get_multimodal_cond_pos_embed(
            embed_dim=self.hidden_size,
            mm_cond_lens=OrderedDict([
                ('timestep', 1),
                ('ctrl_freq', 1),
                ('state', 1),
                ('action', self.horizon),
            ])
        )
        self.x_pos_embed.data.copy_(torch.from_numpy(x_pos_embed).float().unsqueeze(0))

        # 语言条件位置编码
        if self.lang_pos_embed_config is None:
            lang_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(
                self.hidden_size, torch.arange(self.max_lang_cond_len))
        else:
            lang_cond_pos_embed = get_multimodal_cond_pos_embed(
                embed_dim=self.hidden_size,
                mm_cond_lens=OrderedDict(self.lang_pos_embed_config),
                embed_modality=False
            )
        self.lang_cond_pos_embed.data.copy_(
            torch.from_numpy(lang_cond_pos_embed).float().unsqueeze(0))
        
        # 🆕 图像条件位置编码（包含额外的视觉特征）
        if self.img_pos_embed_config is None:
            img_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(
                self.hidden_size, torch.arange(self.actual_img_cond_len))
        else:
            # 构造包含额外视觉特征的配置
            extended_config = OrderedDict(self.img_pos_embed_config)
            if self.use_dinov2_features:
                extended_config["dinov2_cls"] = 1
            if self.use_depth_features:
                extended_config["depth_cls"] = 1
                
            img_cond_pos_embed = get_multimodal_cond_pos_embed(
                embed_dim=self.hidden_size,
                mm_cond_lens=extended_config,
                embed_modality=False
            )
        self.img_cond_pos_embed.data.copy_(
            torch.from_numpy(img_cond_pos_embed).float().unsqueeze(0))

        # 时间步和频率嵌入器初始化
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.freq_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.freq_embedder.mlp[2].weight, std=0.02)
        
        # 🆕 多模态投影器初始化
        for projector_name, projector in self.extra_visual_projectors.items():
            for module in projector:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            print(f"   - {projector_name}投影器初始化完成")
        
        # 🆕 融合层初始化
        if hasattr(self, 'visual_fusion_gate'):
            for module in self.visual_fusion_gate:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
        
        # 最终层零初始化
        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)
        
        # 转换到指定数据类型
        self.to(self.dtype)

    def fuse_visual_features(self, img_c, dinov2_features=None, depth_features=None):
        """
        融合多模态视觉特征到图像条件中
        
        Args:
            img_c: (B, N_siglip, D) SigLIP特征
            dinov2_features: (B, 1, D) DINOv2 CLS token特征
            depth_features: (B, 1, D) DepthAnything CLS token特征
            
        Returns:
            fused_img_c: (B, N_total, D) 融合后的视觉特征
        """
        B, N_siglip, D = img_c.shape
        
        # 收集所有视觉特征
        visual_features = [img_c]  # SigLIP patch tokens
        feature_names = ["siglip"]
        
        if dinov2_features is not None and self.use_dinov2_features:
            # 投影DINOv2特征
            projected_dinov2 = self.extra_visual_projectors['dinov2'](dinov2_features)
            visual_features.append(projected_dinov2)
            feature_names.append("dinov2")
            
        if depth_features is not None and self.use_depth_features:
            # 投影深度特征
            projected_depth = self.extra_visual_projectors['depth'](depth_features)
            visual_features.append(projected_depth)
            feature_names.append("depth")
        
        # 根据融合策略组合特征
        if self.visual_fusion_strategy == "concat":
            # 直接拼接：[SigLIP_patches, DINOv2_cls, Depth_cls]
            fused_features = torch.cat(visual_features, dim=1)
            
        elif self.visual_fusion_strategy == "add":
            # 加权求和（只对相同长度的特征）
            # SigLIP作为基础，额外特征broadcast到所有位置
            fused_features = img_c.clone()
            
            if dinov2_features is not None and self.use_dinov2_features:
                projected_dinov2 = self.extra_visual_projectors['dinov2'](dinov2_features)
                # Broadcast到所有SigLIP位置
                fused_features += projected_dinov2.expand(-1, N_siglip, -1)
                
            if depth_features is not None and self.use_depth_features:
                projected_depth = self.extra_visual_projectors['depth'](depth_features)
                # Broadcast到所有SigLIP位置
                fused_features += projected_depth.expand(-1, N_siglip, -1)
                
        elif self.visual_fusion_strategy == "gate":
            # 门控融合
            # 首先对所有特征进行平均池化得到global representation
            global_features = []
            for feat in visual_features:
                global_feat = feat.mean(dim=1, keepdim=True)  # (B, 1, D)
                global_features.append(global_feat)
            
            # 计算门控权重
            concat_global = torch.cat(global_features, dim=-1)  # (B, 1, D*num_features)
            gates = self.visual_fusion_gate(concat_global)  # (B, 1, num_features)
            
            # 应用门控权重
            weighted_features = []
            for i, feat in enumerate(visual_features):
                weight = gates[:, :, i:i+1]  # (B, 1, 1)
                if feat.shape[1] == 1:  # CLS token
                    weighted_feat = feat * weight
                else:  # patch tokens
                    weighted_feat = feat * weight.expand(-1, feat.shape[1], -1)
                weighted_features.append(weighted_feat)
            
            # 拼接加权特征
            fused_features = torch.cat(weighted_features, dim=1)
            
        else:
            raise ValueError(f"未支持的视觉融合策略: {self.visual_fusion_strategy}")
        
        # 验证输出形状
        expected_len = self.actual_img_cond_len
        actual_len = fused_features.shape[1]
        if actual_len != expected_len:
            print(f"⚠️ 视觉特征长度不匹配: {actual_len} vs 期望的 {expected_len}")
            # 调整到期望长度
            if actual_len > expected_len:
                fused_features = fused_features[:, :expected_len, :]
            else:
                padding = torch.zeros(B, expected_len - actual_len, D, 
                                    device=fused_features.device, 
                                    dtype=fused_features.dtype)
                fused_features = torch.cat([fused_features, padding], dim=1)
        
        print(f"✅ 视觉特征融合完成: {feature_names} -> {fused_features.shape}")
        return fused_features

    def forward(self, x, freq, t, lang_c, img_c, lang_mask=None, img_mask=None,
                dinov2_features=None, depth_features=None):
        """
        前向传播：融合多模态视觉特征
        
        Args:
            x, freq, t, lang_c: 标准DIT输入
            img_c: (B, N_siglip, D) SigLIP图像特征
            dinov2_features: (B, 1, D) DINOv2 CLS特征
            depth_features: (B, 1, D) DepthAnything CLS特征
            
        Returns:
            x: (B, horizon, output_dim) 最终预测
        """
        # 时间步和频率嵌入
        t = self.t_embedder(t).unsqueeze(1)             # (B, 1, D)
        freq = self.freq_embedder(freq).unsqueeze(1)    # (B, 1, D)
        
        # 处理时间步广播
        if t.shape[0] == 1:
            t = t.expand(x.shape[0], -1, -1)
        x = torch.cat([t, freq, x], dim=1)               # (B, T+2, D)
        
        # 🆕 融合多模态视觉特征
        fused_img_c = self.fuse_visual_features(img_c, dinov2_features, depth_features)
        
        # 添加位置编码
        x = x + self.x_pos_embed
        lang_c = lang_c + self.lang_cond_pos_embed[:, :lang_c.shape[1]]
        fused_img_c = fused_img_c + self.img_cond_pos_embed[:, :fused_img_c.shape[1]]

        # 前向传播通过transformer块
        conds = [lang_c, fused_img_c]
        masks = [lang_mask, img_mask]
        for i, block in enumerate(self.blocks):
            c, mask = conds[i%2], masks[i%2]
            x = block(x, c, mask)                       # (B, T+2, D)

        # 最终输出层
        x = self.final_layer(x)                         # (B, T+2, output_dim)

        # 只保留动作token
        x = x[:, -self.horizon:]
        
        return x