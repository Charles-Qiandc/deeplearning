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


class RDT(nn.Module):
    """
    🔄 修改：Robotics Diffusion Transformers with REPA alignment loss
    主要修改：
    1. 减少层数从28到8
    2. 添加REPA对齐投影器
    3. 在第4层提取动作token用于对齐损失
    """
    def __init__(
        self,
        output_dim=128,
        horizon=32,
        hidden_size=1152,
        depth=8,  # 🔄 修改：从28减少到8
        num_heads=16,
        max_lang_cond_len=1024,
        img_cond_len=4096,
        lang_pos_embed_config=None,
        img_pos_embed_config=None,
        dtype=torch.bfloat16,
        # 🆕 REPA相关参数
        enable_repa_loss=True,
        repa_activation_layer=4,
        dinov2_feature_dim=1024,  # DINOv2-L特征维度
    ):
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.max_lang_cond_len = max_lang_cond_len
        self.img_cond_len = img_cond_len
        self.dtype = dtype
        self.lang_pos_embed_config = lang_pos_embed_config
        self.img_pos_embed_config = img_pos_embed_config
        
        # 🆕 REPA配置
        self.enable_repa_loss = enable_repa_loss
        self.repa_activation_layer = repa_activation_layer - 1  # 转换为0-based索引
        self.dinov2_feature_dim = dinov2_feature_dim
        

        # 原有的嵌入器组件
        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        self.freq_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        
        # 位置编码参数
        self.x_pos_embed = nn.Parameter(
            torch.zeros(1, horizon+3, hidden_size))
        self.lang_cond_pos_embed = nn.Parameter(
            torch.zeros(1, max_lang_cond_len, hidden_size))
        self.img_cond_pos_embed = nn.Parameter(
            torch.zeros(1, img_cond_len, hidden_size))

        # 🔄 修改：减少到8层RDT块
        self.blocks = nn.ModuleList([
            RDTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        
        # 🆕 REPA对齐投影器
        if self.enable_repa_loss:
            # 计算合适的投影器维度
            projector_dim = max(hidden_size, dinov2_feature_dim * 2)  # 设置为较大值确保表达能力
            
            self.action_to_vision_projector = nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, dinov2_feature_dim),
            )
            
        
        self.final_layer = FinalLayer(hidden_size, output_dim)
        self.initialize_weights()

    def initialize_weights(self):
        """初始化权重，包括新增的REPA组件"""
        # 基础权重初始化
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # 初始化位置编码
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
        
        # 图像条件位置编码
        if self.img_pos_embed_config is None:
            img_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(
                self.hidden_size, torch.arange(self.img_cond_len))
        else:
            img_cond_pos_embed = get_multimodal_cond_pos_embed(
                embed_dim=self.hidden_size,
                mm_cond_lens=OrderedDict(self.img_pos_embed_config),
                embed_modality=False
            )
        self.img_cond_pos_embed.data.copy_(
            torch.from_numpy(img_cond_pos_embed).float().unsqueeze(0))

        # 时间步和频率嵌入器初始化
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.freq_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.freq_embedder.mlp[2].weight, std=0.02)
            
        # 最终层零初始化
        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)
        
        # 🆕 REPA投影器初始化（较小的初始化以保证训练稳定性）
        if self.enable_repa_loss:
            for module in self.action_to_vision_projector:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.1)  # 较小的初始化
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
        
        # 转换到指定数据类型
        self.to(self.dtype)

    def forward(self, x, freq, t, lang_c, img_c, lang_mask=None, img_mask=None):
        """
        🔄 修改：前向传播，返回预测结果和中间激活
        
        Returns:
            x: (B, horizon, output_dim) 最终预测
            intermediate_activations: dict 包含中间激活，用于REPA损失计算
        """
        # 时间步和频率嵌入
        t = self.t_embedder(t).unsqueeze(1)             # (B, 1, D)
        freq = self.freq_embedder(freq).unsqueeze(1)    # (B, 1, D)
        
        # 处理时间步广播
        if t.shape[0] == 1:
            t = t.expand(x.shape[0], -1, -1)
        x = torch.cat([t, freq, x], dim=1)               # (B, T+2, D)
        
        # 添加位置编码
        x = x + self.x_pos_embed
        lang_c = lang_c + self.lang_cond_pos_embed[:, :lang_c.shape[1]]
        img_c = img_c + self.img_cond_pos_embed

        # 🆕 存储中间激活用于REPA损失
        intermediate_activations = {}
        
        # 前向传播通过transformer块
        conds = [lang_c, img_c]
        masks = [lang_mask, img_mask]
        for i, block in enumerate(self.blocks):
            c, mask = conds[i%2], masks[i%2]
            x = block(x, c, mask)                       # (B, T+2, D)
            
            # 🆕 在指定层提取动作token
            if self.enable_repa_loss and i == self.repa_activation_layer:
                # 提取动作部分 (去除前缀: timestep, freq, state)
                action_tokens = x[:, -self.horizon:, :]  # (B, horizon, hidden_size)
                intermediate_activations['action_tokens_for_repa'] = action_tokens

        # 最终输出层
        x = self.final_layer(x)                         # (B, T+2, output_dim)

        # 只保留动作token
        x = x[:, -self.horizon:]
        
        return x, intermediate_activations