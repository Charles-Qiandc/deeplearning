# models/rdt/model_eval.py - 专门用于评测的RDT模型

from collections import OrderedDict
import torch
import torch.nn as nn

from models.rdt.blocks import (FinalLayer, RDTBlock, TimestepEmbedder,
                               get_1d_sincos_pos_embed_from_grid,
                               get_multimodal_cond_pos_embed)


class RDTEval(nn.Module):
    """
    专门用于评测的RDT模型
    移除了所有训练时的REPA对齐组件，确保推理时的纯净性
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
    ):
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.max_lang_cond_len = max_lang_cond_len
        self.img_cond_len = img_cond_len
        self.dtype = dtype
        self.lang_pos_embed_config = lang_pos_embed_config
        self.img_pos_embed_config = img_pos_embed_config

        print("🔧 创建评测专用RDT模型 (无REPA组件)")

        # 嵌入器组件
        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        self.freq_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        
        # 位置编码参数
        self.x_pos_embed = nn.Parameter(torch.zeros(1, horizon+3, hidden_size))
        self.lang_cond_pos_embed = nn.Parameter(torch.zeros(1, max_lang_cond_len, hidden_size))
        self.img_cond_pos_embed = nn.Parameter(torch.zeros(1, img_cond_len, hidden_size))

        # Transformer块
        self.blocks = nn.ModuleList([
            RDTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        
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
        
        # 🔧 条件性初始化REPA组件
        if self.enable_repa_loss and not self.eval_mode:
            # DINOv2投影器初始化
            if hasattr(self, 'dinov2_to_action_projector'):
                for module in self.dinov2_to_action_projector:
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight, gain=0.5)
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)
            
            # Depth投影器初始化
            if hasattr(self, 'depth_to_action_projector'):
                for module in self.depth_to_action_projector:
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight, gain=0.5)
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)
        
        # 路由网络初始化
        if self.use_dual_teachers and self.enable_repa_loss and not self.eval_mode:
            if hasattr(self, 'routing_network'):
                for module in self.routing_network:
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight, gain=0.5)
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)
        
        # 最终层零初始化
        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)
        
        # 转换到指定数据类型
        self.to(self.dtype)

    def forward(self, x, freq, t, lang_c, img_c, lang_mask=None, img_mask=None):
        """
        前向传播，支持训练模式和评测模式
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

        # 存储中间激活用于REPA损失（只在训练模式下）
        intermediate_activations = {}
        
        # 前向传播通过transformer块
        conds = [lang_c, img_c]
        masks = [lang_mask, img_mask]
        for i, block in enumerate(self.blocks):
            c, mask = conds[i%2], masks[i%2]
            x = block(x, c, mask)                       # (B, T+2, D)
            
            # 🔧 只在训练模式且启用REPA时提取中间激活
            if (not self.eval_mode and 
                self.enable_repa_loss and 
                i == self.repa_activation_layer):
                # 提取动作部分 (去除前缀: timestep, freq, state)
                action_tokens = x[:, -self.horizon:, :]  # (B, horizon, hidden_size)
                intermediate_activations['action_tokens_for_repa'] = action_tokens
                
                # 计算路由权重（如果启用双教师模式）
                if self.use_dual_teachers and hasattr(self, 'routing_network'):
                    routing_logits = self.routing_network(action_tokens)  # (B, T, 2)
                    # 应用温度缩放的softmax
                    routing_weights = torch.softmax(
                        routing_logits / torch.clamp(self.routing_temperature, min=0.1), 
                        dim=-1
                    )
                    intermediate_activations['routing_weights'] = routing_weights
                    intermediate_activations['routing_logits'] = routing_logits

        # 最终输出层
        x = self.final_layer(x)                         # (B, T+2, output_dim)

        # 只保留动作token
        x = x[:, -self.horizon:]
        
        return x, intermediate_activations

    def set_eval_mode(self, eval_mode=True):
        """设置评测模式"""
        self.eval_mode = eval_mode
        if eval_mode:
            print("🔧 RDT模型切换到评测模式")
        else:
            print("🔧 RDT模型切换到训练模式").img_cond_len))
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
        
        # 转换到指定数据类型
        self.to(self.dtype)

    def forward(self, x, freq, t, lang_c, img_c, lang_mask=None, img_mask=None):
        """
        纯净的前向传播，只返回动作预测
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

        # 前向传播通过transformer块 - 纯净版本
        conds = [lang_c, img_c]
        masks = [lang_mask, img_mask]
        for i, block in enumerate(self.blocks):
            c, mask = conds[i%2], masks[i%2]
            x = block(x, c, mask)                       # (B, T+2, D)

        # 最终输出层
        x = self.final_layer(x)                         # (B, T+2, output_dim)

        # 只保留动作token，返回简单格式
        x = x[:, -self.horizon:]
        
        # 为了兼容性，返回空的中间激活
        return x, {}


# 修改原始RDT模型，添加评测模式
class RDT(nn.Module):
    """
    原始RDT模型，添加评测模式支持
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
        # REPA相关参数
        enable_repa_loss=True,
        repa_activation_layer=21,
        dinov2_feature_dim=1024,
        depth_feature_dim=1024,
        # 双教师路由参数
        use_dual_teachers=False,
        routing_hidden_dim=512,
        # 🆕 评测模式参数
        eval_mode=False,
    ):
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.max_lang_cond_len = max_lang_cond_len
        self.img_cond_len = img_cond_len
        self.dtype = dtype
        self.lang_pos_embed_config = lang_pos_embed_config
        self.img_pos_embed_config = img_pos_embed_config
        
        # 🆕 评测模式配置
        self.eval_mode = eval_mode
        if eval_mode:
            print("🔧 RDT模型初始化为评测模式 (禁用REPA)")
            enable_repa_loss = False
            use_dual_teachers = False
        
        # REPA配置
        self.enable_repa_loss = enable_repa_loss
        self.repa_activation_layer = repa_activation_layer - 1 if enable_repa_loss else -1
        self.dinov2_feature_dim = dinov2_feature_dim
        self.depth_feature_dim = depth_feature_dim
        
        # 双教师路由配置
        self.use_dual_teachers = use_dual_teachers

        # 嵌入器组件
        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        self.freq_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        
        # 位置编码参数
        self.x_pos_embed = nn.Parameter(torch.zeros(1, horizon+3, hidden_size))
        self.lang_cond_pos_embed = nn.Parameter(torch.zeros(1, max_lang_cond_len, hidden_size))
        self.img_cond_pos_embed = nn.Parameter(torch.zeros(1, img_cond_len, hidden_size))

        # Transformer块
        self.blocks = nn.ModuleList([
            RDTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        
        # 🔧 条件性创建REPA相关组件
        if self.enable_repa_loss and not eval_mode:
            # DINOv2全局特征投影器
            self.dinov2_to_action_projector = nn.Sequential(
                nn.Linear(dinov2_feature_dim, (dinov2_feature_dim + hidden_size) // 2),
                nn.LayerNorm((dinov2_feature_dim + hidden_size) // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear((dinov2_feature_dim + hidden_size) // 2, hidden_size),
                nn.LayerNorm(hidden_size),
            )
            
            # DepthAnythingV2深度特征投影器
            self.depth_to_action_projector = nn.Sequential(
                nn.Linear(depth_feature_dim, (depth_feature_dim + hidden_size) // 2),
                nn.LayerNorm((depth_feature_dim + hidden_size) // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear((depth_feature_dim + hidden_size) // 2, hidden_size),
                nn.LayerNorm(hidden_size),
            )
        
        # 双教师路由网络
        if self.use_dual_teachers and self.enable_repa_loss and not eval_mode:
            self.routing_network = nn.Sequential(
                nn.Linear(hidden_size, routing_hidden_dim),
                nn.LayerNorm(routing_hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(routing_hidden_dim, routing_hidden_dim),
                nn.LayerNorm(routing_hidden_dim),
                nn.GELU(),
                nn.Linear(routing_hidden_dim, 2),
            )
            
            # 可学习的温度参数
            self.routing_temperature = nn.Parameter(torch.tensor(1.0))
        
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
        
        # 图像条件位置编码
        if self.img_pos_embed_config is None:
            img_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(
                self.hidden_size, torch.arange(self