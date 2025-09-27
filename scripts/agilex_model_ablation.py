import os, sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from configs.state_vec import STATE_VEC_IDX_MAPPING

from pathlib import Path

# get current workspace
current_file = Path(__file__)
sys.path.append(os.path.join(current_file.parent.parent, "models"))

from multimodal_encoder.siglip_encoder import SiglipVisionTower
from multimodal_encoder.t5_encoder import T5Embedder
from multimodal_encoder.dinov2_encoder import create_dinov2_encoder
from multimodal_encoder.depth_encoder import create_depth_encoder
from rdt_runner_ablation import RDTRunnerAblation


def create_model(args, **kwargs):
    """创建消融实验版本的RDT模型"""
    left_arm_dim, right_arm_dim = (
        args["arm_dim"]["left_arm_dim"],
        args["arm_dim"]["right_arm_dim"],
    )
    model = RoboticDiffusionTransformerModelAblation(args, **kwargs)
    pretrained = kwargs.get("pretrained", None)
    if pretrained is not None and os.path.isfile(pretrained):
        model.load_pretrained_weights(pretrained)

    return model


class RoboticDiffusionTransformerModelAblation(object):
    """
    消融实验版RDT模型包装器
    集成输入侧多模态视觉特征融合
    """

    def __init__(
        self,
        args,
        device="cuda",
        dtype=torch.bfloat16,
        image_size=None,
        control_frequency=25,
        pretrained=None,
        pretrained_vision_encoder_name_or_path=None,
    ):
        self.args = args
        self.dtype = dtype
        self.image_size = image_size
        self.device = device
        self.control_frequency = control_frequency
        self.image_processor, self.vision_model = self.get_vision_encoder(pretrained_vision_encoder_name_or_path)
        
        # 🆕 创建额外的视觉编码器
        self.dinov2_encoder = None
        self.depth_encoder = None
        
        # 从配置中读取是否使用额外特征
        self.use_dinov2_features = args.get("use_dinov2_features", False)
        self.use_depth_features = args.get("use_depth_features", False)
        
        if self.use_dinov2_features:
            print("🔧 加载DINOv2编码器用于推理...")
            self.dinov2_encoder = create_dinov2_encoder(model_size="large", select_feature="cls_only")
            self.dinov2_encoder.to(device, dtype=dtype)
            
        if self.use_depth_features:
            print("🔧 加载DepthAnything编码器用于推理...")
            self.depth_encoder = create_depth_encoder(
                model_size="metric_large",
                feature_dim=1024,
                device=device,
                use_metric_model=True
            )
            self.depth_encoder.to(device, dtype=dtype)
        
        self.policy = self.get_policy(pretrained)
        
        self.left_arm_dim, self.right_arm_dim = (
            args["arm_dim"]["left_arm_dim"],
            args["arm_dim"]["right_arm_dim"],
        )

        self.reset()

    def get_policy(self, pretrained):
        """初始化消融实验版策略模型"""
        if pretrained is None or os.path.isfile(pretrained):
            img_cond_len = (self.args["common"]["img_history_size"] * self.args["common"]["num_cameras"] *
                            self.vision_model.num_patches)

            _model = RDTRunnerAblation(
                action_dim=self.args["common"]["state_dim"],
                pred_horizon=self.args["common"]["action_chunk_size"],
                config=self.args["model"],
                lang_token_dim=self.args["model"]["lang_token_dim"],
                img_token_dim=self.args["model"]["img_token_dim"],
                state_token_dim=self.args["model"]["state_token_dim"],
                max_lang_cond_len=self.args["dataset"]["tokenizer_max_length"],
                img_cond_len=img_cond_len,
                img_pos_embed_config=[
                    (
                        "image",
                        (
                            self.args["common"]["img_history_size"],
                            self.args["common"]["num_cameras"],
                            -self.vision_model.num_patches,
                        ),
                    ),
                ],
                lang_pos_embed_config=[
                    ("lang", -self.args["dataset"]["tokenizer_max_length"]),
                ],
                dtype=self.dtype,
                # 🆕 消融实验参数
                use_dinov2_features=self.use_dinov2_features,
                dinov2_feature_dim=1024,
                use_depth_features=self.use_depth_features,
                depth_feature_dim=1024,
                visual_fusion_strategy=self.args.get("visual_fusion_strategy", "concat"),
            )
        else:
            raise NotImplementedError("从hub加载消融版模型暂未实现")

        return _model

    def get_vision_encoder(self, pretrained_vision_encoder_name_or_path):
        vision_encoder = SiglipVisionTower(vision_tower=pretrained_vision_encoder_name_or_path, args=None)
        image_processor = vision_encoder.image_processor
        return image_processor, vision_encoder

    def reset(self):
        """设置模型为评估模式"""
        device = self.device
        weight_dtype = self.dtype
        self.policy.eval()
        self.vision_model.eval()
        
        if self.dinov2_encoder is not None:
            self.dinov2_encoder.eval()
        if self.depth_encoder is not None:
            self.depth_encoder.eval()

        self.policy = self.policy.to(device, dtype=weight_dtype)
        self.vision_model = self.vision_model.to(device, dtype=weight_dtype)
        
        if self.dinov2_encoder is not None:
            self.dinov2_encoder = self.dinov2_encoder.to(device, dtype=weight_dtype)
        if self.depth_encoder is not None:
            self.depth_encoder = self.depth_encoder.to(device, dtype=weight_dtype)

    def load_pretrained_weights(self, pretrained):
        """安全加载预训练权重（消融版本）"""
        try:
            print(f"🔧 开始加载消融实验权重: {pretrained}")
            checkpoint = torch.load(pretrained, map_location='cpu')
            print(f"✅ Checkpoint加载成功，包含 {len(checkpoint)} 个顶级键")
            
            # 获取当前模型的参数名
            model_state_dict = self.policy.state_dict()
            model_keys = set(model_state_dict.keys())
            
            print(f"📋 当前模型包含 {len(model_keys)} 个参数")
            
            # 过滤checkpoint中的参数
            filtered_checkpoint = {}
            skipped_params = []
            
            for key, value in checkpoint.items():
                if key in model_keys:
                    # 检查形状是否匹配
                    if value.shape == model_state_dict[key].shape:
                        filtered_checkpoint[key] = value
                    else:
                        skipped_params.append(f"{key} (形状不匹配: {value.shape} vs {model_state_dict[key].shape})")
                else:
                    # 识别被移除的REPA相关参数
                    if any(keyword in key for keyword in [
                        'dual_teacher_model', 'soft_router', 'routing_network', 
                        'dinov2_to_action_projector', 'depth_to_action_projector',
                        'routing_temperature', 'critical_annotator', 'repa']):
                        skipped_params.append(f"{key} (REPA组件，已移除)")
                    else:
                        skipped_params.append(f"{key} (参数不存在)")
            
            # 加载过滤后的参数
            missing_keys, unexpected_keys = self.policy.load_state_dict(filtered_checkpoint, strict=False)
            
            # 统计结果
            print(f"✅ 成功加载 {len(filtered_checkpoint)} 个参数")
            
            if skipped_params:
                repa_skipped = [p for p in skipped_params if 'REPA组件' in p]
                shape_skipped = [p for p in skipped_params if '形状不匹配' in p]
                other_skipped = [p for p in skipped_params if p not in repa_skipped and p not in shape_skipped]
                
                if repa_skipped:
                    print(f"⚠️  跳过 {len(repa_skipped)} 个REPA相关参数（消融实验已移除）")
                if shape_skipped:
                    print(f"⚠️  跳过 {len(shape_skipped)} 个形状不匹配参数")
                if other_skipped:
                    print(f"⚠️  跳过 {len(other_skipped)} 个其他参数")
                    for param in other_skipped[:3]:  # 只显示前3个
                        print(f"     - {param}")
            
            if missing_keys:
                print(f"⚠️  {len(missing_keys)} 个模型参数未找到对应权重（保持默认初始化）")
                # 这些通常是新加入的视觉特征投影器参数
                for key in missing_keys[:5]:
                    if any(keyword in key for keyword in ['dinov2', 'depth', 'visual']):
                        print(f"     - {key} (新增视觉特征组件)")
            
            print("✅ 消融实验权重加载完成，模型可以正常评估")
            return True
            
        except Exception as e:
            print(f"❌ 权重加载失败: {e}")
            print("ℹ️  将使用默认初始化继续...")
            import traceback
            traceback.print_exc()
            return False

    def _format_joint_to_state(self, joints):
        """格式化关节状态"""
        AGILEX_STATE_INDICES = ([STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"]
                                 for i in range(self.left_arm_dim)] + [STATE_VEC_IDX_MAPPING["left_gripper_open"]] +
                                [STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"]
                                 for i in range(self.right_arm_dim)] + [STATE_VEC_IDX_MAPPING[f"right_gripper_open"]])
        
        # 重新缩放夹爪到[0,1]范围
        joints = joints / torch.tensor(
            [[[1 for i in range(self.left_arm_dim + 1 + self.right_arm_dim + 1)]]],
            device=joints.device,
            dtype=joints.dtype,
        )

        B, N, _ = joints.shape
        state = torch.zeros(
            (B, N, self.args["model"]["state_token_dim"]),
            device=joints.device,
            dtype=joints.dtype,
        )
        # 填充统一状态向量
        state[:, :, AGILEX_STATE_INDICES] = joints
        # 组装掩码
        state_elem_mask = torch.zeros(
            (B, self.args["model"]["state_token_dim"]),
            device=joints.device,
            dtype=joints.dtype,
        )
        state_elem_mask[:, AGILEX_STATE_INDICES] = 1
        return state, state_elem_mask

    def _unformat_action_to_joint(self, action):
        """将统一动作向量转换为关节动作"""
        AGILEX_STATE_INDICES = ([STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"]
                                 for i in range(self.left_arm_dim)] + [STATE_VEC_IDX_MAPPING["left_gripper_open"]] +
                                [STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"]
                                 for i in range(self.right_arm_dim)] + [STATE_VEC_IDX_MAPPING[f"right_gripper_open"]])
        action_indices = AGILEX_STATE_INDICES
        joints = action[:, :, action_indices]

        # 重新缩放夹爪回动作范围
        joints = joints * torch.tensor(
            [[[1 for i in range(self.left_arm_dim + 1 + self.right_arm_dim + 1)]]],
            device=joints.device,
            dtype=joints.dtype,
        )

        return joints

    @torch.no_grad()
    def step(self, proprio, images, text_embeds):
        """
        消融实验版预测步骤：集成多模态视觉特征
        """
        device = self.device
        dtype = self.dtype

        # 背景图像用于填充
        background_color = np.array([int(x * 255) for x in self.image_processor.image_mean],
                                    dtype=np.uint8).reshape(1, 1, 3)
        background_image = (np.ones(
            (
                self.image_processor.size["height"],
                self.image_processor.size["width"],
                3,
            ),
            dtype=np.uint8,
        ) * background_color)

        # 🆕 分离图像用于不同的视觉编码器
        # images顺序: [前t-1, 右t-1, 左t-1, 前t, 右t, 左t]
        current_frame_images = images[3:6]  # 当前帧的三个视角
        
        # 预处理SigLIP图像
        image_tensor_list = []
        for image in images:
            if image is None:
                image = Image.fromarray(background_image)

            if self.image_size is not None:
                image = transforms.Resize(self.image_size)(image)

            if self.args["dataset"].get("auto_adjust_image_brightness", False):
                pixel_values = list(image.getdata())
                average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                if average_brightness <= 0.15:
                    image = transforms.ColorJitter(brightness=(1.75, 1.75))(image)

            if self.args["dataset"].get("image_aspect_ratio", "pad") == "pad":
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))
            image = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            image_tensor_list.append(image)

        image_tensor = torch.stack(image_tensor_list, dim=0).to(device, dtype=dtype)

        # SigLIP编码
        image_embeds = self.vision_model(image_tensor).detach()
        image_embeds = image_embeds.reshape(-1, self.vision_model.hidden_size).unsqueeze(0)

        # 🆕 DINOv2特征提取
        dinov2_features = None
        if self.dinov2_encoder is not None and self.use_dinov2_features:
            # 使用当前帧的前视图像
            front_image = current_frame_images[0]  # 前视相机
            if front_image is not None:
                # 预处理DINOv2图像
                dinov2_input = Image.fromarray(front_image)
                dinov2_input = dinov2_input.resize((518, 518), Image.BILINEAR)
                dinov2_array = np.array(dinov2_input).astype(np.float32) / 255.0
                dinov2_tensor = torch.from_numpy(dinov2_array).permute(2, 0, 1).to(device, dtype=dtype)
                
                # ImageNet标准化
                mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(3, 1, 1)
                dinov2_tensor = (dinov2_tensor - mean) / std
                
                # 添加batch维度并提取特征
                dinov2_tensor = dinov2_tensor.unsqueeze(0)  # (1, 3, 518, 518)
                dinov2_features = self.dinov2_encoder(dinov2_tensor)  # (1, 1, 1024)

        # 🆕 DepthAnything特征提取  
        depth_features = None
        if self.depth_encoder is not None and self.use_depth_features:
            # 使用当前帧的前视图像
            front_image = current_frame_images[0]  # 前视相机
            if front_image is not None:
                # 预处理深度图像
                depth_input = Image.fromarray(front_image)
                depth_input = depth_input.resize((518, 518), Image.BILINEAR)
                depth_array = np.array(depth_input).astype(np.float32) / 255.0
                depth_tensor = torch.from_numpy(depth_array).permute(2, 0, 1).to(device, dtype=dtype)
                
                # ImageNet标准化
                mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(3, 1, 1)
                depth_tensor = (depth_tensor - mean) / std
                
                # 添加batch维度并提取特征
                depth_tensor = depth_tensor.unsqueeze(0)  # (1, 3, 518, 518)
                depth_features_raw, _ = self.depth_encoder(depth_tensor)  # (1, 1370, 1024)
                # 只取CLS token
                depth_features = depth_features_raw[:, 0:1, :]  # (1, 1, 1024)

        # 准备本体感觉状态和控制频率
        joints = proprio.to(device).unsqueeze(0)  # (1, 1, 14)
        states, state_elem_mask = self._format_joint_to_state(joints)  # (1, 1, 128), (1, 128)
        states, state_elem_mask = states.to(device, dtype=dtype), state_elem_mask.to(device, dtype=dtype)
        states = states[:, -1:, :]  # (1, 1, 128)
        ctrl_freqs = torch.tensor([self.control_frequency]).to(device)

        text_embeds = text_embeds.to(device, dtype=dtype)

        # 🆕 使用消融版策略预测（传入额外的视觉特征）
        trajectory = self.policy.predict_action(
            lang_tokens=text_embeds,
            lang_attn_mask=torch.ones(text_embeds.shape[:2], dtype=torch.bool, device=text_embeds.device),
            img_tokens=image_embeds,
            state_tokens=states,
            action_mask=state_elem_mask.unsqueeze(1),
            ctrl_freqs=ctrl_freqs,
            dinov2_features=dinov2_features,
            depth_features=depth_features,
        )
        trajectory = self._unformat_action_to_joint(trajectory).to(torch.float32)

        return trajectory