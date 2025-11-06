import copy
import logging
import math
import os
from pathlib import Path

import diffusers
import torch
import torch.utils.checkpoint
import transformers
import yaml
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from safetensors.torch import load_model

from models.ema_model import EMAModel
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder
from models.rdt_runner import RDTRunner
from train.dataset import DataCollatorForVLAConsumerDataset, VLAConsumerDataset
from train.sample import log_sample_res

# 导入DINOv2和DepthAnythingV2编码器
from models.multimodal_encoder.dinov2_encoder import create_dinov2_encoder
from models.multimodal_encoder.depth_encoder import create_depth_encoder

# 导入关键时间段标注器
from data.critical_timestep_annotator import TaskType

if is_wandb_available():
    import wandb


def save_model_card(repo_id: str, base_model=str, repo_folder=None):
    yaml_header = f"""
---
license: mit
base_model: {base_model}
language:
- en
pipeline_tag: robotics
library_name: transformers
tags:
- robotics
- pytorch
- multimodal
- pretraining
- vla
- diffusion
- rdt
- soft-routing
- dual-teachers
- critical-timestep
- binary-labels
---
    """
    model_card = f"""
# RDT with Soft Routing Dual-Teacher REPA - {repo_id}

This is a RDT model with soft routing dual-teacher REPA alignment loss, task-driven critical timestep annotation 
derived from {base_model}. The weights were trained using [RDT](https://rdt-robotics.github.io/rdt-robotics/) 
with advanced soft routing multi-modal alignment strategies.

## Key Features
- **Soft Routing Strategy**: Rule-driven weight allocation based on binary critical timestep labels
- **Critical Timestep Annotation**: Task-driven annotation for precise temporal alignment
- **Dual Visual Teachers**: DINOv2 (global semantic) + DepthAnythingV2 (depth geometric)
- **Neural Weight Adjustment**: Optional fine-tuning with temporal smoothing
- **Contrastive Learning**: Enhanced feature alignment with contrastive loss

## Weight Allocation Strategy
- **Critical Timesteps (1)**: Global 25%, Depth 75% - Focus on precise manipulation
- **Non-Critical Timesteps (0)**: Global 75%, Depth 25% - Focus on scene understanding

## Architecture Components
1. **Binary Label Soft Router**: Rule-driven weight allocation with optional neural adjustment
2. **Dual Visual Teachers**: DINOv2 (global) + DepthAnythingV2 (geometric)
3. **Temporal Smoothing**: Prevents sudden weight transitions
4. **Contrastive Learning**: Enhances feature alignment quality

## Task Types Supported
- **Grasp Tasks (task_type=1)**: Deceleration → Gripper closing alignment
- **Click Tasks (task_type=2)**: Gripper closing → Deceleration alignment
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml_header + model_card)


def check_critical_alerts(metrics, global_step, logger):
    """
    检查关键异常情况并发出预警
    """
    alerts = []
    
    # 1. 特征对齐失效
    if metrics.get('global_similarity', 1.0) < 0.3 and global_step > 1000:
        alerts.append("WARNING: Low global similarity - feature alignment may be failing")
    
    # 2. 权重分配严重偏离
    if 'critical_global_weight' in metrics:
        expected_critical_global = 0.25
        actual = metrics['critical_global_weight']
        if abs(actual - expected_critical_global) > 0.2:  # 偏离超过20%
            alerts.append(f"WARNING: Critical weight deviation - expected {expected_critical_global}, got {actual:.3f}")
    
    # 3. 权重调整过度
    if metrics.get('weight_drift', 0) > 0.15:
        alerts.append("WARNING: Excessive weight drift - neural adjustment may be too aggressive")
    
    # 4. 数据不平衡
    critical_ratio = metrics.get('critical_ratio', 0.3)
    if critical_ratio < 0.1 or critical_ratio > 0.6:
        alerts.append(f"WARNING: Critical ratio imbalance - {critical_ratio:.3f}")
    
    # 记录警告
    for alert in alerts:
        logger.warning(alert)
    
    return alerts


def train(args, logger):
    # Read the config
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)

    with open(args.model_config_path, "r") as f:
        model_config = yaml.safe_load(f)
    
    args.output_dir = model_config["checkpoint_path"]
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    accelerator = Accelerator(
        deepspeed_plugin=(DeepSpeedPlugin(hf_ds_config=args.deepspeed) if args.deepspeed is not None else None),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    # For mixed precision training
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # ⭐ 获取双教师配置
    global_teacher_type = model_config.get("global_teacher_type", "dinov2")
    depth_teacher_type = model_config.get("depth_teacher_type", "depth_anything_v2")  # ⭐ 新增
    
    # 向后兼容：从旧标志推断
    if "global_teacher_type" not in model_config:
        if model_config.get("use_dinov2_features", False):
            global_teacher_type = "dinov2"
        elif model_config.get("use_siglip_global_features", False):
            global_teacher_type = "siglip"
        else:
            global_teacher_type = "dinov2"
    
    # ⭐ 向后兼容：从旧标志推断深度教师
    if "depth_teacher_type" not in model_config:
        if model_config.get("use_depth_features", True):
            depth_teacher_type = "depth_anything_v2"
        elif model_config.get("use_siglip_depth_features", False):
            depth_teacher_type = "siglip"
        else:
            depth_teacher_type = "depth_anything_v2"
    
    # ⭐ 自动设置特征标志
    use_dinov2_features = (global_teacher_type == "dinov2")
    use_siglip_global_features = (global_teacher_type == "siglip")
    use_depth_anything_v2 = (depth_teacher_type == "depth_anything_v2")
    use_siglip_depth_features = (depth_teacher_type == "siglip")
    
    # 确定特征维度
    if global_teacher_type == "dinov2":
        global_feature_dim = 1024
    elif global_teacher_type == "siglip":
        global_feature_dim = 1152
    else:
        raise ValueError(f"不支持的全局教师类型: {global_teacher_type}")
    
    # ⭐ 确定深度特征维度
    if depth_teacher_type == "depth_anything_v2":
        depth_feature_dim = 1024
    elif depth_teacher_type == "siglip":
        depth_feature_dim = 1152
    else:
        raise ValueError(f"不支持的深度教师类型: {depth_teacher_type}")
    
    logger.info("=" * 70)
    logger.info(f"🎯 双教师配置（自动设置）:")
    logger.info(f"   📊 全局教师:")
    logger.info(f"      - 类型: {global_teacher_type.upper()}")
    logger.info(f"      - 特征维度: {global_feature_dim}")
    logger.info(f"      - use_dinov2_features: {use_dinov2_features}")
    logger.info(f"      - use_siglip_global_features: {use_siglip_global_features}")
    logger.info(f"   📊 深度教师:")
    logger.info(f"      - 类型: {depth_teacher_type.upper()}")
    logger.info(f"      - 特征维度: {depth_feature_dim}")
    logger.info(f"      - use_depth_anything_v2: {use_depth_anything_v2}")
    logger.info(f"      - use_siglip_depth_features: {use_siglip_depth_features}")
    logger.info("=" * 70)
    
    # 获取软路由REPA配置
    enable_soft_routing_repa = model_config.get("enable_soft_routing_repa", True)
    soft_routing_repa_weight = model_config.get("soft_routing_repa_weight", 0.2)
    
    # 关键时间段标注配置
    enable_critical_annotation = model_config.get("enable_critical_annotation", True)
    task_type = model_config.get("task_type", 1)
    critical_annotation_config = model_config.get("critical_annotation_config", {})
    
    # 软路由配置
    soft_routing_config = model_config.get("soft_routing_config", {})
    # ⭐ 自动设置维度（如果未配置）
    if 'global_dim' not in soft_routing_config:
        soft_routing_config['global_dim'] = global_feature_dim
    if 'depth_dim' not in soft_routing_config:
        soft_routing_config['depth_dim'] = depth_feature_dim

    # 文本编码器
    if args.precomp_lang_embed:
        tokenizer, text_encoder = None, None
    else:
        text_embedder = T5Embedder(
            from_pretrained=args.pretrained_text_encoder_name_or_path,
            model_max_length=config["dataset"]["tokenizer_max_length"],
            device=accelerator.device,
        )
        tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model

    # SigLIP视觉编码器（主干）
    vision_encoder = SiglipVisionTower(
        vision_tower=args.pretrained_vision_encoder_name_or_path, 
        args=None
    )
    image_processor = vision_encoder.image_processor

    # ⭐ 创建全局教师编码器
    global_teacher_encoder = None
    if enable_soft_routing_repa:
        if global_teacher_type == "dinov2":
            logger.info("📦 加载DINOv2全局教师编码器...")
            from models.multimodal_encoder.dinov2_encoder import create_dinov2_encoder
            global_teacher_encoder = create_dinov2_encoder(
                model_size="large", 
                select_feature="cls_only"
            )
        elif global_teacher_type == "siglip":
            logger.info("📦 加载SigLIP全局教师编码器...")
            from models.multimodal_encoder.siglip_global_encoder import create_siglip_global_encoder
            global_teacher_encoder = create_siglip_global_encoder(
                model_name=args.pretrained_vision_encoder_name_or_path,
                pooling_strategy="mean",
                feature_dim=global_feature_dim
            )
        else:
            raise ValueError(f"不支持的全局教师类型: {global_teacher_type}")
        
        global_teacher_encoder.to(accelerator.device, dtype=weight_dtype)
        global_teacher_encoder.print_model_info()

    # ⭐ 创建深度教师编码器
    depth_teacher_encoder = None
    if enable_soft_routing_repa:
        if depth_teacher_type == "depth_anything_v2":
            logger.info("📦 加载DepthAnythingV2深度教师编码器...")
            from models.multimodal_encoder.depth_encoder import create_depth_encoder
            depth_teacher_encoder = create_depth_encoder(
                model_size="metric_large",
                feature_dim=1024,
                device=accelerator.device,
                use_metric_model=True
            )
        elif depth_teacher_type == "siglip":
            logger.info("📦 加载SigLIP深度教师编码器...")
            from models.multimodal_encoder.siglip_depth_encoder import create_siglip_depth_encoder
            depth_teacher_encoder = create_siglip_depth_encoder(
                model_name=args.pretrained_vision_encoder_name_or_path,
                feature_dim=depth_feature_dim,
                output_format="patch_tokens",  # 可配置
                device=accelerator.device
            )
        else:
            raise ValueError(f"不支持的深度教师类型: {depth_teacher_type}")
        
        depth_teacher_encoder.to(accelerator.device, dtype=weight_dtype)
        depth_teacher_encoder.print_model_info()

    # ⭐ 构建RDT模型
    logger.info("🔨 构建软路由双教师RDT模型...")
    img_cond_len = (config["common"]["img_history_size"] * 
                    config["common"]["num_cameras"] *
                    vision_encoder.num_patches)
    
    repa_activation_layer = model_config.get("repa_activation_layer", 21)
    
    rdt = RDTRunner(
        action_dim=config["common"]["state_dim"],
        pred_horizon=config["common"]["action_chunk_size"],
        config=config["model"],
        lang_token_dim=config["model"]["lang_token_dim"],
        img_token_dim=config["model"]["img_token_dim"],
        state_token_dim=config["model"]["state_token_dim"],
        max_lang_cond_len=config["dataset"]["tokenizer_max_length"],
        img_cond_len=img_cond_len,
        img_pos_embed_config=[
            ("image", (
                config["common"]["img_history_size"],
                config["common"]["num_cameras"],
                -vision_encoder.num_patches,
            )),
        ],
        lang_pos_embed_config=[
            ("lang", -config["dataset"]["tokenizer_max_length"]),
        ],
        dtype=weight_dtype,
        enable_soft_routing_repa=enable_soft_routing_repa,
        soft_routing_repa_weight=soft_routing_repa_weight,
        global_feature_dim=global_feature_dim,
        depth_feature_dim=depth_feature_dim,  # ⭐ 传递深度维度
        soft_routing_config=soft_routing_config,
        repa_activation_layer=repa_activation_layer
    )
    # 加载预训练权重（如果提供）
    if args.pretrained_model_name_or_path and os.path.isfile(args.pretrained_model_name_or_path):
        logger.info(f"Loading pretrained weights: {args.pretrained_model_name_or_path}")
        ckpt = torch.load(args.pretrained_model_name_or_path, map_location="cpu")

        if isinstance(ckpt, dict) and "module" in ckpt:
            pretrained_sd = ckpt["module"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            pretrained_sd = ckpt["state_dict"]
        else:
            pretrained_sd = ckpt

        own_sd = rdt.state_dict()
        filtered = {}
        for k, v in pretrained_sd.items():
            if k in own_sd and v.shape == own_sd[k].shape:
                filtered[k] = v
            else:
                logger.debug(f"Skipping parameter {k}: checkpoint {tuple(v.shape)} vs model {tuple(own_sd.get(k, v).shape)}")

        rdt.load_state_dict(filtered, strict=False)
        logger.info("Loaded matching pretrained weights; others remain randomly initialized")
    else:
        logger.info("Only using config; skipping pretrained weight loading")

    # EMA模型
    ema_rdt = copy.deepcopy(rdt)
    ema_model = EMAModel(
        ema_rdt,
        update_after_step=config["model"]["ema"]["update_after_step"],
        inv_gamma=config["model"]["ema"]["inv_gamma"],
        power=config["model"]["ema"]["power"],
        min_value=config["model"]["ema"]["min_value"],
        max_value=config["model"]["ema"]["max_value"],
    )

    # 保存钩子
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                model_to_save = model.module if hasattr(model, "module") else model
                if isinstance(model_to_save, type(accelerator.unwrap_model(rdt))):
                    model_to_save.save_pretrained(output_dir)

    accelerator.register_save_state_pre_hook(save_model_hook)

    if args.gradient_checkpointing:
        raise NotImplementedError("Gradient checkpointing is not yet implemented.")

    # Enable TF32 for faster training on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size *
                              accelerator.num_processes)

    # 优化器
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = rdt.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    critical_annotation_config = model_config.get("critical_annotation_config", {})
    use_dinov2_features = (global_teacher_type == "dinov2")
    use_siglip_global_features = (global_teacher_type == "siglip")
    # ⭐ 创建训练数据集（自动传递参数）
    train_dataset = VLAConsumerDataset(
        model_config_path=args.model_config_path,
        config=config["dataset"],
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_cameras=config["common"]["num_cameras"],
        img_history_size=config["common"]["img_history_size"],
        dataset_type=args.dataset_type,
        image_aug=args.image_aug,
        cond_mask_prob=args.cond_mask_prob,
        cam_ext_mask_prob=args.cam_ext_mask_prob,
        state_noise_snr=args.state_noise_snr,
        use_hdf5=args.load_from_hdf5,
        use_precomp_lang_embed=args.precomp_lang_embed,
        # ⭐ 自动设置的特征标志
        use_dinov2_features=use_dinov2_features,
        use_siglip_global_features=use_siglip_global_features,
        use_depth_anything_v2=use_depth_anything_v2,  # ⭐ 新增
        use_siglip_depth_features=use_siglip_depth_features,  # ⭐ 新增
        # 关键时间段标注参数
        task_type=task_type,
        enable_critical_annotation=enable_critical_annotation,
        critical_annotation_config=critical_annotation_config,
    )
    
    # ⭐ 创建采样数据集（自动传递参数）
    sample_dataset = VLAConsumerDataset(
        model_config_path=args.model_config_path,
        config=config["dataset"],
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_cameras=config["common"]["num_cameras"],
        img_history_size=config["common"]["img_history_size"],
        dataset_type=args.dataset_type,
        image_aug=args.image_aug,
        cond_mask_prob=args.cond_mask_prob,
        cam_ext_mask_prob=args.cam_ext_mask_prob,
        state_noise_snr=args.state_noise_snr,
        use_hdf5=args.load_from_hdf5,
        use_precomp_lang_embed=args.precomp_lang_embed,
        # ⭐ 自动设置的特征标志
        use_dinov2_features=use_dinov2_features,
        use_siglip_global_features=use_siglip_global_features,
        use_depth_anything_v2=use_depth_anything_v2,  # ⭐ 新增
        use_siglip_depth_features=use_siglip_depth_features,  # ⭐ 新增
        # 关键时间段标注参数
        task_type=task_type,
        enable_critical_annotation=enable_critical_annotation,
        critical_annotation_config=critical_annotation_config,
    )

    data_collator = DataCollatorForVLAConsumerDataset(tokenizer)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    
    sample_dataloader = torch.utils.data.DataLoader(
        sample_dataset,
        batch_size=args.sample_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    # 学习率调度器
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # 准备训练
    rdt, optimizer, train_dataloader, sample_dataloader, lr_scheduler = (
        accelerator.prepare(rdt, optimizer, train_dataloader, sample_dataloader, lr_scheduler)
    )

    ema_rdt.to(accelerator.device, dtype=weight_dtype)

    if text_encoder is not None:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    if vision_encoder is not None:
        vision_encoder.vision_tower.to(accelerator.device, dtype=weight_dtype)

    # 重新计算训练步数
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # 初始化追踪器
    if accelerator.is_main_process:
        tracker_config = vars(args).copy()
        tracker_config.update({
            'global_teacher_type': global_teacher_type,  # ⭐ 记录全局教师类型
            'global_feature_dim': global_feature_dim,  # ⭐ 记录全局特征维度
            'task_type': task_type,
            'enable_critical_annotation': enable_critical_annotation,
            'critical_annotation_config': critical_annotation_config,
            'enable_soft_routing_repa': enable_soft_routing_repa,
            'soft_routing_config': soft_routing_config,
        })
        # ⭐ 根据全局教师类型设置运行名
        task_name = "grasp" if task_type == 1 else "click"
        run_name = f"RDT_SoftRouting_{global_teacher_type.upper()}_{task_name}_{args.CONFIG_NAME}"
        accelerator.init_trackers(
            "VLA_Soft_Routing_Dual_Teacher_REPA",
            config=tracker_config,
            init_kwargs={"wandb": {
                "name": run_name,
            }},
        )
    # 训练信息
    total_batch_size = (args.train_batch_size * accelerator.num_processes * 
                       args.gradient_accumulation_steps)

    logger.info("***** 开始软路由双教师REPA训练 *****")
    logger.info(f"  示例数量 = {len(train_dataset)}")
    logger.info(f"  Epoch数量 = {args.num_train_epochs}")
    logger.info(f"  全局教师类型 = {global_teacher_type.upper()}")  # ⭐ 新增
    logger.info(f"  深度教师类型 = DepthAnythingV2")
    logger.info(f"  任务类型 = {TaskType(task_type).name}")
    logger.info(f"  路由策略 = 规则驱动 + 可选神经调整")
    
    global_step = 0
    first_epoch = 0

    # 可能从检查点恢复
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' not found. Starting new training.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint: {path}")
            try:
                accelerator.load_state(os.path.join(args.output_dir, path))
            except:
                logger.info("Failed to restore training state. Trying to load model checkpoint only.")
                checkpoint = torch.load(
                    os.path.join(args.output_dir, path, "pytorch_model", "mp_rank_00_model_states.pt"))
                rdt.module.load_state_dict(checkpoint["module"])

            load_model(ema_rdt, os.path.join(args.output_dir, path, "ema", "model.safetensors"))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # 进度条
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    # 精简版统计变量
    soft_routing_stats = {
        'total_samples': 0,
        'critical_timesteps': 0,
    }
    
    # 训练循环中的修改
    for epoch in range(first_epoch, args.num_train_epochs):
        rdt.train()
        accelerator.unwrap_model(rdt).reset_batch_count()

        for batch in train_dataloader:
            with accelerator.accumulate(rdt):
                # 准备输入数据
                images = batch["images"].to(dtype=weight_dtype)
                states = batch["states"].to(dtype=weight_dtype)[:, -1:, :]
                actions = batch["actions"].to(dtype=weight_dtype)
                state_elem_mask = batch["state_elem_mask"].to(dtype=weight_dtype)
                ctrl_freqs = batch["ctrl_freqs"]
                critical_labels = batch.get("critical_labels", None)

                # 编码视觉特征
                with torch.no_grad():
                    # 1️⃣ SigLIP编码（主干）
                    batch_size, _, C, H, W = images.shape
                    image_embeds = vision_encoder(images.reshape(-1, C, H, W)).detach()
                    image_embeds = image_embeds.reshape(
                        (batch_size, -1, vision_encoder.hidden_size)
                    )

                    # 2️⃣ 文本编码
                    lang_attn_mask = batch["lang_attn_mask"]
                    text_embeds = (
                        batch["lang_embeds"].to(dtype=weight_dtype) 
                        if args.precomp_lang_embed 
                        else text_encoder(
                            input_ids=batch["input_ids"], 
                            attention_mask=lang_attn_mask
                        )["last_hidden_state"].detach()
                    )

                    # 3️⃣ ⭐ 全局教师编码
                    global_cls_token = None
                    if global_teacher_encoder is not None:
                        if global_teacher_type == "dinov2" and "dinov2_images" in batch:
                            dinov2_images = batch["dinov2_images"].to(dtype=weight_dtype)
                            dinov2_input = dinov2_images[:, 0]
                            global_cls_token = global_teacher_encoder(dinov2_input)
                        
                        elif global_teacher_type == "siglip" and "siglip_global_images" in batch:
                            siglip_images = batch["siglip_global_images"].to(dtype=weight_dtype)
                            siglip_input = siglip_images[:, 0]
                            global_cls_token = global_teacher_encoder(siglip_input)

                    # 4️⃣ ⭐ 深度教师编码
                    depth_features = None
                    if depth_teacher_encoder is not None:
                        if depth_teacher_type == "depth_anything_v2" and "depth_images" in batch:
                            depth_images = batch["depth_images"].to(dtype=weight_dtype)
                            depth_input = depth_images[:, 0]
                            depth_features, _ = depth_teacher_encoder(depth_input)
                        
                        elif depth_teacher_type == "siglip" and "siglip_depth_images" in batch:
                            siglip_depth_images = batch["siglip_depth_images"].to(dtype=weight_dtype)
                            siglip_depth_input = siglip_depth_images[:, 0]
                            depth_features, _ = depth_teacher_encoder(siglip_depth_input)

                # 计算损失（自动处理不同维度）
                state_elem_mask = state_elem_mask.unsqueeze(1)
                if enable_soft_routing_repa:
                    total_loss, diffusion_loss, repa_loss, detailed_metrics = (
                        accelerator.unwrap_model(rdt).compute_loss(
                            lang_tokens=text_embeds,
                            lang_attn_mask=lang_attn_mask,
                            img_tokens=image_embeds,
                            state_tokens=states,
                            action_gt=actions,
                            action_mask=state_elem_mask,
                            ctrl_freqs=ctrl_freqs,
                            cls_token=global_cls_token,
                            depth_features=depth_features,
                            critical_labels=critical_labels,
                        )
                    )
                    
                    # ⭐⭐⭐ 关键修复：将 total_loss 赋值给 loss
                    loss = total_loss
                    
                    # 精简版指标收集
                    loss_for_log = {
                        "diffusion_loss": diffusion_loss.detach().item(),
                        "repa_loss": repa_loss.detach().item(),
                        "alignment_loss": detailed_metrics.get('soft_routing_alignment_loss', 0.0),
                        'global_similarity': detailed_metrics.get('global_similarity_avg', 0.0),
                        'depth_similarity': detailed_metrics.get('depth_similarity_avg', 0.0),
                        'critical_ratio': detailed_metrics.get('critical_ratio', 0.0),
                        'avg_global_weight': detailed_metrics.get('avg_global_weight', 0.5),
                        'avg_depth_weight': detailed_metrics.get('avg_depth_weight', 0.5),
                    }
                    
                    if 'critical_avg_global_weight' in detailed_metrics:
                        loss_for_log.update({
                            'critical_global_weight': detailed_metrics['critical_avg_global_weight'],
                            'critical_depth_weight': detailed_metrics['critical_avg_depth_weight'],
                        })
                    
                    if 'non_critical_avg_global_weight' in detailed_metrics:
                        loss_for_log.update({
                            'non_critical_global_weight': detailed_metrics['non_critical_avg_global_weight'],
                            'non_critical_depth_weight': detailed_metrics['non_critical_avg_depth_weight'],
                        })
                    
                    if 'weight_drift' in detailed_metrics:
                        loss_for_log['weight_drift'] = detailed_metrics['weight_drift']
                    
                else:
                    loss = rdt(
                        lang_tokens=text_embeds,
                        lang_attn_mask=lang_attn_mask,
                        img_tokens=image_embeds,
                        state_tokens=states,
                        action_gt=actions,
                        action_mask=state_elem_mask,
                        ctrl_freqs=ctrl_freqs,
                    )
                    loss_for_log = {"diffusion_loss": loss.detach().item()}

                # 反向传播
                accelerator.backward(loss)  
                if accelerator.sync_gradients:
                    params_to_clip = rdt.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # EMA更新
            ema_model.step(accelerator.unwrap_model(rdt))

            # 检查点和采样
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_period == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    ema_save_path = os.path.join(save_path, f"ema")
                    accelerator.save_model(ema_rdt, ema_save_path)
                    logger.info(f"💾 保存检查点到 {save_path}")

                # 精简版监控
                if global_step % 500 == 0 and enable_soft_routing_repa:
                    logger.info(f"Step {global_step} - 核心指标:")
                    logger.info(f"  损失: diffusion={loss_for_log.get('diffusion_loss', 0):.4f}, "
                               f"repa={loss_for_log.get('repa_loss', 0):.4f}")
                    logger.info(f"  全局教师: {global_teacher_type.upper()}")  # ⭐ 显示当前教师
                    
                    if 'global_similarity' in loss_for_log:
                        logger.info(f"  对齐质量: global_sim={loss_for_log['global_similarity']:.3f}, "
                                   f"depth_sim={loss_for_log['depth_similarity']:.3f}")
                        logger.info(f"  路由健康: critical_ratio={loss_for_log['critical_ratio']:.3f}, "
                                   f"avg_weights=[{loss_for_log['avg_global_weight']:.3f}, "
                                   f"{loss_for_log['avg_depth_weight']:.3f}]")

                if args.sample_period > 0 and global_step % args.sample_period == 0:
                    sample_loss_for_log = log_sample_res(
                        text_encoder,
                        vision_encoder,
                        rdt,
                        args,
                        accelerator,
                        weight_dtype,
                        sample_dataset.get_dataset_id2name(),
                        sample_dataloader,
                        logger,
                    )
                    logger.info(sample_loss_for_log)
                    accelerator.log(sample_loss_for_log, step=global_step)

            # 记录日志
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            logs.update(loss_for_log)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # 训练结束时的总结
    if accelerator.is_main_process:
        logger.info("训练完成 - 最终总结:")
        logger.info(f"  全局教师类型: {global_teacher_type.upper()}")  # ⭐ 记录
        logger.info(f"  全局特征维度: {global_feature_dim}")
        
        if soft_routing_stats['total_samples'] > 0:
            final_critical_ratio = (soft_routing_stats['critical_timesteps'] / 
                                  soft_routing_stats['total_samples'])
            logger.info(f"  总时间步: {soft_routing_stats['total_samples']}")
            logger.info(f"  关键时间步: {soft_routing_stats['critical_timesteps']}")
            logger.info(f"  最终关键比例: {final_critical_ratio:.3f}")
        
        # 保存训练配置
        task_name = "grasp" if task_type == 1 else "click"
        final_config = {
            'global_teacher_type': global_teacher_type,  # ⭐ 保存教师类型
            'global_feature_dim': global_feature_dim,
            'task_type': task_type,
            'task_name': task_name,
            'enable_critical_annotation': enable_critical_annotation,
            'enable_soft_routing_repa': enable_soft_routing_repa,
            'final_statistics': {
                'total_timesteps': soft_routing_stats['total_samples'],
                'critical_timesteps': soft_routing_stats['critical_timesteps'],
                'critical_ratio': (soft_routing_stats['critical_timesteps'] / 
                                 soft_routing_stats['total_samples'] 
                                 if soft_routing_stats['total_samples'] > 0 else 0.0),
            },
            'training_hyperparameters': {
                'soft_routing_repa_weight': soft_routing_repa_weight,
                'learning_rate': args.learning_rate,
                'train_batch_size': args.train_batch_size,
                'max_train_steps': args.max_train_steps,
            },
        }
        
        import json
        config_filename = f"soft_routing_{global_teacher_type}_training_config.json"
        with open(os.path.join(args.output_dir, config_filename), "w") as f:
            json.dump(final_config, f, indent=2)
        
        logger.info(f"训练配置已保存到: {os.path.join(args.output_dir, config_filename)}")
    # 保存最终模型
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(rdt).save_pretrained(args.output_dir)
        ema_save_path = os.path.join(args.output_dir, f"ema")
        accelerator.save_model(ema_rdt, ema_save_path)

        logger.info(f"Model saved to {args.output_dir}")

        if args.push_to_hub:
            save_model_card(
                repo_id,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of soft routing dual-teacher REPA + critical timestep training",
                token=args.hub_token,
                allow_patterns=["pytorch_model.bin", "*.json", "*.md"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train RDT with Soft Routing Dual-Teacher REPA")
    
    # 基础训练参数
    parser.add_argument("--config_path", type=str, required=True, help="Path to config file")
    parser.add_argument("--model_config_path", type=str, required=True, help="Path to model config file")
    parser.add_argument("--pretrained_model_name_or_path", type=str, help="Path to pretrained model")
    parser.add_argument("--pretrained_text_encoder_name_or_path", type=str, help="Path to pretrained text encoder")
    parser.add_argument("--pretrained_vision_encoder_name_or_path", type=str, help="Path to pretrained vision encoder")
    
    # 训练配置
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="wandb", help="Logging platform")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dataset_type", type=str, default="pretrain", choices=["pretrain", "finetune"])
    
    # 模型和训练参数
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--sample_batch_size", type=int, default=64)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    
    # 优化器参数
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_8bit_adam", action="store_true")
    
    # 学习率调度器
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--scale_lr", action="store_true")
    
    # 检查点和采样
    parser.add_argument("--checkpointing_period", type=int, default=2500)
    parser.add_argument("--sample_period", type=int, default=100)
    parser.add_argument("--num_sample_batches", type=int, default=2)
    parser.add_argument("--checkpoints_total_limit", type=int, default=40)
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to checkpoint to resume from")
    
    # 数据和预处理
    parser.add_argument("--load_from_hdf5", action="store_true", help="Load data from HDF5 files")
    parser.add_argument("--precomp_lang_embed", action="store_true", help="Use precomputed language embeddings")
    parser.add_argument("--image_aug", action="store_true", help="Enable image augmentation")
    parser.add_argument("--cond_mask_prob", type=float, default=0.1, help="Condition masking probability")
    parser.add_argument("--cam_ext_mask_prob", type=float, default=-1.0, help="External camera masking probability")
    parser.add_argument("--state_noise_snr", type=float, help="State noise SNR")
    
    # 系统配置
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--set_grads_to_none", action="store_true")
    parser.add_argument("--deepspeed", type=str, help="Path to DeepSpeed config")
    
    # Hub相关
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, help="Hub model ID")
    parser.add_argument("--hub_token", type=str, help="Hub token")
    
    # 软路由参数
    parser.add_argument("--CONFIG_NAME", type=str, default="soft_routing", help="Configuration name for logging")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    
    # 开始训练
    train(args, logger)
    
    
    
    
    
    
    
    
    
    
    
    
# import copy
# import logging
# import math
# import os
# from pathlib import Path

# import diffusers
# import torch
# import torch.utils.checkpoint
# import transformers
# import yaml
# from accelerate import Accelerator
# from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, set_seed
# from diffusers.optimization import get_scheduler
# from diffusers.utils import is_wandb_available
# from huggingface_hub import create_repo, upload_folder
# from tqdm.auto import tqdm
# from safetensors.torch import load_model

# from models.ema_model import EMAModel
# from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
# from models.multimodal_encoder.t5_encoder import T5Embedder
# from models.rdt_runner import RDTRunner
# from train.dataset import DataCollatorForVLAConsumerDataset, VLAConsumerDataset
# from train.sample import log_sample_res

# # 导入DINOv2和DepthAnythingV2编码器
# from models.multimodal_encoder.dinov2_encoder import create_dinov2_encoder
# from models.multimodal_encoder.depth_encoder import create_depth_encoder



# if is_wandb_available():
#     import wandb


# def save_model_card(repo_id: str, base_model=str, repo_folder=None):
#     yaml_header = f"""
# ---
# license: mit
# base_model: {base_model}
# language:
# - en
# pipeline_tag: robotics
# library_name: transformers
# tags:
# - robotics
# - pytorch
# - multimodal
# - pretraining
# - vla
# - diffusion
# - rdt
# - soft-routing
# - dual-teachers
# - critical-timestep
# - binary-labels
# ---
#     """
#     model_card = f"""
# # RDT with Soft Routing Dual-Teacher REPA - {repo_id}

# This is a RDT model with soft routing dual-teacher REPA alignment loss, task-driven critical timestep annotation 
# derived from {base_model}. The weights were trained using [RDT](https://rdt-robotics.github.io/rdt-robotics/) 
# with advanced soft routing multi-modal alignment strategies.

# ## Key Features
# - **Soft Routing Strategy**: Rule-driven weight allocation based on binary critical timestep labels
# - **Critical Timestep Annotation**: Task-driven annotation for precise temporal alignment
# - **Dual Visual Teachers**: DINOv2 (global semantic) + DepthAnythingV2 (depth geometric)
# - **Neural Weight Adjustment**: Optional fine-tuning with temporal smoothing
# - **Contrastive Learning**: Enhanced feature alignment with contrastive loss

# ## Weight Allocation Strategy
# - **Critical Timesteps (1)**: Global 25%, Depth 75% - Focus on precise manipulation
# - **Non-Critical Timesteps (0)**: Global 75%, Depth 25% - Focus on scene understanding

# ## Architecture Components
# 1. **Binary Label Soft Router**: Rule-driven weight allocation with optional neural adjustment
# 2. **Dual Visual Teachers**: DINOv2 (global) + DepthAnythingV2 (geometric)
# 3. **Temporal Smoothing**: Prevents sudden weight transitions
# 4. **Contrastive Learning**: Enhances feature alignment quality

# ## Task Types Supported
# - **Grasp Tasks (task_type=1)**: Deceleration → Gripper closing alignment
# - **Click Tasks (task_type=2)**: Gripper closing → Deceleration alignment
# """
#     with open(os.path.join(repo_folder, "README.md"), "w") as f:
#         f.write(yaml_header + model_card)


# def check_critical_alerts(metrics, global_step, logger):
#     """
#     检查关键异常情况并发出预警
#     """
#     alerts = []
    
#     # 1. 特征对齐失效
#     if metrics.get('global_similarity', 1.0) < 0.3 and global_step > 1000:
#         alerts.append("WARNING: Low global similarity - feature alignment may be failing")
    
#     # 2. 权重分配严重偏离
#     if 'critical_global_weight' in metrics:
#         expected_critical_global = 0.25
#         actual = metrics['critical_global_weight']
#         if abs(actual - expected_critical_global) > 0.2:  # 偏离超过20%
#             alerts.append(f"WARNING: Critical weight deviation - expected {expected_critical_global}, got {actual:.3f}")
    
#     # 3. 权重调整过度
#     if metrics.get('weight_drift', 0) > 0.15:
#         alerts.append("WARNING: Excessive weight drift - neural adjustment may be too aggressive")
    
#     # 4. 数据不平衡
#     critical_ratio = metrics.get('critical_ratio', 0.3)
#     if critical_ratio < 0.1 or critical_ratio > 0.6:
#         alerts.append(f"WARNING: Critical ratio imbalance - {critical_ratio:.3f}")
    
#     # 记录警告
#     for alert in alerts:
#         logger.warning(alert)
    
#     return alerts


# def train(args, logger):
#     # Read the config
#     with open(args.config_path, "r") as fp:
#         config = yaml.safe_load(fp)

#     with open(args.model_config_path, "r") as f:
#         model_config = yaml.safe_load(f)
    
#     args.output_dir = model_config["checkpoint_path"]
#     logging_dir = Path(args.output_dir, args.logging_dir)

#     accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
#     accelerator = Accelerator(
#         deepspeed_plugin=(DeepSpeedPlugin(hf_ds_config=args.deepspeed) if args.deepspeed is not None else None),
#         gradient_accumulation_steps=args.gradient_accumulation_steps,
#         mixed_precision=args.mixed_precision,
#         log_with=args.report_to,
#         project_dir=logging_dir,
#         project_config=accelerator_project_config,
#     )

#     if args.report_to == "wandb":
#         if not is_wandb_available():
#             raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

#     # Make one log on every process with the configuration for debugging.
#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S",
#         level=logging.INFO,
#     )
#     logger.info(accelerator.state, main_process_only=False)
#     if accelerator.is_local_main_process:
#         transformers.utils.logging.set_verbosity_warning()
#         diffusers.utils.logging.set_verbosity_info()
#     else:
#         transformers.utils.logging.set_verbosity_error()
#         diffusers.utils.logging.set_verbosity_error()

#     # If passed along, set the training seed now.
#     if args.seed is not None:
#         set_seed(args.seed)

#     # Handle the repository creation
#     if accelerator.is_main_process:
#         if args.output_dir is not None:
#             os.makedirs(args.output_dir, exist_ok=True)

#         if args.push_to_hub:
#             repo_id = create_repo(
#                 repo_id=args.hub_model_id or Path(args.output_dir).name,
#                 exist_ok=True,
#                 token=args.hub_token,
#             ).repo_id

#     # For mixed precision training
#     weight_dtype = torch.float32
#     if accelerator.mixed_precision == "fp16":
#         weight_dtype = torch.float16
#     elif accelerator.mixed_precision == "bf16":
#         weight_dtype = torch.bfloat16

#     # 从模型配置读取视觉融合参数
#     enable_vision_fusion = model_config.get("enable_vision_fusion", True)
#     vision_fusion_type = model_config.get("vision_fusion_type", "cross_attention")
#     use_dinov2_features = model_config.get("use_dinov2_features", True)
#     use_depth_features = model_config.get("use_depth_features", True)
    
#     # 🔴 强制关闭REPA和关键时间段标注
#     enable_soft_routing_repa = False
#     enable_critical_annotation = False
    
#     # 🔧 修复：定义所有可能用到的变量，提供默认值
#     task_type = model_config.get("task_type", 1)
#     critical_annotation_config = model_config.get("critical_annotation_config", {})
#     soft_routing_config = model_config.get("soft_routing_config", {})
#     soft_routing_repa_weight = model_config.get("soft_routing_repa_weight", 0.2)
    
#     logger.info(f"🔧 视觉融合配置:")
#     logger.info(f"   - 视觉融合启用: {enable_vision_fusion}")
#     logger.info(f"   - 融合类型: {vision_fusion_type}")
#     logger.info(f"   - DINOv2特征: {use_dinov2_features}")
#     logger.info(f"   - Depth特征: {use_depth_features}")
#     logger.info(f"   - REPA对齐: {enable_soft_routing_repa} (已关闭)")
#     logger.info(f"   - 关键时间段标注: {enable_critical_annotation} (已关闭)")

#     # 文本编码器
#     if args.precomp_lang_embed:
#         tokenizer, text_encoder = None, None
#     else:
#         text_embedder = T5Embedder(
#             from_pretrained=args.pretrained_text_encoder_name_or_path,
#             model_max_length=config["dataset"]["tokenizer_max_length"],
#             device=accelerator.device,
#         )
#         tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model

#     # SigLIP编码器（主干）
#     vision_encoder = SiglipVisionTower(
#         vision_tower=args.pretrained_vision_encoder_name_or_path, 
#         args=None
#     )
#     image_processor = vision_encoder.image_processor

#     # 🆕 DINOv2编码器（全局语义patch tokens）
#     dinov2_encoder = None
#     if use_dinov2_features and enable_vision_fusion:
#         logger.info("📦 加载DINOv2编码器 (Patch Tokens)...")
#         dinov2_encoder = create_dinov2_encoder(
#             model_size="large", 
#             select_feature="patch"  # 🔴 只要patch tokens
#         )
#         dinov2_encoder.to(accelerator.device, dtype=weight_dtype)
#         dinov2_encoder.print_model_info()

#     # 🆕 DepthAnythingV2编码器（深度patch tokens）
#     depth_encoder = None
#     if use_depth_features and enable_vision_fusion:
#         logger.info("📦 加载DepthAnythingV2编码器 (Patch Tokens)...")
#         depth_encoder = create_depth_encoder(
#             model_size="metric_large",
#             feature_dim=1024,
#             device=accelerator.device,
#             use_metric_model=True
#         )
#         depth_encoder.to(accelerator.device, dtype=weight_dtype)
#         depth_encoder.print_model_info()
#     logger.info("🔨 构建带视觉融合的RDT模型...")
#     img_cond_len = (config["common"]["img_history_size"] * 
#                     config["common"]["num_cameras"] *
#                     vision_encoder.num_patches)
    
#     rdt = RDTRunner(
#         action_dim=config["common"]["state_dim"],
#         pred_horizon=config["common"]["action_chunk_size"],
#         config=config["model"],
#         lang_token_dim=config["model"]["lang_token_dim"],
#         img_token_dim=config["model"]["img_token_dim"],
#         state_token_dim=config["model"]["state_token_dim"],
#         max_lang_cond_len=config["dataset"]["tokenizer_max_length"],
#         img_cond_len=img_cond_len,
#         img_pos_embed_config=[
#             ("image", (
#                 config["common"]["img_history_size"],
#                 config["common"]["num_cameras"],
#                 -vision_encoder.num_patches,
#             )),
#         ],
#         lang_pos_embed_config=[
#             ("lang", -config["dataset"]["tokenizer_max_length"]),
#         ],
#         dtype=weight_dtype,
#         # 🆕 视觉融合配置
#         enable_vision_fusion=enable_vision_fusion,
#         vision_fusion_type=vision_fusion_type,
#         dinov2_feature_dim=1024,
#         depth_feature_dim=1024,
#         fusion_num_heads=8,
#         fusion_dropout=0.1,
#         img_history_size=config["common"]["img_history_size"],  # 🆕 传入
#         num_cameras=config["common"]["num_cameras"],            # 🆕 传入
#     )
#     # 加载预训练权重（如果提供）
#     if args.pretrained_model_name_or_path and os.path.isfile(args.pretrained_model_name_or_path):
#         logger.info(f"Loading pretrained weights: {args.pretrained_model_name_or_path}")
#         ckpt = torch.load(args.pretrained_model_name_or_path, map_location="cpu")

#         if isinstance(ckpt, dict) and "module" in ckpt:
#             pretrained_sd = ckpt["module"]
#         elif isinstance(ckpt, dict) and "state_dict" in ckpt:
#             pretrained_sd = ckpt["state_dict"]
#         else:
#             pretrained_sd = ckpt

#         own_sd = rdt.state_dict()
#         filtered = {}
#         for k, v in pretrained_sd.items():
#             if k in own_sd and v.shape == own_sd[k].shape:
#                 filtered[k] = v
#             else:
#                 logger.debug(f"Skipping parameter {k}: checkpoint {tuple(v.shape)} vs model {tuple(own_sd.get(k, v).shape)}")

#         rdt.load_state_dict(filtered, strict=False)
#         logger.info("Loaded matching pretrained weights; others remain randomly initialized")
#     else:
#         logger.info("Only using config; skipping pretrained weight loading")

#     # EMA模型
#     ema_rdt = copy.deepcopy(rdt)
#     ema_model = EMAModel(
#         ema_rdt,
#         update_after_step=config["model"]["ema"]["update_after_step"],
#         inv_gamma=config["model"]["ema"]["inv_gamma"],
#         power=config["model"]["ema"]["power"],
#         min_value=config["model"]["ema"]["min_value"],
#         max_value=config["model"]["ema"]["max_value"],
#     )

#     # 保存钩子
#     def save_model_hook(models, weights, output_dir):
#         if accelerator.is_main_process:
#             for model in models:
#                 model_to_save = model.module if hasattr(model, "module") else model
#                 if isinstance(model_to_save, type(accelerator.unwrap_model(rdt))):
#                     model_to_save.save_pretrained(output_dir)

#     accelerator.register_save_state_pre_hook(save_model_hook)

#     if args.gradient_checkpointing:
#         raise NotImplementedError("Gradient checkpointing is not yet implemented.")

#     # Enable TF32 for faster training on Ampere GPUs
#     if args.allow_tf32:
#         torch.backends.cuda.matmul.allow_tf32 = True

#     if args.scale_lr:
#         args.learning_rate = (args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size *
#                               accelerator.num_processes)

#     # 优化器
#     if args.use_8bit_adam:
#         try:
#             import bitsandbytes as bnb
#         except ImportError:
#             raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")
#         optimizer_class = bnb.optim.AdamW8bit
#     else:
#         optimizer_class = torch.optim.AdamW

#     params_to_optimize = rdt.parameters()
#     optimizer = optimizer_class(
#         params_to_optimize,
#         lr=args.learning_rate,
#         betas=(args.adam_beta1, args.adam_beta2),
#         weight_decay=args.adam_weight_decay,
#         eps=args.adam_epsilon,
#     )

#     train_dataset = VLAConsumerDataset(
#         model_config_path=args.model_config_path,
#         config=config["dataset"],
#         tokenizer=tokenizer,
#         image_processor=image_processor,
#         num_cameras=config["common"]["num_cameras"],
#         img_history_size=config["common"]["img_history_size"],
#         dataset_type=args.dataset_type,
#         image_aug=args.image_aug,
#         cond_mask_prob=args.cond_mask_prob,
#         cam_ext_mask_prob=args.cam_ext_mask_prob,
#         state_noise_snr=args.state_noise_snr,
#         use_hdf5=args.load_from_hdf5,
#         use_precomp_lang_embed=args.precomp_lang_embed,
#         use_dinov2_features=use_dinov2_features,
#         use_depth_features=use_depth_features,
#         # 🔴 关闭关键时间段标注
#         enable_critical_annotation=False,
#     )
    
#     sample_dataset = VLAConsumerDataset(
#         model_config_path=args.model_config_path,
#         config=config["dataset"],
#         tokenizer=tokenizer,
#         image_processor=image_processor,
#         num_cameras=config["common"]["num_cameras"],
#         img_history_size=config["common"]["img_history_size"],
#         dataset_type=args.dataset_type,
#         image_aug=False,
#         cond_mask_prob=0,
#         cam_ext_mask_prob=-1,
#         state_noise_snr=None,
#         use_hdf5=args.load_from_hdf5,
#         use_precomp_lang_embed=args.precomp_lang_embed,
#         use_dinov2_features=use_dinov2_features,
#         use_depth_features=use_depth_features,
#         enable_critical_annotation=False,
#     )

#     data_collator = DataCollatorForVLAConsumerDataset(tokenizer)

#     train_dataloader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=args.train_batch_size,
#         shuffle=True,
#         collate_fn=data_collator,
#         num_workers=args.dataloader_num_workers,
#         pin_memory=True,
#         persistent_workers=True,
#     )
    
#     sample_dataloader = torch.utils.data.DataLoader(
#         sample_dataset,
#         batch_size=args.sample_batch_size,
#         shuffle=True,
#         collate_fn=data_collator,
#         num_workers=args.dataloader_num_workers,
#         pin_memory=True,
#         persistent_workers=True,
#     )

#     # 学习率调度器
#     overrode_max_train_steps = False
#     num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
#     if args.max_train_steps is None:
#         args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
#         overrode_max_train_steps = True

#     lr_scheduler = get_scheduler(
#         args.lr_scheduler,
#         optimizer=optimizer,
#         num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
#         num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
#         num_cycles=args.lr_num_cycles,
#         power=args.lr_power,
#     )

#     # 准备训练
#     rdt, optimizer, train_dataloader, sample_dataloader, lr_scheduler = (
#         accelerator.prepare(rdt, optimizer, train_dataloader, sample_dataloader, lr_scheduler)
#     )

#     ema_rdt.to(accelerator.device, dtype=weight_dtype)

#     if text_encoder is not None:
#         text_encoder.to(accelerator.device, dtype=weight_dtype)

#     if vision_encoder is not None:
#         vision_encoder.vision_tower.to(accelerator.device, dtype=weight_dtype)

#     # 重新计算训练步数
#     num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
#     if overrode_max_train_steps:
#         args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
#     args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

#     # 初始化追踪器
#     if accelerator.is_main_process:
#         tracker_config = vars(args).copy()
        
#         # 基础配置
#         tracker_config.update({
#             'enable_vision_fusion': enable_vision_fusion,
#             'vision_fusion_type': vision_fusion_type,
#             'use_dinov2_features': use_dinov2_features,
#             'use_depth_features': use_depth_features,
#             'enable_soft_routing_repa': enable_soft_routing_repa,
#             'enable_critical_annotation': enable_critical_annotation,
#         })
        
#         # 只在启用时添加相关配置
#         if enable_critical_annotation:
#             tracker_config['task_type'] = task_type
#             tracker_config['critical_annotation_config'] = critical_annotation_config
        
#         if enable_soft_routing_repa:
#             tracker_config['soft_routing_config'] = soft_routing_config
        
#         # 🔧 根据模式设置项目名和运行名
#         if enable_vision_fusion:
#             project_name = "VLA_Vision_Fusion"
#             run_name = f"RDT_VisionFusion_{vision_fusion_type}_{args.CONFIG_NAME}"
#         elif enable_soft_routing_repa:
#             project_name = "VLA_Soft_Routing_Dual_Teacher_REPA"
#             task_name = "grasp" if task_type == 1 else "click"
#             run_name = f"RDT_SoftRouting_{task_name}_{args.CONFIG_NAME}"
#         else:
#             project_name = "VLA_Training"
#             run_name = f"RDT_{args.CONFIG_NAME}"
        
#         accelerator.init_trackers(
#             project_name,
#             config=tracker_config,
#             init_kwargs={"wandb": {
#                 "name": run_name,
#             }},
#         )

#     # 训练信息
#     total_batch_size = (args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps)

#     logger.info("***** Starting Soft Routing Dual-Teacher REPA Training *****")
#     logger.info(f"  Num examples = {len(train_dataset)}")
#     logger.info(f"  Num Epochs = {args.num_train_epochs}")
#     logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
#     logger.info(f"  Total train batch size = {total_batch_size}")
#     logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
#     logger.info(f"  Total optimization steps = {args.max_train_steps}")
#     task_name = "grasp" if task_type == 1 else ("click" if task_type == 2 else "unknown")
#     logger.info(f"  Task Type = {task_name}")
#     logger.info(f"  Routing Strategy = Rule-driven + Optional Neural Adjustment")
    
#     global_step = 0
#     first_epoch = 0

#     # 可能从检查点恢复
#     if args.resume_from_checkpoint:
#         if args.resume_from_checkpoint != "latest":
#             path = os.path.basename(args.resume_from_checkpoint)
#         else:
#             dirs = os.listdir(args.output_dir)
#             dirs = [d for d in dirs if d.startswith("checkpoint")]
#             dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
#             path = dirs[-1] if len(dirs) > 0 else None

#         if path is None:
#             accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' not found. Starting new training.")
#             args.resume_from_checkpoint = None
#         else:
#             accelerator.print(f"Resuming from checkpoint: {path}")
#             try:
#                 accelerator.load_state(os.path.join(args.output_dir, path))
#             except:
#                 logger.info("Failed to restore training state. Trying to load model checkpoint only.")
#                 checkpoint = torch.load(
#                     os.path.join(args.output_dir, path, "pytorch_model", "mp_rank_00_model_states.pt"))
#                 rdt.module.load_state_dict(checkpoint["module"])

#             load_model(ema_rdt, os.path.join(args.output_dir, path, "ema", "model.safetensors"))
#             global_step = int(path.split("-")[1])

#             resume_global_step = global_step * args.gradient_accumulation_steps
#             first_epoch = global_step // num_update_steps_per_epoch
#             resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

#     # 进度条
#     progress_bar = tqdm(
#         range(global_step, args.max_train_steps),
#         disable=not accelerator.is_local_main_process,
#     )
#     progress_bar.set_description("Steps")

#     # 精简版统计变量
#     soft_routing_stats = {
#         'total_samples': 0,
#         'critical_timesteps': 0,
#     }
    
#     # 训练循环
#     for epoch in range(first_epoch, args.num_train_epochs):
#         rdt.train()
        
#         # 每个epoch开始时重置batch计数
#         accelerator.unwrap_model(rdt).reset_batch_count()

#         if args.resume_from_checkpoint and epoch == first_epoch:
#             progress_bar.update(resume_step // args.gradient_accumulation_steps)

#         for batch in train_dataloader:
#             with accelerator.accumulate(rdt):
#                 # 准备输入数据
#                 images = batch["images"].to(dtype=weight_dtype)
#                 states = batch["states"].to(dtype=weight_dtype)[:, -1:, :]
#                 actions = batch["actions"].to(dtype=weight_dtype)
#                 state_elem_mask = batch["state_elem_mask"].to(dtype=weight_dtype)
#                 ctrl_freqs = batch["ctrl_freqs"]

#                 # ========================================
#                 # 🆕 三路视觉特征提取
#                 # ========================================
                
#                 with torch.no_grad():
#                     # 1️⃣ SigLIP编码（主干）
#                     batch_size, _, C, H, W = images.shape
#                     image_embeds = vision_encoder(
#                         images.reshape(-1, C, H, W)
#                     ).detach()
#                     image_embeds = image_embeds.reshape(
#                         (batch_size, -1, vision_encoder.hidden_size)
#                     )  # (B, img_history_size*num_cameras*729, 1152)

#                     # 2️⃣ 文本编码
#                     lang_attn_mask = batch["lang_attn_mask"]
#                     text_embeds = (
#                         batch["lang_embeds"].to(dtype=weight_dtype) 
#                         if args.precomp_lang_embed 
#                         else text_encoder(
#                             input_ids=batch["input_ids"], 
#                             attention_mask=lang_attn_mask
#                         )["last_hidden_state"].detach()
#                     )

#                     # 3️⃣ DINOv2编码（patch tokens）
#                     dinov2_features = None
#                     if dinov2_encoder is not None and "dinov2_images" in batch:
#                         dinov2_images = batch["dinov2_images"].to(dtype=weight_dtype)
#                         dinov2_input = dinov2_images[:, 0]  # (B, 3, 518, 518)
#                         dinov2_features = dinov2_encoder(dinov2_input)  # (B, 1369, 1024)

#                     # 4️⃣ DepthAnythingV2编码（patch tokens）
#                     depth_features = None
#                     if depth_encoder is not None and "depth_images" in batch:
#                         depth_images = batch["depth_images"].to(dtype=weight_dtype)
#                         depth_input = depth_images[:, 0]  # (B, 3, 518, 518)
#                         depth_features, _ = depth_encoder(depth_input)  # (B, 1370, 1024)

#                 # ========================================
#                 # 🆕 计算损失（视觉融合在内部完成）
#                 # ========================================
                
#                 state_elem_mask = state_elem_mask.unsqueeze(1)
#                 loss = accelerator.unwrap_model(rdt).compute_loss(
#                     lang_tokens=text_embeds,
#                     lang_attn_mask=lang_attn_mask,
#                     img_tokens=image_embeds,  # SigLIP tokens
#                     state_tokens=states,
#                     action_gt=actions,
#                     action_mask=state_elem_mask,
#                     ctrl_freqs=ctrl_freqs,
#                     # 🆕 传入DINOv2和Depth特征
#                     dinov2_features=dinov2_features,
#                     depth_features=depth_features,
#                 )

#                 # 反向传播（保持不变）
#                 accelerator.backward(loss)
#                 if accelerator.sync_gradients:
#                     params_to_clip = rdt.parameters()
#                     accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
#                 optimizer.step()
#                 lr_scheduler.step()
#                 optimizer.zero_grad(set_to_none=args.set_grads_to_none)

#             # EMA更新（保持不变）
#             ema_model.step(accelerator.unwrap_model(rdt))

#             # 检查点和日志（保持不变）
#             if accelerator.sync_gradients:
#                 progress_bar.update(1)
#                 global_step += 1

#                 if global_step % args.checkpointing_period == 0:
#                     save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
#                     accelerator.save_state(save_path)
#                     ema_save_path = os.path.join(save_path, f"ema")
#                     accelerator.save_model(ema_rdt, ema_save_path)
#                     logger.info(f"💾 保存检查点到 {save_path}")

#                 # 🆕 采样评估时也传入编码器
#                 if args.sample_period > 0 and global_step % args.sample_period == 0:
#                     sample_loss_for_log = log_sample_res(
#                     text_encoder,
#                     vision_encoder,
#                     rdt,
#                     args,
#                     accelerator,
#                     weight_dtype,
#                     sample_dataset.get_dataset_id2name(),
#                     sample_dataloader,
#                     logger,
#                     # 🆕 传入额外的编码器
#                     dinov2_encoder=dinov2_encoder,
#                     depth_encoder=depth_encoder,
#                     )
#                     logger.info(sample_loss_for_log)
#                     accelerator.log(sample_loss_for_log, step=global_step)

#             # 记录日志
#             logs = {
#                 "loss": loss.detach().item(), 
#                 "lr": lr_scheduler.get_last_lr()[0]
#             }
#             accelerator.log(logs, step=global_step)
#             # 在训练循环中添加调试
#             print(f"dinov2_images shape: {batch['dinov2_images'].shape if 'dinov2_images' in batch else 'None'}")
#             print(f"depth_images shape: {batch['depth_images'].shape if 'depth_images' in batch else 'None'}")
#             print(f"images shape: {batch['images'].shape}")
#             if global_step >= args.max_train_steps:
#                 break

#     # 训练结束时的精简统计总结
#     if accelerator.is_main_process:
#         logger.info("Training Complete - Final Summary:")
        
#         if soft_routing_stats['total_samples'] > 0:
#             final_critical_ratio = soft_routing_stats['critical_timesteps'] / soft_routing_stats['total_samples']
#             logger.info(f"  Total Timesteps Processed: {soft_routing_stats['total_samples']}")
#             logger.info(f"  Critical Timesteps: {soft_routing_stats['critical_timesteps']}")
#             logger.info(f"  Final Critical Ratio: {final_critical_ratio:.3f}")
        
#         # 获取最终的软路由配置
#         final_soft_routing_stats = accelerator.unwrap_model(rdt).get_soft_routing_statistics()
#         if final_soft_routing_stats:
#             logger.info(f"  Final Routing Temperature: {final_soft_routing_stats.get('routing_temperature', 1.0):.4f}")
#             logger.info(f"  Neural Adjustment Enabled: {final_soft_routing_stats.get('enable_neural_adjustment', False)}")
#             logger.info(f"  Temporal Smoothing: {final_soft_routing_stats.get('temporal_smoothing', 0.0):.2f}")

#         # 保存精简版训练配置
#         task_name = "grasp" if task_type == 1 else ("click" if task_type == 2 else "unknown")
#         final_config = {
#             'task_type': task_type,
#             'task_name': task_name,  # 🔧 使用变量而不是 TaskType
#             'enable_critical_annotation': enable_critical_annotation,
#             'enable_soft_routing_repa': enable_soft_routing_repa,
#             'enable_vision_fusion': enable_vision_fusion,
#             'vision_fusion_type': vision_fusion_type,
#             'final_statistics': {
#                 'total_timesteps': soft_routing_stats['total_samples'],
#                 'critical_timesteps': soft_routing_stats['critical_timesteps'],
#                 'critical_ratio': soft_routing_stats['critical_timesteps'] / soft_routing_stats['total_samples'] if soft_routing_stats['total_samples'] > 0 else 0.0,
#             },
#             'training_hyperparameters': {
#                 'soft_routing_repa_weight': soft_routing_repa_weight,
#                 'learning_rate': args.learning_rate,
#                 'train_batch_size': args.train_batch_size,
#                 'max_train_steps': args.max_train_steps,
#             },
#             'soft_routing_final_state': final_soft_routing_stats,
#         }
        
#         import json
#         config_filename = "vision_fusion_training_config.json" if enable_vision_fusion else "soft_routing_training_config.json"
#         with open(os.path.join(args.output_dir, config_filename), "w") as f:
#             json.dump(final_config, f, indent=2)
        
#         logger.info(f"Training config saved to: {os.path.join(args.output_dir, config_filename)}")

#     # 保存最终模型
#     accelerator.wait_for_everyone()
#     if accelerator.is_main_process:
#         accelerator.unwrap_model(rdt).save_pretrained(args.output_dir)
#         ema_save_path = os.path.join(args.output_dir, f"ema")
#         accelerator.save_model(ema_rdt, ema_save_path)

#         logger.info(f"Model saved to {args.output_dir}")

#         if args.push_to_hub:
#             save_model_card(
#                 repo_id,
#                 base_model=args.pretrained_model_name_or_path,
#                 repo_folder=args.output_dir,
#             )
#             upload_folder(
#                 repo_id=repo_id,
#                 folder_path=args.output_dir,
#                 commit_message="End of soft routing dual-teacher REPA + critical timestep training",
#                 token=args.hub_token,
#                 allow_patterns=["pytorch_model.bin", "*.json", "*.md"],
#             )

#     accelerator.end_training()


# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description="Train RDT with Soft Routing Dual-Teacher REPA")
    
#     # 基础训练参数
#     parser.add_argument("--config_path", type=str, required=True, help="Path to config file")
#     parser.add_argument("--model_config_path", type=str, required=True, help="Path to model config file")
#     parser.add_argument("--pretrained_model_name_or_path", type=str, help="Path to pretrained model")
#     parser.add_argument("--pretrained_text_encoder_name_or_path", type=str, help="Path to pretrained text encoder")
#     parser.add_argument("--pretrained_vision_encoder_name_or_path", type=str, help="Path to pretrained vision encoder")
    
#     # 训练配置
#     parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
#     parser.add_argument("--report_to", type=str, default="wandb", help="Logging platform")
#     parser.add_argument("--logging_dir", type=str, default="logs")
#     parser.add_argument("--seed", type=int, default=42, help="Random seed")
#     parser.add_argument("--dataset_type", type=str, default="pretrain", choices=["pretrain", "finetune"])
    
#     # 模型和训练参数
#     parser.add_argument("--train_batch_size", type=int, default=32)
#     parser.add_argument("--sample_batch_size", type=int, default=64)
#     parser.add_argument("--num_train_epochs", type=int, default=3)
#     parser.add_argument("--max_train_steps", type=int, default=20000)
#     parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
#     parser.add_argument("--dataloader_num_workers", type=int, default=8)
    
#     # 优化器参数
#     parser.add_argument("--learning_rate", type=float, default=1e-4)
#     parser.add_argument("--adam_beta1", type=float, default=0.9)
#     parser.add_argument("--adam_beta2", type=float, default=0.999)
#     parser.add_argument("--adam_weight_decay", type=float, default=0.01)
#     parser.add_argument("--adam_epsilon", type=float, default=1e-8)
#     parser.add_argument("--max_grad_norm", type=float, default=1.0)
#     parser.add_argument("--use_8bit_adam", action="store_true")
    
#     # 学习率调度器
#     parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
#     parser.add_argument("--lr_warmup_steps", type=int, default=500)
#     parser.add_argument("--lr_num_cycles", type=int, default=1)
#     parser.add_argument("--lr_power", type=float, default=1.0)
#     parser.add_argument("--scale_lr", action="store_true")
    
#     # 检查点和采样
#     parser.add_argument("--checkpointing_period", type=int, default=2500)
#     parser.add_argument("--sample_period", type=int, default=100)
#     parser.add_argument("--num_sample_batches", type=int, default=2)
#     parser.add_argument("--checkpoints_total_limit", type=int, default=40)
#     parser.add_argument("--resume_from_checkpoint", type=str, help="Path to checkpoint to resume from")
    
#     # 数据和预处理
#     parser.add_argument("--load_from_hdf5", action="store_true", help="Load data from HDF5 files")
#     parser.add_argument("--precomp_lang_embed", action="store_true", help="Use precomputed language embeddings")
#     parser.add_argument("--image_aug", action="store_true", help="Enable image augmentation")
#     parser.add_argument("--cond_mask_prob", type=float, default=0.1, help="Condition masking probability")
#     parser.add_argument("--cam_ext_mask_prob", type=float, default=-1.0, help="External camera masking probability")
#     parser.add_argument("--state_noise_snr", type=float, help="State noise SNR")
    
#     # 系统配置
#     parser.add_argument("--gradient_checkpointing", action="store_true")
#     parser.add_argument("--allow_tf32", action="store_true")
#     parser.add_argument("--set_grads_to_none", action="store_true")
#     parser.add_argument("--deepspeed", type=str, help="Path to DeepSpeed config")
    
#     # Hub相关
#     parser.add_argument("--push_to_hub", action="store_true")
#     parser.add_argument("--hub_model_id", type=str, help="Hub model ID")
#     parser.add_argument("--hub_token", type=str, help="Hub token")
    
#     # 软路由参数
#     parser.add_argument("--CONFIG_NAME", type=str, default="soft_routing", help="Configuration name for logging")
    
#     args = parser.parse_args()
    
#     # 设置日志
#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S",
#         level=logging.INFO,
#     )
#     logger = logging.getLogger(__name__)
    
#     # 开始训练
#     train(args, logger)