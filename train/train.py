#!/usr/bin/env python
# coding=utf-8

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

# 🆕 导入DINOv2和DepthAnythingV2编码器
from models.multimodal_encoder.dinov2_encoder import create_dinov2_encoder
from models.multimodal_encoder.depth_encoder import create_depth_encoder

# 🆕 导入关键时间段标注器
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
- repa
- dual-teachers
- critical-timestep
- constrained-weights
---
    """
    model_card = f"""
# RDT with Dual-Teacher REPA, Critical Timestep and Constrained Adaptive Weights - {repo_id}

This is a RDT model with dual-teacher REPA alignment loss, task-driven critical timestep annotation, 
and constrained adaptive weight learning derived from {base_model}. 
The weights were trained using [RDT](https://rdt-robotics.github.io/rdt-robotics/) 
with advanced multi-modal alignment strategies.

## Key Features
- **Dual-Teacher Alignment Strategy**: Dynamic routing between global semantic and depth geometric experts
- **Critical Timestep Annotation**: Task-driven annotation for precise temporal alignment
- **Constrained Adaptive Weights**: Model learns optimal weight allocation within task-driven constraints
- **Routing Network**: Learns to infer critical vs non-critical timesteps from action tokens

## Architecture Components
1. **Routing Network**: Predicts timestep criticality from action tokens
2. **Constrained Weight Learner**: Learns fine-grained weight allocation within constraints
3. **Dual Visual Teachers**: DINOv2 (global) + DepthAnythingV2 (geometric)

## Task Types Supported
- **Grasp Tasks (task_type=1)**: Deceleration → Gripper closing alignment
- **Click Tasks (task_type=2)**: Gripper closing → Deceleration alignment
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml_header + model_card)


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

    # 🆕 获取双教师REPA和关键时间段标注配置
    enable_repa_loss = model_config.get("enable_repa_loss", True)
    repa_loss_weight = model_config.get("repa_loss_weight", 0.2)
    use_dinov2_features = model_config.get("use_dinov2_features", True)
    use_depth_features = model_config.get("use_depth_features", True)
    routing_loss_weight = model_config.get("routing_loss_weight", 0.1)
    
    # 🆕 关键时间段标注配置
    enable_critical_annotation = model_config.get("enable_critical_annotation", True)
    task_type = model_config.get("task_type", 1)  # 1=抓取类, 2=点击类
    critical_annotation_config = model_config.get("critical_annotation_config", {})
    
    # 🆕 约束权重配置
    enable_constrained_weights = model_config.get("enable_constrained_weights", True)
    constrained_weight_config = model_config.get("constrained_weight_config", {})
    
    logger.info(f"🔧 双教师REPA + 关键时间段标注 + 约束权重学习配置:")
    logger.info(f"   - REPA损失启用: {enable_repa_loss}")
    logger.info(f"   - REPA损失权重: {repa_loss_weight}")
    logger.info(f"   - 使用DINOv2特征: {use_dinov2_features}")
    logger.info(f"   - 使用深度特征: {use_depth_features}")
    logger.info(f"   - 路由损失权重: {routing_loss_weight}")
    logger.info(f"   - 关键时间段标注: {enable_critical_annotation}")
    logger.info(f"   - 任务类型: {TaskType(task_type).name} ({task_type})")
    logger.info(f"   - 约束权重学习: {enable_constrained_weights}")
    if critical_annotation_config:
        logger.info(f"   - 标注配置: {critical_annotation_config}")
    if constrained_weight_config:
        logger.info(f"   - 约束权重配置: {constrained_weight_config}")

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

    # 视觉编码器
    vision_encoder = SiglipVisionTower(vision_tower=args.pretrained_vision_encoder_name_or_path, args=None)
    image_processor = vision_encoder.image_processor

    # 🆕 创建DINOv2编码器（全局语义特征）
    dinov2_encoder = None
    if use_dinov2_features and enable_repa_loss:
        logger.info("🔧 加载DINOv2编码器（全局语义教师）...")
        dinov2_encoder = create_dinov2_encoder(model_size="large", select_feature="cls_only")
        dinov2_encoder.to(accelerator.device, dtype=weight_dtype)
        dinov2_encoder.print_model_info()

    # 🆕 创建DepthAnythingV2编码器（深度几何特征）
    depth_encoder = None
    if use_depth_features and enable_repa_loss:
        logger.info("🔧 加载DepthAnythingV2编码器（深度几何教师）...")
        depth_encoder = create_depth_encoder(
            model_size="metric_large",
            feature_dim=1024,
            device=accelerator.device,
            use_metric_model=True
        )
        depth_encoder.to(accelerator.device, dtype=weight_dtype)
        depth_encoder.print_model_info()

    # 构建RDT模型
    logger.info("🔧 构建双教师约束权重RDT模型...")
    img_cond_len = (config["common"]["img_history_size"] * config["common"]["num_cameras"] *
                    vision_encoder.num_patches)
    
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
        enable_repa_loss=enable_repa_loss,
        repa_loss_weight=repa_loss_weight,
        use_dual_teachers=use_depth_features,  # 启用双教师模式
        routing_loss_weight=routing_loss_weight,
        # 🆕 约束权重参数
        enable_constrained_weights=enable_constrained_weights,
        constrained_weight_config=constrained_weight_config,
    )
    
    # 加载预训练权重（如果提供）
    if args.pretrained_model_name_or_path and os.path.isfile(args.pretrained_model_name_or_path):
        logger.info(f"📥 加载预训练权重: {args.pretrained_model_name_or_path}")
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
                logger.debug(f"跳过参数 {k}: checkpoint {tuple(v.shape)} vs model {tuple(own_sd.get(k, v).shape)}")

        rdt.load_state_dict(filtered, strict=False)
        logger.info("✅ 加载匹配的预训练权重；其余保持随机初始化")
    else:
        logger.info("🎲 仅使用配置；跳过预训练权重加载")

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

    # 🆕 数据集和数据加载器（集成关键时间段标注）
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
        use_dinov2_features=use_dinov2_features,
        use_depth_features=use_depth_features,
        # 🆕 关键时间段标注参数
        task_type=task_type,
        enable_critical_annotation=enable_critical_annotation,
        critical_annotation_config=critical_annotation_config,
    )
    
    sample_dataset = VLAConsumerDataset(
        model_config_path=args.model_config_path,
        config=config["dataset"],
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_cameras=config["common"]["num_cameras"],
        img_history_size=config["common"]["img_history_size"],
        dataset_type=args.dataset_type,
        image_aug=False,
        cond_mask_prob=0,
        cam_ext_mask_prob=-1,
        state_noise_snr=None,
        use_hdf5=args.load_from_hdf5,
        use_precomp_lang_embed=args.precomp_lang_embed,
        use_dinov2_features=use_dinov2_features,
        use_depth_features=use_depth_features,
        # 🆕 关键时间段标注参数（采样时也启用）
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
            'task_type': task_type,
            'enable_critical_annotation': enable_critical_annotation,
            'critical_annotation_config': critical_annotation_config,
            'enable_constrained_weights': enable_constrained_weights,
            'constrained_weight_config': constrained_weight_config,
        })
        
        accelerator.init_trackers(
            "VLA_Dual_Teacher_REPA_Critical_Constrained",
            config=tracker_config,
            init_kwargs={"wandb": {
                "name": f"RDT_DualTeacher_Constrained_{TaskType(task_type).name}_{args.CONFIG_NAME}",
            }},
        )

    # 训练信息
    total_batch_size = (args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps)

    logger.info("***** 开始双教师REPA + 关键时间段标注 + 约束权重学习训练 *****")
    logger.info(f"  样本数量 = {len(train_dataset)}")
    logger.info(f"  每epoch批次数 = {len(train_dataloader)}")
    logger.info(f"  Epoch数 = {args.num_train_epochs}")
    logger.info(f"  每设备瞬时批次大小 = {args.train_batch_size}")
    logger.info(f"  总训练批次大小 = {total_batch_size}")
    logger.info(f"  梯度累积步数 = {args.gradient_accumulation_steps}")
    logger.info(f"  总优化步数 = {args.max_train_steps}")
    logger.info(f"  任务类型 = {TaskType(task_type).name}")
    logger.info(f"  约束权重学习 = {enable_constrained_weights}")
    
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
            accelerator.print(f"检查点 '{args.resume_from_checkpoint}' 不存在。开始新的训练。")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"从检查点恢复: {path}")
            try:
                accelerator.load_state(os.path.join(args.output_dir, path))
            except:
                logger.info("恢复训练状态失败。尝试仅加载模型检查点。")
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

    # 🆕 用于记录约束权重学习统计的变量
    constraint_stats = {
        'total_samples': 0,
        'critical_timesteps': 0,
        'routing_accuracy_sum': 0.0,
        'routing_accuracy_count': 0,
        'depth_preference_ratio_sum': 0.0,
        'weight_temperature_sum': 0.0,
    }
    
    loss_for_log = {}
    
    # 训练循环
    for epoch in range(first_epoch, args.num_train_epochs):
        rdt.train()

        if args.resume_from_checkpoint and epoch == first_epoch:
            progress_bar.update(resume_step // args.gradient_accumulation_steps)

        for batch in train_dataloader:
            with accelerator.accumulate(rdt):
                # 准备输入数据
                images = batch["images"].to(dtype=weight_dtype)
                states = batch["states"].to(dtype=weight_dtype)[:, -1:, :]  # 只使用最后一个状态
                actions = batch["actions"].to(dtype=weight_dtype)
                state_elem_mask = batch["state_elem_mask"].to(dtype=weight_dtype)
                ctrl_freqs = batch["ctrl_freqs"]

                # 🆕 获取关键时间段标签
                critical_labels = batch.get("critical_labels", None)
                if critical_labels is not None:
                    critical_labels = critical_labels.to(accelerator.device)
                    
                    # 统计关键时间段信息
                    batch_size, seq_len = critical_labels.shape
                    constraint_stats['total_samples'] += batch_size * seq_len
                    constraint_stats['critical_timesteps'] += critical_labels.sum().item()

                # 编码原始图像（SigLIP）
                with torch.no_grad():
                    batch_size, _, C, H, W = images.shape
                    image_embeds = vision_encoder(images.reshape(-1, C, H, W)).detach()
                    image_embeds = image_embeds.reshape((batch_size, -1, vision_encoder.hidden_size))

                    lang_attn_mask = batch["lang_attn_mask"]
                    text_embeds = (batch["lang_embeds"].to(dtype=weight_dtype) 
                                 if args.precomp_lang_embed 
                                 else text_encoder(input_ids=batch["input_ids"], 
                                                 attention_mask=lang_attn_mask)["last_hidden_state"].detach())

                # 🆕 提取DINOv2全局语义特征
                cls_token = None
                if dinov2_encoder is not None and "dinov2_images" in batch:
                    with torch.no_grad():
                        dinov2_images = batch["dinov2_images"].to(dtype=weight_dtype)
                        dinov2_input = dinov2_images[:, 0]  # (B, 3, 518, 518)
                        cls_token = dinov2_encoder(dinov2_input)  # (B, 1, 1024)

                # 🆕 提取DepthAnythingV2深度几何特征
                depth_features = None
                if depth_encoder is not None and "depth_images" in batch:
                    with torch.no_grad():
                        depth_images = batch["depth_images"].to(dtype=weight_dtype)
                        depth_input = depth_images[:, 0]  # (B, 3, 518, 518)
                        depth_features, _ = depth_encoder(depth_input)  # (B, 1370, 1024)

                # 🆕 计算约束权重双教师REPA损失
                state_elem_mask = state_elem_mask.unsqueeze(1)
                if enable_repa_loss:
                    total_loss, diffusion_loss, repa_loss, routing_loss, detailed_metrics = accelerator.unwrap_model(rdt).compute_loss(
                        lang_tokens=text_embeds,
                        lang_attn_mask=lang_attn_mask,
                        img_tokens=image_embeds,
                        state_tokens=states,
                        action_gt=actions,
                        action_mask=state_elem_mask,
                        ctrl_freqs=ctrl_freqs,
                        cls_token=cls_token,              
                        depth_features=depth_features,   
                        critical_labels=critical_labels,  # 传入关键时间段标签
                    )
                    loss = total_loss
                    
                    # 🆕 记录约束权重的详细损失
                    loss_for_log["diffusion_loss"] = diffusion_loss.detach().item()
                    loss_for_log["repa_loss"] = repa_loss.detach().item()
                    loss_for_log["routing_loss"] = routing_loss.detach().item()
                    
                    # 🆕 记录约束权重学习的详细指标
                    if detailed_metrics:
                        # 主要损失组件
                        loss_for_log.update({
                            'constrained_alignment_loss': detailed_metrics.get('alignment_loss', 0.0),
                            'constrained_diversity_loss': detailed_metrics.get('diversity_loss', 0.0),
                            'constrained_smoothness_loss': detailed_metrics.get('smoothness_loss', 0.0),
                            'total_constrained_loss': detailed_metrics.get('total_constrained_loss', 0.0),
                            
                            # 权重学习器状态
                            'weight_temperature': detailed_metrics.get('weight_temperature', 1.0),
                            'depth_preference_ratio': detailed_metrics.get('depth_preference_ratio', 0.5),
                            
                            # 约束后的权重统计
                            'constrained_avg_global_weight': detailed_metrics.get('constrained_avg_global_weight', 0.5),
                            'constrained_avg_depth_weight': detailed_metrics.get('constrained_avg_depth_weight', 0.5),
                            'constrained_global_weight_std': detailed_metrics.get('constrained_global_weight_std', 0.0),
                            'constrained_depth_weight_std': detailed_metrics.get('constrained_depth_weight_std', 0.0),
                            
                            # 路由权重统计（用于对比）
                            'routing_avg_global_weight': detailed_metrics.get('routing_avg_global_weight', 0.5),
                            'routing_avg_depth_weight': detailed_metrics.get('routing_avg_depth_weight', 0.5),
                            
                            # 基础损失（未加权）
                            'raw_global_loss': detailed_metrics.get('raw_global_loss', 0.0),
                            'raw_depth_loss': detailed_metrics.get('raw_depth_loss', 0.0),
                            
                            # 路由监督损失
                            'routing_supervision_loss': detailed_metrics.get('routing_supervision_loss', 0.0),
                        })
                        
                        # 关键时间段分析
                        if critical_labels is not None:
                            loss_for_log.update({
                                'critical_constrained_global': detailed_metrics.get('critical_constrained_global', 0.5),
                                'critical_constrained_depth': detailed_metrics.get('critical_constrained_depth', 0.5),
                                'critical_routing_global': detailed_metrics.get('critical_routing_global', 0.5),
                                'critical_routing_depth': detailed_metrics.get('critical_routing_depth', 0.5),
                                'non_critical_constrained_global': detailed_metrics.get('non_critical_constrained_global', 0.5),
                                'non_critical_constrained_depth': detailed_metrics.get('non_critical_constrained_depth', 0.5),
                                'non_critical_routing_global': detailed_metrics.get('non_critical_routing_global', 0.5),
                                'non_critical_routing_depth': detailed_metrics.get('non_critical_routing_depth', 0.5),
                                
                                # 路由网络准确性
                                'routing_critical_accuracy': detailed_metrics.get('routing_critical_accuracy', 0.5),
                                'routing_non_critical_accuracy': detailed_metrics.get('routing_non_critical_accuracy', 0.5),
                                'routing_overall_accuracy': detailed_metrics.get('routing_overall_accuracy', 0.5),
                            })
                            
                            # 更新累积统计
                            if 'routing_overall_accuracy' in detailed_metrics:
                                constraint_stats['routing_accuracy_sum'] += detailed_metrics['routing_overall_accuracy']
                                constraint_stats['routing_accuracy_count'] += 1
                            
                            if 'depth_preference_ratio' in detailed_metrics:
                                constraint_stats['depth_preference_ratio_sum'] += detailed_metrics['depth_preference_ratio']
                            
                            if 'weight_temperature' in detailed_metrics:
                                constraint_stats['weight_temperature_sum'] += detailed_metrics['weight_temperature']
                else:
                    # 原始方式（兼容性）
                    loss = rdt(
                        lang_tokens=text_embeds,
                        lang_attn_mask=lang_attn_mask,
                        img_tokens=image_embeds,
                        state_tokens=states,
                        action_gt=actions,
                        action_mask=state_elem_mask,
                        ctrl_freqs=ctrl_freqs,
                    )
                    loss_for_log["diffusion_loss"] = loss.detach().item()

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
                    logger.info(f"保存状态到 {save_path}")

                # 每100步记录约束权重学习统计信息
                if global_step % 100 == 0 and detailed_metrics and 'adaptive_weights_tensor' in detailed_metrics:
                    adaptive_weights = detailed_metrics['adaptive_weights_tensor']  # (B, T, 2)
                    routing_weights = detailed_metrics['routing_weights_tensor']   # (B, T, 2)
                    
                    logger.info(f"约束权重学习统计 (步骤 {global_step}):")
                    logger.info(f"   - 路由网络权重: 全局={routing_weights[:, :, 0].mean():.3f}, 深度={routing_weights[:, :, 1].mean():.3f}")
                    logger.info(f"   - 约束后权重: 全局={adaptive_weights[:, :, 0].mean():.3f}, 深度={adaptive_weights[:, :, 1].mean():.3f}")
                    logger.info(f"   - 深度偏好比例: {detailed_metrics.get('depth_preference_ratio', 0.5):.3f}")
                    logger.info(f"   - 权重温度: {detailed_metrics.get('weight_temperature', 1.0):.3f}")
                    
                    if critical_labels is not None and 'routing_overall_accuracy' in detailed_metrics:
                        logger.info(f"   - 路由网络准确率: {detailed_metrics['routing_overall_accuracy']:.3f}")
                    
                    # 计算总体统计
                    if constraint_stats['total_samples'] > 0:
                        overall_critical_ratio = constraint_stats['critical_timesteps'] / constraint_stats['total_samples']
                        logger.info(f"   - 总体关键时间段比例: {overall_critical_ratio:.3f}")
                    
                    if constraint_stats['routing_accuracy_count'] > 0:
                        avg_routing_accuracy = constraint_stats['routing_accuracy_sum'] / constraint_stats['routing_accuracy_count']
                        logger.info(f"   - 平均路由准确率: {avg_routing_accuracy:.3f}")

                # 每1000步打印更详细的约束权重分析
                if global_step % 1000 == 0 and critical_labels is not None and detailed_metrics:
                    logger.info(f"详细约束权重分析 (步骤 {global_step}):")
                    logger.info(f"   关键时间段:")
                    logger.info(f"     - 路由权重: 全局={detailed_metrics.get('critical_routing_global', 0.5):.3f}, 深度={detailed_metrics.get('critical_routing_depth', 0.5):.3f}")
                    logger.info(f"     - 约束权重: 全局={detailed_metrics.get('critical_constrained_global', 0.5):.3f}, 深度={detailed_metrics.get('critical_constrained_depth', 0.5):.3f}")
                    logger.info(f"   非关键时间段:")
                    logger.info(f"     - 路由权重: 全局={detailed_metrics.get('non_critical_routing_global', 0.5):.3f}, 深度={detailed_metrics.get('non_critical_routing_depth', 0.5):.3f}")
                    logger.info(f"     - 约束权重: 全局={detailed_metrics.get('non_critical_constrained_global', 0.5):.3f}, 深度={detailed_metrics.get('non_critical_constrained_depth', 0.5):.3f}")
                    
                    logger.info(f"   损失组件:")
                    logger.info(f"     - 对齐损失: {detailed_metrics.get('alignment_loss', 0.0):.4f}")
                    logger.info(f"     - 多样性损失: {detailed_metrics.get('diversity_loss', 0.0):.4f}")
                    logger.info(f"     - 平滑性损失: {detailed_metrics.get('smoothness_loss', 0.0):.4f}")
                    logger.info(f"     - 路由监督损失: {detailed_metrics.get('routing_supervision_loss', 0.0):.4f}")

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
            progress_bar.set_postfix(**logs)
            logs.update(loss_for_log)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # 训练结束时的约束权重学习统计总结
    if accelerator.is_main_process:
        logger.info("训练完成 - 约束权重学习统计总结:")
        
        if constraint_stats['total_samples'] > 0:
            final_critical_ratio = constraint_stats['critical_timesteps'] / constraint_stats['total_samples']
            logger.info(f"   - 处理的总时间步数: {constraint_stats['total_samples']}")
            logger.info(f"   - 关键时间步数: {constraint_stats['critical_timesteps']}")
            logger.info(f"   - 最终关键时间段比例: {final_critical_ratio:.3f}")
        
        if constraint_stats['routing_accuracy_count'] > 0:
            final_routing_accuracy = constraint_stats['routing_accuracy_sum'] / constraint_stats['routing_accuracy_count']
            logger.info(f"   - 路由网络最终平均准确率: {final_routing_accuracy:.3f}")
            
            avg_depth_preference = constraint_stats['depth_preference_ratio_sum'] / constraint_stats['routing_accuracy_count']
            logger.info(f"   - 平均深度偏好比例: {avg_depth_preference:.3f}")
            
            avg_temperature = constraint_stats['weight_temperature_sum'] / constraint_stats['routing_accuracy_count']
            logger.info(f"   - 平均权重学习温度: {avg_temperature:.3f}")

        # 保存训练配置和统计信息
        final_config = {
            'task_type': task_type,
            'task_name': TaskType(task_type).name,
            'enable_critical_annotation': enable_critical_annotation,
            'critical_annotation_config': critical_annotation_config,
            'enable_constrained_weights': enable_constrained_weights,
            'constrained_weight_config': constrained_weight_config,
            'final_statistics': {
                'total_timesteps': constraint_stats['total_samples'],
                'critical_timesteps': constraint_stats['critical_timesteps'],
                'critical_ratio': constraint_stats['critical_timesteps'] / constraint_stats['total_samples'] if constraint_stats['total_samples'] > 0 else 0.0,
                'avg_routing_accuracy': constraint_stats['routing_accuracy_sum'] / constraint_stats['routing_accuracy_count'] if constraint_stats['routing_accuracy_count'] > 0 else 0.0,
                'avg_depth_preference_ratio': constraint_stats['depth_preference_ratio_sum'] / constraint_stats['routing_accuracy_count'] if constraint_stats['routing_accuracy_count'] > 0 else 0.0,
                'avg_weight_temperature': constraint_stats['weight_temperature_sum'] / constraint_stats['routing_accuracy_count'] if constraint_stats['routing_accuracy_count'] > 0 else 0.0,
            },
            'training_hyperparameters': {
                'repa_loss_weight': repa_loss_weight,
                'routing_loss_weight': routing_loss_weight,
                'learning_rate': args.learning_rate,
                'train_batch_size': args.train_batch_size,
                'max_train_steps': args.max_train_steps,
            }
        }
        
        import json
        with open(os.path.join(args.output_dir, "constrained_training_config.json"), "w") as f:
            json.dump(final_config, f, indent=2)
        
        logger.info(f"约束权重学习配置已保存到: {os.path.join(args.output_dir, 'constrained_training_config.json')}")

    # 保存最终模型
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(rdt).save_pretrained(args.output_dir)
        ema_save_path = os.path.join(args.output_dir, f"ema")
        accelerator.save_model(ema_rdt, ema_save_path)

        logger.info(f"保存模型到 {args.output_dir}")

        if args.push_to_hub:
            save_model_card(
                repo_id,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of dual-teacher REPA + critical timestep + constrained weights training",
                token=args.hub_token,
                allow_patterns=["pytorch_model.bin", "*.json", "*.md"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train RDT with Dual-Teacher REPA and Constrained Adaptive Weights")
    
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
    
    # 约束权重学习参数
    parser.add_argument("--CONFIG_NAME", type=str, default="constrained_weights", help="Configuration name for logging")
    
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