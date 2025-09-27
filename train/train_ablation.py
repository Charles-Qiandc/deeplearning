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
from models.rdt_runner_ablation import RDTRunnerAblation
from train.dataset import DataCollatorForVLAConsumerDataset, VLAConsumerDataset
from train.sample import log_sample_res

# 导入DINOv2和DepthAnythingV2编码器
from models.multimodal_encoder.dinov2_encoder import create_dinov2_encoder
from models.multimodal_encoder.depth_encoder import create_depth_encoder

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
- ablation-study
- input-fusion
---
    """
    model_card = f"""
# RDT Ablation Study: Input-side Visual Feature Fusion - {repo_id}

This is an ablation study version of RDT derived from {base_model}. The weights were trained using input-side 
visual feature fusion instead of REPA alignment, combining DINOv2 and DepthAnythingV2 features with SigLIP at the input level.

## Ablation Study Features
- **REPA Alignment Removed**: No routing or alignment mechanisms
- **Input-side Fusion**: DINOv2 CLS + DepthAnything CLS tokens concatenated with SigLIP patches
- **Preserved Architecture**: Same transformer blocks as original RDT
- **Visual Feature Sources**: SigLIP (patch tokens) + DINOv2 (global semantic) + DepthAnything (depth geometric)

## Architecture Changes
1. **Removed Components**: All REPA alignment losses, routing networks, teacher alignment
2. **Added Components**: Visual feature projection layers for DINOv2/DepthAnything
3. **Modified Input**: Extended image condition length to accommodate extra CLS tokens
4. **Fusion Strategy**: Direct concatenation of projected features

## Usage
This model follows the same inference API as standard RDT but processes additional visual modalities internally.
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml_header + model_card)


def train_ablation(args, logger):
    """消融实验训练函数"""
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

    # 🆕 获取消融实验配置
    use_dinov2_features = model_config.get("use_dinov2_features", True)
    use_depth_features = model_config.get("use_depth_features", True)
    visual_fusion_strategy = model_config.get("visual_fusion_strategy", "concat")
    
    logger.info(f"🔬 消融实验配置:")
    logger.info(f"   - 移除REPA对齐机制")
    logger.info(f"   - DINOv2特征: {use_dinov2_features}")
    logger.info(f"   - 深度特征: {use_depth_features}")
    logger.info(f"   - 视觉融合策略: {visual_fusion_strategy}")

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

    # 🆕 创建额外的视觉编码器
    dinov2_encoder = None
    if use_dinov2_features:
        logger.info("加载DINOv2编码器（全局语义特征提取）...")
        dinov2_encoder = create_dinov2_encoder(model_size="large", select_feature="cls_only")
        dinov2_encoder.to(accelerator.device, dtype=weight_dtype)
        dinov2_encoder.print_model_info()

    depth_encoder = None
    if use_depth_features:
        logger.info("加载DepthAnythingV2编码器（深度几何特征提取）...")
        depth_encoder = create_depth_encoder(
            model_size="metric_large",
            feature_dim=1024,
            device=accelerator.device,
            use_metric_model=True
        )
        depth_encoder.to(accelerator.device, dtype=weight_dtype)
        depth_encoder.print_model_info()

    # 🆕 构建消融实验版RDT模型
    logger.info("构建消融实验版RDT模型...")
    img_cond_len = (config["common"]["img_history_size"] * config["common"]["num_cameras"] *
                    vision_encoder.num_patches)
    
    rdt = RDTRunnerAblation(
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
        # 🆕 消融实验参数
        use_dinov2_features=use_dinov2_features,
        dinov2_feature_dim=1024,
        use_depth_features=use_depth_features,
        depth_feature_dim=1024,
        visual_fusion_strategy=visual_fusion_strategy,
    )
    
    # 加载预训练权重（如果提供）
    if args.pretrained_model_name_or_path and os.path.isfile(args.pretrained_model_name_or_path):
        logger.info(f"加载预训练权重: {args.pretrained_model_name_or_path}")
        ckpt = torch.load(args.pretrained_model_name_or_path, map_location="cpu")

        if isinstance(ckpt, dict) and "module" in ckpt:
            pretrained_sd = ckpt["module"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            pretrained_sd = ckpt["state_dict"]
        else:
            pretrained_sd = ckpt

        own_sd = rdt.state_dict()
        filtered = {}
        skipped_count = 0
        for k, v in pretrained_sd.items():
            if k in own_sd and v.shape == own_sd[k].shape:
                filtered[k] = v
            else:
                skipped_count += 1
                if skipped_count <= 10:  # 只显示前10个跳过的参数
                    logger.debug(f"跳过参数 {k}: 检查点 {tuple(v.shape)} vs 模型 {tuple(own_sd.get(k, v).shape)}")

        rdt.load_state_dict(filtered, strict=False)
        logger.info(f"加载了匹配的预训练权重，跳过了 {skipped_count} 个不匹配的参数")
    else:
        logger.info("只使用配置初始化，跳过预训练权重加载")

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

    # 🆕 数据集配置（支持视觉特征提取）
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
        # 消融实验不需要关键时间段标注
        enable_critical_annotation=False,
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
        enable_critical_annotation=False,
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
            'ablation_study': True,
            'use_dinov2_features': use_dinov2_features,
            'use_depth_features': use_depth_features,
            'visual_fusion_strategy': visual_fusion_strategy,
            'repa_alignment': False,  # 标识移除了REPA
        })
        
        accelerator.init_trackers(
            "RDT_Ablation_Input_Fusion",
            config=tracker_config,
            init_kwargs={"wandb": {
                "name": f"RDT_Ablation_InputFusion_{args.CONFIG_NAME}",
            }},
        )

    # 训练信息
    total_batch_size = (args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps)

    logger.info("***** 开始消融实验训练 *****")
    logger.info(f"  示例数量 = {len(train_dataset)}")
    logger.info(f"  训练轮数 = {args.num_train_epochs}")
    logger.info(f"  设备批大小 = {args.train_batch_size}")
    logger.info(f"  总训练批大小 = {total_batch_size}")
    logger.info(f"  梯度累积步数 = {args.gradient_accumulation_steps}")
    logger.info(f"  总优化步数 = {args.max_train_steps}")
    logger.info(f"  消融模式 = 输入侧视觉特征融合")
    
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
            accelerator.print(f"检查点 '{args.resume_from_checkpoint}' 未找到。开始新训练。")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"从检查点恢复: {path}")
            try:
                accelerator.load_state(os.path.join(args.output_dir, path))
            except:
                logger.info("恢复训练状态失败。尝试只加载模型检查点。")
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

    # 训练循环
    for epoch in range(first_epoch, args.num_train_epochs):
        rdt.train()

        if args.resume_from_checkpoint and epoch == first_epoch:
            progress_bar.update(resume_step // args.gradient_accumulation_steps)

        for batch in train_dataloader:
            with accelerator.accumulate(rdt):
                # 准备输入数据
                images = batch["images"].to(dtype=weight_dtype)
                states = batch["states"].to(dtype=weight_dtype)[:, -1:, :]
                actions = batch["actions"].to(dtype=weight_dtype)
                state_elem_mask = batch["state_elem_mask"].to(dtype=weight_dtype)
                ctrl_freqs = batch["ctrl_freqs"]

                # 🆕 编码多模态视觉特征
                dinov2_features = None
                depth_features = None
                
                with torch.no_grad():
                    # SigLIP特征（原有）
                    batch_size, _, C, H, W = images.shape
                    image_embeds = vision_encoder(images.reshape(-1, C, H, W)).detach()
                    image_embeds = image_embeds.reshape((batch_size, -1, vision_encoder.hidden_size))

                    # 语言特征（原有）
                    lang_attn_mask = batch["lang_attn_mask"]
                    text_embeds = (batch["lang_embeds"].to(dtype=weight_dtype) 
                                 if args.precomp_lang_embed 
                                 else text_encoder(input_ids=batch["input_ids"], 
                                                 attention_mask=lang_attn_mask)["last_hidden_state"].detach())

                    # 🆕 DINOv2特征提取
                    if dinov2_encoder is not None and "dinov2_images" in batch:
                        dinov2_images = batch["dinov2_images"].to(dtype=weight_dtype)
                        dinov2_input = dinov2_images[:, 0]  # (B, 3, 518, 518)
                        dinov2_features = dinov2_encoder(dinov2_input)  # (B, 1, 1024)

                    # 🆕 DepthAnything特征提取
                    if depth_encoder is not None and "depth_images" in batch:
                        depth_images = batch["depth_images"].to(dtype=weight_dtype)
                        depth_input = depth_images[:, 0]  # (B, 3, 518, 518)
                        depth_features_raw, _ = depth_encoder(depth_input)  # (B, 1370, 1024)
                        # 只取CLS token
                        depth_features = depth_features_raw[:, 0:1, :]  # (B, 1, 1024)

                # 🆕 计算消融版损失（只有扩散损失）
                state_elem_mask = state_elem_mask.unsqueeze(1)
                total_loss, diffusion_loss, _, detailed_metrics = accelerator.unwrap_model(rdt).compute_loss(
                    lang_tokens=text_embeds,
                    lang_attn_mask=lang_attn_mask,
                    img_tokens=image_embeds,
                    state_tokens=states,
                    action_gt=actions,
                    action_mask=state_elem_mask,
                    ctrl_freqs=ctrl_freqs,
                    dinov2_features=dinov2_features,
                    depth_features=depth_features,
                )
                
                loss = total_loss

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

                # 🆕 消融实验监控
                if global_step % 500 == 0:
                    logger.info(f"步骤 {global_step} - 消融实验指标:")
                    logger.info(f"  扩散损失: {detailed_metrics.get('diffusion_loss', 0):.4f}")
                    logger.info(f"  总损失: {detailed_metrics.get('total_loss', 0):.4f}")
                    
                    if detailed_metrics.get('dinov2_features_used', False):
                        logger.info(f"  DINOv2特征: 已使用")
                    if detailed_metrics.get('depth_features_used', False):
                        logger.info(f"  深度特征: 已使用")
                        
                    logger.info(f"  消融模式: 输入侧视觉特征融合")

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
            logs.update(detailed_metrics)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # 训练结束统计
    if accelerator.is_main_process:
        logger.info("消融实验训练完成 - 最终总结:")
        logger.info(f"  模式: 输入侧视觉特征融合")
        logger.info(f"  DINOv2特征使用: {use_dinov2_features}")
        logger.info(f"  深度特征使用: {use_depth_features}")
        logger.info(f"  视觉融合策略: {visual_fusion_strategy}")
        
        # 保存消融实验配置
        final_config = {
            'ablation_study': True,
            'experiment_type': 'input_side_visual_fusion',
            'repa_alignment_removed': True,
            'visual_features': {
                'siglip_patches': True,
                'dinov2_cls': use_dinov2_features,
                'depth_cls': use_depth_features,
            },
            'fusion_strategy': visual_fusion_strategy,
            'training_hyperparameters': {
                'learning_rate': args.learning_rate,
                'train_batch_size': args.train_batch_size,
                'max_train_steps': args.max_train_steps,
            },
        }
        
        import json
        with open(os.path.join(args.output_dir, "ablation_study_config.json"), "w") as f:
            json.dump(final_config, f, indent=2)
        
        logger.info(f"消融实验配置保存到: {os.path.join(args.output_dir, 'ablation_study_config.json')}")

    # 保存最终模型
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(rdt).save_pretrained(args.output_dir)
        ema_save_path = os.path.join(args.output_dir, f"ema")
        accelerator.save_model(ema_rdt, ema_save_path)

        logger.info(f"模型保存到 {args.output_dir}")

        if args.push_to_hub:
            save_model_card(
                repo_id,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="消融实验训练完成: 输入侧视觉特征融合",
                token=args.hub_token,
                allow_patterns=["pytorch_model.bin", "*.json", "*.md"],
            )

    accelerator.end_training()