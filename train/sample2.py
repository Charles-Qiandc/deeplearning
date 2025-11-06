# from collections import defaultdict

# import torch
# import torch.nn.functional as F


# @torch.no_grad()
# def log_sample_res(
#     text_encoder,
#     vision_encoder,
#     rdt,
#     args,
#     accelerator,
#     weight_dtype,
#     dataset_id2name,
#     dataloader,
#     logger,
# ):
#     with torch.autocast(device_type="cuda", dtype=torch.float16):
#         logger.info(f"Running sampling for {args.num_sample_batches} batches...")

#         rdt.eval()

#         loss_for_log = defaultdict(float)
#         loss_counter = defaultdict(int)
#         for step, batch in enumerate(dataloader):
#             if step >= args.num_sample_batches:
#                 break

#             data_indices = batch["data_indices"]
#             ctrl_freqs = batch["ctrl_freqs"]
#             state_norm = batch["state_norm"].to(dtype=weight_dtype)
#             images = batch["images"].to(dtype=weight_dtype)
#             states = batch["states"].to(dtype=weight_dtype)
#             # We only use the last state as input
#             states = states[:, -1:, :]
#             actions = batch["actions"].to(dtype=weight_dtype)
#             state_elem_mask = batch["state_elem_mask"].to(dtype=weight_dtype)

#             batch_size, _, C, H, W = images.shape
#             image_embeds = vision_encoder(images.reshape(-1, C, H, W)).detach()
#             image_embeds = image_embeds.reshape((batch_size, -1, vision_encoder.hidden_size))

#             lang_attn_mask = batch["lang_attn_mask"]
#             text_embeds = (batch["lang_embeds"].to(dtype=weight_dtype) if args.precomp_lang_embed else text_encoder(
#                 input_ids=batch["input_ids"], attention_mask=lang_attn_mask)["last_hidden_state"].detach())

#             pred_actions = rdt.predict_action(
#                 lang_tokens=text_embeds,
#                 lang_attn_mask=lang_attn_mask,
#                 img_tokens=image_embeds,
#                 state_tokens=states,
#                 action_mask=state_elem_mask.unsqueeze(1),
#                 ctrl_freqs=ctrl_freqs,
#             )

#             num_steps = pred_actions.shape[1]
#             expanded_state_elem_mask = (state_elem_mask.unsqueeze(1).tile((1, num_steps, 1)).float())
#             expanded_state_norm = (state_norm.unsqueeze(1).tile((1, num_steps, 1)).float())

#             loss = F.mse_loss(pred_actions, actions, reduction="none").float()

#             mse_loss_per_entry = (loss * expanded_state_elem_mask).reshape(
#                 (batch_size, -1)).sum(1) / expanded_state_elem_mask.reshape((batch_size, -1)).sum(1)
#             l2_loss_per_entry = loss.sqrt() / (expanded_state_norm + 1e-3)
#             l2_loss_per_entry = (l2_loss_per_entry * expanded_state_elem_mask).reshape(
#                 (batch_size, -1)).sum(1) / expanded_state_elem_mask.reshape((batch_size, -1)).sum(1)

#             dataset_indices, mse_losses, l2_losses = accelerator.gather_for_metrics((
#                 torch.LongTensor(data_indices).to(device=pred_actions.device),
#                 mse_loss_per_entry,
#                 l2_loss_per_entry,
#             ), )
#             dataset_indices = dataset_indices.tolist()
#             if accelerator.is_main_process:
#                 for loss_suffix, losses in zip(["_sample_mse", "_sample_l2err"], [mse_losses, l2_losses]):
#                     for dataset_idx, loss_tensor in zip(dataset_indices, losses):
#                         loss_name = dataset_id2name[dataset_idx] + loss_suffix
#                         loss_for_log[loss_name] += loss_tensor.item()
#                         loss_counter[loss_name] += 1

#             mse_loss = (loss * expanded_state_elem_mask).sum() / expanded_state_elem_mask.sum()
#             mse_loss_scaler = accelerator.gather(mse_loss).mean().item()
#             loss_for_log["overall_avg_sample_mse"] += mse_loss_scaler

#             l2_loss = loss.sqrt() / (expanded_state_norm + 1e-3)
#             l2_loss = (l2_loss * expanded_state_elem_mask).sum() / expanded_state_elem_mask.sum()
#             l2_loss_scaler = accelerator.gather(l2_loss).mean().item()
#             loss_for_log["overall_avg_sample_l2err"] += l2_loss_scaler

#         for name in loss_for_log:
#             if name in ["overall_avg_sample_mse", "overall_avg_sample_l2err"]:
#                 loss_scaler = loss_for_log[name]
#                 loss_for_log[name] = round(loss_scaler / (args.num_sample_batches), 4)
#             else:
#                 loss_for_log[name] = round(loss_for_log[name] / loss_counter[name], 4)

#         rdt.train()
#         torch.cuda.empty_cache()

#         return dict(loss_for_log)
    
    
    
    
from collections import defaultdict
import torch
import torch.nn.functional as F


@torch.no_grad()
def log_sample_res(
    text_encoder,
    vision_encoder,
    rdt,
    args,
    accelerator,
    weight_dtype,
    dataset_id2name,
    sample_dataloader,
    logger,
    # 🆕 新增参数
    dinov2_encoder=None,
    depth_encoder=None,
):
    """评估采样（支持视觉融合模式）"""
    rdt.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(sample_dataloader):
            if batch_idx >= args.num_sample_batches:
                break
            
            # 准备输入
            images = batch["images"].to(dtype=weight_dtype)
            states = batch["states"].to(dtype=weight_dtype)[:, -1:, :]
            actions = batch["actions"].to(dtype=weight_dtype)
            state_elem_mask = batch["state_elem_mask"].to(dtype=weight_dtype)
            ctrl_freqs = batch["ctrl_freqs"]
            
            # 编码
            batch_size, _, C, H, W = images.shape
            image_embeds = vision_encoder(images.reshape(-1, C, H, W)).detach()
            image_embeds = image_embeds.reshape((batch_size, -1, vision_encoder.hidden_size))
            
            lang_attn_mask = batch["lang_attn_mask"]
            text_embeds = (
                batch["lang_embeds"].to(dtype=weight_dtype) 
                if args.precomp_lang_embed 
                else text_encoder(
                    input_ids=batch["input_ids"], 
                    attention_mask=lang_attn_mask
                )["last_hidden_state"].detach()
            )
            
            # 🆕 编码DINOv2和Depth
            dinov2_features = None
            if dinov2_encoder is not None and "dinov2_images" in batch:
                dinov2_images = batch["dinov2_images"].to(dtype=weight_dtype)
                dinov2_input = dinov2_images[:, 0]
                dinov2_features = dinov2_encoder(dinov2_input)
            
            depth_features = None
            if depth_encoder is not None and "depth_images" in batch:
                depth_images = batch["depth_images"].to(dtype=weight_dtype)
                depth_input = depth_images[:, 0]
                depth_features, _ = depth_encoder(depth_input)
            
            # 计算损失
            state_elem_mask = state_elem_mask.unsqueeze(1)
            loss = accelerator.unwrap_model(rdt).compute_loss(
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
            
            total_loss += loss.item()
            num_batches += 1
    
    rdt.train()
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return {
        "sample_loss": avg_loss,
        "num_sample_batches": num_batches
    }