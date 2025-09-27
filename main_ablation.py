import argparse
import os
from train.train_ablation import train_ablation
from accelerate.logging import get_logger


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="消融实验：输入侧视觉特征融合RDT训练")
    parser.add_argument(
        "--model_config_path",
        type=str,
        default="model_config/ablation_input_fusion_config.yaml",
        help="消融实验配置文件路径",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/base.yaml",
        help="基础配置文件路径",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="DeepSpeed配置文件路径",
    )
    parser.add_argument(
        "--pretrained_text_encoder_name_or_path",
        type=str,
        default=None,
        help="预训练文本编码器路径",
    )
    parser.add_argument(
        "--pretrained_vision_encoder_name_or_path",
        type=str,
        default=None,
        help="预训练视觉编码器路径",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="输出目录",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子")

    parser.add_argument(
        "--load_from_hdf5",
        action="store_true",
        default=False,
        help="是否从HDF5文件加载数据集",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="训练批大小",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=8,
        help="采样批大小",
    )
    parser.add_argument(
        "--num_sample_batches",
        type=int,
        default=2,
        help="采样批次数",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="最大训练步数",
    )
    parser.add_argument(
        "--checkpointing_period",
        type=int,
        default=500,
        help="检查点保存间隔",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="检查点最大数量",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="从检查点恢复",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="预训练模型路径",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="梯度累积步数",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="是否使用梯度检查点",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="学习率",
    )
    parser.add_argument(
        "--cond_mask_prob",
        type=float,
        default=0.1,
        help="条件掩码概率",
    )
    parser.add_argument(
        "--cam_ext_mask_prob",
        type=float,
        default=-1.0,
        help="外部相机掩码概率",
    )
    parser.add_argument(
        "--state_noise_snr",
        type=float,
        default=None,
        help="状态噪声信噪比",
    )
    parser.add_argument(
        "--image_aug",
        action="store_true",
        default=False,
        help="是否使用图像增强",
    )
    parser.add_argument(
        "--precomp_lang_embed",
        action="store_true",
        default=False,
        help="是否使用预计算语言嵌入",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="是否缩放学习率",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="学习率调度器类型",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="学习率预热步数",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="学习率循环次数",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="多项式调度器幂次",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="是否使用8位Adam优化器",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="数据加载器工作进程数",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Adam优化器beta1参数",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Adam优化器beta2参数",
    )
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="权重衰减")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Adam优化器epsilon值",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="最大梯度范数")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="是否推送到Hub",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="Hub token",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Hub模型ID",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="日志目录",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="是否允许TF32",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="报告平台",
    )
    parser.add_argument(
        "--sample_period",
        type=int,
        default=-1,
        help="采样周期",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="混合精度训练",
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="分布式训练本地排名",
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help="是否将梯度设置为None",
    )

    parser.add_argument(
        "--dataset_type",
        type=str,
        default="pretrain",
        required=False,
        help="数据集类型",
    )

    parser.add_argument(
        "--CONFIG_NAME",
        type=str,
        default="ablation_input_fusion",
        required=True,
        help="消融实验配置名称",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


if __name__ == "__main__":
    logger = get_logger(__name__)
    args = parse_args()
    train_ablation(args, logger)