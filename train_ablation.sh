#!/bin/bash

CONFIG_NAME="$1"
FUSION_STRATEGY="${2:-concat}"  # 默认使用concat策略

if [ -z "$CONFIG_NAME" ]; then
    echo "用法: ./train_ablation.sh CONFIG_NAME [FUSION_STRATEGY]"
    echo "示例: ./train_ablation.sh ablation_input_fusion concat"
    echo "FUSION_STRATEGY可选: concat, add, gate"
    exit 1
fi

CONFIG_FILE="model_config/${CONFIG_NAME}.yml"

echo "🔬 开始RDT消融实验训练: 输入侧视觉特征融合"
echo "CONFIG_NAME: $CONFIG_NAME"
echo "CONFIG_FILE: $CONFIG_FILE"
echo "FUSION_STRATEGY: $FUSION_STRATEGY"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件 $CONFIG_FILE 不存在!"
    echo "📝 创建模板配置文件..."
    
    # 创建配置目录
    mkdir -p "model_config"
    
    # 创建消融实验配置文件
    cat > "$CONFIG_FILE" << EOF
# 消融实验配置: 输入侧视觉特征融合
model: $CONFIG_NAME
data_path: training_data/$CONFIG_NAME
checkpoint_path: checkpoints/$CONFIG_NAME
pretrained_model_name_or_path: "../weights/RDT/rdt-1b"
cuda_visible_device: "0,1,2,3"

# 训练超参数
train_batch_size: 16
sample_batch_size: 32
max_train_steps: 20000
checkpointing_period: 2500
sample_period: 100
checkpoints_total_limit: 40
learning_rate: 1e-4
dataloader_num_workers: 8
state_noise_snr: 40
gradient_accumulation_steps: 2
lr_scheduler: "constant_with_warmup"
dataset_type: "finetune"

# 🔬 消融实验配置
ablation_study: true
experiment_type: "input_side_visual_fusion"

# 🚫 移除REPA相关组件
enable_repa_loss: false
enable_critical_annotation: false
enable_soft_routing_repa: false

# ✅ 多模态视觉特征配置
use_dinov2_features: true
dinov2_feature_dim: 1024
use_depth_features: true
depth_feature_dim: 1024
visual_fusion_strategy: "$FUSION_STRATEGY"
EOF
    
    echo "✅ 模板配置文件已创建: $CONFIG_FILE"
    echo "📋 请检查并修改配置文件，然后重新运行脚本"
    exit 0
fi

# 环境设置
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="../weights/RDT/siglip-so400m-patch14-384"
export PRETRAINED_RDT="/home/deng_xiang/qian_daichao/RoboTwin/policy/weights/RDT/rdt-1b/pytorch_model.bin"
export WANDB_PROJECT="RDT_Ablation_InputFusion"
export WANDB_DEFAULT_RUN_NAME="${CONFIG_NAME}_${FUSION_STRATEGY}"

# 读取配置的辅助函数
read_yaml_config() {
    local file="$1"
    local key="$2"
    if [ -f "scripts/read_yaml.py" ]; then
        python scripts/read_yaml.py "$file" "$key" 2>/dev/null
    else
        # 备用方案使用grep
        grep "^$key:" "$file" | sed "s/^$key: *//" | tr -d '"' | head -1
    fi
}

# 从配置文件读取关键设置
TRAIN_BATCH_SIZE=$(read_yaml_config "$CONFIG_FILE" train_batch_size)
SAMPLE_BATCH_SIZE=$(read_yaml_config "$CONFIG_FILE" sample_batch_size)
MAX_TRAIN_STEPS=$(read_yaml_config "$CONFIG_FILE" max_train_steps)
CHECKPOINTING_PERIOD=$(read_yaml_config "$CONFIG_FILE" checkpointing_period)
SAMPLE_PERIOD=$(read_yaml_config "$CONFIG_FILE" sample_period)
CHECKPOINTS_TOTAL_LIMIT=$(read_yaml_config "$CONFIG_FILE" checkpoints_total_limit)
LR_SCHEDULER=$(read_yaml_config "$CONFIG_FILE" lr_scheduler)
LEARNING_RATE=$(read_yaml_config "$CONFIG_FILE" learning_rate)
DATALOADER_NUM_WORKERS=$(read_yaml_config "$CONFIG_FILE" dataloader_num_workers)
DATASET_TYPE=$(read_yaml_config "$CONFIG_FILE" dataset_type)
STATE_NOISE_SNR=$(read_yaml_config "$CONFIG_FILE" state_noise_snr)
GRAD_ACCUM_STEPS=$(read_yaml_config "$CONFIG_FILE" gradient_accumulation_steps)
OUTPUT_DIR=$(read_yaml_config "$CONFIG_FILE" checkpoint_path)
CUDA_USE=$(read_yaml_config "$CONFIG_FILE" cuda_visible_device)

# 消融实验特定设置
USE_DINOV2=$(read_yaml_config "$CONFIG_FILE" use_dinov2_features)
USE_DEPTH=$(read_yaml_config "$CONFIG_FILE" use_depth_features)
VISUAL_FUSION_STRATEGY=$(read_yaml_config "$CONFIG_FILE" visual_fusion_strategy)

# 清理引号
CUDA_USE=$(echo "$CUDA_USE" | tr -d '"')
OUTPUT_DIR=$(echo "$OUTPUT_DIR" | tr -d '"')
LR_SCHEDULER=$(echo "$LR_SCHEDULER" | tr -d '"')
DATASET_TYPE=$(echo "$DATASET_TYPE" | tr -d '"')
VISUAL_FUSION_STRATEGY=$(echo "$VISUAL_FUSION_STRATEGY" | tr -d '"')

# 设置默认值
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
SAMPLE_BATCH_SIZE=${SAMPLE_BATCH_SIZE:-32}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-20000}
CHECKPOINTING_PERIOD=${CHECKPOINTING_PERIOD:-2500}
SAMPLE_PERIOD=${SAMPLE_PERIOD:-100}
CHECKPOINTS_TOTAL_LIMIT=${CHECKPOINTS_TOTAL_LIMIT:-40}
LR_SCHEDULER=${LR_SCHEDULER:-constant_with_warmup}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
DATASET_TYPE=${DATASET_TYPE:-finetune}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
CUDA_USE=${CUDA_USE:-0}
USE_DINOV2=${USE_DINOV2:-true}
USE_DEPTH=${USE_DEPTH:-true}
VISUAL_FUSION_STRATEGY=${VISUAL_FUSION_STRATEGY:-$FUSION_STRATEGY}

# 创建输出目录
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "✅ 创建输出目录: $OUTPUT_DIR"
fi

# 复制配置文件到输出目录
cp "$CONFIG_FILE" "$OUTPUT_DIR/training_config.yaml"

export CUDA_VISIBLE_DEVICES=$CUDA_USE

echo ""
echo "🔧 消融实验配置总结:"
echo "   - 实验类型: 输入侧视觉特征融合"
echo "   - 模型配置: $CONFIG_NAME"
echo "   - 批大小: $TRAIN_BATCH_SIZE"
echo "   - 最大步数: $MAX_TRAIN_STEPS"
echo "   - 学习率: $LEARNING_RATE"
echo "   - 输出目录: $OUTPUT_DIR"
echo "   - CUDA设备: $CUDA_USE"
echo "   - 数据集类型: $DATASET_TYPE"
echo ""
echo "🔬 消融实验特性:"
echo "   - REPA对齐: 已移除"
echo "   - DINOv2特征: $USE_DINOV2"
echo "   - 深度特征: $USE_DEPTH"
echo "   - 融合策略: $VISUAL_FUSION_STRATEGY"
echo "   - 关键时间段标注: 已禁用"
echo ""

# 计算数据集统计信息
echo "📊 计算数据集统计信息..."
python -m data.compute_dataset_stat_hdf5 --task_name $CONFIG_NAME

# 验证必需文件
if [ ! -f "$PRETRAINED_RDT" ]; then
    echo "⚠️  警告: 预训练RDT模型未找到: $PRETRAINED_RDT"
    echo "请更新脚本中的PRETRAINED_RDT路径。"
fi

# 检查main_ablation.py是否存在
if [ ! -f "main_ablation.py" ]; then
    echo "❌ main_ablation.py未在当前目录找到!"
    echo "请确保从项目根目录运行此脚本。"
    exit 1
fi

# 启动消融实验训练
echo "🏃 启动消融实验训练..."
echo "📝 命令: accelerate launch --main_process_port=28499 main_ablation.py ..."
echo ""

accelerate launch --main_process_port=28499 main_ablation.py \
    --deepspeed="./configs/zero2.json" \
    --pretrained_model_name_or_path="$PRETRAINED_RDT" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=$TRAIN_BATCH_SIZE \
    --sample_batch_size=$SAMPLE_BATCH_SIZE \
    --max_train_steps=$MAX_TRAIN_STEPS \
    --checkpointing_period=$CHECKPOINTING_PERIOD \
    --sample_period=$SAMPLE_PERIOD \
    --checkpoints_total_limit=$CHECKPOINTS_TOTAL_LIMIT \
    --lr_scheduler=$LR_SCHEDULER \
    --learning_rate=$LEARNING_RATE \
    --mixed_precision="bf16" \
    --dataloader_num_workers=$DATALOADER_NUM_WORKERS \
    --image_aug \
    --dataset_type=$DATASET_TYPE \
    --state_noise_snr=$STATE_NOISE_SNR \
    --load_from_hdf5 \
    --report_to=wandb \
    --precomp_lang_embed \
    --gradient_accumulation_steps=$GRAD_ACCUM_STEPS \
    --model_config_path=$CONFIG_FILE \
    --CONFIG_NAME=$CONFIG_NAME \
    --config_path="configs/base.yaml"

TRAINING_EXIT_CODE=$?

echo ""
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ 消融实验训练成功完成!"
    echo "📁 模型保存到: $OUTPUT_DIR"
    echo "📊 训练配置保存到: $OUTPUT_DIR/training_config.yaml"
    echo "📈 查看wandb获取训练指标和可视化"
    
    # 显示最终统计信息
    if [ -f "$OUTPUT_DIR/ablation_study_config.json" ]; then
        echo ""
        echo "📋 消融实验最终统计:"
        python -c "
import json
try:
    with open('$OUTPUT_DIR/ablation_study_config.json', 'r') as f:
        config = json.load(f)
        print(f'   - 实验类型: {config.get(\"experiment_type\", \"N/A\")}')
        print(f'   - REPA对齐已移除: {config.get(\"repa_alignment_removed\", \"N/A\")}')
        visual_features = config.get('visual_features', {})
        print(f'   - SigLIP patches: {visual_features.get(\"siglip_patches\", \"N/A\")}')
        print(f'   - DINOv2 CLS: {visual_features.get(\"dinov2_cls\", \"N/A\")}')
        print(f'   - Depth CLS: {visual_features.get(\"depth_cls\", \"N/A\")}')
        print(f'   - 融合策略: {config.get(\"fusion_strategy\", \"N/A\")}')
except:
    print('   统计文件未找到或损坏。')
"
    fi
else
    echo "❌ 训练失败，退出代码: $TRAINING_EXIT_CODE"
    echo "📝 请检查上方日志获取错误详情"
    exit $TRAINING_EXIT_CODE
fi

echo ""
echo "🎯 下一步骤:"
echo "   1. 评估消融实验模型性能"
echo "   2. 对比输入侧融合vs REPA对齐的效果"
echo "   3. 分析不同视觉融合策略的影响"
echo "   4. 验证多模态视觉特征的有效性"
echo "   5. 生成消融实验报告"
echo ""