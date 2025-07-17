#!/bin/bash
# RDT with REPA Training Script
# Usage: ./train_repa.sh CONFIG_NAME

CONFIG_NAME="$1"
CONFIG_FILE="model_config/${CONFIG_NAME}.yml"

if [ -z "$CONFIG_NAME" ]; then
    echo "Usage: ./train_repa.sh CONFIG_NAME"
    echo "Example: ./train_repa.sh my_task_repa_config"
    exit 1
fi

echo "🚀 Starting RDT + REPA Training"
echo "CONFIG_NAME: $CONFIG_NAME"
echo "CONFIG_FILE: $CONFIG_FILE"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file $CONFIG_FILE does not exist!"
    exit 1
fi

# Environment setup
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="../weights/RDT/siglip-so400m-patch14-384"
export WANDB_PROJECT="RDT_REPA"
export WANDB_DEFAULT_RUN_NAME="${CONFIG_NAME}"

# Read key training settings from config
TRAIN_BATCH_SIZE=$(python scripts/read_yaml.py "$CONFIG_FILE" train_batch_size)
SAMPLE_BATCH_SIZE=$(python scripts/read_yaml.py "$CONFIG_FILE" sample_batch_size)
MAX_TRAIN_STEPS=$(python scripts/read_yaml.py "$CONFIG_FILE" max_train_steps)
CHECKPOINTING_PERIOD=$(python scripts/read_yaml.py "$CONFIG_FILE" checkpointing_period)
SAMPLE_PERIOD=$(python scripts/read_yaml.py "$CONFIG_FILE" sample_period)
CHECKPOINTS_TOTAL_LIMIT=$(python scripts/read_yaml.py "$CONFIG_FILE" checkpoints_total_limit)
LR_SCHEDULER=$(python scripts/read_yaml.py "$CONFIG_FILE" lr_scheduler)
LEARNING_RATE=$(python scripts/read_yaml.py "$CONFIG_FILE" learning_rate)
DATALOADER_NUM_WORKERS=$(python scripts/read_yaml.py "$CONFIG_FILE" dataloader_num_workers)
DATASET_TYPE=$(python scripts/read_yaml.py "$CONFIG_FILE" dataset_type)
STATE_NOISE_SNR=$(python scripts/read_yaml.py "$CONFIG_FILE" state_noise_snr)
GRAD_ACCUM_STEPS=$(python scripts/read_yaml.py "$CONFIG_FILE" gradient_accumulation_steps)
OUTPUT_DIR=$(python scripts/read_yaml.py "$CONFIG_FILE" checkpoint_path)
CUDA_USE=$(python scripts/read_yaml.py "$CONFIG_FILE" cuda_visible_device)

# Clean up quotes
CUDA_USE=$(echo "$CUDA_USE" | tr -d '"')
OUTPUT_DIR=$(echo "$OUTPUT_DIR" | tr -d '"')

# Create output directory
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "✅ Created output directory: $OUTPUT_DIR"
fi

# Copy config file to output directory for reference
cp "$CONFIG_FILE" "$OUTPUT_DIR/training_config.yaml"

export CUDA_VISIBLE_DEVICES=$CUDA_USE

echo ""
echo "🔧 Configuration Summary:"
echo "   - Batch Size: $TRAIN_BATCH_SIZE"
echo "   - Max Steps: $MAX_TRAIN_STEPS"
echo "   - Learning Rate: $LEARNING_RATE"
echo "   - Output Directory: $OUTPUT_DIR"
echo "   - CUDA Devices: $CUDA_USE"
echo ""

# Compute dataset statistics if needed
python -m data.compute_dataset_stat_hdf5 --task_name $CONFIG_NAME

# Launch training (only pass main.py支持的参数)
echo "🏃 Launching training..."
accelerate launch --main_process_port=28499 main.py \
    --deepspeed="./configs/zero2.json" \
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
    --dataset_type="finetune" \
    --state_noise_snr=$STATE_NOISE_SNR \
    --load_from_hdf5 \
    --report_to=wandb \
    --precomp_lang_embed \
    --gradient_accumulation_steps=$GRAD_ACCUM_STEPS \
    --model_config_path=$CONFIG_FILE \
    --CONFIG_NAME=$CONFIG_NAME

echo "✅ Training completed!"
