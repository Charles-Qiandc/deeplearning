#!/bin/bash
# train_vision_fusion.sh
# RDT + Vision Fusion (DINOv2 + DepthAnythingV2) Training Script
# 关闭REPA对齐和关键时间段标注，使用输入侧视觉特征融合

CONFIG_NAME="$1"
CONFIG_FILE="model_config/${CONFIG_NAME}.yml"

if [ -z "$CONFIG_NAME" ]; then
    echo "Usage: ./train_vision_fusion.sh CONFIG_NAME"
    echo "Example: ./train_vision_fusion.sh vision_fusion_agilex"
    exit 1
fi

echo "🚀 Starting RDT + Vision Fusion Training"
echo "CONFIG_NAME: $CONFIG_NAME"
echo "CONFIG_FILE: $CONFIG_FILE"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file $CONFIG_FILE does not exist!"
    echo "📝 Creating a template config file..."
    
    # Create template config if it doesn't exist
    mkdir -p "model_config"
    cat > "$CONFIG_FILE" << EOF
# Generated template for $CONFIG_NAME
model: $CONFIG_NAME
data_path: training_data/$CONFIG_NAME
checkpoint_path: checkpoints/$CONFIG_NAME
pretrained_model_name_or_path: "../weights/RDT/rdt-1b"
cuda_visible_device: "0,1,2,3"
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

# ========================================
# 🆕 Vision Fusion Configuration
# ========================================

# Enable vision fusion (input-side feature fusion)
enable_vision_fusion: true
vision_fusion_type: "cross_attention"  # "cross_attention" or "simple"

# Visual feature encoders
use_dinov2_features: true   # DINOv2 for global semantic features
use_depth_features: true     # DepthAnythingV2 for depth geometric features

# Fusion module hyperparameters
fusion_num_heads: 8          # Number of attention heads
fusion_dropout: 0.1          # Dropout rate for fusion module

# 🔴 Disable REPA alignment and critical timestep annotation
enable_repa_loss: false
enable_soft_routing_repa: false
enable_critical_annotation: false

# Visual encoder paths (optional, will use defaults if not specified)
# dinov2_encoder_path: "facebook/dinov2-large"
# depth_encoder_path: "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
EOF
    
    echo "✅ Template config file created at: $CONFIG_FILE"
    echo "📋 Please review and modify the configuration as needed, then run the script again."
    exit 0
fi

# Environment setup
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="../weights/RDT/siglip-so400m-patch14-384"
export PRETRAINED_RDT="/home/deng_xiang/qian_daichao/RoboTwin/policy/weights/RDT/rdt-1b/pytorch_model.bin"
export WANDB_PROJECT="RDT_Vision_Fusion"
export WANDB_DEFAULT_RUN_NAME="${CONFIG_NAME}"

# Helper function to read YAML config
read_yaml_config() {
    local file="$1"
    local key="$2"
    if [ -f "scripts/read_yaml.py" ]; then
        python scripts/read_yaml.py "$file" "$key" 2>/dev/null
    else
        # Fallback to grep if read_yaml.py is not available
        grep "^$key:" "$file" | sed "s/^$key: *//" | tr -d '"' | head -1
    fi
}

# Read key training settings from config
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

# 🆕 Read vision fusion specific settings
ENABLE_VISION_FUSION=$(read_yaml_config "$CONFIG_FILE" enable_vision_fusion)
VISION_FUSION_TYPE=$(read_yaml_config "$CONFIG_FILE" vision_fusion_type)
USE_DINOV2_FEATURES=$(read_yaml_config "$CONFIG_FILE" use_dinov2_features)
USE_DEPTH_FEATURES=$(read_yaml_config "$CONFIG_FILE" use_depth_features)
FUSION_NUM_HEADS=$(read_yaml_config "$CONFIG_FILE" fusion_num_heads)
FUSION_DROPOUT=$(read_yaml_config "$CONFIG_FILE" fusion_dropout)

# Clean up quotes from values
CUDA_USE=$(echo "$CUDA_USE" | tr -d '"')
OUTPUT_DIR=$(echo "$OUTPUT_DIR" | tr -d '"')
LR_SCHEDULER=$(echo "$LR_SCHEDULER" | tr -d '"')
DATASET_TYPE=$(echo "$DATASET_TYPE" | tr -d '"')
VISION_FUSION_TYPE=$(echo "$VISION_FUSION_TYPE" | tr -d '"')

# Set defaults if values are empty
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
ENABLE_VISION_FUSION=${ENABLE_VISION_FUSION:-true}
VISION_FUSION_TYPE=${VISION_FUSION_TYPE:-cross_attention}
USE_DINOV2_FEATURES=${USE_DINOV2_FEATURES:-true}
USE_DEPTH_FEATURES=${USE_DEPTH_FEATURES:-true}
FUSION_NUM_HEADS=${FUSION_NUM_HEADS:-8}
FUSION_DROPOUT=${FUSION_DROPOUT:-0.1}

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
echo "   - Model Config: $CONFIG_NAME"
echo "   - Batch Size: $TRAIN_BATCH_SIZE"
echo "   - Max Steps: $MAX_TRAIN_STEPS"
echo "   - Learning Rate: $LEARNING_RATE"
echo "   - Output Directory: $OUTPUT_DIR"
echo "   - CUDA Devices: $CUDA_USE"
echo "   - Dataset Type: $DATASET_TYPE"
echo ""
echo "🎨 Vision Fusion Configuration:"
echo "   - Vision Fusion Enabled: $ENABLE_VISION_FUSION"
echo "   - Fusion Type: $VISION_FUSION_TYPE"
echo "   - DINOv2 Features: $USE_DINOV2_FEATURES"
echo "   - Depth Features: $USE_DEPTH_FEATURES"
echo "   - Attention Heads: $FUSION_NUM_HEADS"
echo "   - Dropout: $FUSION_DROPOUT"
echo ""
echo "🔴 Disabled Features:"
echo "   - REPA Alignment: false"
echo "   - Critical Timestep Annotation: false"
echo "   - Soft Routing: false"
echo ""

# Compute dataset statistics if needed
echo "📊 Computing dataset statistics..."
python -m data.compute_dataset_stat_hdf5 --task_name $CONFIG_NAME

# Validate required files exist
if [ ! -f "$PRETRAINED_RDT" ]; then
    echo "⚠️  Warning: Pretrained RDT model not found at: $PRETRAINED_RDT"
    echo "Please update the PRETRAINED_RDT path in this script."
fi

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "❌ main.py not found in current directory!"
    echo "Please run this script from the project root directory."
    exit 1
fi

# Test module imports before training
echo "🧪 Testing module imports..."
python -c "
import sys
try:
    from models.multimodal_encoder.vision_fusion_module import create_vision_fusion_module
    print('✅ Vision fusion module import successful')
except Exception as e:
    print(f'❌ Failed to import vision fusion module: {e}')
    sys.exit(1)

try:
    from models.multimodal_encoder.dinov2_encoder import create_dinov2_encoder
    print('✅ DINOv2 encoder import successful')
except Exception as e:
    print(f'❌ Failed to import DINOv2 encoder: {e}')
    sys.exit(1)

try:
    from models.multimodal_encoder.depth_encoder import create_depth_encoder
    print('✅ Depth encoder import successful')
except Exception as e:
    print(f'❌ Failed to import depth encoder: {e}')
    sys.exit(1)

print('✅ All module imports successful!')
" || {
    echo "❌ Module import test failed!"
    echo "Please ensure all required modules are properly installed."
    exit 1
}

# Launch training using main.py (which calls train/train.py)
echo "🏃 Launching vision fusion training..."
echo "📝 Command: accelerate launch --main_process_port=28499 main.py ..."
echo ""

accelerate launch --main_process_port=28499 main.py \
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
    echo "✅ Vision fusion training completed successfully!"
    echo "📁 Model saved to: $OUTPUT_DIR"
    echo "📊 Training config saved to: $OUTPUT_DIR/training_config.yaml"
    echo "📈 Check wandb for training metrics and visualizations"
    
    # Display final statistics if available
    if [ -f "$OUTPUT_DIR/vision_fusion_info.json" ]; then
        echo ""
        echo "📋 Vision Fusion Training Info:"
        python -c "
import json
try:
    with open('$OUTPUT_DIR/vision_fusion_info.json', 'r') as f:
        info = json.load(f)
        print(f'   - Fusion Type: {info.get(\"fusion_type\", \"N/A\")}')
        print(f'   - DINOv2 Enabled: {info.get(\"use_dinov2\", \"N/A\")}')
        print(f'   - Depth Enabled: {info.get(\"use_depth\", \"N/A\")}')
        print(f'   - Fusion Module Params: {info.get(\"fusion_params\", \"N/A\")}')
        print(f'   - Total Training Steps: {info.get(\"total_steps\", \"N/A\")}')
except:
    print('   Info file not found or corrupted.')
"
    fi
    
    # Create a summary file
    echo ""
    echo "📝 Creating training summary..."
    cat > "$OUTPUT_DIR/TRAINING_SUMMARY.txt" << EOF
========================================
RDT Vision Fusion Training Summary
========================================

Training Configuration:
- Config Name: $CONFIG_NAME
- Dataset Type: $DATASET_TYPE
- Total Steps: $MAX_TRAIN_STEPS
- Batch Size: $TRAIN_BATCH_SIZE
- Learning Rate: $LEARNING_RATE

Vision Fusion Setup:
- Fusion Type: $VISION_FUSION_TYPE
- DINOv2 Features: $USE_DINOV2_FEATURES
- Depth Features: $USE_DEPTH_FEATURES
- Attention Heads: $FUSION_NUM_HEADS
- Dropout: $FUSION_DROPOUT

Architecture:
- Input: SigLIP ($(grep "img_history_size" configs/base.yaml | awk '{print $2}') frames × $(grep "num_cameras" configs/base.yaml | awk '{print $2}') cameras × 729 patches)
- Fusion: DINOv2 (1369 patches) + Depth (1369 patches)
- Output: Fused tokens with same shape as SigLIP input

Model Checkpoints:
- Directory: $OUTPUT_DIR
- Checkpointing Period: every $CHECKPOINTING_PERIOD steps
- Total Checkpoints Kept: $CHECKPOINTS_TOTAL_LIMIT

Training Date: $(date)
Training Duration: See wandb logs for details

Next Steps:
1. Evaluate model performance on validation set
2. Compare with baseline RDT (without fusion)
3. Analyze fusion module attention weights
4. Test on real robot deployment

========================================
EOF
    echo "✅ Training summary saved to: $OUTPUT_DIR/TRAINING_SUMMARY.txt"
    
else
    echo "❌ Training failed with exit code: $TRAINING_EXIT_CODE"
    echo "📝 Check the logs above for error details"
    
    # Save error log
    echo "💾 Saving error information..."
    cat > "$OUTPUT_DIR/TRAINING_ERROR.txt" << EOF
Training failed with exit code: $TRAINING_EXIT_CODE
Timestamp: $(date)
Config: $CONFIG_NAME

Common issues to check:
1. CUDA out of memory -> Reduce batch size or use simpler fusion type
2. Module import errors -> Check if vision_fusion_module.py exists
3. Shape mismatch -> Verify img_history_size and num_cameras in config
4. Missing pretrained weights -> Check PRETRAINED_RDT path

See training logs above for detailed error messages.
EOF
    echo "❌ Error info saved to: $OUTPUT_DIR/TRAINING_ERROR.txt"
    exit $TRAINING_EXIT_CODE
fi

echo ""
echo "🎯 Next Steps:"
echo "   1. Evaluate the trained model performance"
echo "   2. Compare fusion vs baseline (no fusion)"
echo "   3. Analyze attention weights and fusion quality"
echo "   4. Visualize feature representations (DINOv2 + Depth)"
echo "   5. Test generalization on unseen scenarios"
echo ""
echo "📚 Useful Commands:"
echo "   # Verify checkpoint"
echo "   python scripts/verify_training.py --checkpoint_path $OUTPUT_DIR/checkpoint-5000"
echo ""
echo "   # Resume training"
echo "   ./train_vision_fusion.sh $CONFIG_NAME --resume_from_checkpoint $OUTPUT_DIR/checkpoint-5000"
echo ""
echo "   # Monitor training"
echo "   tail -f logs/${CONFIG_NAME}_training.log"
echo ""