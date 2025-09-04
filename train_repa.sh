CONFIG_NAME="$1"
CONFIG_FILE="model_config/${CONFIG_NAME}.yml"

if [ -z "$CONFIG_NAME" ]; then
    echo "Usage: ./train_constrained_weights.sh CONFIG_NAME"
    echo "Example: ./train_constrained_weights.sh constrained_weights_grasp"
    exit 1
fi

echo "🚀 Starting RDT + Dual-Teacher REPA + Constrained Weights Training"
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

# 🆕 Dual-Teacher REPA + Constrained Weights Configuration
enable_repa_loss: true
repa_loss_weight: 0.2
use_dinov2_features: true
use_depth_features: true
routing_loss_weight: 0.1

# 🆕 Critical Timestep Annotation Configuration
enable_critical_annotation: true
task_type: 1  # 1=grasp, 2=click
critical_annotation_config:
  relative_low_speed_ratio: 0.15
  min_deceleration_threshold: -0.0008
  gripper_close_delta_threshold: -0.01
  smooth: true
  verbose: false

# 🆕 Constrained Adaptive Weight Learning Configuration
enable_constrained_weights: true
constrained_weight_config:
  critical_depth_min: 0.6      # Critical timesteps: depth weight minimum
  critical_depth_max: 0.9      # Critical timesteps: depth weight maximum
  non_critical_global_min: 0.6 # Non-critical timesteps: global weight minimum
  non_critical_global_max: 0.9 # Non-critical timesteps: global weight maximum
  temperature_init: 1.0        # Initial temperature for weight learning
  hidden_dim: 512              # Hidden dimension for weight predictor network
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
export WANDB_PROJECT="RDT_Constrained_Weights"
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

# 🆕 Read constrained weights specific settings
ENABLE_REPA_LOSS=$(read_yaml_config "$CONFIG_FILE" enable_repa_loss)
REPA_LOSS_WEIGHT=$(read_yaml_config "$CONFIG_FILE" repa_loss_weight)
USE_DINOV2_FEATURES=$(read_yaml_config "$CONFIG_FILE" use_dinov2_features)
USE_DEPTH_FEATURES=$(read_yaml_config "$CONFIG_FILE" use_depth_features)
ROUTING_LOSS_WEIGHT=$(read_yaml_config "$CONFIG_FILE" routing_loss_weight)
ENABLE_CRITICAL_ANNOTATION=$(read_yaml_config "$CONFIG_FILE" enable_critical_annotation)
TASK_TYPE=$(read_yaml_config "$CONFIG_FILE" task_type)
ENABLE_CONSTRAINED_WEIGHTS=$(read_yaml_config "$CONFIG_FILE" enable_constrained_weights)

# Clean up quotes from values
CUDA_USE=$(echo "$CUDA_USE" | tr -d '"')
OUTPUT_DIR=$(echo "$OUTPUT_DIR" | tr -d '"')
LR_SCHEDULER=$(echo "$LR_SCHEDULER" | tr -d '"')
DATASET_TYPE=$(echo "$DATASET_TYPE" | tr -d '"')

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
echo "🎯 Constrained Weights Features:"
echo "   - REPA Loss: ${ENABLE_REPA_LOSS:-true}"
echo "   - DINOv2 Features: ${USE_DINOV2_FEATURES:-true}"
echo "   - Depth Features: ${USE_DEPTH_FEATURES:-true}"
echo "   - Critical Annotation: ${ENABLE_CRITICAL_ANNOTATION:-true}"
echo "   - Task Type: ${TASK_TYPE:-1}"
echo "   - Constrained Weights: ${ENABLE_CONSTRAINED_WEIGHTS:-true}"
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

# Launch training using main.py (which calls train/train.py)
echo "🏃 Launching constrained weights training..."
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
    echo "✅ Constrained weights training completed successfully!"
    echo "📁 Model saved to: $OUTPUT_DIR"
    echo "📊 Training config saved to: $OUTPUT_DIR/training_config.yaml"
    echo "📈 Check wandb for training metrics and visualizations"
    
    # Display final statistics if available
    if [ -f "$OUTPUT_DIR/constrained_training_config.json" ]; then
        echo ""
        echo "📋 Final Training Statistics:"
        python -c "
import json
try:
    with open('$OUTPUT_DIR/constrained_training_config.json', 'r') as f:
        config = json.load(f)
        stats = config.get('final_statistics', {})
        print(f'   - Total Timesteps: {stats.get(\"total_timesteps\", \"N/A\")}')
        print(f'   - Critical Ratio: {stats.get(\"critical_ratio\", \"N/A\"):.3f}')
        print(f'   - Routing Accuracy: {stats.get(\"avg_routing_accuracy\", \"N/A\"):.3f}')
        print(f'   - Task Type: {config.get(\"task_name\", \"N/A\")}')
except:
    print('   Statistics file not found or corrupted.')
"
    fi
else
    echo "❌ Training failed with exit code: $TRAINING_EXIT_CODE"
    echo "📝 Check the logs above for error details"
    exit $TRAINING_EXIT_CODE
fi

echo ""
echo "🎯 Next Steps:"
echo "   1. Evaluate the trained model performance"
echo "   2. Analyze the constrained weight learning effectiveness"
echo "   3. Check routing network accuracy and convergence"
echo "   4. Validate critical timestep annotation quality"
echo ""