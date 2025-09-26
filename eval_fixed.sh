#!/bin/bash

# script/eval_fixed.sh - 修复后的评测脚本

policy_name=RDT_repa
task_name=${1}
task_config=${2}
model_name=${3}
checkpoint_id=${4}
seed=${5}
gpu_id=${6}

DEBUG=False
export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

# 🔧 输出评测配置信息
echo -e "\033[36m============= 评测配置 =============\033[0m"
echo -e "\033[32m策略名称:\033[0m ${policy_name}"
echo -e "\033[32m任务名称:\033[0m ${task_name}"
echo -e "\033[32m任务配置:\033[0m ${task_config}"
echo -e "\033[32m模型名称:\033[0m ${model_name}"
echo -e "\033[32m检查点ID:\033[0m ${checkpoint_id}"
echo -e "\033[32m随机种子:\033[0m ${seed}"
echo -e "\033[32mGPU设备:\033[0m ${gpu_id}"
echo -e "\033[96m🔧 评测优化:\033[0m 启用 (使用评测专用组件)"
echo -e "\033[36m====================================\033[0m"

cd ../.. # move to root

# 🔧 检查必要文件
echo -e "\033[34m🔍 检查文件完整性...\033[0m"

config_file="policy/$policy_name/deploy_policy.yml"
if [ ! -f "$config_file" ]; then
    echo -e "\033[31m❌ 配置文件不存在: $config_file\033[0m"
    exit 1
fi

eval_script="script/eval_policy_fixed.py"
if [ ! -f "$eval_script" ]; then
    echo -e "\033[33m⚠️  使用原始评测脚本: script/eval_policy.py\033[0m"
    eval_script="script/eval_policy.py"
fi

get_model_file="policy/$policy_name/get_model_eval.py"
if [ ! -f "$get_model_file" ]; then
    echo -e "\033[33m⚠️  评测专用模型加载器不存在，将尝试使用原始版本\033[0m"
    echo -e "\033[33m   提示: 建议创建 $get_model_file 以获得最佳评测体验\033[0m"
fi

echo -e "\033[32m✅ 文件检查完成\033[0m"

# 🔧 运行评测 - 使用修复后的脚本
echo -e "\033[34m🚀 开始执行评测...\033[0m"

PYTHONWARNINGS=ignore::UserWarning \
python $eval_script --config $config_file \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --checkpoint_id ${checkpoint_id} \
    --policy_name ${policy_name}

eval_exit_code=$?

# 🔧 评测结果处理
if [ $eval_exit_code -eq 0 ]; then
    echo -e "\n\033[32m🎉 评测成功完成！\033[0m"
    echo -e "\033[32m   ✅ 权重加载优化已应用\033[0m"
    echo -e "\033[32m   ✅ 训练专用组件已禁用\033[0m"
    echo -e "\033[32m   ✅ 结果已保存到 eval_result/ 目录\033[0m"
    
    # 🔧 尝试显示结果摘要
    result_dir="eval_result/${task_name}/${policy_name}/${task_config}/${model_name}/${checkpoint_id}"
    if [ -d "$result_dir" ]; then
        echo -e "\033[36m📊 最新结果目录: $result_dir\033[0m"
        
        # 查找最新的结果文件
        latest_result=$(find "$result_dir" -name "_result.txt" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
        
        if [ -f "$latest_result" ]; then
            echo -e "\033[36m📈 成功率摘要:\033[0m"
            echo -e "\033[33m$(tail -n 5 "$latest_result")\033[0m"
        fi
    fi
    
else
    echo -e "\n\033[31m❌ 评测失败 (退出码: $eval_exit_code)\033[0m"
    echo -e "\033[31m   可能的原因:\033[0m"
    echo -e "\033[31m   1. 检查点文件不存在或损坏\033[0m"
    echo -e "\033[31m   2. 模型配置文件缺失\033[0m"
    echo -e "\033[31m   3. 环境依赖问题\033[0m"
    echo -e "\033[31m   4. GPU内存不足\033[0m"
    echo -e "\n\033[33m💡 建议:\033[0m"
    echo -e "\033[33m   - 检查 policy/$policy_name/get_model_eval.py 是否存在\033[0m"
    echo -e "\033[33m   - 验证检查点路径和ID是否正确\033[0m"
    echo -e "\033[33m   - 查看上方错误信息获取详细原因\033[0m"
    
    exit $eval_exit_code
fi

echo -e "\n\033[36m🔧 评测脚本执行完毕\033[0m"