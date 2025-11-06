#!/usr/bin/env python3
"""
修复 dataset.py 中的 critical_labels 引用
"""

print("🔧 修复 train/dataset.py 中的 critical_labels...")

with open('train/dataset.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

modified = False
new_lines = []
in_collator = False
skip_block = False
block_indent = 0

for i, line in enumerate(lines):
    line_num = i + 1
    
    # 检测是否在 DataCollator 的 __call__ 方法中
    if 'class DataCollatorForVLAConsumerDataset' in line:
        in_collator = True
    
    if in_collator and 'def __call__' in line:
        in_collator = True
    
    # 方法1: 注释掉 critical_labels 的定义
    if 'critical_labels = []' in line and not line.strip().startswith('#'):
        new_lines.append(line.replace('critical_labels = []', '# critical_labels = []  # 🔴 视觉融合模式不需要'))
        modified = True
        print(f"✅ 第 {line_num} 行: 注释掉 critical_labels 定义")
        continue
    
    # 方法2: 注释掉收集 critical_labels 的代码
    if '"critical_labels" in instance' in line and not line.strip().startswith('#'):
        # 标记开始跳过
        skip_block = True
        block_indent = len(line) - len(line.lstrip())
        new_lines.append(line.replace('if "critical_labels"', '# if "critical_labels"  # 🔴 视觉融合模式不需要'))
        modified = True
        print(f"✅ 第 {line_num} 行: 注释掉 critical_labels 收集")
        continue
    
    # 如果在跳过块中
    if skip_block:
        current_indent = len(line) - len(line.lstrip())
        # 如果缩进回到原来的级别或更少，结束跳过
        if line.strip() and current_indent <= block_indent:
            skip_block = False
        else:
            # 注释这一行
            if line.strip() and not line.strip().startswith('#'):
                new_lines.append('#' + line)
                continue
    
    # 方法3: 注释掉检查 critical_labels 的代码
    if 'if len(critical_labels)' in line and not line.strip().startswith('#'):
        skip_block = True
        block_indent = len(line) - len(line.lstrip())
        new_lines.append(line.replace('if len(critical_labels)', '# if len(critical_labels)  # 🔴 视觉融合模式不需要'))
        modified = True
        print(f"✅ 第 {line_num} 行: 注释掉 critical_labels 检查")
        continue
    
    # 方法4: 注释掉堆叠 critical_labels 的代码
    if 'critical_labels' in line and 'torch.stack' in line and not line.strip().startswith('#'):
        new_lines.append('#' + line.replace('\n', '  # 🔴 视觉融合模式不需要\n'))
        modified = True
        print(f"✅ 第 {line_num} 行: 注释掉 critical_labels 堆叠")
        continue
    
    # 保留其他行
    new_lines.append(line)

# 保存
if modified:
    with open('train/dataset.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("\n✅ 修复完成")
else:
    print("\nℹ️  没有发现需要修复的内容")

# 验证
print("\n🔍 验证 critical_labels 引用:")
with open('train/dataset.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

remaining = []
for i, line in enumerate(lines, 1):
    if 'critical_labels' in line and not line.strip().startswith('#'):
        remaining.append((i, line.strip()[:80]))

if remaining:
    print(f"⚠️  仍有 {len(remaining)} 处 critical_labels 引用:")
    for line_num, line in remaining:
        print(f"   第 {line_num} 行: {line}")
else:
    print("✅ 所有 critical_labels 引用已清理")

