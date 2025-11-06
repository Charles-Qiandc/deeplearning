#!/usr/bin/env python3
"""
速度计算检查工具
详细检查每一步的关节角度、末端位置、速度计算过程
"""

import os
import sys
import numpy as np
import h5py

# 导入我们的标注器
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from critical_timestep_annotator import AgilexForwardKinematics, AgilexCriticalAnnotator


def detailed_velocity_inspection(file_path: str, max_steps: int = 50):
    """
    详细检查速度计算过程
    
    Args:
        file_path: HDF5文件路径
        max_steps: 最大检查步数
    """
    print("🔍 详细速度计算检查")
    print("=" * 60)
    print(f"文件: {file_path}")
    
    # 1. 读取数据
    with h5py.File(file_path, 'r') as f:
        qpos = f['observations']['qpos'][:]
        
    print(f"总步数: {len(qpos)}")
    print(f"检查前 {min(max_steps, len(qpos))} 步")
    print()
    
    # 2. 初始化正运动学计算器
    fk_calculator = AgilexForwardKinematics()
    
    # 3. 逐步计算并显示
    print("="*120)
    print("步骤 | 左臂关节角度                    | 右臂关节角度                    | 左臂末端位置        | 右臂末端位置        | 左臂速度   | 右臂速度")
    print("="*120)
    
    left_ee_positions = []
    right_ee_positions = []
    
    for t in range(min(max_steps, len(qpos))):
        # 提取关节角度
        left_joints = qpos[t, :6]
        right_joints = qpos[t, 7:13]
        
        # 计算末端位置
        left_ee_pos = fk_calculator.compute_forward_kinematics(
            left_joints, fk_calculator.left_dh_params, fk_calculator.left_base_offset)
        right_ee_pos = fk_calculator.compute_forward_kinematics(
            right_joints, fk_calculator.right_dh_params, fk_calculator.right_base_offset)
        
        left_ee_positions.append(left_ee_pos)
        right_ee_positions.append(right_ee_pos)
        
        # 计算速度（从第2步开始）
        if t > 0:
            left_velocity = np.linalg.norm(left_ee_pos - left_ee_positions[t-1])
            right_velocity = np.linalg.norm(right_ee_pos - right_ee_positions[t-1])
        else:
            left_velocity = 0.0
            right_velocity = 0.0
        
        # 格式化输出
        left_joints_str = "[" + ", ".join([f"{x:6.3f}" for x in left_joints]) + "]"
        right_joints_str = "[" + ", ".join([f"{x:6.3f}" for x in right_joints]) + "]"
        left_pos_str = f"[{left_ee_pos[0]:6.3f}, {left_ee_pos[1]:6.3f}, {left_ee_pos[2]:6.3f}]"
        right_pos_str = f"[{right_ee_pos[0]:6.3f}, {right_ee_pos[1]:6.3f}, {right_ee_pos[2]:6.3f}]"
        
        print(f"{t:3d}  | {left_joints_str:30s} | {right_joints_str:30s} | {left_pos_str:18s} | {right_pos_str:18s} | {left_velocity:8.5f} | {right_velocity:8.5f}")
        
        # 每10步停顿一下
        if (t + 1) % 10 == 0:
            print("-" * 120)
    
    print("=" * 120)
    
    # 4. 完整的轨迹速度计算
    left_ee_positions = np.array(left_ee_positions)
    right_ee_positions = np.array(right_ee_positions)
    
    # 计算完整的速度序列
    left_diff = np.diff(left_ee_positions, axis=0, prepend=left_ee_positions[0:1])
    right_diff = np.diff(right_ee_positions, axis=0, prepend=right_ee_positions[0:1])
    
    left_velocities = np.linalg.norm(left_diff, axis=-1)
    right_velocities = np.linalg.norm(right_diff, axis=-1)
    
    # 5. 统计分析
    print(f"\n📊 速度统计分析 (前{len(left_velocities)}步):")
    print(f"左臂速度: 平均={np.mean(left_velocities):.5f}, 最大={np.max(left_velocities):.5f}, 最小={np.min(left_velocities):.5f}")
    print(f"右臂速度: 平均={np.mean(right_velocities):.5f}, 最大={np.max(right_velocities):.5f}, 最小={np.min(right_velocities):.5f}")
    
    # 6. 检查关节角度变化
    print(f"\n📈 关节角度变化分析:")
    left_joint_changes = np.max(np.abs(np.diff(qpos[:max_steps, :6], axis=0)), axis=0)
    right_joint_changes = np.max(np.abs(np.diff(qpos[:max_steps, 7:13], axis=0)), axis=0)
    
    print(f"左臂各关节最大变化: {left_joint_changes}")
    print(f"右臂各关节最大变化: {right_joint_changes}")
    
    # 7. 位置轨迹合理性检查
    print(f"\n🎯 末端位置合理性检查:")
    left_pos_range = np.max(left_ee_positions, axis=0) - np.min(left_ee_positions, axis=0)
    right_pos_range = np.max(right_ee_positions, axis=0) - np.min(right_ee_positions, axis=0)
    
    print(f"左臂末端位置范围: X={left_pos_range[0]:.3f}, Y={left_pos_range[1]:.3f}, Z={left_pos_range[2]:.3f}")
    print(f"右臂末端位置范围: X={right_pos_range[0]:.3f}, Y={right_pos_range[1]:.3f}, Z={right_pos_range[2]:.3f}")
    
    # 8. 检查是否有异常值
    print(f"\n⚠️  异常值检查:")
    left_high_speed = np.where(left_velocities > 0.1)[0]
    right_high_speed = np.where(right_velocities > 0.1)[0]
    
    if len(left_high_speed) > 0:
        print(f"左臂高速步骤 (>0.1): {left_high_speed[:10]}...")  # 只显示前10个
    if len(right_high_speed) > 0:
        print(f"右臂高速步骤 (>0.1): {right_high_speed[:10]}...")
    
    if len(left_high_speed) == 0 and len(right_high_speed) == 0:
        print("✅ 无异常高速值")
    
    return left_velocities, right_velocities, left_ee_positions, right_ee_positions


def compare_with_annotator(file_path: str, velocity_threshold: float = 0.01):
    """
    与标注器结果对比
    """
    print(f"\n🔄 与标注器结果对比 (阈值={velocity_threshold})")
    print("=" * 50)
    
    # 使用标注器处理
    annotator = AgilexCriticalAnnotator(velocity_threshold=velocity_threshold)
    
    with h5py.File(file_path, 'r') as f:
        qpos = f['observations']['qpos'][:]
    
    critical_labels, analysis_info = annotator.annotate(qpos)
    
    left_velocities = analysis_info['left_velocity']
    right_velocities = analysis_info['right_velocity']
    stats = analysis_info['statistics']
    
    print(f"标注器结果:")
    print(f"  总步数: {stats['total_steps']}")
    print(f"  关键步数: {stats['critical_steps']}")
    print(f"  关键比例: {stats['critical_ratio']:.3f}")
    print(f"  左臂平均速度: {stats['left_avg_velocity']:.5f}")
    print(f"  右臂平均速度: {stats['right_avg_velocity']:.5f}")
    
    # 显示前20步的标注结果
    print(f"\n前20步标注结果:")
    print("步骤 | 左臂速度 | 右臂速度 | 左<阈值 | 右<阈值 | 关键?")
    print("-" * 55)
    
    for t in range(min(20, len(critical_labels))):
        left_low = "✓" if left_velocities[t] < velocity_threshold else "✗"
        right_low = "✓" if right_velocities[t] < velocity_threshold else "✗"
        critical = "🔴" if critical_labels[t] == 1 else "⚪"
        
        print(f"{t:3d}  | {left_velocities[t]:7.5f} | {right_velocities[t]:7.5f} |   {left_low}    |   {right_low}    | {critical}")


def create_simple_visualization(left_velocities, right_velocities, critical_labels=None):
    """
    创建简单的文本可视化
    """
    print(f"\n📊 速度变化趋势 (文本图表)")
    print("=" * 50)
    
    # 归一化速度到0-20的范围用于显示
    max_vel = max(np.max(left_velocities), np.max(right_velocities))
    
    if max_vel > 0:
        left_norm = (left_velocities / max_vel * 20).astype(int)
        right_norm = (right_velocities / max_vel * 20).astype(int)
    else:
        left_norm = np.zeros_like(left_velocities, dtype=int)
        right_norm = np.zeros_like(right_velocities, dtype=int)
    
    print(f"速度范围: 0 到 {max_vel:.5f}")
    print("图例: L=左臂, R=右臂, *=关键时间段")
    print()
    
    for t in range(min(30, len(left_velocities))):
        # 创建图表行
        chart_line = [' '] * 25
        
        # 标记左臂速度
        if left_norm[t] < 25:
            chart_line[left_norm[t]] = 'L'
        
        # 标记右臂速度
        if right_norm[t] < 25:
            if chart_line[right_norm[t]] == 'L':
                chart_line[right_norm[t]] = 'B'  # Both
            else:
                chart_line[right_norm[t]] = 'R'
        
        # 添加关键时间段标记
        critical_marker = "*" if critical_labels is not None and critical_labels[t] == 1 else " "
        
        chart_str = ''.join(chart_line)
        print(f"{t:2d} |{chart_str}| {critical_marker} L:{left_velocities[t]:.5f} R:{right_velocities[t]:.5f}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="速度计算详细检查工具")
    parser.add_argument("--file", type=str, help="指定HDF5文件路径")
    parser.add_argument("--max_steps", type=int, default=30, help="检查的最大步数")
    parser.add_argument("--threshold", type=float, default=0.01, help="速度阈值")
    
    args = parser.parse_args()
    
    # 查找测试文件
    if args.file:
        test_file = args.file
    else:
        # 自动查找
        test_file = None
        search_paths = ["../processed_data", "../../processed_data", "processed_data"]
        
        for search_path in search_paths:
            if os.path.exists(search_path):
                for root, dirs, files in os.walk(search_path):
                    for file in files:
                        if file.endswith(".hdf5"):
                            test_file = os.path.join(root, file)
                            break
                    if test_file:
                        break
            if test_file:
                break
    
    if not test_file or not os.path.exists(test_file):
        print("❌ 未找到HDF5文件")
        print("请使用: python velocity_inspector.py --file=/path/to/your/file.hdf5")
        return
    
    print("🔧 Agilex速度计算详细检查工具")
    print("=" * 60)
    
    # 1. 详细检查速度计算
    left_vels, right_vels, left_pos, right_pos = detailed_velocity_inspection(test_file, args.max_steps)
    
    # 2. 与标注器对比
    compare_with_annotator(test_file, args.threshold)
    
    # 3. 简单可视化
    with h5py.File(test_file, 'r') as f:
        qpos = f['observations']['qpos'][:]
    
    annotator = AgilexCriticalAnnotator(velocity_threshold=args.threshold)
    critical_labels, _ = annotator.annotate(qpos)
    
    create_simple_visualization(left_vels, right_vels, critical_labels[:len(left_vels)])
    
    print(f"\n✅ 检查完成!")
    print(f"\n💡 如果速度看起来不合理:")
    print(f"   1. 检查关节角度是否有跳跃")
    print(f"   2. 检查DH参数是否正确")
    print(f"   3. 检查基座偏移是否合理")
    print(f"   4. 调整velocity_threshold参数")


if __name__ == "__main__":
    main()