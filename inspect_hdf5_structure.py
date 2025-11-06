#!/usr/bin/env python3
"""
抓取任务诊断脚本
分析为什么抓取类任务识别不到关键时间段
"""

from data.critical_timestep_annotator import TaskDrivenCriticalTimestepAnnotator, TaskType
import h5py
import numpy as np
import os
import glob

def find_grasp_files():
    """查找抓取任务文件"""
    grasp_patterns = [
        "processed_data/lift_pot/**/*.hdf5",
        "processed_data/grasp*/**/*.hdf5", 
        "processed_data/pick*/**/*.hdf5",
        "processed_data/*lift*/**/*.hdf5"
    ]
    
    files = []
    for pattern in grasp_patterns:
        files.extend(glob.glob(pattern, recursive=True))
    
    return files[:3]  # 只取前3个文件分析

def diagnose_grasp_file(file_path):
    """诊断单个抓取任务文件"""
    print(f"\n📁 诊断文件: {file_path}")
    print("-" * 50)
    
    try:
        with h5py.File(file_path, 'r') as f:
            qpos = f['observations']['qpos'][:]
        
        print(f"数据形状: {qpos.shape}")
        
        # 1. 分析夹爪行为
        left_gripper = qpos[:, 6]
        right_gripper = qpos[:, 13]
        
        print(f"\n🤏 夹爪数据分析:")
        print(f"左臂: 范围[{left_gripper.min():.3f}, {left_gripper.max():.3f}], 标准差{left_gripper.std():.4f}")
        print(f"右臂: 范围[{right_gripper.min():.3f}, {right_gripper.max():.3f}], 标准差{right_gripper.std():.4f}")
        
        # 判断哪个臂是活跃的
        left_active = left_gripper.std() > 0.01
        right_active = right_gripper.std() > 0.01
        print(f"活跃机械臂: {'左臂' if left_active else ''}{'右臂' if right_active else ''}{'都不活跃' if not (left_active or right_active) else ''}")
        
        # 2. 分析夹爪变化模式
        left_delta = np.diff(left_gripper, prepend=left_gripper[0])
        right_delta = np.diff(right_gripper, prepend=right_gripper[0])
        
        print(f"\n📈 夹爪变化模式:")
        print(f"左臂最大闭合变化: {left_delta.min():.4f}")
        print(f"左臂最大张开变化: {left_delta.max():.4f}")
        print(f"右臂最大闭合变化: {right_delta.min():.4f}")
        print(f"右臂最大张开变化: {right_delta.max():.4f}")
        
        # 找出显著变化的时间点
        left_significant = np.where(np.abs(left_delta) > 0.05)[0]
        right_significant = np.where(np.abs(right_delta) > 0.05)[0]
        
        if len(left_significant) > 0:
            print(f"左臂显著变化点: {left_significant[:5]}...")  # 只显示前5个
        if len(right_significant) > 0:
            print(f"右臂显著变化点: {right_significant[:5]}...")
        
        # 3. 使用不同阈值测试
        print(f"\n🔧 测试不同参数:")
        
        test_configs = [
            ("默认参数", -0.05),
            ("更敏感", -0.03),
            ("最敏感", -0.01),
        ]
        
        for config_name, threshold in test_configs:
            annotator = TaskDrivenCriticalTimestepAnnotator(
                task_type=TaskType.GRASP,
                gripper_close_delta_threshold=threshold,
                verbose=False  # 避免输出太多
            )
            
            critical_labels, analysis_info = annotator.annotate(qpos)
            critical_steps = critical_labels.sum()
            segments = analysis_info.get('all_segments', [])
            
            print(f"  {config_name:10s} (阈值={threshold:5.2f}): 关键步数={critical_steps:3d}, 时间段数={len(segments)}")
        
        # 4. 详细分析一次，看具体检测过程
        print(f"\n🔍 详细检测过程:")
        annotator = TaskDrivenCriticalTimestepAnnotator(
            task_type=TaskType.GRASP,
            gripper_close_delta_threshold=-0.03,  # 使用敏感参数
            verbose=True  # 显示详细过程
        )
        
        critical_labels, analysis_info = annotator.annotate(qpos)
        
        # 5. 分析夹爪时序
        print(f"\n⏰ 夹爪时序分析:")
        
        # 找到夹爪变化的时间点
        active_gripper = right_gripper if right_active else left_gripper
        active_arm_name = "右臂" if right_active else "左臂"
        
        if left_active or right_active:
            delta = np.diff(active_gripper, prepend=active_gripper[0])
            
            # 找到闭合和张开的时间点
            close_points = np.where(delta < -0.03)[0]  # 闭合
            open_points = np.where(delta > 0.03)[0]    # 张开
            
            print(f"{active_arm_name}夹爪闭合时间点: {close_points}")
            print(f"{active_arm_name}夹爪张开时间点: {open_points}")
            
            # 分析时序：是先闭合后张开，还是先张开后闭合？
            if len(close_points) > 0 and len(open_points) > 0:
                first_close = close_points[0] if len(close_points) > 0 else float('inf')
                first_open = open_points[0] if len(open_points) > 0 else float('inf')
                
                if first_close < first_open:
                    print("时序模式: 先闭合后张开 (典型的抓取模式)")
                else:
                    print("时序模式: 先张开后闭合 (可能是放置或预抓取)")
        else:
            print("夹爪无显著变化，可能是静态抓取")
        
        return True
        
    except Exception as e:
        print(f"❌ 诊断失败: {e}")
        return False

def main():
    """主诊断函数"""
    print("🔍 抓取任务关键时间段检测诊断")
    print("=" * 60)
    
    # 查找抓取任务文件
    files = find_grasp_files()
    
    if not files:
        print("❌ 未找到抓取任务文件")
        print("请检查以下目录是否存在抓取相关的HDF5文件:")
        print("- processed_data/lift_pot/")
        print("- processed_data/grasp*/")
        print("- processed_data/pick*/")
        return
    
    print(f"找到 {len(files)} 个抓取任务文件进行诊断")
    
    # 逐个诊断
    for file_path in files:
        success = diagnose_grasp_file(file_path)
        if not success:
            continue
    
    print(f"\n💡 常见问题和解决方案:")
    print("1. 如果夹爪变化很小 (<0.05):")
    print("   - 使用更敏感的阈值: gripper_close_delta_threshold = -0.01")
    
    print("2. 如果是先张开后闭合的模式:")
    print("   - 可能需要检测张开动作而不是闭合动作")
    print("   - 或者检查任务类型是否正确")
    
    print("3. 如果夹爪基本不变:")
    print("   - 可能是静态抓取，夹爪一直保持闭合状态")
    print("   - 需要基于其他特征（如速度、加速度）来检测关键时间段")
    
    print("4. 如果减速点检测不到:")
    print("   - 调整 min_deceleration_threshold = -0.0001")
    print("   - 调整 relative_low_speed_ratio = 0.25")

if __name__ == "__main__":
    main()