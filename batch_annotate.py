#!/usr/bin/env python3
"""
简单批量关键时间段标注脚本
输入任务名称和任务类型，批量标注该任务下的所有轨迹
"""

import os
import h5py
import numpy as np
import glob
from pathlib import Path
from data.critical_timestep_annotator import TaskDrivenCriticalTimestepAnnotator, TaskType

def batch_annotate_task():
    """批量标注指定任务的所有轨迹"""
    
    # 用户输入
    task_name = input("输入任务名称 (如: click_bell): ").strip()
    task_type_input = input("输入任务类型 (1=抓取, 2=点击): ").strip()
    
    try:
        task_type = TaskType(int(task_type_input))
    except ValueError:
        print("错误: 任务类型必须是 1 或 2")
        return
    
    # 查找任务文件夹
    data_dirs = ["processed_data", "data", "."]
    task_dir = None
    
    for base_dir in data_dirs:
        potential_path = os.path.join(base_dir, task_name)
        if os.path.exists(potential_path):
            task_dir = potential_path
            break
        
        # 也尝试在子目录中查找
        for root, dirs, files in os.walk(base_dir):
            if task_name in dirs:
                task_dir = os.path.join(root, task_name)
                break
    
    if not task_dir:
        print(f"错误: 未找到任务目录 '{task_name}'")
        return
    
    print(f"找到任务目录: {task_dir}")
    
    # 查找所有HDF5文件
    hdf5_files = []
    for root, dirs, files in os.walk(task_dir):
        for file in files:
            if file.endswith('.hdf5'):
                hdf5_files.append(os.path.join(root, file))
    
    if not hdf5_files:
        print(f"错误: 在 {task_dir} 中未找到HDF5文件")
        return
    
    print(f"找到 {len(hdf5_files)} 个轨迹文件")
    
    # 创建标注器
    annotator = TaskDrivenCriticalTimestepAnnotator(
        task_type=task_type,
        gripper_close_delta_threshold=-0.01,
        verbose=False  # 关闭详细输出以提高效率
    )
    
    # 批量处理
    results = []
    successful = 0
    failed = 0
    
    print(f"\n开始批量标注...")
    print("-" * 60)
    
    for i, file_path in enumerate(hdf5_files):
        episode_name = os.path.basename(os.path.dirname(file_path))
        
        try:
            # 读取数据
            with h5py.File(file_path, 'r') as f:
                qpos = f['observations']['qpos'][:]
            
            # 执行标注
            critical_labels, analysis_info = annotator.annotate(qpos)
            
            # 统计结果
            total_steps = len(critical_labels)
            critical_steps = critical_labels.sum()
            critical_ratio = critical_steps / total_steps
            num_segments = len(analysis_info['all_segments'])
            
            # 提取时间段
            segments = []
            if critical_steps > 0:
                critical_indices = np.where(critical_labels == 1)[0]
                start = critical_indices[0]
                for j in range(1, len(critical_indices)):
                    if critical_indices[j] - critical_indices[j-1] > 1:
                        segments.append((start, critical_indices[j-1]))
                        start = critical_indices[j]
                segments.append((start, critical_indices[-1]))
            
            # 记录结果
            result = {
                'file_path': file_path,
                'episode': episode_name,
                'total_steps': total_steps,
                'critical_steps': int(critical_steps),
                'critical_ratio': critical_ratio,
                'segments': segments,
                'success': True
            }
            
            results.append(result)
            successful += 1
            
            # 打印进度和关键段
            status = f"[{i+1:3d}/{len(hdf5_files)}] {episode_name:20s} "
            status += f"步数:{total_steps:3d} 关键:{critical_steps:2d} "
            status += f"比例:{critical_ratio:.3f} "
            
            if segments:
                segments_str = " ".join([f"({start}-{end})" for start, end in segments])
                status += f"关键段:{segments_str}"
            else:
                status += "关键段:无"
            
            print(status)
            
        except Exception as e:
            result = {
                'file_path': file_path,
                'episode': episode_name,
                'error': str(e),
                'success': False
            }
            results.append(result)
            failed += 1
            print(f"[{i+1:3d}/{len(hdf5_files)}] {episode_name:20s} 失败: {e}")
    
    # 生成总结报告
    print("\n" + "=" * 60)
    print("批量标注完成")
    print("=" * 60)
    
    print(f"任务名称: {task_name}")
    print(f"任务类型: {'点击' if task_type == TaskType.CLICK else '抓取'}")
    print(f"处理文件: {len(hdf5_files)}")
    print(f"成功: {successful}")
    print(f"失败: {failed}")
    print(f"成功率: {successful/len(hdf5_files):.1%}")
    
    if successful > 0:
        successful_results = [r for r in results if r['success']]
        
        # 统计分析
        total_steps = [r['total_steps'] for r in successful_results]
        critical_steps = [r['critical_steps'] for r in successful_results]
        critical_ratios = [r['critical_ratio'] for r in successful_results]
        segment_counts = [len(r['segments']) for r in successful_results]
        
        print(f"\n统计分析:")
        print(f"  平均轨迹长度: {np.mean(total_steps):.1f} 步")
        print(f"  平均关键步数: {np.mean(critical_steps):.1f} 步")
        print(f"  平均关键比例: {np.mean(critical_ratios):.3f}")
        print(f"  平均时间段数: {np.mean(segment_counts):.1f}")
        print(f"  关键比例范围: [{min(critical_ratios):.3f}, {max(critical_ratios):.3f}]")
        
        # 显示一些典型例子
        print(f"\n典型例子:")
        
        # 关键比例最高的
        max_ratio_result = max(successful_results, key=lambda x: x['critical_ratio'])
        print(f"  关键比例最高: {max_ratio_result['episode']} ({max_ratio_result['critical_ratio']:.3f})")
        
        # 关键比例最低的（但>0）
        positive_results = [r for r in successful_results if r['critical_ratio'] > 0]
        if positive_results:
            min_ratio_result = min(positive_results, key=lambda x: x['critical_ratio'])
            print(f"  关键比例最低: {min_ratio_result['episode']} ({min_ratio_result['critical_ratio']:.3f})")
        
        # 时间段最多的
        max_segments_result = max(successful_results, key=lambda x: len(x['segments']))
        print(f"  时间段最多: {max_segments_result['episode']} ({len(max_segments_result['segments'])}段)")
        
        # 没有检测到关键时间段的
        no_segments = [r for r in successful_results if len(r['segments']) == 0]
        if no_segments:
            print(f"  无关键时间段: {len(no_segments)} 个轨迹")
    
    # 询问是否保存结果
    save_results = input(f"\n是否保存结果到文件? (y/N): ").strip().lower()
    
    if save_results in ['y', 'yes']:
        output_file = f"{task_name}_annotation_results.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"任务名称: {task_name}\n")
            f.write(f"任务类型: {'点击' if task_type == TaskType.CLICK else '抓取'}\n")
            f.write(f"处理时间: {np.datetime64('now')}\n")
            f.write(f"成功率: {successful}/{len(hdf5_files)} ({successful/len(hdf5_files):.1%})\n\n")
            
            f.write("详细结果:\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Episode':20s} {'总步数':>8s} {'关键步数':>8s} {'关键比例':>8s} {'关键时间段':>30s} {'状态':>8s}\n")
            f.write("-" * 100 + "\n")
            
            for result in results:
                if result['success']:
                    segments_str = "无" if not result['segments'] else " ".join([f"({start}-{end})" for start, end in result['segments']])
                    
                    f.write(f"{result['episode']:20s} ")
                    f.write(f"{result['total_steps']:8d} ")
                    f.write(f"{result['critical_steps']:8d} ")
                    f.write(f"{result['critical_ratio']:8.3f} ")
                    f.write(f"{segments_str:30s} ")
                    f.write(f"{'成功':8s}\n")
                else:
                    f.write(f"{result['episode']:20s} ")
                    f.write(f"{'N/A':8s} {'N/A':8s} {'N/A':8s} {'错误':30s} ")
                    f.write(f"{'失败':8s}\n")
        
        print(f"结果已保存到: {output_file}")
    
    print(f"\n批量标注完成!")

if __name__ == "__main__":
    batch_annotate_task()