# test_critical_annotator_real.py

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import yaml

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.critical_timestep_annotator import SimpleCriticalTimestepAnnotator
from data.hdf5_vla_dataset import HDF5VLADataset


def test_single_trajectory(dataset, index, annotator, save_dir="results"):
    """
    测试单个轨迹的标注效果
    
    Args:
        dataset: HDF5数据集
        index: 轨迹索引
        annotator: 标注器
        save_dir: 结果保存目录
    """
    # 获取轨迹数据
    sample = dataset.get_item(index=index, state_only=True)
    trajectory = sample['action']  # (T, D)
    
    print(f"\n=== 轨迹 {index} ===")
    print(f"轨迹形状: {trajectory.shape}")
    
    # 执行标注
    critical_mask, confidence, velocity = annotator.annotate(trajectory)
    stats = annotator.get_statistics(critical_mask, velocity)
    
    # 打印统计
    print(f"关键步数: {stats['critical_steps']}/{stats['total_steps']} ({stats['critical_ratio']:.1%})")
    print(f"关键段数: {stats['num_segments']}")
    print(f"关键段: {stats['segments']}")
    print(f"速度范围: [{stats['min_velocity']:.4f}, {stats['max_velocity']:.4f}]")
    print(f"平均速度: {stats['avg_velocity']:.4f}")
    
    # 可视化
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    
    # 1. 速度曲线
    axes[0].plot(velocity, 'b-', linewidth=1.5)
    axes[0].axhline(y=annotator.velocity_threshold, color='r', linestyle='--', 
                    alpha=0.5, label=f'Threshold={annotator.velocity_threshold}')
    axes[0].set_ylabel('Velocity')
    axes[0].set_title(f'Trajectory {index}: Velocity Profile')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. 关键段标记
    for start, end in stats['segments']:
        axes[0].axvspan(start, end, alpha=0.2, color='red')
    
    # 3. 关键时间步
    axes[1].fill_between(range(len(critical_mask)), 
                         critical_mask.astype(float), 
                         alpha=0.7, color='red', label='Critical')
    axes[1].set_ylabel('Critical Mask')
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_title(f'Critical Timesteps (Total: {stats["critical_steps"]})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 4. 置信度
    axes[2].plot(confidence, 'g-', linewidth=1.5)
    axes[2].fill_between(range(len(confidence)), confidence, alpha=0.3, color='green')
    axes[2].set_ylabel('Confidence')
    axes[2].set_title('Annotation Confidence')
    axes[2].set_ylim(0, 1.1)
    axes[2].grid(True, alpha=0.3)
    
    # 5. 轨迹的几个关键维度（可视化前几个关节的运动）
    dims_to_show = min(3, trajectory.shape[1])
    for d in range(dims_to_show):
        axes[3].plot(trajectory[:, d], alpha=0.7, label=f'Dim {d}')
    axes[3].set_ylabel('Joint Position')
    axes[3].set_xlabel('Time Step')
    axes[3].set_title('Sample Joint Trajectories')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # 标记关键段
    for start, end in stats['segments']:
        axes[3].axvspan(start, end, alpha=0.15, color='red')
    
    plt.tight_layout()
    
    # 保存图像
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'trajectory_{index}_analysis.png')
    plt.savefig(save_path, dpi=150)
    print(f"图表已保存到: {save_path}")
    plt.show()
    
    return stats


def test_multiple_trajectories(dataset, num_samples=5, annotator=None):
    """
    测试多个轨迹，统计整体表现
    
    Args:
        dataset: HDF5数据集
        num_samples: 要测试的轨迹数量
        annotator: 标注器
    """
    if annotator is None:
        annotator = SimpleCriticalTimestepAnnotator(
            velocity_threshold=0.01,
            window_size=5,
            expand_steps=3,
            smooth=True
        )
    
    all_stats = []
    
    # 确保不超过数据集大小
    num_samples = min(num_samples, len(dataset))
    
    print(f"\n测试 {num_samples} 个轨迹...")
    
    for i in range(num_samples):
        sample = dataset.get_item(index=i, state_only=True)
        trajectory = sample['action']
        
        critical_mask, confidence, velocity = annotator.annotate(trajectory)
        stats = annotator.get_statistics(critical_mask, velocity)
        all_stats.append(stats)
        
        print(f"轨迹 {i}: 关键比例={stats['critical_ratio']:.1%}, "
              f"段数={stats['num_segments']}, "
              f"平均速度={stats['avg_velocity']:.4f}")
    
    # 汇总统计
    print("\n=== 汇总统计 ===")
    critical_ratios = [s['critical_ratio'] for s in all_stats]
    num_segments = [s['num_segments'] for s in all_stats]
    avg_velocities = [s['avg_velocity'] for s in all_stats]
    
    print(f"平均关键比例: {np.mean(critical_ratios):.1%} ± {np.std(critical_ratios):.1%}")
    print(f"平均段数: {np.mean(num_segments):.1f} ± {np.std(num_segments):.1f}")
    print(f"平均速度: {np.mean(avg_velocities):.4f} ± {np.std(avg_velocities):.4f}")
    
    # 可视化分布
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].hist(critical_ratios, bins=10, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_xlabel('Critical Ratio')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Critical Ratios')
    axes[0].axvline(np.mean(critical_ratios), color='red', linestyle='--', label='Mean')
    axes[0].legend()
    
    axes[1].hist(num_segments, bins=10, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_xlabel('Number of Segments')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of Segment Counts')
    axes[1].axvline(np.mean(num_segments), color='red', linestyle='--', label='Mean')
    axes[1].legend()
    
    axes[2].hist(avg_velocities, bins=10, alpha=0.7, color='orange', edgecolor='black')
    axes[2].set_xlabel('Average Velocity')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Distribution of Average Velocities')
    axes[2].axvline(np.mean(avg_velocities), color='red', linestyle='--', label='Mean')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('results/statistics_distribution.png', dpi=150)
    plt.show()
    
    return all_stats


def test_different_thresholds(dataset, trajectory_index=0):
    """
    在同一轨迹上测试不同阈值的效果
    
    Args:
        dataset: HDF5数据集
        trajectory_index: 要测试的轨迹索引
    """
    # 获取轨迹
    sample = dataset.get_item(index=trajectory_index, state_only=True)
    trajectory = sample['action']
    
    print(f"\n在轨迹 {trajectory_index} 上测试不同阈值")
    print(f"轨迹形状: {trajectory.shape}")
    
    # 测试的阈值
    thresholds = [0.005, 0.01, 0.02, 0.05]
    
    fig, axes = plt.subplots(len(thresholds) + 1, 1, figsize=(14, 12))
    
    # 先计算速度
    annotator_base = SimpleCriticalTimestepAnnotator(velocity_threshold=0.01)
    _, _, velocity = annotator_base.annotate(trajectory)
    
    # 显示速度曲线
    axes[0].plot(velocity, 'b-', linewidth=2)
    axes[0].set_ylabel('Velocity')
    axes[0].set_title(f'Trajectory {trajectory_index}: Velocity Profile')
    axes[0].grid(True, alpha=0.3)
    
    # 测试每个阈值
    results = []
    for idx, threshold in enumerate(thresholds):
        annotator = SimpleCriticalTimestepAnnotator(
            velocity_threshold=threshold,
            window_size=5,
            expand_steps=2,
            smooth=True
        )
        
        critical_mask, confidence, velocity = annotator.annotate(trajectory)
        stats = annotator.get_statistics(critical_mask, velocity)
        results.append((threshold, stats))
        
        ax = axes[idx + 1]
        
        # 绘制速度曲线
        ax.plot(velocity, 'b-', alpha=0.5, label='Velocity')
        
        # 绘制阈值线
        ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, 
                  label=f'Threshold={threshold}')
        
        # 标记关键时间段
        for start, end in stats['segments']:
            ax.axvspan(start, end, alpha=0.3, color='red')
        
        # 添加统计信息
        info_text = (f"Critical: {stats['critical_steps']}/{stats['total_steps']} "
                    f"({stats['critical_ratio']:.1%}), Segments: {stats['num_segments']}")
        ax.set_title(f'Threshold = {threshold:.3f} | {info_text}')
        ax.set_ylabel('Velocity')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time Step')
    plt.tight_layout()
    plt.savefig(f'results/threshold_comparison_traj_{trajectory_index}.png', dpi=150)
    plt.show()
    
    # 打印比较结果
    print("\n阈值比较结果:")
    print("-" * 60)
    print(f"{'Threshold':<12} {'Critical%':<12} {'Segments':<10} {'Avg Seg Len':<12}")
    print("-" * 60)
    for threshold, stats in results:
        avg_seg_len = stats['avg_segment_length'] if stats['num_segments'] > 0 else 0
        print(f"{threshold:<12.3f} {stats['critical_ratio']*100:<12.1f} "
              f"{stats['num_segments']:<10} {avg_seg_len:<12.1f}")
    
    return results


def find_optimal_threshold(dataset, num_samples=10):
    """
    通过多个轨迹找出最佳阈值
    
    Args:
        dataset: HDF5数据集
        num_samples: 用于评估的轨迹数量
    """
    print(f"\n使用 {num_samples} 个轨迹寻找最佳阈值...")
    
    # 测试的阈值范围
    thresholds = np.linspace(0.001, 0.05, 20)
    
    # 收集每个阈值的统计
    threshold_stats = {t: {'ratios': [], 'segments': []} for t in thresholds}
    
    num_samples = min(num_samples, len(dataset))
    
    for i in range(num_samples):
        sample = dataset.get_item(index=i, state_only=True)
        trajectory = sample['action']
        
        for threshold in thresholds:
            annotator = SimpleCriticalTimestepAnnotator(
                velocity_threshold=threshold,
                window_size=5,
                expand_steps=2,
                smooth=True
            )
            
            critical_mask, _, velocity = annotator.annotate(trajectory)
            stats = annotator.get_statistics(critical_mask, velocity)
            
            threshold_stats[threshold]['ratios'].append(stats['critical_ratio'])
            threshold_stats[threshold]['segments'].append(stats['num_segments'])
    
    # 计算平均值
    avg_ratios = []
    avg_segments = []
    std_segments = []
    
    for threshold in thresholds:
        avg_ratios.append(np.mean(threshold_stats[threshold]['ratios']))
        avg_segments.append(np.mean(threshold_stats[threshold]['segments']))
        std_segments.append(np.std(threshold_stats[threshold]['segments']))
    
    # 可视化
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    axes[0].plot(thresholds, avg_ratios, 'b-', linewidth=2)
    axes[0].fill_between(thresholds, 0, avg_ratios, alpha=0.3)
    axes[0].set_xlabel('Velocity Threshold')
    axes[0].set_ylabel('Average Critical Ratio')
    axes[0].set_title('Critical Ratio vs Threshold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0.3, color='g', linestyle='--', alpha=0.5, label='Target: 30%')
    axes[0].legend()
    
    axes[1].errorbar(thresholds, avg_segments, yerr=std_segments, 
                     fmt='g-', linewidth=2, capsize=3)
    axes[1].set_xlabel('Velocity Threshold')
    axes[1].set_ylabel('Average Number of Segments')
    axes[1].set_title('Number of Segments vs Threshold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=3, color='r', linestyle='--', alpha=0.5, label='Target: 3 segments')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('results/optimal_threshold_search.png', dpi=150)
    plt.show()
    
    # 找出最佳阈值（目标：约30%关键时间，3-4个段）
    target_ratio = 0.3
    target_segments = 3
    
    # 计算综合得分
    scores = []
    for i, t in enumerate(thresholds):
        ratio_error = abs(avg_ratios[i] - target_ratio)
        segment_error = abs(avg_segments[i] - target_segments)
        score = ratio_error + 0.1 * segment_error  # 加权组合
        scores.append(score)
    
    best_idx = np.argmin(scores)
    best_threshold = thresholds[best_idx]
    
    print(f"\n推荐的最佳阈值: {best_threshold:.4f}")
    print(f"  - 平均关键比例: {avg_ratios[best_idx]:.1%}")
    print(f"  - 平均段数: {avg_segments[best_idx]:.1f}")
    
    return best_threshold


def main():
    """主测试函数"""
    # 配置文件路径
    config_path = "/home/deng_xiang/qian_daichao/RoboTwin/policy/RDT_flare/model_config/multitask.yml"  # 修改为您的配置文件路径
    
    # 加载数据集
    print("加载HDF5数据集...")
    dataset = HDF5VLADataset(config_path)
    print(f"数据集大小: {len(dataset)} 个轨迹")
    
    # 创建结果目录
    os.makedirs("results", exist_ok=True)
    
    # 1. 测试单个轨迹（详细分析）
    print("\n" + "="*60)
    print("1. 单轨迹详细分析")
    print("="*60)
    annotator = SimpleCriticalTimestepAnnotator(
        velocity_threshold=0.01,
        window_size=5,
        expand_steps=3,
        smooth=True
    )
    test_single_trajectory(dataset, index=0, annotator=annotator)
    
    # 2. 测试不同阈值
    print("\n" + "="*60)
    print("2. 不同阈值效果对比")
    print("="*60)
    test_different_thresholds(dataset, trajectory_index=0)
    
    # 3. 多轨迹统计
    print("\n" + "="*60)
    print("3. 多轨迹统计分析")
    print("="*60)
    test_multiple_trajectories(dataset, num_samples=min(20, len(dataset)))
    
    # 4. 寻找最佳阈值
    print("\n" + "="*60)
    print("4. 最佳阈值搜索")
    print("="*60)
    best_threshold = find_optimal_threshold(dataset, num_samples=min(10, len(dataset)))
    
    # 5. 使用最佳阈值重新测试
    print("\n" + "="*60)
    print("5. 使用最佳阈值测试")
    print("="*60)
    best_annotator = SimpleCriticalTimestepAnnotator(
        velocity_threshold=best_threshold,
        window_size=5,
        expand_steps=3,
        smooth=True
    )
    print(f"使用最佳阈值 {best_threshold:.4f} 进行测试:")
    test_single_trajectory(dataset, index=1, annotator=best_annotator, 
                          save_dir="results/best_threshold")
    
    print("\n" + "="*60)
    print("测试完成！所有结果已保存到 results/ 目录")
    print("="*60)


if __name__ == "__main__":
    main()