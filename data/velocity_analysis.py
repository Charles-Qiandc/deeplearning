import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from typing import List, Dict, Tuple

class VelocityAnalyzer:
    """
    速度计算验证和可视化分析工具
    """
    
    def __init__(self):
        # 从你的代码复制正运动学计算器
        from critical_timestep_annotator import AgilexForwardKinematics
        self.fk_calculator = AgilexForwardKinematics()
    
    def analyze_velocity_calculation(self, qpos_trajectory: np.ndarray) -> Dict:
        """
        详细分析速度计算过程
        """
        print("🔍 检查速度计算方式")
        print("=" * 50)
        
        # 1. 检查输入数据
        T, joints = qpos_trajectory.shape
        print(f"轨迹信息:")
        print(f"  时间步数: {T}")
        print(f"  关节数量: {joints}")
        print(f"  期望格式: 14维 [左臂6关节+夹爪, 右臂6关节+夹爪]")
        
        if joints != 14:
            print(f"  ⚠️ 警告: 关节数量不是14维!")
        
        # 2. 检查关节角度范围
        print(f"\n关节角度范围检查:")
        for i in range(min(14, joints)):
            joint_values = qpos_trajectory[:, i]
            joint_range = np.max(joint_values) - np.min(joint_values)
            joint_name = self.get_joint_name(i)
            print(f"  {joint_name}: 范围={joint_range:.4f}rad ({np.degrees(joint_range):.1f}°)")
        
        # 3. 计算末端位置
        left_ee_pos, right_ee_pos = self.fk_calculator.compute_end_effector_positions(qpos_trajectory)
        
        # 4. 检查末端位置合理性
        print(f"\n末端位置范围检查:")
        left_range = self.calculate_position_range(left_ee_pos)
        right_range = self.calculate_position_range(right_ee_pos)
        
        print(f"  左臂末端位置:")
        print(f"    X: {left_range['x'][0]:.3f} ~ {left_range['x'][1]:.3f} (范围: {left_range['x'][1]-left_range['x'][0]:.3f}m)")
        print(f"    Y: {left_range['y'][0]:.3f} ~ {left_range['y'][1]:.3f} (范围: {left_range['y'][1]-left_range['y'][0]:.3f}m)")
        print(f"    Z: {left_range['z'][0]:.3f} ~ {left_range['z'][1]:.3f} (范围: {left_range['z'][1]-left_range['z'][0]:.3f}m)")
        
        print(f"  右臂末端位置:")
        print(f"    X: {right_range['x'][0]:.3f} ~ {right_range['x'][1]:.3f} (范围: {right_range['x'][1]-right_range['x'][0]:.3f}m)")
        print(f"    Y: {right_range['y'][0]:.3f} ~ {right_range['y'][1]:.3f} (范围: {right_range['y'][1]-right_range['y'][0]:.3f}m)")
        print(f"    Z: {right_range['z'][0]:.3f} ~ {right_range['z'][1]:.3f} (范围: {right_range['z'][1]-right_range['z'][0]:.3f}m)")
        
        # 5. 计算速度 - 使用你当前的方法
        left_velocity_current = self.compute_velocity_current_method(left_ee_pos)
        right_velocity_current = self.compute_velocity_current_method(right_ee_pos)
        
        # 6. 尝试其他速度计算方法进行对比
        left_velocity_simple = self.compute_velocity_simple(left_ee_pos)
        right_velocity_simple = self.compute_velocity_simple(right_ee_pos)
        
        # 7. 检查采样频率影响
        sampling_info = self.analyze_sampling_frequency(qpos_trajectory)
        
        print(f"\n速度计算结果对比:")
        print(f"  左臂 (当前方法): 平均={np.mean(left_velocity_current):.6f}, 最大={np.max(left_velocity_current):.6f}")
        print(f"  左臂 (简单方法): 平均={np.mean(left_velocity_simple):.6f}, 最大={np.max(left_velocity_simple):.6f}")
        print(f"  右臂 (当前方法): 平均={np.mean(right_velocity_current):.6f}, 最大={np.max(right_velocity_current):.6f}")
        print(f"  右臂 (简单方法): 平均={np.mean(right_velocity_simple):.6f}, 最大={np.max(right_velocity_simple):.6f}")
        
        return {
            'trajectory_info': {'steps': T, 'joints': joints},
            'joint_ranges': [np.max(qpos_trajectory[:, i]) - np.min(qpos_trajectory[:, i]) for i in range(min(14, joints))],
            'position_ranges': {'left': left_range, 'right': right_range},
            'velocities': {
                'left_current': left_velocity_current,
                'right_current': right_velocity_current,
                'left_simple': left_velocity_simple,
                'right_simple': right_velocity_simple,
            },
            'positions': {'left': left_ee_pos, 'right': right_ee_pos},
            'sampling_info': sampling_info
        }
    
    def get_joint_name(self, joint_idx: int) -> str:
        """获取关节名称"""
        joint_names = [
            "左臂joint1", "左臂joint2", "左臂joint3", "左臂joint4", "左臂joint5", "左臂joint6", "左夹爪",
            "右臂joint1", "右臂joint2", "右臂joint3", "右臂joint4", "右臂joint5", "右臂joint6", "右夹爪"
        ]
        if joint_idx < len(joint_names):
            return joint_names[joint_idx]
        return f"joint_{joint_idx}"
    
    def calculate_position_range(self, positions: np.ndarray) -> Dict:
        """计算位置范围"""
        return {
            'x': [np.min(positions[:, 0]), np.max(positions[:, 0])],
            'y': [np.min(positions[:, 1]), np.max(positions[:, 1])],
            'z': [np.min(positions[:, 2]), np.max(positions[:, 2])],
        }
    
    def compute_velocity_current_method(self, trajectory: np.ndarray) -> np.ndarray:
        """你当前使用的速度计算方法"""
        from scipy.signal import savgol_filter
        
        # 相邻帧差分
        diff = np.diff(trajectory, axis=0, prepend=trajectory[0:1])
        velocity = np.linalg.norm(diff, axis=-1)
        
        # 平滑处理
        if len(velocity) > 5:
            try:
                window_length = min(5, len(velocity) // 2 * 2 + 1)
                velocity = savgol_filter(velocity, window_length, 3)
            except:
                # 降级到简单平滑
                velocity = np.convolve(velocity, np.ones(3)/3, mode='same')
        
        return velocity
    
    def compute_velocity_simple(self, trajectory: np.ndarray) -> np.ndarray:
        """简单的速度计算方法（无平滑）"""
        diff = np.diff(trajectory, axis=0, prepend=trajectory[0:1])
        velocity = np.linalg.norm(diff, axis=-1)
        return velocity
    
    def analyze_sampling_frequency(self, qpos_trajectory: np.ndarray) -> Dict:
        """分析采样频率"""
        # 检查是否有时间信息
        T = len(qpos_trajectory)
        
        # 估算采样频率（基于典型机器人控制频率）
        # 假设轨迹时长在10-30秒之间
        estimated_duration_range = [5, 30]  # 秒
        estimated_freq_range = [T/estimated_duration_range[1], T/estimated_duration_range[0]]
        
        print(f"\n采样频率估算:")
        print(f"  轨迹步数: {T}")
        print(f"  假设时长: {estimated_duration_range[0]}-{estimated_duration_range[1]}秒")
        print(f"  估算频率: {estimated_freq_range[0]:.1f}-{estimated_freq_range[1]:.1f} Hz")
        print(f"  典型机器人控制频率: 10-100 Hz")
        
        return {
            'steps': T,
            'estimated_frequency_range': estimated_freq_range,
            'typical_robot_frequency': [10, 100]
        }
    
    def visualize_multiple_trajectories(self, file_paths: List[str], max_files: int = 5):
        """可视化多个轨迹的速度曲线对比"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        successful_analyses = []
        
        for i, file_path in enumerate(file_paths[:max_files]):
            try:
                with h5py.File(file_path, 'r') as f:
                    qpos = f['observations']['qpos'][:]
                
                analysis = self.analyze_velocity_calculation(qpos)
                successful_analyses.append({
                    'file': os.path.basename(file_path),
                    'analysis': analysis
                })
                
                # 绘制速度曲线
                velocities = analysis['velocities']
                time_steps = range(len(velocities['left_current']))
                color = colors[i % len(colors)]
                
                # 左臂速度
                axes[0, 0].plot(time_steps, velocities['left_current'], 
                               color=color, alpha=0.7, label=f'Episode {i+1}')
                
                # 右臂速度
                axes[0, 1].plot(time_steps, velocities['right_current'], 
                               color=color, alpha=0.7, label=f'Episode {i+1}')
                
                # 左臂位置轨迹
                positions = analysis['positions']
                axes[1, 0].plot(positions['left'][:, 0], positions['left'][:, 1], 
                               color=color, alpha=0.7, label=f'Episode {i+1}')
                
                # 右臂位置轨迹
                axes[1, 1].plot(positions['right'][:, 0], positions['right'][:, 1], 
                               color=color, alpha=0.7, label=f'Episode {i+1}')
                
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
        
        # 设置图表
        axes[0, 0].set_title('左臂速度曲线对比')
        axes[0, 0].set_xlabel('时间步')
        axes[0, 0].set_ylabel('速度 (m/s)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('右臂速度曲线对比')
        axes[0, 1].set_xlabel('时间步')
        axes[0, 1].set_ylabel('速度 (m/s)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('左臂XY轨迹对比')
        axes[1, 0].set_xlabel('X (m)')
        axes[1, 0].set_ylabel('Y (m)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axis('equal')
        
        axes[1, 1].set_title('右臂XY轨迹对比')
        axes[1, 1].set_xlabel('X (m)')
        axes[1, 1].set_ylabel('Y (m)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axis('equal')
        
        plt.tight_layout()
        plt.savefig('velocity_analysis.png', dpi=150, bbox_inches='tight')
        print(f"\n📊 可视化结果已保存到: velocity_analysis.png")
        plt.show()
        
        return successful_analyses
    
    def detect_potential_issues(self, analyses: List[Dict]):
        """检测潜在的速度计算问题"""
        print(f"\n🔍 潜在问题检测:")
        print("=" * 50)
        
        # 1. 检查速度量级一致性
        all_left_max = [a['analysis']['velocities']['left_current'].max() for a in analyses]
        all_right_max = [a['analysis']['velocities']['right_current'].max() for a in analyses]
        
        left_max_std = np.std(all_left_max)
        right_max_std = np.std(all_right_max)
        
        print(f"速度量级一致性检查:")
        print(f"  左臂最大速度变异系数: {left_max_std/np.mean(all_left_max):.2f}")
        print(f"  右臂最大速度变异系数: {right_max_std/np.mean(all_right_max):.2f}")
        
        if left_max_std/np.mean(all_left_max) > 0.5:
            print(f"  ⚠️ 左臂速度变异性较大，可能存在问题")
        if right_max_std/np.mean(all_right_max) > 0.5:
            print(f"  ⚠️ 右臂速度变异性较大，可能存在问题")
        
        # 2. 检查异常高速
        for i, analysis in enumerate(analyses):
            velocities = analysis['analysis']['velocities']
            left_max = velocities['left_current'].max()
            right_max = velocities['right_current'].max()
            
            if left_max > 0.1:  # 10cm/s 对于精细操作来说算高速
                print(f"  ⚠️ {analysis['file']}: 左臂出现异常高速 {left_max:.4f} m/s")
            if right_max > 0.1:
                print(f"  ⚠️ {analysis['file']}: 右臂出现异常高速 {right_max:.4f} m/s")
        
        # 3. 检查末端位置合理性
        print(f"\n末端位置合理性检查:")
        for i, analysis in enumerate(analyses):
            pos_ranges = analysis['analysis']['position_ranges']
            
            # 检查是否在合理的机器人工作空间内
            left_range = pos_ranges['left']
            right_range = pos_ranges['right']
            
            # 假设机器人工作空间大约在基座周围1米范围内
            for arm, ranges in [('左臂', left_range), ('右臂', right_range)]:
                total_range = (ranges['x'][1] - ranges['x'][0] + 
                             ranges['y'][1] - ranges['y'][0] + 
                             ranges['z'][1] - ranges['z'][0])
                
                if total_range > 2.0:  # 总变化范围超过2米
                    print(f"  ⚠️ {analysis['file']}: {arm}运动范围异常大 ({total_range:.3f}m)")


def quick_velocity_analysis():
    """快速速度分析"""
    print("🔍 机器人速度计算验证与可视化分析")
    print("=" * 60)
    
    # 查找HDF5文件
    test_files = []
    search_paths = [
        "../processed_data/click_bell",
        "processed_data/click_bell",
        "../processed_data", 
        "processed_data",
    ]
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if file.endswith(".hdf5"):
                        test_files.append(os.path.join(root, file))
                        if len(test_files) >= 5:  # 分析5个文件
                            break
                if len(test_files) >= 5:
                    break
        if len(test_files) >= 5:
            break
    
    if not test_files:
        print("❌ 未找到HDF5测试文件")
        return
    
    print(f"找到 {len(test_files)} 个文件，分析前 5 个:")
    for f in test_files[:5]:
        print(f"  - {os.path.basename(f)}")
    
    # 执行分析
    analyzer = VelocityAnalyzer()
    analyses = analyzer.visualize_multiple_trajectories(test_files, max_files=5)
    
    # 检测问题
    analyzer.detect_potential_issues(analyses)
    
    print(f"\n✅ 分析完成!")
    print(f"💡 建议:")
    print(f"   1. 检查生成的 velocity_analysis.png 图像")
    print(f"   2. 关注速度曲线的一致性和合理性")
    print(f"   3. 检查末端位置轨迹是否符合预期")
    print(f"   4. 如发现异常，可能需要检查:")
    print(f"      - DH参数设置")
    print(f"      - 关节角度数据质量")
    print(f"      - 速度计算方法")

if __name__ == "__main__":
    quick_velocity_analysis()