# agilex_precise_critical_annotator.py

import os
import numpy as np
import h5py
import fnmatch
from typing import Tuple, Dict
from scipy.signal import savgol_filter
from scipy.ndimage import binary_dilation


class AgilexForwardKinematics:
    """
    Agilex双臂机器人正运动学计算器
    基于实际URDF参数和14维qpos格式：[左臂6关节+夹爪, 右臂6关节+夹爪]
    """
    
    def __init__(self):
        """基于你的URDF文件配置DH参数"""
        
        # 从URDF分析得到的前臂DH参数 (front-left arm)
        # 基于fl_joint1-6的配置
        self.left_dh_params = np.array([
            # [a,     alpha,    d,      theta_offset]
            [0.025,   0,       0.058,   0],          # fl_joint1 (base)
            [-0.264,  -np.pi,  0.0045,  -np.pi/2],  # fl_joint2 -> fl_joint3
            [0.246,   0,      -0.06,    0],          # fl_joint3 -> fl_joint4  
            [0.068,   0,      -0.085,   -np.pi/2],  # fl_joint4 -> fl_joint5
            [0.031,   -np.pi,  0.085,   0],          # fl_joint5 -> fl_joint6
            [0.085,   0,       0.001,   0],          # fl_joint6 -> 末端
        ])
        
        # 右臂DH参数 (front-right arm)
        # 基于fr_joint1-6的配置，与左臂镜像
        self.right_dh_params = np.array([
            [0.025,   0,       0.058,   0],          # fr_joint1 (base)
            [-0.264,  -np.pi,  0.0045,  -np.pi/2],  # fr_joint2 -> fr_joint3
            [0.246,   0,      -0.06,    0],          # fr_joint3 -> fr_joint4
            [0.068,   0,      -0.085,   -np.pi/2],  # fr_joint4 -> fr_joint5
            [0.031,   -np.pi,  0.085,   0],          # fr_joint5 -> fr_joint6
            [0.085,   0,       0.001,   0],          # fr_joint6 -> 末端
        ])
        
        # 左右臂基座位置偏移（基于URDF中的joint位置）
        self.left_base_offset = np.array([0.2305, 0.297, 0.782])   # fl_base_joint位置
        self.right_base_offset = np.array([0.2315, -0.3063, 0.781]) # fr_base_joint位置
        
    def dh_transform(self, a, alpha, d, theta):
        """标准DH变换矩阵"""
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        
        return np.array([
            [ct,    -st*ca,  st*sa,   a*ct],
            [st,    ct*ca,   -ct*sa,  a*st], 
            [0,     sa,      ca,      d],
            [0,     0,       0,       1]
        ])
    
    def compute_forward_kinematics(self, joint_angles, dh_params, base_offset):
        """计算单臂正运动学"""
        T = np.eye(4)
        T[:3, 3] = base_offset  # 设置基座偏移
        
        for i, (a, alpha, d, theta_offset) in enumerate(dh_params):
            if i < len(joint_angles):
                theta = joint_angles[i] + theta_offset
                T_i = self.dh_transform(a, alpha, d, theta)
                T = T @ T_i
        
        return T[:3, 3]  # 返回位置
    
    def compute_end_effector_positions(self, qpos_trajectory: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        从qpos轨迹计算双臂末端执行器位置
        
        Args:
            qpos_trajectory: (T, 14) 关节角度轨迹
                格式：[左臂6关节, 左夹爪, 右臂6关节, 右夹爪]
                
        Returns:
            left_ee_positions: (T, 3) 左臂末端位置
            right_ee_positions: (T, 3) 右臂末端位置
        """
        T = qpos_trajectory.shape[0]
        left_ee_positions = np.zeros((T, 3))
        right_ee_positions = np.zeros((T, 3))
        
        for t in range(T):
            # 提取关节角度（跳过夹爪）
            left_joints = qpos_trajectory[t, :6]      # 前6个关节
            right_joints = qpos_trajectory[t, 7:13]   # 第8-13个关节
            
            # 计算末端位置
            left_ee_positions[t] = self.compute_forward_kinematics(
                left_joints, self.left_dh_params, self.left_base_offset)
            
            right_ee_positions[t] = self.compute_forward_kinematics(
                right_joints, self.right_dh_params, self.right_base_offset)
                
        return left_ee_positions, right_ee_positions


class AgilexCriticalAnnotator:
    """
    Agilex机器人关键时间段标注器
    基于双臂末端执行器速度的关键时间段检测
    """
    
    def __init__(self, 
                 velocity_threshold: float = 0.01,
                 expand_steps: int = 3,
                 smooth: bool = True):
        """
        Args:
            velocity_threshold: 速度阈值，两臂都低于此值视为关键时间段
            expand_steps: 关键段前后扩展步数
            smooth: 是否平滑速度曲线
        """
        self.velocity_threshold = velocity_threshold
        self.expand_steps = expand_steps
        self.smooth = smooth
        
        # 初始化正运动学计算器
        self.fk_calculator = AgilexForwardKinematics()
        
    def compute_velocity(self, trajectory: np.ndarray) -> np.ndarray:
        """计算轨迹速度（欧氏范数）"""
        # 相邻帧差分
        diff = np.diff(trajectory, axis=0, prepend=trajectory[0:1])
        velocity = np.linalg.norm(diff, axis=-1)
        
        # 平滑处理
        if self.smooth and len(velocity) > 5:
            try:
                window_length = min(5, len(velocity) // 2 * 2 + 1)
                velocity = savgol_filter(velocity, window_length, 3)
            except:
                # 降级到简单平滑
                velocity = np.convolve(velocity, np.ones(3)/3, mode='same')
        
        return velocity
    
    def annotate(self, qpos_trajectory: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        从qpos轨迹标注关键时间段
        
        Args:
            qpos_trajectory: (T, 14) 关节角度轨迹
            
        Returns:
            critical_labels: (T,) 关键时间段标签，1表示关键，0表示非关键
            analysis_info: 分析信息字典
        """
        # 1. 计算末端执行器位置
        left_ee_pos, right_ee_pos = self.fk_calculator.compute_end_effector_positions(qpos_trajectory)
        
        # 2. 计算速度
        left_velocity = self.compute_velocity(left_ee_pos)
        right_velocity = self.compute_velocity(right_ee_pos)
        
        # 3. 双臂都低速时标记为关键时间段
        left_low_speed = left_velocity < self.velocity_threshold
        right_low_speed = right_velocity < self.velocity_threshold
        critical_mask = left_low_speed & right_low_speed
        
        # 4. 扩展关键区域
        if self.expand_steps > 0:
            structure = np.ones(2 * self.expand_steps + 1)
            critical_mask = binary_dilation(critical_mask, structure=structure)
        
        # 5. 强制起始和结束为关键时间段
        critical_mask[0] = True
        critical_mask[-1] = True
        
        # 6. 转换为0/1标签
        critical_labels = critical_mask.astype(int)
        
        # 7. 计算统计信息
        T = len(critical_labels)
        critical_count = np.sum(critical_labels)
        
        analysis_info = {
            'left_velocity': left_velocity,
            'right_velocity': right_velocity,
            'left_ee_positions': left_ee_pos,
            'right_ee_positions': right_ee_pos,
            'statistics': {
                'total_steps': T,
                'critical_steps': int(critical_count),
                'critical_ratio': float(critical_count / T),
                'velocity_threshold': self.velocity_threshold,
                'left_avg_velocity': float(np.mean(left_velocity)),
                'right_avg_velocity': float(np.mean(right_velocity)),
                'left_max_velocity': float(np.max(left_velocity)),
                'right_max_velocity': float(np.max(right_velocity)),
            }
        }
        
        return critical_labels, analysis_info


def process_hdf5_file(file_path: str, velocity_threshold: float = 0.01) -> Dict:
    """
    处理单个HDF5文件
    
    Args:
        file_path: HDF5文件路径
        velocity_threshold: 速度阈值
        
    Returns:
        结果字典，包含critical_labels和分析信息
    """
    annotator = AgilexCriticalAnnotator(velocity_threshold=velocity_threshold)
    
    try:
        with h5py.File(file_path, 'r') as f:
            qpos = f['observations']['qpos'][:]
            
            # 跳过初始静止步骤
            qpos_delta = np.abs(qpos - qpos[0:1])
            moving_indices = np.where(np.any(qpos_delta > 1e-3, axis=1))[0]
            
            if len(moving_indices) > 0:
                start_idx = max(0, moving_indices[0] - 1)
                qpos_active = qpos[start_idx:]
            else:
                qpos_active = qpos
                start_idx = 0
            
            # 执行标注
            critical_labels, analysis_info = annotator.annotate(qpos_active)
            
            # 如果跳过了步骤，调整标签长度
            if start_idx > 0:
                full_labels = np.zeros(len(qpos), dtype=int)
                full_labels[start_idx:] = critical_labels
                critical_labels = full_labels
            
            return {
                'file_path': file_path,
                'critical_labels': critical_labels,
                'analysis_info': analysis_info,
                'success': True
            }
            
    except Exception as e:
        return {
            'file_path': file_path,
            'error': str(e),
            'success': False
        }


def batch_process_dataset(data_dir: str = None, 
                         velocity_threshold: float = 0.01,
                         max_files: int = 10) -> Dict:
    """
    批量处理数据集
    
    Returns:
        包含所有结果的字典
    """
    # 如果没有指定数据目录，自动搜索
    if data_dir is None:
        possible_dirs = [
            "processed_data/adjust_bottle",
            "../processed_data/adjust_bottle", 
            "../../processed_data/adjust_bottle",
            "processed_data",
            "../processed_data",
            "../../processed_data",
        ]
        
        for d in possible_dirs:
            if os.path.exists(d):
                data_dir = d
                print(f"🔍 自动找到数据目录: {data_dir}")
                break
        
        if data_dir is None:
            print("❌ 未找到数据目录，请手动指定 --data_dir")
            return {'results': [], 'successful_count': 0, 'total_count': 0}
    
    # 查找HDF5文件
    hdf5_files = []
    for root, dirs, files in os.walk(data_dir):
        for filename in fnmatch.filter(files, "*.hdf5"):
            hdf5_files.append(os.path.join(root, filename))
            if len(hdf5_files) >= max_files:
                break
        if len(hdf5_files) >= max_files:
            break
    
    print(f"找到 {len(hdf5_files)} 个HDF5文件，处理前 {max_files} 个")
    
    results = []
    for i, file_path in enumerate(hdf5_files[:max_files]):
        print(f"处理 {i+1}/{max_files}: {os.path.basename(file_path)}")
        
        result = process_hdf5_file(file_path, velocity_threshold)
        results.append(result)
        
        if result['success']:
            stats = result['analysis_info']['statistics']
            print(f"  ✅ 关键比例: {stats['critical_ratio']:.3f}")
        else:
            print(f"  ❌ 失败: {result['error']}")
    
    # 计算总体统计
    successful_results = [r for r in results if r['success']]
    if successful_results:
        critical_ratios = [r['analysis_info']['statistics']['critical_ratio'] 
                          for r in successful_results]
        avg_ratio = np.mean(critical_ratios)
        print(f"\n总体统计: 平均关键比例 {avg_ratio:.3f}")
    
    return {
        'results': results,
        'successful_count': len(successful_results),
        'total_count': len(results),
        'config': {
            'velocity_threshold': velocity_threshold,
            'data_dir': data_dir
        }
    }


def create_routing_labels_for_training(batch_results: Dict, save_path: str = "agilex_routing_labels.npy"):
    """
    为训练创建路由标签数据集
    
    Args:
        batch_results: batch_process_dataset的返回结果
        save_path: 保存路径
    """
    routing_dataset = {
        'file_paths': [],
        'critical_labels': [],
        'episode_lengths': [],
        'critical_ratios': [],
        'config': batch_results['config']
    }
    
    for result in batch_results['results']:
        if result['success']:
            routing_dataset['file_paths'].append(result['file_path'])
            routing_dataset['critical_labels'].append(result['critical_labels'])
            routing_dataset['episode_lengths'].append(len(result['critical_labels']))
            routing_dataset['critical_ratios'].append(
                result['analysis_info']['statistics']['critical_ratio']
            )
    
    # 保存
    np.save(save_path, routing_dataset, allow_pickle=True)
    print(f"路由标签数据集已保存到: {save_path}")
    print(f"包含 {len(routing_dataset['file_paths'])} 个episode的标注")
    
    return routing_dataset


def quick_test():
    """快速测试功能"""
    print("🧪 Agilex关键时间段标注器快速测试")
    print("=" * 40)
    
    # 查找测试文件 - 扩展搜索路径
    test_files = []
    search_paths = [
        "processed_data",           # 当前目录
        "../processed_data",        # 上级目录
        "../../processed_data",     # 上上级目录
        "../processed_data/adjust_bottle",  # 具体路径
        "../../processed_data/adjust_bottle",
        ".",                        # 当前目录的所有子目录
    ]
    
    print("🔍 搜索HDF5文件...")
    for search_path in search_paths:
        if os.path.exists(search_path):
            print(f"检查路径: {search_path}")
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if file.endswith(".hdf5"):
                        full_path = os.path.join(root, file)
                        test_files.append(full_path)
                        print(f"  找到: {full_path}")
                        if len(test_files) >= 2:
                            break
                if len(test_files) >= 2:
                    break
        if len(test_files) >= 2:
            break
    
    if not test_files:
        print("❌ 未找到HDF5测试文件")
        print("请确认以下路径中存在.hdf5文件:")
        for path in search_paths:
            abs_path = os.path.abspath(path)
            exists = "✅" if os.path.exists(path) else "❌"
            print(f"  {exists} {abs_path}")
        
        # 提示用户手动指定路径
        print("\n💡 如果文件在其他位置，请使用:")
        print("python critical_timestep_annotator.py --data_dir=/path/to/your/hdf5/files")
        return
    
    print(f"\n📁 使用测试文件: {test_files[0]}")
    
    # 测试不同阈值
    thresholds = [0.005, 0.01, 0.02]
    print("\n阈值测试结果:")
    print("阈值   | 关键比例 | 建议")
    print("-" * 25)
    
    for threshold in thresholds:
        result = process_hdf5_file(test_files[0], threshold)
        if result['success']:
            ratio = result['analysis_info']['statistics']['critical_ratio']
            
            if ratio > 0.8:
                suggestion = "过高"
            elif ratio < 0.2:
                suggestion = "过低"
            else:
                suggestion = "✅合理"
                
            print(f"{threshold:5.3f} | {ratio:7.3f} | {suggestion}")
        else:
            print(f"{threshold:5.3f} | 错误   | {result['error']}")
    
    print("\n✅ 快速测试完成")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # 无参数时运行快速测试
        quick_test()
    else:
        # 命令行参数处理
        import argparse
        parser = argparse.ArgumentParser(description="Agilex关键时间段标注器")
        parser.add_argument("--data_dir", default="processed_data/adjust_bottle")
        parser.add_argument("--velocity_threshold", type=float, default=0.01)
        parser.add_argument("--max_files", type=int, default=10)
        parser.add_argument("--save_labels", default="agilex_routing_labels.npy")
        
        args = parser.parse_args()
        
        # 批量处理
        batch_results = batch_process_dataset(
            args.data_dir, args.velocity_threshold, args.max_files
        )
        
        # 创建训练用的路由标签
        create_routing_labels_for_training(batch_results, args.save_labels)