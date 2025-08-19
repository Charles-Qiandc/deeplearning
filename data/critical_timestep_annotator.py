import os
import numpy as np
import h5py
import fnmatch
from typing import Tuple, Dict, List
from scipy.signal import savgol_filter
from scipy.ndimage import binary_dilation


class AgilexForwardKinematics:
    """
    Agilex双臂机器人正运动学计算器
    基于实际URDF参数和14维qpos格式：[左臂6关节+夹爪, 右臂6关节+夹爪]
    """
    
    def __init__(self):
        """基于URDF文件的精确DH参数配置"""
        
        # 基于URDF文件的精确前臂DH参数 (fl_)
        self.left_dh_params = np.array([
            # [a,        alpha,     d,       theta_offset]
            [0.0,       0,        0.058,   0],           # fl_joint1: xyz="0 0 0.058"
            [0.025013, -np.pi/2,  0.042,   0],           # fl_joint2: xyz="0.025013 0.00060169 0.042"  
            [-0.26396,  0,        0,       np.pi],       # fl_joint3: xyz="-0.26396 0.0044548 0" rpy="-3.1416 0 -0.015928"
            [0.246,     0,       -0.06,    0],           # fl_joint4: xyz="0.246 -0.00025 -0.06"
            [0.06775,   np.pi/2, -0.0855,  0],           # fl_joint5: xyz="0.06775 0.0015 -0.0855"
            [0.03095,   0,        0.0855,  np.pi],       # fl_joint6: xyz="0.03095 0 0.0855" rpy="-3.1416 0 0"
        ])
        
        # 右臂DH参数 (fr_) - 结构与左臂相同
        self.right_dh_params = np.array([
            [0.0,       0,        0.058,   0],           
            [0.025013, -np.pi/2,  0.042,   0],           
            [-0.26396,  0,        0,       np.pi],       
            [0.246,     0,       -0.06,    0],           
            [0.06775,   np.pi/2, -0.0855,  0],           
            [0.03095,   0,        0.0855,  np.pi],       
        ])
        
        # 基座位置偏移（与URDF完全一致）
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


class AgilexDualKeypointAnnotator:
    """
    Agilex机器人双关键点区间标注器
    检测两个关键点（减速+低速），然后将两点之间的所有动作标记为关键时间段
    双臂联合检测：任一臂满足条件就标注，两臂都满足就都标注
    """
    
    def __init__(self, 
                 relative_low_speed_ratio: float = 0.1,      # 10%
                 min_deceleration_threshold: float = -0.0005, # 更宽松
                 min_interval_steps: int = 5,
                 max_interval_steps: int = 100,
                 keypoint_skip_steps: int = 10,              # 新增：检测到关键点后跳过的步数
                 smooth: bool = True):
        """
        Args:
            relative_low_speed_ratio: 相对低速比例，当前速度低于轨迹最大速度的这个比例时认为是低速（默认10%）
            min_deceleration_threshold: 最小减速度阈值，加速度小于此值认为是减速（默认-0.0005，更宽松）
            min_interval_steps: 两个关键点之间的最小间隔步数
            max_interval_steps: 两个关键点之间的最大间隔步数
            keypoint_skip_steps: 检测到关键点后跳过的步数，避免连续关键点（默认10）
            smooth: 是否平滑速度曲线
        """
        self.relative_low_speed_ratio = relative_low_speed_ratio
        self.min_deceleration_threshold = min_deceleration_threshold
        self.min_interval_steps = min_interval_steps
        self.max_interval_steps = max_interval_steps
        self.keypoint_skip_steps = keypoint_skip_steps
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
                velocity = np.maximum(velocity, 0.0)
            except:
                velocity = np.convolve(velocity, np.ones(3)/3, mode='same')
                velocity = np.maximum(velocity, 0.0)
        
        velocity = np.clip(velocity, 0.0, None)
        return velocity
    
    def compute_acceleration(self, velocity: np.ndarray) -> np.ndarray:
        """计算加速度（速度的一阶差分）"""
        acceleration = np.diff(velocity, prepend=velocity[0])
        return acceleration
    
    def detect_keypoints(self, velocity: np.ndarray, acceleration: np.ndarray, 
                        low_speed_threshold: float, arm_name: str) -> List[int]:
        """
        检测关键点：同时满足减速和低速条件的点
        新增跳过逻辑：检测到关键点后跳过指定步数，避免连续关键点
        
        Args:
            velocity: 速度序列
            acceleration: 加速度序列
            low_speed_threshold: 低速阈值
            arm_name: 臂名称（用于日志）
            
        Returns:
            关键点索引列表
        """
        keypoints = []
        i = 0
        
        while i < len(velocity):
            # 检查是否同时满足减速和低速条件
            is_low_speed = velocity[i] < low_speed_threshold
            is_decelerating = acceleration[i] < self.min_deceleration_threshold
            
            if is_low_speed and is_decelerating:
                keypoints.append(i)
                print(f"    🎯 {arm_name}臂关键点: 步骤 {i}, 速度={velocity[i]:.6f}, 加速度={acceleration[i]:.6f}")
                
                # 跳过后续指定步数，避免连续关键点
                skip_steps = self.keypoint_skip_steps
                next_i = i + skip_steps + 1
                if next_i < len(velocity):
                    print(f"    ⏭️ {arm_name}臂跳过 {skip_steps} 步: 从步骤 {i+1} 跳到 {next_i}")
                i = next_i
            else:
                i += 1
        
        return keypoints
    
    def find_valid_intervals(self, keypoints: List[int], arm_name: str) -> List[Tuple[int, int]]:
        """
        从关键点中找到有效的区间 - 使用配对逻辑
        第1个和第2个配对，第3个和第4个配对，以此类推
        
        Args:
            keypoints: 关键点索引列表
            arm_name: 臂名称（用于日志）
            
        Returns:
            有效区间列表 [(start, end), ...]
        """
        if len(keypoints) < 2:
            print(f"    ⚠️ {arm_name}臂关键点不足2个，无法形成区间")
            return []
        
        intervals = []
        
        # 配对逻辑：(0,1), (2,3), (4,5), ...
        for i in range(0, len(keypoints) - 1, 2):
            start_point = keypoints[i]
            end_point = keypoints[i + 1]
            interval_length = end_point - start_point
            
            # 检查区间长度是否在合理范围内
            if self.min_interval_steps <= interval_length <= self.max_interval_steps:
                intervals.append((start_point, end_point))
                print(f"    ✅ {arm_name}臂有效区间: 关键点{i+1}-{i+2}, 步骤 {start_point}-{end_point} (长度{interval_length})")
            elif interval_length < self.min_interval_steps:
                print(f"    ❌ {arm_name}臂区间太短: 关键点{i+1}-{i+2}, 步骤 {start_point}-{end_point} (长度{interval_length} < {self.min_interval_steps})")
            elif interval_length > self.max_interval_steps:
                print(f"    ❌ {arm_name}臂区间太长: 关键点{i+1}-{i+2}, 步骤 {start_point}-{end_point} (长度{interval_length} > {self.max_interval_steps})")
        
        # 如果关键点数量是奇数，最后一个点无法配对
        if len(keypoints) % 2 == 1:
            print(f"    ⚠️ {arm_name}臂最后一个关键点(第{len(keypoints)}个)无法配对，已忽略")
        
        return intervals
    
    def annotate(self, qpos_trajectory: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        基于双关键点区间的关键时间段标注
        双臂联合检测：任一臂满足条件就标注，两臂都满足就都标注
        
        Args:
            qpos_trajectory: (T, 14) 关节角度轨迹
            
        Returns:
            critical_labels: (T,) 关键时间段标签，1表示关键，0表示非关键
            analysis_info: 分析信息字典
        """
        print("🎯 开始双关键点区间标注（双臂联合检测）")
        
        # 1. 计算末端执行器位置
        left_ee_pos, right_ee_pos = self.fk_calculator.compute_end_effector_positions(qpos_trajectory)
        
        # 2. 计算速度和加速度
        left_velocity = self.compute_velocity(left_ee_pos)
        right_velocity = self.compute_velocity(right_ee_pos)
        left_acceleration = self.compute_acceleration(left_velocity)
        right_acceleration = self.compute_acceleration(right_velocity)
        
        # 3. 计算低速阈值
        left_max_velocity = np.max(left_velocity)
        right_max_velocity = np.max(right_velocity)
        left_low_speed_threshold = left_max_velocity * self.relative_low_speed_ratio
        right_low_speed_threshold = right_max_velocity * self.relative_low_speed_ratio
        
        print(f"速度统计:")
        print(f"  左臂: 平均={np.mean(left_velocity):.6f}, 最大={left_max_velocity:.6f}")
        print(f"  右臂: 平均={np.mean(right_velocity):.6f}, 最大={right_max_velocity:.6f}")
        print(f"检测阈值:")
        print(f"  左臂低速阈值: {left_low_speed_threshold:.6f} (最大速度的{self.relative_low_speed_ratio:.1%})")
        print(f"  右臂低速阈值: {right_low_speed_threshold:.6f} (最大速度的{self.relative_low_speed_ratio:.1%})")
        print(f"  减速阈值: {self.min_deceleration_threshold:.6f} (更宽松设置)")
        print(f"  关键点跳过步数: {self.keypoint_skip_steps} (避免连续关键点)")
        
        # 4. 检测关键点
        print(f"\n🔍 检测关键点:")
        left_keypoints = self.detect_keypoints(
            left_velocity, left_acceleration, left_low_speed_threshold, "左"
        )
        right_keypoints = self.detect_keypoints(
            right_velocity, right_acceleration, right_low_speed_threshold, "右"
        )
        
        print(f"  左臂关键点: {len(left_keypoints)}个 - {left_keypoints}")
        print(f"  右臂关键点: {len(right_keypoints)}个 - {right_keypoints}")
        
        # 5. 找到有效区间
        print(f"\n📏 寻找有效区间:")
        left_intervals = self.find_valid_intervals(left_keypoints, "左")
        right_intervals = self.find_valid_intervals(right_keypoints, "右")
        
        # 6. 双臂联合标记关键时间段
        print(f"\n🤝 双臂联合标注:")
        T = len(left_velocity)
        critical_mask = np.zeros(T, dtype=bool)
        
        all_intervals = []
        
        # 标记左臂区间
        if left_intervals:
            print(f"  左臂贡献 {len(left_intervals)} 个区间:")
            for start, end in left_intervals:
                critical_mask[start:end+1] = True
                all_intervals.append((start, end, 'left'))
                print(f"    ✅ 左臂区间: 步骤 {start}-{end} (长度{end-start+1})")
        else:
            print(f"  左臂无有效区间")
        
        # 标记右臂区间
        if right_intervals:
            print(f"  右臂贡献 {len(right_intervals)} 个区间:")
            for start, end in right_intervals:
                critical_mask[start:end+1] = True
                all_intervals.append((start, end, 'right'))
                print(f"    ✅ 右臂区间: 步骤 {start}-{end} (长度{end-start+1})")
        else:
            print(f"  右臂无有效区间")
        
        # 检查是否有重叠区间
        if left_intervals and right_intervals:
            overlaps = []
            for l_start, l_end in left_intervals:
                for r_start, r_end in right_intervals:
                    # 检查区间重叠
                    overlap_start = max(l_start, r_start)
                    overlap_end = min(l_end, r_end)
                    if overlap_start <= overlap_end:
                        overlaps.append((overlap_start, overlap_end))
            
            if overlaps:
                print(f"  🔗 发现双臂重叠区间 {len(overlaps)} 个:")
                for start, end in overlaps:
                    print(f"    双臂重叠: 步骤 {start}-{end} (长度{end-start+1})")
        
        # 7. 结束点总是关键的
        critical_mask[-1] = True
        
        # 8. 转换为0/1标签
        critical_labels = critical_mask.astype(int)
        
        # 9. 计算统计信息
        critical_count = np.sum(critical_labels)
        
        print(f"\n📊 最终标注结果:")
        print(f"  总步数: {T}")
        print(f"  左臂关键点: {len(left_keypoints)}个")
        print(f"  右臂关键点: {len(right_keypoints)}个")
        print(f"  左臂有效区间: {len(left_intervals)}个")
        print(f"  右臂有效区间: {len(right_intervals)}个")
        print(f"  总标注区间: {len(all_intervals)}个")
        print(f"  关键步数: {critical_count}")
        print(f"  关键比例: {critical_count/T:.3f}")
        print(f"  联合检测策略: 任一臂满足条件即标注 ✓")
        
        # 详细区间信息
        if all_intervals:
            print(f"  所有区间详情:")
            for start, end, arm in sorted(all_intervals, key=lambda x: x[0]):
                duration = end - start + 1
                print(f"    {arm}臂: 步骤 {start}-{end} (持续{duration}步)")
        else:
            print("  ⚠️ 未检测到有效的关键区间")
            print("  💡 当前使用宽松参数设置:")
            print(f"    - 低速比例: {self.relative_low_speed_ratio:.1%} (10%)")
            print(f"    - 减速阈值: {self.min_deceleration_threshold} (宽松)")
            print("  💡 可进一步调整:")
            print("    - 继续提高低速比例 (如15%)")
            print("    - 进一步放宽减速阈值 (如-0.0003)")
            print("    - 减小最小区间长度要求")
        
        analysis_info = {
            'left_velocity': left_velocity,
            'right_velocity': right_velocity,
            'left_acceleration': left_acceleration,
            'right_acceleration': right_acceleration,
            'left_ee_positions': left_ee_pos,
            'right_ee_positions': right_ee_pos,
            'left_keypoints': left_keypoints,
            'right_keypoints': right_keypoints,
            'left_intervals': left_intervals,
            'right_intervals': right_intervals,
            'all_intervals': all_intervals,
            'velocity_thresholds': {
                'left_low_speed_threshold': left_low_speed_threshold,
                'right_low_speed_threshold': right_low_speed_threshold,
                'left_max_velocity': left_max_velocity,
                'right_max_velocity': right_max_velocity,
                'min_deceleration_threshold': self.min_deceleration_threshold,
            },
            'statistics': {
                'total_steps': T,
                'critical_steps': int(critical_count),
                'critical_ratio': float(critical_count / T),
                'left_keypoints_count': len(left_keypoints),
                'right_keypoints_count': len(right_keypoints),
                'left_intervals_count': len(left_intervals),
                'right_intervals_count': len(right_intervals),
                'total_intervals_count': len(all_intervals),
                'joint_detection': True,  # 标记使用了联合检测
                'config': {
                    'relative_low_speed_ratio': self.relative_low_speed_ratio,
                    'min_deceleration_threshold': self.min_deceleration_threshold,
                    'min_interval_steps': self.min_interval_steps,
                    'max_interval_steps': self.max_interval_steps,
                    'keypoint_skip_steps': self.keypoint_skip_steps,
                }
            }
        }
        
        return critical_labels, analysis_info


def process_hdf5_file(file_path: str, 
                     relative_low_speed_ratio: float = 0.1,      # 10%
                     min_deceleration_threshold: float = -0.0005, # 更宽松
                     min_interval_steps: int = 5,
                     max_interval_steps: int = 100,
                     keypoint_skip_steps: int = 10) -> Dict:      # 新增参数
    """
    处理单个HDF5文件 - 使用双关键点区间方法（双臂联合检测）
    
    Args:
        file_path: HDF5文件路径
        relative_low_speed_ratio: 相对低速比例（默认10%）
        min_deceleration_threshold: 最小减速度阈值（默认-0.0005，更宽松）
        min_interval_steps: 最小区间长度
        max_interval_steps: 最大区间长度
        keypoint_skip_steps: 关键点跳过步数（默认10）
        
    Returns:
        结果字典，包含critical_labels和分析信息
    """
    annotator = AgilexDualKeypointAnnotator(
        relative_low_speed_ratio=relative_low_speed_ratio,
        min_deceleration_threshold=min_deceleration_threshold,
        min_interval_steps=min_interval_steps,
        max_interval_steps=max_interval_steps,
        keypoint_skip_steps=keypoint_skip_steps
    )
    
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
                         relative_low_speed_ratio: float = 0.1,        # 10%
                         min_deceleration_threshold: float = -0.0005,  # 更宽松
                         min_interval_steps: int = 5,
                         max_interval_steps: int = 100,
                         keypoint_skip_steps: int = 10,                # 新增参数
                         max_files: int = 10) -> Dict:
    """
    批量处理数据集 - 使用双关键点区间方法（双臂联合检测）
    """
    # 如果没有指定数据目录，自动搜索
    if data_dir is None:
        possible_dirs = [
            "processed_data/click_bell",
            "../processed_data/click_bell", 
            "../../processed_data/click_bell",
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
    print(f"使用双关键点区间方法（双臂联合检测）")
    print(f"参数设置: 低速比例={relative_low_speed_ratio:.1%}, 减速阈值={min_deceleration_threshold}, 跳过步数={keypoint_skip_steps}")
    
    results = []
    for i, file_path in enumerate(hdf5_files[:max_files]):
        print(f"\n处理 {i+1}/{max_files}: {os.path.basename(file_path)}")
        print("-" * 50)
        
        result = process_hdf5_file(
            file_path, 
            relative_low_speed_ratio,
            min_deceleration_threshold,
            min_interval_steps,
            max_interval_steps,
            keypoint_skip_steps
        )
        results.append(result)
        
        if result['success']:
            stats = result['analysis_info']['statistics']
            left_kp = stats['left_keypoints_count']
            right_kp = stats['right_keypoints_count']
            total_intervals = stats['total_intervals_count']
            critical_ratio = stats['critical_ratio']
            print(f"✅ 成功 - 左臂点数:{left_kp}, 右臂点数:{right_kp}, 区间数:{total_intervals}, 关键比例:{critical_ratio:.3f}")
        else:
            print(f"❌ 失败: {result['error']}")
    
    # 计算总体统计
    successful_results = [r for r in results if r['success']]
    if successful_results:
        critical_ratios = [r['analysis_info']['statistics']['critical_ratio'] 
                          for r in successful_results]
        left_keypoints_counts = [r['analysis_info']['statistics']['left_keypoints_count']
                               for r in successful_results]
        right_keypoints_counts = [r['analysis_info']['statistics']['right_keypoints_count']
                                for r in successful_results]
        interval_counts = [r['analysis_info']['statistics']['total_intervals_count']
                          for r in successful_results]
        
        avg_ratio = np.mean(critical_ratios)
        avg_left_kp = np.mean(left_keypoints_counts)
        avg_right_kp = np.mean(right_keypoints_counts)
        avg_intervals = np.mean(interval_counts)
        
        print(f"\n📊 总体统计:")
        print(f"  成功处理: {len(successful_results)}/{len(results)}")
        print(f"  平均关键比例: {avg_ratio:.3f}")
        print(f"  平均左臂关键点: {avg_left_kp:.1f}")
        print(f"  平均右臂关键点: {avg_right_kp:.1f}")
        print(f"  平均区间数: {avg_intervals:.1f}")
        print(f"  🤝 使用双臂联合检测策略:")
        print(f"    低速比例: {relative_low_speed_ratio:.1%}")
        print(f"    减速阈值: {min_deceleration_threshold}")
        print(f"    区间长度: {min_interval_steps}-{max_interval_steps}")
        print(f"    跳过步数: {keypoint_skip_steps} (避免连续关键点)")
    
    return {
        'results': results,
        'successful_count': len(successful_results),
        'total_count': len(results),
        'config': {
            'relative_low_speed_ratio': relative_low_speed_ratio,
            'min_deceleration_threshold': min_deceleration_threshold,
            'min_interval_steps': min_interval_steps,
            'max_interval_steps': max_interval_steps,
            'keypoint_skip_steps': keypoint_skip_steps,
            'data_dir': data_dir,
            'method': 'dual_keypoint_joint_detection_with_skip'
        }
    }


def quick_test():
    """快速测试双关键点区间方法（双臂联合检测）"""
    print("🧪 Agilex双关键点区间标注器测试（双臂联合检测）")
    print("=" * 80)
    
    # 查找测试文件
    test_files = []
    search_paths = [
        "processed_data",
        "../processed_data", 
        "../../processed_data",
    ]
    
    print("🔍 搜索HDF5文件...")
    for search_path in search_paths:
        if os.path.exists(search_path):
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if file.endswith(".hdf5"):
                        full_path = os.path.join(root, file)
                        test_files.append(full_path)
                        print(f"  找到: {full_path}")
                        if len(test_files) >= 1:
                            break
                if len(test_files) >= 1:
                    break
        if len(test_files) >= 1:
            break
    
    if not test_files:
        print("❌ 未找到HDF5测试文件")
        return
    
    print(f"\n📁 使用测试文件: {test_files[0]}")
    
    # 测试不同参数配置
    test_configs = [
        {
            'relative_low_speed_ratio': 0.1,      # 10%，当前默认值
            'min_deceleration_threshold': -0.0005, # 宽松加速度阈值
            'min_interval_steps': 5,
            'max_interval_steps': 50,
            'keypoint_skip_steps': 10              # 跳过10步
        },
        {
            'relative_low_speed_ratio': 0.08,     # 8%，稍微严格
            'min_deceleration_threshold': -0.0003, # 更宽松的加速度阈值
            'min_interval_steps': 3,
            'max_interval_steps': 80,
            'keypoint_skip_steps': 8               # 跳过8步
        },
        {
            'relative_low_speed_ratio': 0.15,     # 15%，更宽松
            'min_deceleration_threshold': -0.0008, # 中等宽松
            'min_interval_steps': 10,
            'max_interval_steps': 100,
            'keypoint_skip_steps': 15              # 跳过15步
        }
    ]
    
    print(f"\n🎯 双关键点区间方法测试结果 (双臂联合检测 + 跳过逻辑):")
    print("低速比例 | 减速阈值 | 跳过步数 | 左臂点数 | 右臂点数 | 总区间数 | 关键比例")
    print("-" * 75)
    
    for i, config in enumerate(test_configs):
        result = process_hdf5_file(test_files[0], **config)
        
        if result['success']:
            stats = result['analysis_info']['statistics']
            left_kp = stats['left_keypoints_count']
            right_kp = stats['right_keypoints_count']
            total_intervals = stats['total_intervals_count']
            critical_ratio = stats['critical_ratio']
            
            skip_steps = config['keypoint_skip_steps']
            
            print(f"{config['relative_low_speed_ratio']:7.2f} | {config['min_deceleration_threshold']:9.4f} | "
                  f"{skip_steps:8d} | {left_kp:8d} | {right_kp:8d} | {total_intervals:8d} | {critical_ratio:7.3f}")
        else:
            print(f"配置{i+1}: 错误 - {result.get('error', 'Unknown error')[:30]}")
    
    print(f"\n✅ 测试完成!")
    print(f"💡 双关键点区间检测说明 (已更新 - 新增跳过逻辑):")
    print(f"   📊 当前默认参数:")
    print(f"      - 低速比例: 10% (relative_low_speed_ratio=0.1)")
    print(f"      - 减速阈值: -0.0005 (更宽松，原来是-0.001)")
    print(f"      - 跳过步数: 10 (keypoint_skip_steps=10) 🆕")
    print(f"   🤝 双臂联合检测逻辑:")
    print(f"      - 左臂和右臂分别检测关键点和区间")
    print(f"      - 任一臂有有效区间就标注该区间")
    print(f"      - 两臂都有区间时，两臂的区间都会被标注")
    print(f"      - 重叠区间会被自动合并")
    print(f"   🔗 配对规则:")
    print(f"      - 第1和第2个关键点配对，第3和第4个配对，以此类推")
    print(f"      - 奇数个关键点时，最后一个点会被忽略")
    print(f"   ⏭️ 跳过逻辑 (新增):")
    print(f"      - 检测到关键点后跳过{test_configs[0]['keypoint_skip_steps']}步再继续检测")
    print(f"      - 避免连续的关键点造成过短区间")
    print(f"      - 提高关键点间距，增加有效区间的概率")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # 无参数时运行快速测试
        quick_test()
    else:
        # 命令行参数处理
        import argparse
        parser = argparse.ArgumentParser(description="Agilex双关键点区间标注器（双臂联合检测）")
        parser.add_argument("--file", type=str, help="指定HDF5文件路径")
        parser.add_argument("--data_dir", type=str, help="数据目录路径")
        parser.add_argument("--max_files", type=int, default=10, help="最大处理文件数")
        parser.add_argument("--relative_low_speed_ratio", type=float, default=0.1, help="相对低速比例（默认10%）")
        parser.add_argument("--min_deceleration_threshold", type=float, default=-0.0005, help="最小减速度阈值（默认-0.0005，更宽松）")
        parser.add_argument("--min_interval_steps", type=int, default=5, help="最小区间长度")
        parser.add_argument("--max_interval_steps", type=int, default=100, help="最大区间长度")
        parser.add_argument("--keypoint_skip_steps", type=int, default=10, help="检测到关键点后跳过的步数（默认10）")
        
        args = parser.parse_args()
        
        if args.file:
            # 处理单个文件
            result = process_hdf5_file(
                args.file,
                args.relative_low_speed_ratio,
                args.min_deceleration_threshold,
                args.min_interval_steps,
                args.max_interval_steps,
                args.keypoint_skip_steps
            )
            
            if result['success']:
                stats = result['analysis_info']['statistics']
                print(f"✅ 处理成功")
                print(f"左臂关键点: {stats['left_keypoints_count']}")
                print(f"右臂关键点: {stats['right_keypoints_count']}")
                print(f"有效区间数: {stats['total_intervals_count']}")
                print(f"关键比例: {stats['critical_ratio']:.3f}")
            else:
                print(f"❌ 处理失败: {result.get('error')}")
        elif args.data_dir:
            # 批量处理
            batch_results = batch_process_dataset(
                args.data_dir, 
                args.relative_low_speed_ratio, 
                args.min_deceleration_threshold,
                args.min_interval_steps,
                args.max_interval_steps,
                args.keypoint_skip_steps,
                args.max_files
            )
        else:
            # 运行测试
            quick_test()