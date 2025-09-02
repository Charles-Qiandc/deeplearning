import os
import numpy as np
import h5py
import fnmatch
from typing import Tuple, Dict, List, Optional
from scipy.signal import savgol_filter
from scipy.ndimage import binary_dilation
from enum import IntEnum


class TaskType(IntEnum):
    """任务类型枚举"""
    GRASP = 1  # 抓取类任务
    CLICK = 2  # 点击类任务


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


class TaskDrivenCriticalTimestepAnnotator:
    """
    任务驱动的关键时间段标注器
    
    🎯 核心创新：
    1. 基于任务类型的不同标注逻辑
    2. 夹爪状态作为关键时间节点
    3. 双臂独立标注但联合输出
    
    标注策略：
    - 抓取类：减速对准 → 夹爪闭合（关键时间段）
    - 点击类：夹爪闭合 → 减速对准（关键时间段）
    """
    
    def __init__(self, 
                 task_type: TaskType = TaskType.GRASP,
                 relative_low_speed_ratio: float = 0.15,
                 min_deceleration_threshold: float = -0.0008,
                 gripper_close_delta_threshold: float = 0.01,  # 🔧 夹爪闭合变化阈值
                 smooth: bool = True,
                 verbose: bool = False):
        """
        初始化任务驱动的关键时间段标注器
        
        Args:
            task_type: 任务类型（GRASP=1, CLICK=2）
            relative_low_speed_ratio: 相对低速比例（默认15%）
            min_deceleration_threshold: 最小减速度阈值（默认-0.0008）
            gripper_close_delta_threshold: 夹爪闭合变化阈值（默认0.01，检测夹爪开始闭合）
            smooth: 是否平滑速度曲线
            verbose: 是否打印详细信息
        """
        self.task_type = task_type
        self.relative_low_speed_ratio = relative_low_speed_ratio
        self.min_deceleration_threshold = min_deceleration_threshold
        self.gripper_close_delta_threshold = gripper_close_delta_threshold
        self.smooth = smooth
        self.verbose = verbose
        
        # 初始化正运动学计算器
        self.fk_calculator = AgilexForwardKinematics()
        
        # 任务类型验证
        if task_type not in [TaskType.GRASP, TaskType.CLICK]:
            raise ValueError(f"不支持的任务类型: {task_type}")
        
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
    
    def detect_gripper_events(self, gripper_trajectory: np.ndarray, arm_name: str) -> List[int]:
        """
        检测夹爪开始闭合事件（基于变化量，不是绝对值）
        
        Args:
            gripper_trajectory: (T,) 夹爪开度轨迹
            arm_name: 机械臂名称（用于日志）
            
        Returns:
            gripper_close_points: 夹爪开始闭合的时间点列表
        """
        gripper_close_points = []
    
        # 🔧 新增：检测静态臂
        gripper_range = gripper_trajectory.max() - gripper_trajectory.min()
        gripper_std = gripper_trajectory.std()
        
        # 如果夹爪基本不动，认为是静态臂，跳过检测
        if gripper_range < 0.01 and gripper_std < 0.01:
            if self.verbose:
                print(f"    ℹ️  {arm_name}臂检测为静态臂（范围:{gripper_range:.4f}, 标准差:{gripper_std:.4f}），跳过夹爪检测")
            return []
        
        # 计算夹爪开度的变化率（负值表示闭合）
        gripper_delta = np.diff(gripper_trajectory, prepend=gripper_trajectory[0])
        
        # 🔧 修正：只检测真正显著的闭合动作
        for t in range(1, len(gripper_delta)):
            # 条件1：变化为负且超过阈值（闭合动作）
            is_closing = gripper_delta[t] < self.gripper_close_delta_threshold
            
            # 条件2：变化量足够大，排除数值误差 (1%的变化才认为是真实动作)
            is_significant = abs(gripper_delta[t]) > 0.01
            
            # 条件3：确保不是从0开始的误判（防止初始值问题）
            is_valid_timing = t > 0
            
            if is_closing and is_significant and is_valid_timing:
                gripper_close_points.append(t)
                if self.verbose:
                    print(f"    🤏 {arm_name}臂夹爪开始闭合: 步骤 {t}, 开度={gripper_trajectory[t]:.4f}, 变化量={gripper_delta[t]:.4f}")
        
        # 🔧 改进去重：如果连续多个时间点都检测到闭合，只保留第一个
        if len(gripper_close_points) > 1:
            filtered_points = [gripper_close_points[0]]
            for point in gripper_close_points[1:]:
                # 至少间隔3步才认为是新的闭合动作
                if point - filtered_points[-1] > 3:
                    filtered_points.append(point)
            
            if len(filtered_points) < len(gripper_close_points) and self.verbose:
                print(f"    🔄 {arm_name}臂去重后夹爪闭合点: {filtered_points}")
                
            gripper_close_points = filtered_points
        
        return gripper_close_points
    
    def detect_deceleration_low_speed_points(self, velocity: np.ndarray, 
                                           acceleration: np.ndarray,
                                           low_speed_threshold: float, 
                                           arm_name: str) -> List[int]:
        """
        检测减速且低速的时间点
        
        Args:
            velocity: 速度轨迹
            acceleration: 加速度轨迹
            low_speed_threshold: 低速阈值
            arm_name: 机械臂名称
            
        Returns:
            decel_low_speed_points: 减速且低速的时间点列表
        """
        decel_low_speed_points = []
        
        for t in range(len(velocity)):
            is_low_speed = velocity[t] < low_speed_threshold
            is_decelerating = acceleration[t] < self.min_deceleration_threshold
            
            if is_low_speed and is_decelerating:
                decel_low_speed_points.append(t)
                if self.verbose:
                    print(f"    🎯 {arm_name}臂减速低速点: 步骤 {t}, 速度={velocity[t]:.6f}, 加速度={acceleration[t]:.6f}")
        
        return decel_low_speed_points
    
    def find_critical_segment_grasp_mode(self, gripper_points: List[int], 
                                       decel_points: List[int],
                                       arm_name: str) -> Optional[Tuple[int, int]]:
        """
        抓取模式：简化检测流程
        🔧 1. 找到第一个减速点
        🔧 2. 在减速点之后找到第一个夹爪闭合点
        🔧 3. 两点之间即为关键时间段
        
        Args:
            gripper_points: 夹爪闭合时间点
            decel_points: 减速低速时间点
            arm_name: 机械臂名称
            
        Returns:
            critical_segment: (start, end) 或 None
        """
        if not gripper_points or not decel_points:
            if self.verbose:
                print(f"    ❌ {arm_name}臂抓取模式：缺少关键点（夹爪点:{len(gripper_points)}, 减速点:{len(decel_points)}）")
            return None
        
        # 🔧 步骤1：找到第一个减速点
        first_decel_point = min(decel_points)
        if self.verbose:
            print(f"    🎯 {arm_name}臂第一个减速点: 步骤 {first_decel_point}")
        
        # 🔧 步骤2：在减速点之后找到第一个夹爪闭合点
        following_gripper_points = [gp for gp in gripper_points if gp > first_decel_point]
        
        if not following_gripper_points:
            if self.verbose:
                print(f"    ❌ {arm_name}臂：减速点({first_decel_point})之后无夹爪闭合点")
            return None
        
        first_gripper_point = min(following_gripper_points)
        if self.verbose:
            print(f"    🤏 {arm_name}臂减速后第一个夹爪闭合点: 步骤 {first_gripper_point}")
        
        # 🔧 步骤3：两点之间即为关键时间段
        start_point = first_decel_point
        end_point = first_gripper_point
        
        if self.verbose:
            print(f"    ✅ {arm_name}臂抓取关键段: 步骤 {start_point}-{end_point} (长度{end_point-start_point+1})")
        
        return (start_point, end_point)
    
    def find_critical_segment_click_mode(self, gripper_points: List[int], 
                                       decel_points: List[int],
                                       arm_name: str) -> Optional[Tuple[int, int]]:
        """
        点击模式：简化检测流程
        🔧 1. 找到第一个夹爪闭合点
        🔧 2. 在夹爪闭合点之后找到第一个减速点
        🔧 3. 两点之间即为关键时间段
        
        Args:
            gripper_points: 夹爪闭合时间点
            decel_points: 减速低速时间点
            arm_name: 机械臂名称
            
        Returns:
            critical_segment: (start, end) 或 None
        """
        if not gripper_points or not decel_points:
            if self.verbose:
                print(f"    ❌ {arm_name}臂点击模式：缺少关键点（夹爪点:{len(gripper_points)}, 减速点:{len(decel_points)}）")
            return None
        
        # 🔧 步骤1：找到第一个夹爪闭合点
        first_gripper_point = min(gripper_points)
        if self.verbose:
            print(f"    🤏 {arm_name}臂第一个夹爪闭合点: 步骤 {first_gripper_point}")
        
        # 🔧 步骤2：在夹爪闭合点之后找到第一个减速点
        following_decel_points = [dp for dp in decel_points if dp > first_gripper_point]
        
        if not following_decel_points:
            if self.verbose:
                print(f"    ❌ {arm_name}臂：夹爪闭合点({first_gripper_point})之后无减速点")
            return None
        
        first_decel_point = min(following_decel_points)
        if self.verbose:
            print(f"    🎯 {arm_name}臂夹爪闭合后第一个减速点: 步骤 {first_decel_point}")
        
        # 🔧 步骤3：两点之间即为关键时间段
        start_point = first_gripper_point
        end_point = first_decel_point
        
        if self.verbose:
            print(f"    ✅ {arm_name}臂点击关键段: 步骤 {start_point}-{end_point} (长度{end_point-start_point+1})")
        
        return (start_point, end_point)
    
    def annotate_single_arm(self, ee_positions: np.ndarray, 
                          gripper_trajectory: np.ndarray,
                          arm_name: str) -> Optional[Tuple[int, int]]:
        """
        单臂关键时间段标注
        
        Args:
            ee_positions: (T, 3) 末端执行器位置轨迹
            gripper_trajectory: (T,) 夹爪开度轨迹
            arm_name: 机械臂名称（'左' 或 '右'）
            
        Returns:
            critical_segment: (start, end) 或 None
        """
        if self.verbose:
            print(f"  🔍 标注{arm_name}臂（任务类型: {'抓取' if self.task_type == TaskType.GRASP else '点击'}）")
        
        # 1. 计算运动学特征
        velocity = self.compute_velocity(ee_positions)
        acceleration = self.compute_acceleration(velocity)
        
        # 2. 计算阈值
        max_velocity = np.max(velocity)
        low_speed_threshold = max_velocity * self.relative_low_speed_ratio
        
        if self.verbose:
            print(f"    速度统计: 平均={np.mean(velocity):.6f}, 最大={max_velocity:.6f}")
            print(f"    低速阈值: {low_speed_threshold:.6f} (最大速度的{self.relative_low_speed_ratio:.1%})")
        
        # 3. 检测关键事件点
        gripper_points = self.detect_gripper_events(gripper_trajectory, arm_name)
        decel_points = self.detect_deceleration_low_speed_points(
            velocity, acceleration, low_speed_threshold, arm_name)
        
        # 4. 根据任务类型找到关键时间段
        if self.task_type == TaskType.GRASP:
            critical_segment = self.find_critical_segment_grasp_mode(
                gripper_points, decel_points, arm_name)
        else:  # TaskType.CLICK
            critical_segment = self.find_critical_segment_click_mode(
                gripper_points, decel_points, arm_name)
        
        return critical_segment
    
    def annotate(self, qpos_trajectory: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        主标注函数：基于任务类型的智能关键时间段标注
        
        Args:
            qpos_trajectory: (T, 14) 关节角度轨迹
            
        Returns:
            critical_labels: (T,) 关键时间段标签 (0/1)
            analysis_info: 分析信息字典
        """
        task_name = "抓取类" if self.task_type == TaskType.GRASP else "点击类"
        
        if self.verbose:
            print(f"🎯 开始任务驱动标注（{task_name}）")
            print("=" * 50)
        
        # 1. 计算末端执行器位置
        left_ee_pos, right_ee_pos = self.fk_calculator.compute_end_effector_positions(qpos_trajectory)
        
        # 2. 提取夹爪轨迹
        left_gripper = qpos_trajectory[:, 6]   # 左臂夹爪（第7列）
        right_gripper = qpos_trajectory[:, 13]  # 右臂夹爪（第14列）
        
        # 3. 双臂独立标注
        left_segment = self.annotate_single_arm(left_ee_pos, left_gripper, "左")
        right_segment = self.annotate_single_arm(right_ee_pos, right_gripper, "右")
        
        # 4. 生成标注标签
        T = len(qpos_trajectory)
        critical_mask = np.zeros(T, dtype=bool)
        
        segments = []
        
        # 左臂时间段
        if left_segment is not None:
            start, end = left_segment
            critical_mask[start:end+1] = True
            segments.append((start, end, 'left'))
            if self.verbose:
                print(f"  ✅ 左臂关键时间段: 步骤 {start}-{end} (长度{end-start+1})")
        
        # 右臂时间段
        if right_segment is not None:
            start, end = right_segment
            critical_mask[start:end+1] = True
            segments.append((start, end, 'right'))
            if self.verbose:
                print(f"  ✅ 右臂关键时间段: 步骤 {start}-{end} (长度{end-start+1})")
        
        # 5. 转换为标签
        critical_labels = critical_mask.astype(int)
        critical_count = np.sum(critical_labels)
        
        # 6. 生成分析信息
        analysis_info = {
            'task_type': self.task_type,
            'task_name': task_name,
            'left_segment': left_segment,
            'right_segment': right_segment,
            'all_segments': segments,
            'left_ee_positions': left_ee_pos,
            'right_ee_positions': right_ee_pos,
            'left_gripper': left_gripper,
            'right_gripper': right_gripper,
            'statistics': {
                'total_steps': T,
                'critical_steps': int(critical_count),
                'critical_ratio': float(critical_count / T),
                'left_has_segment': left_segment is not None,
                'right_has_segment': right_segment is not None,
                'total_segments': len(segments),
                'config': {
                    'task_type': int(self.task_type),
                    'relative_low_speed_ratio': self.relative_low_speed_ratio,
                    'min_deceleration_threshold': self.min_deceleration_threshold,
                    'gripper_close_delta_threshold': self.gripper_close_delta_threshold,
                }
            }
        }
        
        
        if self.verbose:
            print(f"\n📊 标注结果:")
            print(f"  任务类型: {task_name}")
            print(f"  总步数: {T}")
            print(f"  关键步数: {critical_count}")
            print(f"  关键比例: {critical_count/T:.3f}")
            print(f"  左臂段: {'有' if left_segment else '无'}")
            print(f"  右臂段: {'有' if right_segment else '无'}")
            print(f"  总时间段数: {len(segments)}")
            
            if segments:
                print(f"  详细段落:")
                for start, end, arm in segments:
                    duration = end - start + 1
                    print(f"    {arm}臂: 步骤 {start}-{end} (持续{duration}步)")
        
        return critical_labels, analysis_info


def create_task_annotator(task_type: TaskType, verbose: bool = False):
    """创建任务驱动标注器的工厂函数"""
    return TaskDrivenCriticalTimestepAnnotator(
        task_type=task_type,
        relative_low_speed_ratio=0.15,
        min_deceleration_threshold=-0.0008,
        gripper_close_delta_threshold=-0.01,  # 🔧 夹爪闭合变化阈值
        smooth=True,
        verbose=verbose
    )


def process_hdf5_file_with_task(file_path: str, task_type: TaskType) -> Dict:
    """
    使用任务驱动标注器处理HDF5文件
    
    Args:
        file_path: HDF5文件路径
        task_type: 任务类型（GRASP=1, CLICK=2）
        
    Returns:
        结果字典
    """
    annotator = create_task_annotator(task_type, verbose=True)
    
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
                'task_type': task_type,
                'critical_labels': critical_labels,
                'analysis_info': analysis_info,
                'success': True
            }
            
    except Exception as e:
        return {
            'file_path': file_path,
            'task_type': task_type,
            'error': str(e),
            'success': False
        }


def test_task_annotators():
    """测试不同任务类型的标注器"""
    print("🧪 任务驱动关键时间段标注器测试")
    print("=" * 60)
    
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
                        if len(test_files) >= 2:  # 只需要2个测试文件
                            break
                if len(test_files) >= 2:
                    break
        if len(test_files) >= 2:
            break
    
    if not test_files:
        print("❌ 未找到HDF5测试文件")
        return
    
    # 测试两种任务类型
    test_configs = [
        (TaskType.GRASP, "抓取类任务"),
        (TaskType.CLICK, "点击类任务")
    ]
    
    print(f"\n🎯 任务驱动标注测试结果:")
    print("任务类型 | 文件 | 左臂段 | 右臂段 | 总段数 | 关键比例 | 状态")
    print("-" * 80)
    
    for task_type, task_name in test_configs:
        for i, file_path in enumerate(test_files[:1]):  # 每种任务测试1个文件
            result = process_hdf5_file_with_task(file_path, task_type)
            
            if result['success']:
                stats = result['analysis_info']['statistics']
                left_has = "✓" if stats['left_has_segment'] else "✗"
                right_has = "✓" if stats['right_has_segment'] else "✗"
                total_segments = stats['total_segments']
                critical_ratio = stats['critical_ratio']
                
                file_short = os.path.basename(file_path)[:20] + "..."
                print(f"{task_name:8s} | {file_short:20s} | {left_has:6s} | {right_has:6s} | {total_segments:6d} | {critical_ratio:7.3f} | 成功")
            else:
                file_short = os.path.basename(file_path)[:20] + "..."
                print(f"{task_name:8s} | {file_short:20s} | {'错误':6s} | {'错误':6s} | {'0':6s} | {'0.000':7s} | 失败")
    
    print(f"\n✅ 测试完成!")
    print(f"💡 任务驱动标注说明:")
    print(f"   📊 抓取类任务 (task_type=1):")
    print(f"      - 检测逻辑: 减速对准 → 夹爪闭合")
    print(f"      - 关键时间段: 从减速低速点到夹爪闭合点")
    print(f"      - 适用场景: 精细抓取、物体操纵")
    print(f"   🖱️ 点击类任务 (task_type=2):")
    print(f"      - 检测逻辑: 夹爪闭合 → 减速对准")
    print(f"      - 关键时间段: 从夹爪闭合点到减速低速点")
    print(f"      - 适用场景: 按钮点击、触摸交互")
    print(f"   🤖 双臂策略:")
    print(f"      - 每臂独立检测，最多一个关键时间段")
    print(f"      - 任何一臂处于关键时间段即为关键时间段")
    print(f"      - 支持双臂协调操作的标注")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # 无参数时运行测试
        test_task_annotators()
    else:
        # 命令行参数处理
        import argparse
        parser = argparse.ArgumentParser(description="任务驱动关键时间段标注器")
        parser.add_argument("--file", type=str, help="指定HDF5文件路径")
        parser.add_argument("--task_type", type=int, choices=[1, 2], required=True, 
                          help="任务类型: 1=抓取类, 2=点击类")
        parser.add_argument("--verbose", action="store_true", help="显示详细信息")
        
        args = parser.parse_args()
        
        if args.file:
            # 处理单个文件
            result = process_hdf5_file_with_task(args.file, TaskType(args.task_type))
            
            if result['success']:
                stats = result['analysis_info']['statistics']
                task_name = result['analysis_info']['task_name']
                print(f"✅ {task_name}标注成功")
                print(f"   左臂时间段: {'有' if stats['left_has_segment'] else '无'}")
                print(f"   右臂时间段: {'有' if stats['right_has_segment'] else '无'}")
                print(f"   总时间段数: {stats['total_segments']}")
                print(f"   关键比例: {stats['critical_ratio']:.3f}")
            else:
                print(f"❌ 标注失败: {result.get('error')}")
        else:
            print("❌ 请提供文件路径 --file")


# 🆕 集成到数据集的便捷函数
def create_silent_task_annotator(task_type: TaskType):
    """创建静默的任务标注器（用于训练）"""
    return TaskDrivenCriticalTimestepAnnotator(
        task_type=task_type,
        relative_low_speed_ratio=0.15,
        min_deceleration_threshold=-0.0008,
        gripper_close_delta_threshold=0.01,  # 🔧 夹爪闭合变化阈值  
        smooth=True,
        verbose=False  # 🔧 关闭所有打印信息
    )