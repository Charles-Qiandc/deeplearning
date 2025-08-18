# data/critical_timestep_annotator_simple.py

import numpy as np
from typing import Tuple
from scipy.signal import savgol_filter
from scipy.ndimage import binary_dilation


class SimpleCriticalTimestepAnnotator:
    """
    简化版关键时间段标注器
    仅使用末端速度来判断关键时间段
    """
    
    def __init__(self, 
                 velocity_threshold: float = 0.01,
                 window_size: int = 5,
                 expand_steps: int = 3,
                 smooth: bool = True):
        """
        初始化标注器
        
        Args:
            velocity_threshold: 速度阈值，低于此值视为关键时间段
            window_size: 平滑窗口大小
            expand_steps: 关键段前后扩展步数
            smooth: 是否对速度进行平滑
        """
        self.velocity_threshold = velocity_threshold
        self.window_size = window_size
        self.expand_steps = expand_steps
        self.smooth = smooth
        
    def compute_velocity(self, trajectory: np.ndarray) -> np.ndarray:
        """
        计算轨迹的末端速度
        
        Args:
            trajectory: (T, D) 动作轨迹
            
        Returns:
            velocity: (T,) 速度序列
        """
        # 计算相邻帧之间的差分
        diff = np.diff(trajectory, axis=0, prepend=trajectory[0:1])
        
        # 计算速度（欧氏范数）
        velocity = np.linalg.norm(diff, axis=-1)
        
        # 可选：平滑处理
        if self.smooth and len(velocity) > self.window_size:
            velocity = savgol_filter(
                velocity, 
                window_length=min(self.window_size, len(velocity)),
                polyorder=min(3, self.window_size-1)
            )
            
        return velocity
    
    def annotate(self, trajectory: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        标注轨迹中的关键时间段
        
        Args:
            trajectory: (T, D) 动作轨迹
            
        Returns:
            critical_mask: (T,) 关键时间段布尔掩码
            confidence: (T,) 置信度分数 [0, 1]
            velocity: (T,) 速度序列（用于可视化）
        """
        # 输入验证
        if not isinstance(trajectory, np.ndarray):
            trajectory = np.array(trajectory)
            
        if len(trajectory.shape) != 2:
            raise ValueError(f"期望2D轨迹，得到{len(trajectory.shape)}D")
            
        T, D = trajectory.shape
        
        # 计算速度
        velocity = self.compute_velocity(trajectory)
        
        # 标注低速段为关键时间段
        critical_mask = velocity < self.velocity_threshold
        
        # 置信度：速度越低，置信度越高
        # 使用反比例函数，确保在[0, 1]范围内
        confidence = 1.0 - np.clip(velocity / (self.velocity_threshold * 2), 0, 1)
        
        # 扩展关键区域
        if self.expand_steps > 0:
            structure = np.ones(2 * self.expand_steps + 1)
            critical_mask = binary_dilation(critical_mask, structure=structure)
            
            # 扩展区域的置信度稍低
            expanded_only = critical_mask & (velocity >= self.velocity_threshold)
            confidence[expanded_only] = 0.3
            
        # 确保起始和结束时刻被标注为关键
        if T > 0:
            critical_mask[0] = True
            critical_mask[-1] = True
            confidence[0] = max(confidence[0], 0.8)
            confidence[-1] = max(confidence[-1], 0.8)
            
        return critical_mask.astype(bool), confidence, velocity
    
    def get_statistics(self, critical_mask: np.ndarray, velocity: np.ndarray) -> dict:
        """
        获取标注统计信息
        
        Returns:
            stats: 统计信息字典
        """
        T = len(critical_mask)
        
        # 找出连续的关键段
        segments = []
        in_segment = False
        start = 0
        
        for t, is_critical in enumerate(critical_mask):
            if is_critical and not in_segment:
                start = t
                in_segment = True
            elif not is_critical and in_segment:
                segments.append((start, t))
                in_segment = False
                
        if in_segment:
            segments.append((start, T))
            
        stats = {
            'total_steps': T,
            'critical_steps': int(np.sum(critical_mask)),
            'critical_ratio': float(np.mean(critical_mask)),
            'num_segments': len(segments),
            'segments': segments,
            'avg_velocity': float(np.mean(velocity)),
            'min_velocity': float(np.min(velocity)),
            'max_velocity': float(np.max(velocity)),
            'velocity_threshold': self.velocity_threshold
        }
        
        # 计算每个段的平均长度
        if segments:
            segment_lengths = [end - start for start, end in segments]
            stats['avg_segment_length'] = float(np.mean(segment_lengths))
            stats['max_segment_length'] = max(segment_lengths)
            stats['min_segment_length'] = min(segment_lengths)
        else:
            stats['avg_segment_length'] = 0
            stats['max_segment_length'] = 0
            stats['min_segment_length'] = 0
            
        return stats