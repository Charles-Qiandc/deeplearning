import os
import fnmatch
import json

import h5py
import yaml
import cv2
import numpy as np
import torch

from configs.state_vec import STATE_VEC_IDX_MAPPING

# 🆕 导入关键时间段标注器
try:
    from data.critical_timestep_annotator import TaskType, create_silent_task_annotator
    CRITICAL_ANNOTATION_AVAILABLE = True
except ImportError:
    print("⚠️ 关键时间段标注器不可用，将跳过关键时间段生成")
    CRITICAL_ANNOTATION_AVAILABLE = False


class HDF5VLADataset:
    """
    This class is used to sample episodes from the embododiment dataset
    stored in HDF5.
    🆕 集成关键时间段标注功能
    
    数据结构: 每个HDF5文件包含一个完整episode
    """

    def __init__(self, model_config_path) -> None:
        # [Modify] The path to the HDF5 dataset directory
        # Each HDF5 file contains one episode
        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)
        HDF5_DIR = model_config["data_path"]
        self.DATASET_NAME = model_config.get("dataset_name", "agilex")

        self.file_paths = []
        for root, _, files in os.walk(HDF5_DIR):
            for filename in fnmatch.filter(files, "*.hdf5"):
                file_path = os.path.join(root, filename)
                self.file_paths.append(file_path)

        if not self.file_paths:
            raise ValueError(f"未在 {HDF5_DIR} 中找到HDF5文件")

        # Load the config
        with open("configs/base.yaml", "r") as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config["common"]["action_chunk_size"]
        self.IMG_HISORY_SIZE = config["common"]["img_history_size"]
        self.STATE_DIM = config["common"]["state_dim"]

        # 🆕 关键时间段标注配置
        self.enable_critical_annotation = model_config.get("enable_critical_annotation", False)
        self.task_type = TaskType(model_config.get("task_type", 1)) if CRITICAL_ANNOTATION_AVAILABLE else None
        
        if self.enable_critical_annotation and CRITICAL_ANNOTATION_AVAILABLE:
            # 创建关键时间段标注器
            annotation_config = model_config.get("critical_annotation_config", {})
            self.critical_annotator = create_silent_task_annotator(self.task_type)
            
            # 如果有自定义配置，更新标注器参数
            if annotation_config:
                for key, value in annotation_config.items():
                    if hasattr(self.critical_annotator, key):
                        setattr(self.critical_annotator, key, value)
                        
            print(f"🎯 关键时间段标注器已启用 (任务类型: {self.task_type.name})")
        else:
            self.critical_annotator = None
            if self.enable_critical_annotation:
                print("⚠️ 关键时间段标注已禁用 (标注器不可用)")

        # Get each episode's len
        episode_lens = []
        valid_file_paths = []
        
        print(f"📂 扫描HDF5文件: 找到 {len(self.file_paths)} 个文件")
        
        for i, file_path in enumerate(self.file_paths):
            try:
                valid, res = self.parse_hdf5_file_state_only(file_path)
                if valid:
                    _len = res["state"].shape[0]
                    episode_lens.append(_len)
                    valid_file_paths.append(file_path)
                else:
                    print(f"⚠️ 跳过无效文件: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"⚠️ 处理文件失败 {os.path.basename(file_path)}: {e}")
                
        if not valid_file_paths:
            raise ValueError("未找到有效的episode数据")
            
        self.file_paths = valid_file_paths
        self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)
        
        print(f"✅ 数据集初始化完成: {len(self.file_paths)} 个有效episodes")
        print(f"   - 关键时间段标注: {'启用' if self.enable_critical_annotation and self.critical_annotator else '禁用'}")

    def __len__(self):
        return len(self.file_paths)

    def get_dataset_name(self):
        return self.DATASET_NAME

    def _generate_critical_labels(self, qpos_trajectory: np.ndarray, action_horizon: int = None) -> torch.Tensor:
        """
        🆕 生成关键时间段标签
        
        Args:
            qpos_trajectory: (T, N) numpy数组，完整的qpos轨迹
            action_horizon: 动作预测范围，如果为None则使用CHUNK_SIZE
            
        Returns:
            critical_labels: (action_horizon,) torch tensor，关键时间段标签
        """
        if not self.enable_critical_annotation or not self.critical_annotator:
            # 如果没有启用或标注器不可用，返回全0标签
            horizon = action_horizon or self.CHUNK_SIZE
            return torch.zeros(horizon, dtype=torch.long)
        
        if action_horizon is None:
            action_horizon = self.CHUNK_SIZE
            
        try:
            # 使用标注器生成标签
            critical_labels, analysis_info = self.critical_annotator.annotate(qpos_trajectory)
            
            # 转换为torch tensor
            critical_labels = torch.from_numpy(critical_labels).long()
            
            # 确保标签长度匹配action_horizon
            if len(critical_labels) > action_horizon:
                # 如果轨迹比action_horizon长，截取对应部分
                critical_labels = critical_labels[:action_horizon]
            elif len(critical_labels) < action_horizon:
                # 如果轨迹比action_horizon短，用0填充
                padding = torch.zeros(action_horizon - len(critical_labels), dtype=torch.long)
                critical_labels = torch.cat([critical_labels, padding])
                
            return critical_labels
            
        except Exception as e:
            print(f"⚠️ 关键时间段标注失败: {e}")
            # 失败时返回全0标签
            return torch.zeros(action_horizon, dtype=torch.long)

    def get_item(self, index: int = None, state_only=False):
        """Get a training sample at a random timestep.

        Args:
            index (int, optional): the index of the episode.
                If not provided, a random episode will be selected.
            state_only (bool, optional): Whether to return only the state.
                In this way, the sample will contain a complete trajectory rather
                than a single timestep. Defaults to False.

        Returns:
           sample (dict): a dictionary containing the training sample.
        """
        while True:
            if index is None:
                file_path = np.random.choice(self.file_paths, p=self.episode_sample_weights)
            else:
                file_path = self.file_paths[index]
            valid, sample = (self.parse_hdf5_file(file_path)
                             if not state_only else self.parse_hdf5_file_state_only(file_path))
            if valid:
                return sample
            else:
                index = np.random.randint(0, len(self.file_paths))

    def parse_hdf5_file(self, file_path):
        """🔄 修改现有方法，添加关键时间段标签生成"""
        try:
            with h5py.File(file_path, "r") as f:
                qpos = f["observations"]["qpos"][:]
                left_arm_dim = f["observations"]["left_arm_dim"][:]
                right_arm_dim = f["observations"]["right_arm_dim"][:]
                num_steps = qpos.shape[0]
                
                # [Optional] We drop too-short episode
                # if num_steps < 128:
                #     return False, None

                # [Optional] We skip the first few still steps
                EPS = 1e-2
                # Get the idx of the first qpos whose delta exceeds the threshold
                qpos_delta = np.abs(qpos - qpos[0:1])
                indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
                if len(indices) > 0:
                    first_idx = indices[0]
                else:
                    # 如果没有找到运动，使用第一个索引
                    first_idx = 0
                    print(f"⚠️ 文件 {os.path.basename(file_path)} 中未检测到明显运动")

                # 🆕 保存完整的qpos轨迹用于关键时间段分析
                qpos_trajectory = qpos.copy()

                # We randomly sample a timestep
                step_id = np.random.randint(max(first_idx - 1, 0), num_steps)

                # Load the instruction
                dir_path = os.path.dirname(file_path)

                # You can also use precomputed language embeddings (recommended)
                instruction = "perform manipulation task"  # 默认指令
                
                instructions_path = os.path.join(dir_path, "instructions")
                if os.path.exists(instructions_path):
                    instructions_names = []
                    for filename in os.listdir(instructions_path):
                        # 检查文件名是否以.pt结尾
                        if filename.endswith(".pt"):
                            instructions_names.append(os.path.join(instructions_path, filename))
                    
                    if instructions_names:
                        instruction = np.random.choice(instructions_names)

                # Assemble the meta
                meta = {
                    "dataset_name": self.DATASET_NAME,
                    "#steps": num_steps,
                    "step_id": step_id,
                    "instruction": instruction,
                }

                # Rescale gripper to [0, 1]
                total_joints = left_arm_dim[0] + 1 + right_arm_dim[0] + 1
                qpos = qpos / np.array([[1 for i in range(total_joints)]])
                target_qpos = f["action"][step_id:step_id + self.CHUNK_SIZE] / np.array(
                    [[1 for i in range(total_joints)]])

                # Parse the state and action
                state = qpos[step_id:step_id + 1]
                state_std = np.std(qpos, axis=0)
                state_mean = np.mean(qpos, axis=0)
                state_norm = np.sqrt(np.mean(qpos**2, axis=0))
                actions = target_qpos
                
                if actions.shape[0] < self.CHUNK_SIZE:
                    # Pad the actions using the last action
                    actions = np.concatenate(
                        [
                            actions,
                            np.tile(actions[-1:], (self.CHUNK_SIZE - actions.shape[0], 1)),
                        ],
                        axis=0,
                    )

                # 🆕 生成关键时间段标签
                # 为当前采样的动作序列生成标签
                action_start_idx = step_id
                action_end_idx = min(step_id + self.CHUNK_SIZE, qpos_trajectory.shape[0])
                
                # 提取对应时间段的qpos用于标注
                qpos_segment = qpos_trajectory[max(0, action_start_idx-10):action_end_idx+10]  # 稍微扩展上下文
                critical_labels = self._generate_critical_labels(qpos_segment, self.CHUNK_SIZE)

                # Fill the state/action into the unified vector
                def fill_in_state(values):
                    # Target indices corresponding to your state space
                    # In this example: 6 joints + 1 gripper for each arm
                    UNI_STATE_INDICES = (
                        [STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"]
                         for i in range(left_arm_dim[0])] + [STATE_VEC_IDX_MAPPING["left_gripper_open"]] +
                        [STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"]
                         for i in range(right_arm_dim[0])] + [STATE_VEC_IDX_MAPPING["right_gripper_open"]])
                    uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM, ))
                    uni_vec[..., UNI_STATE_INDICES] = values
                    return uni_vec

                state = fill_in_state(state)
                state_indicator = fill_in_state(np.ones_like(state_std))
                state_std = fill_in_state(state_std)
                state_mean = fill_in_state(state_mean)
                state_norm = fill_in_state(state_norm)
                # If action's format is different from state's,
                # you may implement fill_in_action()
                actions = fill_in_state(actions)

                # Parse the images
                def parse_img(key):
                    try:
                        imgs = []
                        for i in range(max(step_id - self.IMG_HISORY_SIZE + 1, 0), step_id + 1):
                            if i < f["observations"]["images"][key].shape[0]:
                                img_bits = f["observations"]["images"][key][i]
                                img = cv2.imdecode(np.frombuffer(img_bits, np.uint8), cv2.IMREAD_COLOR)
                                imgs.append(img)
                            else:
                                # 如果索引超出范围，使用最后一张图像
                                if imgs:
                                    imgs.append(imgs[-1])
                                else:
                                    # 创建黑色图像作为占位符
                                    imgs.append(np.zeros((224, 224, 3), dtype=np.uint8))
                        
                        imgs = np.stack(imgs) if imgs else np.zeros((self.IMG_HISORY_SIZE, 224, 224, 3), dtype=np.uint8)
                        
                        if imgs.shape[0] < self.IMG_HISORY_SIZE:
                            # Pad the images using the first image
                            imgs = np.concatenate(
                                [
                                    np.tile(
                                        imgs[:1],
                                        (self.IMG_HISORY_SIZE - imgs.shape[0], 1, 1, 1),
                                    ),
                                    imgs,
                                ],
                                axis=0,
                            )
                        return imgs
                    except Exception as e:
                        print(f"⚠️ 解析图像 {key} 失败: {e}")
                        return np.zeros((self.IMG_HISORY_SIZE, 224, 224, 3), dtype=np.uint8)

                # `cam_high` is the external camera image
                cam_high = parse_img("cam_high")
                # For step_id = first_idx - 1, the valid_len should be one
                valid_len = min(step_id - max(first_idx - 1, 0) + 1, self.IMG_HISORY_SIZE)
                cam_high_mask = np.array([False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len)
                cam_left_wrist = parse_img("cam_left_wrist")
                cam_left_wrist_mask = cam_high_mask.copy()
                cam_right_wrist = parse_img("cam_right_wrist")
                cam_right_wrist_mask = cam_high_mask.copy()

                # Return the resulting sample
                # For unavailable images, return zero-shape arrays, i.e., (IMG_HISORY_SIZE, 0, 0, 0)
                # E.g., return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0)) for the key "cam_left_wrist",
                # if the left-wrist camera is unavailable on your robot
                sample = {
                    "meta": meta,
                    "state": state,
                    "state_std": state_std,
                    "state_mean": state_mean,
                    "state_norm": state_norm,
                    "actions": actions,
                    "state_indicator": state_indicator,
                    "cam_high": cam_high,
                    "cam_high_mask": cam_high_mask,
                    "cam_left_wrist": cam_left_wrist,
                    "cam_left_wrist_mask": cam_left_wrist_mask,
                    "cam_right_wrist": cam_right_wrist,
                    "cam_right_wrist_mask": cam_right_wrist_mask,
                    "qpos_trajectory": qpos_trajectory,  # 🆕 添加完整qpos轨迹
                    "critical_labels": critical_labels,  # 🆕 添加关键时间段标签
                }
                
                return True, sample

        except Exception as e:
            print(f"❌ 解析HDF5文件失败 {os.path.basename(file_path)}: {e}")
            return False, None

    def parse_hdf5_file_state_only(self, file_path):
        """🔄 修改现有方法，添加关键时间段标签生成"""
        try:
            with h5py.File(file_path, "r") as f:
                qpos = f["observations"]["qpos"][:]
                left_arm_dim = f["observations"]["left_arm_dim"][:]
                right_arm_dim = f["observations"]["right_arm_dim"][:]

                num_steps = qpos.shape[0]
                # [Optional] We drop too-short episode
                if num_steps < 10:  # 至少需要10步
                    return False, None

                # [Optional] We skip the first few still steps
                EPS = 1e-2
                # Get the idx of the first qpos whose delta exceeds the threshold
                qpos_delta = np.abs(qpos - qpos[0:1])
                indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
                if len(indices) > 0:
                    first_idx = indices[0]
                else:
                    first_idx = 0  # 如果没有运动，从第一个时间步开始

                # 🆕 保存完整的qpos轨迹
                qpos_trajectory = qpos.copy()

                # 🆕 生成完整轨迹的关键时间段标签
                full_critical_labels = self._generate_critical_labels(qpos_trajectory)

                # Rescale gripper to [0, 1]
                total_joints = left_arm_dim[0] + right_arm_dim[0] + 2
                qpos = qpos / np.array([[1 for i in range(total_joints)]])
                target_qpos = f["action"][:] / np.array([[1 for i in range(total_joints)]])

                # Parse the state and action
                state = qpos[max(first_idx - 1, 0):]
                action = target_qpos[max(first_idx - 1, 0):]

                # Fill the state/action into the unified vector
                def fill_in_state(values):
                    # Target indices corresponding to your state space
                    # In this example: 6 joints + 1 gripper for each arm
                    UNI_STATE_INDICES = (
                        [STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"]
                         for i in range(left_arm_dim[0])] + [STATE_VEC_IDX_MAPPING["left_gripper_open"]] +
                        [STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"]
                         for i in range(right_arm_dim[0])] + [STATE_VEC_IDX_MAPPING["right_gripper_open"]])
                    uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM, ))
                    uni_vec[..., UNI_STATE_INDICES] = values
                    return uni_vec

                state = fill_in_state(state)
                action = fill_in_state(action)

                # 调整关键时间段标签长度以匹配action长度
                if len(full_critical_labels) > len(action):
                    adjusted_critical_labels = full_critical_labels[:len(action)]
                elif len(full_critical_labels) < len(action):
                    # 用0填充
                    padding = torch.zeros(len(action) - len(full_critical_labels), dtype=torch.long)
                    adjusted_critical_labels = torch.cat([full_critical_labels, padding])
                else:
                    adjusted_critical_labels = full_critical_labels

                # Return the resulting sample
                return True, {
                    "state": state, 
                    "action": action,
                    "qpos_trajectory": qpos_trajectory,  # 🆕 添加完整qpos轨迹
                    "critical_labels": adjusted_critical_labels,  # 🆕 添加关键时间段标签
                }

        except Exception as e:
            print(f"❌ 解析HDF5文件失败 {os.path.basename(file_path)}: {e}")
            return False, None


if __name__ == "__main__":
    # 测试代码
    import tempfile
    
    # 创建临时配置文件
    test_config = {
        "data_path": "/path/to/your/hdf5/data",  # 请修改为你的实际路径
        "dataset_name": "agilex",
        "enable_critical_annotation": True,
        "task_type": 1,  # 1=抓取类, 2=点击类
        "critical_annotation_config": {
            "relative_low_speed_ratio": 0.15,
            "min_deceleration_threshold": -0.0008,
            "gripper_close_delta_threshold": -0.01,
            "verbose": True  # 测试时启用详细输出
        }
    }
    
    # 保存临时配置
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        import yaml
        yaml.dump(test_config, f)
        config_path = f.name
    
    try:
        print("🧪 测试HDF5VLA数据集...")
        ds = HDF5VLADataset(config_path)
        
        print(f"数据集长度: {len(ds)}")
        print(f"数据集名称: {ds.get_dataset_name()}")
        
        # 测试获取样本
        if len(ds) > 0:
            sample = ds.get_item(0)
            print("\n样本信息:")
            for key, value in sample.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, torch.Tensor):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, dict):
                    print(f"  {key}: dict with keys {list(value.keys())}")
                else:
                    print(f"  {key}: {type(value)}")
            
            # 检查关键时间段标签
            if "critical_labels" in sample:
                critical_labels = sample["critical_labels"]
                critical_ratio = critical_labels.float().mean().item() if hasattr(critical_labels, 'float') else 0
                print(f"\n🎯 关键时间段信息:")
                print(f"  标签形状: {critical_labels.shape}")
                print(f"  关键时间段比例: {critical_ratio:.3f}")
                print(f"  关键时间步数: {critical_labels.sum()}")
                
        print("✅ 测试完成!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理临时文件
        os.unlink(config_path)