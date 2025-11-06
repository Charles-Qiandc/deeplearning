# # train/dataset.py - 更新版本，集成任务驱动的关键时间段标注

# import traceback
# import time
# import os
# import json
# import math
# import random
# from typing import Dict, Sequence

# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms
# from PIL import Image
# import transformers

# from data.filelock import FileLock
# from data.hdf5_vla_dataset import HDF5VLADataset
# from train.image_corrupt import image_corrupt

# # 🆕 导入任务驱动的关键时间段标注器
# from data.critical_timestep_annotator import (
#     TaskType, 
#     create_silent_task_annotator,
#     TaskDrivenCriticalTimestepAnnotator
# )


# def get_clean_item(chunk_dir):
#     """Get indexes of clean items in a chunk."""
#     dirty_bit = read_dirty_bit(chunk_dir)
#     return np.where(1 - dirty_bit)[0].tolist()


# def save_dirty_bit(chunk_dir, dirty_bit):
#     """Save the dirty bit to the chunk directory."""
#     time_stmp = time.time()
#     while time.time() - time_stmp < 10.0:
#         try:
#             file_path = os.path.join(chunk_dir, "dirty_bit")
#             lock = FileLock(file_path)
#             lock.acquire_write_lock()
#             with open(file_path, "wb") as file:
#                 file.write(dirty_bit.tobytes())
#             lock.release_lock()
#             return
#         except KeyboardInterrupt:
#             lock.release_lock()
#             raise KeyboardInterrupt
#         except BaseException:
#             lock.release_lock()
#             continue
#     raise RuntimeError("Failed to save dirty bit.")


# def read_dirty_bit(chunk_dir):
#     """Read the dirty bit from the chunk directory."""
#     time_stmp = time.time()
#     while time.time() - time_stmp < 10.0:
#         try:
#             file_path = os.path.join(chunk_dir, "dirty_bit")
#             lock = FileLock(file_path)
#             lock.acquire_read_lock()
#             with open(file_path, "rb") as file:
#                 dirty_bit = np.frombuffer(file.read(), dtype=np.uint8).copy()
#             lock.release_lock()
#             assert len(dirty_bit) > 0
#             return dirty_bit
#         except KeyboardInterrupt:
#             lock.release_lock()
#             raise KeyboardInterrupt
#         except BaseException:
#             lock.release_lock()
#             continue
#     raise RuntimeError("Failed to read dirty bit.")


# class VLAConsumerDataset(Dataset):
#     """A vision-language-action Dataset for supervised training.
#     🆕 集成任务驱动的关键时间段标注机制
#     """

#     def __init__(
#         self,
#         model_config_path,
#         config,
#         tokenizer,
#         image_processor,
#         num_cameras,
#         img_history_size,
#         image_size=None,
#         auto_adjust_image_brightness=False,
#         image_aug=False,
#         dataset_type="pretrain",
#         cond_mask_prob=0.1,
#         cam_ext_mask_prob=-1.0,
#         state_noise_snr=None,
#         use_hdf5=False,
#         use_precomp_lang_embed=False,
#         use_dinov2_features=False,
#         use_depth_features=False,
#         # 🆕 关键时间段标注相关参数
#         task_type: int = 1,  # 1=抓取类, 2=点击类
#         enable_critical_annotation: bool = True,
#         critical_annotation_config: Dict = None,
#     ):
#         super(VLAConsumerDataset, self).__init__()

#         # Load the control frequency for each dataset
#         with open("configs/dataset_control_freq.json", "r") as fp:
#             self.control_freq = json.load(fp)
#         # Load the dataset names
#         dataset_names_cfg = ("configs/pretrain_datasets.json"
#                              if dataset_type == "pretrain" else "configs/finetune_datasets.json")
#         with open(dataset_names_cfg, "r") as file:
#             DATASET_NAMES = json.load(file)
#         # Create the mapping between dataset name and id
#         self.dataset_name2id = {name: i for i, name in enumerate(DATASET_NAMES)}
#         self.dataset_id2name = {i: name for i, name in enumerate(DATASET_NAMES)}

#         self.image_processor = image_processor
#         self.model_config_path = model_config_path
#         self.buffer_dir = config["buf_path"]
#         self.num_chunks = config["buf_num_chunks"]
#         self.chunk_size = config["buf_chunk_size"]
#         self.tokenizer_max_length = config["tokenizer_max_length"]
#         self.image_aspect_ratio = config["image_aspect_ratio"]
#         self.state_noise_snr = state_noise_snr
#         self.num_cameras = num_cameras
#         self.img_history_size = img_history_size
#         self.cond_mask_prob = cond_mask_prob
#         self.cam_ext_mask_prob = cam_ext_mask_prob
#         self.use_hdf5 = use_hdf5
#         self.hdf5_dataset = None
#         if use_hdf5:
#             self.hdf5_dataset = HDF5VLADataset(self.model_config_path)
#         self.use_precomp_lang_embed = use_precomp_lang_embed
#         if use_precomp_lang_embed:
#             self.empty_lang_embed = torch.load("data/empty_lang_embed.pt")
        
#         # DINOv2相关配置
#         self.use_dinov2_features = use_dinov2_features
#         self.dinov2_image_size = 518
        
#         # DepthAnythingV2相关配置
#         self.use_depth_features = use_depth_features
#         self.depth_image_size = 518

#         # Load dataset stat
#         with open("configs/dataset_stat.json", "r") as f:
#             dataset_stat = json.load(f)
#         self.dataset_stat = dataset_stat

#         self.tokenizer = tokenizer
#         self.image_size = image_size
#         self.auto_adjust_image_brightness = auto_adjust_image_brightness
#         self.image_aug = image_aug

#         self.last_content = None
#         self.last_meta = None
        
#         # 🆕 关键时间段标注配置
#         self.task_type = TaskType(task_type)
#         self.enable_critical_annotation = enable_critical_annotation
        
#         if enable_critical_annotation:
#             # 设置默认配置
#             default_config = {
#                 'relative_low_speed_ratio': 0.15,
#                 'min_deceleration_threshold': -0.0008,
#                 'gripper_close_delta_threshold': -0.01,
#                 'smooth': True,
#                 'verbose': False
#             }
            
#             # 合并用户配置
#             if critical_annotation_config:
#                 default_config.update(critical_annotation_config)
            
#             # 创建标注器
#             self.critical_annotator = TaskDrivenCriticalTimestepAnnotator(
#                 task_type=self.task_type,
#                 **default_config
#             )
            
#             print(f"🎯 关键时间段标注器初始化:")
#             print(f"   - 任务类型: {self.task_type.name} ({self.task_type.value})")
#             print(f"   - 配置: {default_config}")
#         else:
#             self.critical_annotator = None
#             print("⚠️  关键时间段标注已禁用")

#     def get_dataset_name2id(self):
#         return self.dataset_name2id

#     def get_dataset_id2name(self):
#         return self.dataset_id2name

#     @staticmethod
#     def pairwise(iterable):
#         a = iter(iterable)
#         return zip(a, a)

#     @staticmethod
#     def _load_data_from_chunk(chunk_dir, chunk_item_idx):
#         time_stmp = time.time()
#         while time.time() - time_stmp < 10.0:
#             try:
#                 locks = []
#                 file_path = os.path.join(chunk_dir, f"json_content_{chunk_item_idx}.json")
#                 lock = FileLock(file_path)
#                 locks.append(lock)
#                 lock.acquire_read_lock()
#                 with open(file_path, "r") as file:
#                     json_content = json.load(file)
#                 lock.release_lock()
#                 file_path = os.path.join(chunk_dir, f"sample_{chunk_item_idx}.npz")
#                 lock = FileLock(file_path)
#                 locks.append(lock)
#                 lock.acquire_read_lock()
#                 with open(file_path, "rb") as file:
#                     sample_dict = np.load(file)
#                     meta = tuple(sample_dict.values())
#                 lock.release_lock()
#                 return json_content, meta
#             except KeyboardInterrupt:
#                 for lock in locks:
#                     lock.release_lock()
#                 raise KeyboardInterrupt
#             except BaseException:
#                 for lock in locks:
#                     lock.release_lock()
#                 continue
#         raise RuntimeError("Failed to load sample.")

#     def __len__(self) -> int:
#         if self.use_hdf5:
#             return len(self.hdf5_dataset)
#         else:
#             return self.num_chunks * self.chunk_size

#     def _safe_load(self, index):
#         read_chunk_item_indices = []
#         # Start searching from a random chunk
#         read_chunk_idx = index // self.chunk_size
#         while len(read_chunk_item_indices) == 0:
#             read_chunk_dir = os.path.join(self.buffer_dir, f"chunk_{read_chunk_idx}")
#             try:
#                 read_chunk_item_indices = get_clean_item(read_chunk_dir)
#             except BaseException as e:
#                 print("Error catched when searching a clean chunk:", e)
#                 traceback.print_exc()
#                 read_chunk_item_indices = []
#             read_chunk_idx = (read_chunk_idx + 1) % self.num_chunks

#         random_item_index = index % len(read_chunk_item_indices)
#         read_chunk_item_index = read_chunk_item_indices[random_item_index]

#         # Modify the dirty bit
#         try:
#             dirty_bit = read_dirty_bit(read_chunk_dir)
#             dirty_bit[read_chunk_item_index] = 1
#             save_dirty_bit(read_chunk_dir, dirty_bit)
#         except BaseException as e:
#             print("Error catched when modifying the dirty bit:", e)
#             traceback.print_exc()

#         # load the sample
#         try:
#             content, meta = self._load_data_from_chunk(read_chunk_dir, read_chunk_item_index)
#             self.last_content, self.last_meta = content, meta
#         except BaseException as e:
#             print("Error catched when loading sample:", e)
#             traceback.print_exc()
#             content, meta = self.last_content, self.last_meta

#         return (content, *meta)

#     def _generate_critical_labels(self, qpos_trajectory, action_horizon=64):
#         """
#         🆕 生成关键时间段标签
#         🔧 修复版本：确保输出形状正确
        
#         Args:
#             qpos_trajectory: (T, 14) numpy数组，关节角度轨迹
#             action_horizon: 动作预测的时间范围
            
#         Returns:
#             critical_labels: (action_horizon,) torch tensor，关键时间段标签
#         """
#         try:
#             if self.critical_annotator is None:
#                 # 如果没有标注器，返回合理的默认标签
#                 # 使用简单的启发式：前25%和后25%为关键时间段
#                 critical_labels = np.zeros(action_horizon, dtype=np.int64)
                
#                 # 开始阶段（前25%）和结束阶段（后25%）标记为关键
#                 start_critical = int(action_horizon * 0.25)
#                 end_critical = int(action_horizon * 0.75)
                
#                 critical_labels[:start_critical] = 1  # 开始阶段
#                 critical_labels[end_critical:] = 1    # 结束阶段
                
#                 return torch.from_numpy(critical_labels).long()
            
#             # 有标注器时使用标注器
#             if qpos_trajectory is None or len(qpos_trajectory) == 0:
#                 print("⚠️ qpos_trajectory为空，使用默认标签")
#                 # 返回交替模式的标签
#                 critical_labels = np.zeros(action_horizon, dtype=np.int64)
#                 critical_labels[action_horizon//4:action_horizon//2] = 1  # 中间一段为关键
#                 return torch.from_numpy(critical_labels).long()
            
#             # 使用标注器生成标签
#             critical_labels, analysis_info = self.critical_annotator.annotate(qpos_trajectory)
            
#             # 🔧 修复：确保标签是正确的数据类型
#             if not isinstance(critical_labels, np.ndarray):
#                 critical_labels = np.array(critical_labels, dtype=np.int64)
#             else:
#                 critical_labels = critical_labels.astype(np.int64)
            
#             # 🔧 修复：处理长度不匹配的问题
#             if len(critical_labels) > action_horizon:
#                 # 如果标注结果太长，截取前action_horizon个
#                 critical_labels = critical_labels[:action_horizon]
#             elif len(critical_labels) < action_horizon:
#                 # 如果标注结果太短，使用策略填充
#                 padding_len = action_horizon - len(critical_labels)
                
#                 if len(critical_labels) > 0:
#                     # 使用最后一个值填充
#                     last_value = critical_labels[-1]
#                     padding = np.full(padding_len, last_value, dtype=np.int64)
#                 else:
#                     # 如果完全没有标注，使用0填充
#                     padding = np.zeros(padding_len, dtype=np.int64)
                
#                 critical_labels = np.concatenate([critical_labels, padding])
            
#             # 🔧 修复：验证输出形状
#             assert len(critical_labels) == action_horizon, f"标签长度不匹配: {len(critical_labels)} vs {action_horizon}"
#             assert critical_labels.dtype == np.int64, f"标签类型错误: {critical_labels.dtype}"
            
#             # 转换为torch tensor
#             critical_labels_tensor = torch.from_numpy(critical_labels).long()
            
#             # 🔧 验证值的范围
#             if torch.any(critical_labels_tensor < 0) or torch.any(critical_labels_tensor > 1):
#                 print(f"⚠️ 标签值超出范围 [0,1]: {critical_labels_tensor.unique()}")
#                 # 将所有非0值转换为1
#                 critical_labels_tensor = torch.clamp(critical_labels_tensor, 0, 1)
            
#             return critical_labels_tensor
            
#         except Exception as e:
#             print(f"⚠️ 关键时间段标注失败: {e}")
#             import traceback
#             traceback.print_exc()
            
#             # 🔧 降级方案：返回安全的默认标签
#             print("🔧 使用安全的默认标签")
#             critical_labels = np.zeros(action_horizon, dtype=np.int64)
            
#             # 简单策略：中间30%为关键时间段
#             start_idx = int(action_horizon * 0.35)
#             end_idx = int(action_horizon * 0.65)
#             critical_labels[start_idx:end_idx] = 1
            
#             return torch.from_numpy(critical_labels).long()

#     def __getitem__(self, index):
#         # For robustness, we will try to load the data until we succeed
#         while True:
#             data_dict = None
#             try:
#                 if self.use_hdf5:
#                     res = self.hdf5_dataset.get_item()
#                     content = res["meta"]
#                     states = res["state"]
#                     actions = res["actions"]
#                     state_elem_mask = res["state_indicator"]
#                     image_metas = [
#                         res["cam_high"],
#                         res["cam_high_mask"],
#                         res["cam_right_wrist"],
#                         res["cam_right_wrist_mask"],
#                         res["cam_left_wrist"],
#                         res["cam_left_wrist_mask"],
#                     ]
#                     state_std = res["state_std"]
#                     state_mean = res["state_mean"]
#                     state_norm = res["state_norm"]
                    
#                     # 🆕 获取qpos轨迹用于关键时间段分析
#                     qpos_trajectory = res.get("qpos_trajectory", None)
#                 else:
#                     (
#                         content,
#                         _,
#                         states,
#                         _,
#                         actions,
#                         _,
#                         state_elem_mask,
#                         *image_metas,
#                         state_std,
#                         state_mean,
#                         state_norm,
#                     ) = self._safe_load(index)
#                     qpos_trajectory = None  # 非HDF5模式暂不支持

#                 data_dict = {}
#                 data_dict["dataset_name"] = content["dataset_name"]
#                 data_dict["data_idx"] = self.dataset_name2id[data_dict["dataset_name"]]
#                 data_dict["ctrl_freq"] = (self.control_freq[data_dict["dataset_name"]]
#                                           if random.random() > self.cond_mask_prob else 0)

#                 if self.state_noise_snr is not None:
#                     states += np.random.normal(
#                         0.0,
#                         state_std / np.sqrt(10**(self.state_noise_snr / 10)),
#                         states.shape,
#                     )
#                 ds_state_mean = np.array(self.dataset_stat[data_dict["dataset_name"]]["state_mean"])
#                 ds_state_mean = np.tile(ds_state_mean[None], (states.shape[0], 1))
                
#                 data_dict["states"] = (states if random.random() > self.cond_mask_prob else ds_state_mean)
#                 data_dict["actions"] = actions
#                 data_dict["state_elem_mask"] = (state_elem_mask if random.random() > self.cond_mask_prob else
#                                                 np.zeros_like(state_elem_mask))
#                 data_dict["state_norm"] = state_norm

#                 # 🆕 生成关键时间段标签 - 修复版本
#                 action_horizon = actions.shape[0]  # 动作序列长度
                
#                 if self.enable_critical_annotation:
#                     try:
#                         critical_labels = self._generate_critical_labels(qpos_trajectory, action_horizon)
                        
#                         # 🔧 额外验证
#                         if critical_labels.shape[0] != action_horizon:
#                             print(f"⚠️ 样本 {index}: 标签形状不匹配 {critical_labels.shape[0]} vs {action_horizon}")
#                             # 重新生成正确形状的标签
#                             critical_labels = torch.zeros(action_horizon, dtype=torch.long)
#                             critical_labels[action_horizon//3:2*action_horizon//3] = 1
                        
#                         data_dict["critical_labels"] = critical_labels
                        
#                         # 可选：记录标注统计信息（用于调试）
#                         if hasattr(self.critical_annotator, 'verbose') and self.critical_annotator and self.critical_annotator.verbose:
#                             critical_ratio = critical_labels.float().mean().item()
#                             print(f"📊 样本 {index}: 关键时间段比例 = {critical_ratio:.3f}, 形状 = {critical_labels.shape}")
                            
#                     except Exception as e:
#                         print(f"⚠️ 样本 {index} 关键时间段标注失败: {e}")
#                         # 🔧 失败时使用安全的默认标签
#                         critical_labels = torch.zeros(action_horizon, dtype=torch.long)
#                         # 简单模式：每4个时间步有1个关键时间段
#                         critical_labels[::4] = 1
#                         data_dict["critical_labels"] = critical_labels
#                 else:
#                     # 如果没有启用标注，使用简单的默认模式
#                     critical_labels = torch.zeros(action_horizon, dtype=torch.long)
#                     # 使用启发式：中间50%为关键时间段
#                     start_critical = action_horizon // 4
#                     end_critical = 3 * action_horizon // 4
#                     critical_labels[start_critical:end_critical] = 1
#                     data_dict["critical_labels"] = critical_labels

#                 # 🔧 最终验证critical_labels
#                 final_critical_labels = data_dict["critical_labels"]
#                 if not isinstance(final_critical_labels, torch.Tensor):
#                     final_critical_labels = torch.tensor(final_critical_labels, dtype=torch.long)
#                     data_dict["critical_labels"] = final_critical_labels
                
#                 # 验证形状和类型
#                 assert final_critical_labels.shape == (action_horizon,), f"标签形状错误: {final_critical_labels.shape} vs ({action_horizon},)"
#                 assert final_critical_labels.dtype == torch.long, f"标签类型错误: {final_critical_labels.dtype}"
#                 assert torch.all(final_critical_labels >= 0) and torch.all(final_critical_labels <= 1), f"标签值超出范围: {final_critical_labels.unique()}"

#                 # ... 继续处理其他数据（图像等）
#                 # Background image for padding/masking
#                 background_color = np.array(
#                     [int(x * 255) for x in self.image_processor.image_mean],
#                     dtype=np.uint8,
#                 ).reshape(1, 1, 3)
#                 background_image = (np.ones(
#                     (
#                         self.image_processor.size["height"],
#                         self.image_processor.size["width"],
#                         3,
#                     ),
#                     dtype=np.uint8,
#                 ) * background_color)

#                 image_metas = list(self.pairwise(image_metas))
#                 mask_probs = [self.cond_mask_prob] * self.num_cameras
#                 if self.cam_ext_mask_prob >= 0.0:
#                     mask_probs[0] = self.cam_ext_mask_prob
                
#                 rearranged_images = []
#                 for i in range(self.img_history_size):
#                     for j in range(self.num_cameras):
#                         images, image_mask = image_metas[j]
#                         image, valid = images[i], image_mask[i]
#                         if (valid and (math.prod(image.shape) > 0) and (random.random() > mask_probs[j])):
#                             rearranged_images.append((image, True))
#                         else:
#                             rearranged_images.append((background_image.copy(), False))

#                 # 为DINOv2准备单独的图像
#                 if self.use_dinov2_features:
#                     camera_idx = 0
#                     frame_idx = self.img_history_size - 1
                    
#                     if camera_idx >= len(image_metas):
#                         raise ValueError(f"相机索引 {camera_idx} 超出范围")
                    
#                     images, image_mask = image_metas[camera_idx]
#                     image, valid = images[frame_idx], image_mask[frame_idx]
                    
#                     if not valid or math.prod(image.shape) <= 0:
#                         raise ValueError(f"DINOv2图像无效")
                    
#                     # 预处理DINOv2图像
#                     pil_image = Image.fromarray(image)
#                     pil_image = pil_image.resize((self.dinov2_image_size, self.dinov2_image_size), Image.BILINEAR)
#                     image_array = np.array(pil_image).astype(np.float32) / 255.0
#                     image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
                    
#                     # ImageNet标准化
#                     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#                     std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
#                     image_tensor = (image_tensor - mean) / std
                    
#                     data_dict["dinov2_images"] = image_tensor.unsqueeze(0)

#                 # 为DepthAnythingV2准备深度图像
#                 if self.use_depth_features:
#                     camera_idx = 0
#                     frame_idx = self.img_history_size - 1
                    
#                     if camera_idx >= len(image_metas):
#                         raise ValueError(f"深度编码器相机索引 {camera_idx} 超出范围")
                    
#                     images, image_mask = image_metas[camera_idx]
#                     image, valid = images[frame_idx], image_mask[frame_idx]
                    
#                     if not valid or math.prod(image.shape) <= 0:
#                         raise ValueError(f"深度编码器图像无效")
                    
#                     # DepthAnythingV2预处理
#                     pil_image = Image.fromarray(image)
#                     pil_image = pil_image.resize((self.depth_image_size, self.depth_image_size), Image.BILINEAR)
#                     image_array = np.array(pil_image).astype(np.float32) / 255.0
#                     image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
                    
#                     # ImageNet标准化
#                     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#                     std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
#                     image_tensor = (image_tensor - mean) / std
                    
#                     data_dict["depth_images"] = image_tensor.unsqueeze(0)

#                 # 处理原始图像（SigLIP编码用）
#                 preprocessed_images = []
#                 processor = self.image_processor
#                 for image, valid in rearranged_images:
#                     image = Image.fromarray(image)
#                     if self.image_size is not None:
#                         image = transforms.Resize(self.image_size)(image)

#                     if valid and self.auto_adjust_image_brightness:
#                         pixel_values = list(image.getdata())
#                         average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
#                         if average_brightness <= 0.15:
#                             image = transforms.ColorJitter(brightness=(1.75, 1.75))(image)

#                     # 图像增强
#                     if valid and self.image_aug and (random.random() > 0.5):
#                         aug_type = random.choice(["corrput_only", "color_only", "both"])
#                         if aug_type != "corrput_only":
#                             image = transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5,
#                                                            hue=0.03)(image)
#                         if aug_type != "color_only":
#                             image = image_corrupt(image)

#                     if self.image_aspect_ratio == "pad":
#                         def expand2square(pil_img, background_color):
#                             width, height = pil_img.size
#                             if width == height:
#                                 return pil_img
#                             elif width > height:
#                                 result = Image.new(pil_img.mode, (width, width), background_color)
#                                 result.paste(pil_img, (0, (width - height) // 2))
#                                 return result
#                             else:
#                                 result = Image.new(pil_img.mode, (height, height), background_color)
#                                 result.paste(pil_img, ((height - width) // 2, 0))
#                                 return result

#                         image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
#                     image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
#                     preprocessed_images.append(image)
#                 data_dict["images"] = preprocessed_images

#                 # 处理语言输入
#                 if self.use_precomp_lang_embed:
#                     if content["instruction"][-1] == ".":
#                         content["instruction"] = content["instruction"][:-1]
#                     data_dict["lang_embed"] = (torch.load(content["instruction"])
#                                                if random.random() > self.cond_mask_prob else self.empty_lang_embed)
#                 else:
#                     instruction = (content["instruction"] if random.random() > self.cond_mask_prob else "")
#                     data_dict["input_ids"] = self.tokenizer(
#                         instruction,
#                         return_tensors="pt",
#                         padding="longest",
#                         truncation=False,
#                     ).input_ids[0]

#                     assert (
#                         len(data_dict["input_ids"]) <= self.tokenizer_max_length
#                     ), f"Instruction length {len(data_dict['input_ids'])} exceeds the maximum length {self.tokenizer_max_length}."

#                 # 转换numpy数组为tensor
#                 for k, v in data_dict.items():
#                     if isinstance(v, np.ndarray):
#                         data_dict[k] = torch.from_numpy(v)

#                 # 🔧 最终检查所有tensor
#                 for k, v in data_dict.items():
#                     assert not isinstance(v, np.ndarray), f"key: {k}, value: {v}"

#                 # 🔧 最终验证critical_labels（再次检查）
#                 if "critical_labels" in data_dict:
#                     labels = data_dict["critical_labels"]
#                     if not isinstance(labels, torch.Tensor):
#                         print(f"⚠️ critical_labels不是tensor: {type(labels)}")
#                         data_dict["critical_labels"] = torch.tensor(labels, dtype=torch.long)
#                     elif labels.dtype != torch.long:
#                         print(f"⚠️ critical_labels类型错误: {labels.dtype}")
#                         data_dict["critical_labels"] = labels.long()

#                 return data_dict
                
#             except BaseException as e:
#                 if data_dict is not None:
#                     print(f"Error catched when processing sample from {data_dict.get('dataset_name')}: {e}")
#                 else:
#                     print(f"Error catched when processing sample: {e}")
#                 traceback.print_exc()
#                 index = (index + 1) % len(self)

# # 修复 train/dataset.py 中的 DataCollatorForVLAConsumerDataset 部分

# class DataCollatorForVLAConsumerDataset(object):
#     """Collate examples for supervised training.
#     🆕 支持关键时间段标签的数据收集
#     🔧 修复版本：增强错误处理和形状验证
#     """

#     def __init__(self, tokenizer: transformers.PreTrainedTokenizer) -> None:
#         self.tokenizer = tokenizer

#     def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
#         batch = {
#             "states": [],
#             "actions": [],
#             "state_elem_mask": [],
#             "state_norm": [],
#             "images": [],
#             "data_indices": [],
#             "ctrl_freqs": [],
#         }
        
#         # 视觉特征收集
#         dinov2_images = []
#         depth_images = []
        
#         # 🆕 关键时间段标签收集 - 修复版本
#         critical_labels = []
        
#         # 语言特征收集
#         input_ids = []
#         lang_embeds = []
#         lang_embed_lens = []

#         for idx, instance in enumerate(instances):
#             try:
#                 # 收集基础数据
#                 keys_to_check = ["states", "actions", "state_elem_mask", "state_norm"]
#                 for key in keys_to_check:
#                     if isinstance(instance[key], torch.Tensor):
#                         item = instance[key]
#                     else:
#                         item = torch.from_numpy(instance[key])
#                     batch[key].append(item)

#                 # 收集语言数据
#                 if "input_ids" in instance:
#                     input_ids.append(instance["input_ids"])
#                 else:
#                     lang_embeds.append(instance["lang_embed"])
#                     lang_embed_lens.append(instance["lang_embed"].shape[0])

#                 # 收集图像和其他数据
#                 batch["images"].append(torch.stack(instance["images"], dim=0))
#                 batch["data_indices"].append(instance["data_idx"])
#                 batch["ctrl_freqs"].append(instance["ctrl_freq"])
                
#                 # 收集DINOv2图像
#                 if "dinov2_images" in instance:
#                     dinov2_images.append(instance["dinov2_images"])
                
#                 # 收集DepthAnythingV2图像
#                 if "depth_images" in instance:
#                     depth_images.append(instance["depth_images"])
                
#                 # 🆕 收集关键时间段标签 - 修复版本
#                 if "critical_labels" in instance:
#                     labels = instance["critical_labels"]
                    
#                     # 🔧 验证和转换标签
#                     if not isinstance(labels, torch.Tensor):
#                         labels = torch.tensor(labels, dtype=torch.long)
#                     elif labels.dtype != torch.long:
#                         labels = labels.long()
                    
#                     # 🔧 验证标签值范围
#                     if torch.any(labels < 0) or torch.any(labels > 1):
#                         print(f"⚠️ 实例 {idx}: 标签值超出范围 [0,1]: {labels.unique()}")
#                         labels = torch.clamp(labels, 0, 1)
                    
#                     # 🔧 验证形状
#                     if labels.dim() != 1:
#                         print(f"⚠️ 实例 {idx}: 标签维度错误: {labels.shape}，应该是1D")
#                         if labels.numel() > 0:
#                             labels = labels.flatten()
#                         else:
#                             # 如果为空，创建默认标签
#                             action_len = instance["actions"].shape[0]
#                             labels = torch.zeros(action_len, dtype=torch.long)
                    
#                     critical_labels.append(labels)
#                 else:
#                     # 如果没有标签，创建默认标签
#                     action_len = instance["actions"].shape[0]
#                     default_labels = torch.zeros(action_len, dtype=torch.long)
#                     # 简单启发式：中间部分为关键时间段
#                     start_critical = action_len // 4
#                     end_critical = 3 * action_len // 4
#                     default_labels[start_critical:end_critical] = 1
#                     critical_labels.append(default_labels)
                    
#             except Exception as e:
#                 print(f"❌ 处理实例 {idx} 时出错: {e}")
#                 import traceback
#                 traceback.print_exc()
                
#                 # 🔧 创建降级数据
#                 action_len = 64  # 默认长度
#                 if "actions" in instance:
#                     try:
#                         action_len = instance["actions"].shape[0]
#                     except:
#                         pass
                
#                 # 添加默认的critical_labels
#                 default_labels = torch.zeros(action_len, dtype=torch.long)
#                 default_labels[action_len//3:2*action_len//3] = 1  # 中间1/3为关键
#                 critical_labels.append(default_labels)
                
#                 # 继续处理其他必要字段...
#                 continue

#         # 🔧 验证所有实例都有标签
#         if len(critical_labels) != len(instances):
#             print(f"⚠️ 标签数量不匹配: {len(critical_labels)} vs {len(instances)}")
#             # 补充缺失的标签
#             while len(critical_labels) < len(instances):
#                 default_labels = torch.zeros(64, dtype=torch.long)
#                 critical_labels.append(default_labels)

#         # 堆叠基础数据
#         keys_to_stack = ["states", "actions", "state_elem_mask", "state_norm", "images"]
#         for key in keys_to_stack:
#             batch[key] = torch.stack(batch[key], dim=0)

#         batch["ctrl_freqs"] = torch.tensor(batch["ctrl_freqs"])

#         # 处理语言数据
#         if len(input_ids) > 0:
#             input_ids = torch.nn.utils.rnn.pad_sequence(input_ids,
#                                                         batch_first=True,
#                                                         padding_value=self.tokenizer.pad_token_id)
#             batch["input_ids"] = input_ids
#             batch["lang_attn_mask"] = input_ids.ne(self.tokenizer.pad_token_id)
#         else:
#             lang_embeds = torch.nn.utils.rnn.pad_sequence(lang_embeds, batch_first=True, padding_value=0)
#             input_lang_attn_mask = torch.zeros(lang_embeds.shape[0], lang_embeds.shape[1], dtype=torch.bool)
#             for i, l in enumerate(lang_embed_lens):
#                 input_lang_attn_mask[i, :l] = True
#             batch["lang_embeds"] = lang_embeds
#             batch["lang_attn_mask"] = input_lang_attn_mask
        
#         # 堆叠DINOv2图像
#         if len(dinov2_images) > 0:
#             batch["dinov2_images"] = torch.stack(dinov2_images, dim=0)
        
#         # 堆叠DepthAnythingV2图像
#         if len(depth_images) > 0:
#             batch["depth_images"] = torch.stack(depth_images, dim=0)
        
#         # 🆕 堆叠关键时间段标签 - 修复版本
#         if len(critical_labels) > 0:
#             try:
#                 # 🔧 找到最大长度进行填充
#                 max_len = max(labels.shape[0] for labels in critical_labels)
                
#                 # 🔧 验证所有标签都是1D
#                 validated_labels = []
#                 for i, labels in enumerate(critical_labels):
#                     if labels.dim() != 1:
#                         print(f"⚠️ 标签 {i} 维度错误: {labels.shape}")
#                         labels = labels.flatten()
                    
#                     # 🔧 填充到统一长度
#                     if labels.shape[0] < max_len:
#                         padding_len = max_len - labels.shape[0]
#                         # 使用0填充（非关键时间段）
#                         padding = torch.zeros(padding_len, dtype=labels.dtype)
#                         labels = torch.cat([labels, padding], dim=0)
#                     elif labels.shape[0] > max_len:
#                         # 截断到最大长度
#                         labels = labels[:max_len]
                    
#                     validated_labels.append(labels)
                
#                 # 🔧 堆叠标签
#                 batch["critical_labels"] = torch.stack(validated_labels, dim=0)
                
#                 # 🔧 最终验证
#                 final_shape = batch["critical_labels"].shape
#                 expected_shape = (len(instances), max_len)
                
#                 if final_shape != expected_shape:
#                     print(f"⚠️ 最终标签形状不匹配: {final_shape} vs 期望的 {expected_shape}")
#                     # 创建正确形状的默认标签
#                     batch["critical_labels"] = torch.zeros(len(instances), max_len, dtype=torch.long)
#                     # 使用简单模式填充
#                     for i in range(len(instances)):
#                         # 中间40%为关键时间段
#                         start_idx = int(max_len * 0.3)
#                         end_idx = int(max_len * 0.7)
#                         batch["critical_labels"][i, start_idx:end_idx] = 1
                
#                 # 🔧 验证数据类型和值范围
#                 if batch["critical_labels"].dtype != torch.long:
#                     batch["critical_labels"] = batch["critical_labels"].long()
                
#                 # 验证值在[0,1]范围内
#                 if torch.any(batch["critical_labels"] < 0) or torch.any(batch["critical_labels"] > 1):
#                     print(f"⚠️ 批次中有标签值超出范围: {batch['critical_labels'].unique()}")
#                     batch["critical_labels"] = torch.clamp(batch["critical_labels"], 0, 1)
#             except Exception as e:
#                 print(f"❌ 处理关键时间段标签时出错: {e}")
#                 import traceback
#                 traceback.print_exc()
                
#                 # 🔧 创建安全的默认标签
#                 print("🔧 使用安全的默认标签")
#                 batch_size = len(instances)
#                 default_seq_len = 64  # 默认序列长度
                
#                 # 尝试从actions获取真实长度
#                 if "actions" in batch and len(batch["actions"]) > 0:
#                     try:
#                         default_seq_len = batch["actions"].shape[1]
#                     except:
#                         pass
                
#                 # 创建默认标签张量
#                 batch["critical_labels"] = torch.zeros(batch_size, default_seq_len, dtype=torch.long)
                
#                 # 为每个序列设置简单的关键时间段模式
#                 for i in range(batch_size):
#                     # 策略：开始25%和结束25%为关键时间段
#                     quarter_len = default_seq_len // 4
#                     batch["critical_labels"][i, :quarter_len] = 1          # 开始阶段
#                     batch["critical_labels"][i, -quarter_len:] = 1         # 结束阶段
        
#         else:
#             # 🔧 如果完全没有标签，创建默认批次
#             print("⚠️ 没有任何关键时间段标签，创建默认批次")
#             batch_size = len(instances)
#             default_seq_len = 64
            
#             if "actions" in batch and len(batch["actions"]) > 0:
#                 try:
#                     default_seq_len = batch["actions"].shape[1]
#                 except:
#                     pass
            
#             batch["critical_labels"] = torch.zeros(batch_size, default_seq_len, dtype=torch.long)
#             # 简单策略：中间50%为关键时间段
#             start_critical = default_seq_len // 4
#             end_critical = 3 * default_seq_len // 4
#             batch["critical_labels"][:, start_critical:end_critical] = 1

#         return batch
    
    
    
    
    
    
    
    
    
# train/dataset.py - 更新版本，集成任务驱动的关键时间段标注

import traceback
import time
import os
import json
import math
import random
from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import transformers

from data.filelock import FileLock
from data.hdf5_vla_dataset import HDF5VLADataset
from train.image_corrupt import image_corrupt

# 🆕 导入任务驱动的关键时间段标注器
from data.critical_timestep_annotator import (
    TaskType, 
    create_silent_task_annotator,
    TaskDrivenCriticalTimestepAnnotator
)


def get_clean_item(chunk_dir):
    """Get indexes of clean items in a chunk."""
    dirty_bit = read_dirty_bit(chunk_dir)
    return np.where(1 - dirty_bit)[0].tolist()


def save_dirty_bit(chunk_dir, dirty_bit):
    """Save the dirty bit to the chunk directory."""
    time_stmp = time.time()
    while time.time() - time_stmp < 10.0:
        try:
            file_path = os.path.join(chunk_dir, "dirty_bit")
            lock = FileLock(file_path)
            lock.acquire_write_lock()
            with open(file_path, "wb") as file:
                file.write(dirty_bit.tobytes())
            lock.release_lock()
            return
        except KeyboardInterrupt:
            lock.release_lock()
            raise KeyboardInterrupt
        except BaseException:
            lock.release_lock()
            continue
    raise RuntimeError("Failed to save dirty bit.")


def read_dirty_bit(chunk_dir):
    """Read the dirty bit from the chunk directory."""
    time_stmp = time.time()
    while time.time() - time_stmp < 10.0:
        try:
            file_path = os.path.join(chunk_dir, "dirty_bit")
            lock = FileLock(file_path)
            lock.acquire_read_lock()
            with open(file_path, "rb") as file:
                dirty_bit = np.frombuffer(file.read(), dtype=np.uint8).copy()
            lock.release_lock()
            assert len(dirty_bit) > 0
            return dirty_bit
        except KeyboardInterrupt:
            lock.release_lock()
            raise KeyboardInterrupt
        except BaseException:
            lock.release_lock()
            continue
    raise RuntimeError("Failed to read dirty bit.")


class VLAConsumerDataset(Dataset):
    """A vision-language-action Dataset for supervised training.
    🆕 集成任务驱动的关键时间段标注机制
    """

    def __init__(
        self,
        model_config_path,
        config,
        tokenizer,
        image_processor,
        num_cameras,
        img_history_size,
        image_size=None,
        auto_adjust_image_brightness=False,
        image_aug=False,
        dataset_type="pretrain",
        cond_mask_prob=0.1,
        cam_ext_mask_prob=-1.0,
        state_noise_snr=None,
        use_hdf5=False,
        use_precomp_lang_embed=False,
        use_dinov2_features=False,
        use_depth_features=False,
        # 🆕 关键时间段标注相关参数
        task_type: int = 1,  # 1=抓取类, 2=点击类
        enable_critical_annotation: bool = True,
        critical_annotation_config: Dict = None,
    ):
        super(VLAConsumerDataset, self).__init__()

        # Load the control frequency for each dataset
        with open("configs/dataset_control_freq.json", "r") as fp:
            self.control_freq = json.load(fp)
        # Load the dataset names
        dataset_names_cfg = ("configs/pretrain_datasets.json"
                             if dataset_type == "pretrain" else "configs/finetune_datasets.json")
        with open(dataset_names_cfg, "r") as file:
            DATASET_NAMES = json.load(file)
        # Create the mapping between dataset name and id
        self.dataset_name2id = {name: i for i, name in enumerate(DATASET_NAMES)}
        self.dataset_id2name = {i: name for i, name in enumerate(DATASET_NAMES)}

        self.image_processor = image_processor
        self.model_config_path = model_config_path
        self.buffer_dir = config["buf_path"]
        self.num_chunks = config["buf_num_chunks"]
        self.chunk_size = config["buf_chunk_size"]
        self.tokenizer_max_length = config["tokenizer_max_length"]
        self.image_aspect_ratio = config["image_aspect_ratio"]
        self.state_noise_snr = state_noise_snr
        self.num_cameras = num_cameras
        self.img_history_size = img_history_size
        self.cond_mask_prob = cond_mask_prob
        self.cam_ext_mask_prob = cam_ext_mask_prob
        self.use_hdf5 = use_hdf5
        self.hdf5_dataset = None
        if use_hdf5:
            self.hdf5_dataset = HDF5VLADataset(self.model_config_path)
        self.use_precomp_lang_embed = use_precomp_lang_embed
        if use_precomp_lang_embed:
            self.empty_lang_embed = torch.load("data/empty_lang_embed.pt")
        
        # DINOv2相关配置
        self.use_dinov2_features = use_dinov2_features
        self.dinov2_image_size = 518
        
        # DepthAnythingV2相关配置
        self.use_depth_features = use_depth_features
        self.depth_image_size = 518

        # Load dataset stat
        with open("configs/dataset_stat.json", "r") as f:
            dataset_stat = json.load(f)
        self.dataset_stat = dataset_stat

        self.tokenizer = tokenizer
        self.image_size = image_size
        self.auto_adjust_image_brightness = auto_adjust_image_brightness
        self.image_aug = image_aug

        self.last_content = None
        self.last_meta = None
        
        # 🆕 关键时间段标注配置
        self.task_type = TaskType(task_type)
        self.enable_critical_annotation = enable_critical_annotation
        
        if enable_critical_annotation:
            # 设置默认配置
            default_config = {
                'relative_low_speed_ratio': 0.15,
                'min_deceleration_threshold': -0.0008,
                'gripper_close_delta_threshold': -0.01,
                'smooth': True,
                'verbose': False
            }
            
            # 合并用户配置
            if critical_annotation_config:
                default_config.update(critical_annotation_config)
            
            # 创建标注器
            self.critical_annotator = TaskDrivenCriticalTimestepAnnotator(
                task_type=self.task_type,
                **default_config
            )
            
            print(f"🎯 关键时间段标注器初始化:")
            print(f"   - 任务类型: {self.task_type.name} ({self.task_type.value})")
            print(f"   - 配置: {default_config}")
        else:
            self.critical_annotator = None
            print("⚠️  关键时间段标注已禁用")

    def get_dataset_name2id(self):
        return self.dataset_name2id

    def get_dataset_id2name(self):
        return self.dataset_id2name

    @staticmethod
    def pairwise(iterable):
        a = iter(iterable)
        return zip(a, a)

    @staticmethod
    def _load_data_from_chunk(chunk_dir, chunk_item_idx):
        time_stmp = time.time()
        while time.time() - time_stmp < 10.0:
            try:
                locks = []
                file_path = os.path.join(chunk_dir, f"json_content_{chunk_item_idx}.json")
                lock = FileLock(file_path)
                locks.append(lock)
                lock.acquire_read_lock()
                with open(file_path, "r") as file:
                    json_content = json.load(file)
                lock.release_lock()
                file_path = os.path.join(chunk_dir, f"sample_{chunk_item_idx}.npz")
                lock = FileLock(file_path)
                locks.append(lock)
                lock.acquire_read_lock()
                with open(file_path, "rb") as file:
                    sample_dict = np.load(file)
                    meta = tuple(sample_dict.values())
                lock.release_lock()
                return json_content, meta
            except KeyboardInterrupt:
                for lock in locks:
                    lock.release_lock()
                raise KeyboardInterrupt
            except BaseException:
                for lock in locks:
                    lock.release_lock()
                continue
        raise RuntimeError("Failed to load sample.")

    def __len__(self) -> int:
        if self.use_hdf5:
            return len(self.hdf5_dataset)
        else:
            return self.num_chunks * self.chunk_size

    def _safe_load(self, index):
        read_chunk_item_indices = []
        # Start searching from a random chunk
        read_chunk_idx = index // self.chunk_size
        while len(read_chunk_item_indices) == 0:
            read_chunk_dir = os.path.join(self.buffer_dir, f"chunk_{read_chunk_idx}")
            try:
                read_chunk_item_indices = get_clean_item(read_chunk_dir)
            except BaseException as e:
                print("Error catched when searching a clean chunk:", e)
                traceback.print_exc()
                read_chunk_item_indices = []
            read_chunk_idx = (read_chunk_idx + 1) % self.num_chunks

        random_item_index = index % len(read_chunk_item_indices)
        read_chunk_item_index = read_chunk_item_indices[random_item_index]

        # Modify the dirty bit
        try:
            dirty_bit = read_dirty_bit(read_chunk_dir)
            dirty_bit[read_chunk_item_index] = 1
            save_dirty_bit(read_chunk_dir, dirty_bit)
        except BaseException as e:
            print("Error catched when modifying the dirty bit:", e)
            traceback.print_exc()

        # load the sample
        try:
            content, meta = self._load_data_from_chunk(read_chunk_dir, read_chunk_item_index)
            self.last_content, self.last_meta = content, meta
        except BaseException as e:
            print("Error catched when loading sample:", e)
            traceback.print_exc()
            content, meta = self.last_content, self.last_meta

        return (content, *meta)

    def _generate_critical_labels(self, qpos_trajectory, action_horizon=64):
        """
        🆕 生成关键时间段标签
        🔧 修复版本：确保输出形状正确
        
        Args:
            qpos_trajectory: (T, 14) numpy数组，关节角度轨迹
            action_horizon: 动作预测的时间范围
            
        Returns:
            critical_labels: (action_horizon,) torch tensor，关键时间段标签
        """
        try:
            if self.critical_annotator is None:
                # 如果没有标注器，返回合理的默认标签
                # 使用简单的启发式：前25%和后25%为关键时间段
                critical_labels = np.zeros(action_horizon, dtype=np.int64)
                
                # 开始阶段（前25%）和结束阶段（后25%）标记为关键
                start_critical = int(action_horizon * 0.25)
                end_critical = int(action_horizon * 0.75)
                
                critical_labels[:start_critical] = 1  # 开始阶段
                critical_labels[end_critical:] = 1    # 结束阶段
                
                return torch.from_numpy(critical_labels).long()
            
            # 有标注器时使用标注器
            if qpos_trajectory is None or len(qpos_trajectory) == 0:
                print("⚠️ qpos_trajectory为空，使用默认标签")
                # 返回交替模式的标签
                critical_labels = np.zeros(action_horizon, dtype=np.int64)
                critical_labels[action_horizon//4:action_horizon//2] = 1  # 中间一段为关键
                return torch.from_numpy(critical_labels).long()
            
            # 使用标注器生成标签
            critical_labels, analysis_info = self.critical_annotator.annotate(qpos_trajectory)
            
            # 🔧 修复：确保标签是正确的数据类型
            if not isinstance(critical_labels, np.ndarray):
                critical_labels = np.array(critical_labels, dtype=np.int64)
            else:
                critical_labels = critical_labels.astype(np.int64)
            
            # 🔧 修复：处理长度不匹配的问题
            # if len(critical_labels)  # 🔴 视觉融合模式不需要 > action_horizon:
                # 如果标注结果太长，截取前action_horizon个
#                critical_labels = critical_labels[:action_horizon]
            el# if len(critical_labels)  # 🔴 视觉融合模式不需要 < action_horizon:
                # 如果标注结果太短，使用策略填充
#                padding_len = action_horizon - len(critical_labels)
                
#                if len(critical_labels) > 0:
                    # 使用最后一个值填充
#                    last_value = critical_labels[-1]
#                    padding = np.full(padding_len, last_value, dtype=np.int64)
#                else:
                    # 如果完全没有标注，使用0填充
#                    padding = np.zeros(padding_len, dtype=np.int64)
                
#                critical_labels = np.concatenate([critical_labels, padding])
            
            # 🔧 修复：验证输出形状
            assert len(critical_labels) == action_horizon, f"标签长度不匹配: {len(critical_labels)} vs {action_horizon}"
            assert critical_labels.dtype == np.int64, f"标签类型错误: {critical_labels.dtype}"
            
            # 转换为torch tensor
            critical_labels_tensor = torch.from_numpy(critical_labels).long()
            
            # 🔧 验证值的范围
            if torch.any(critical_labels_tensor < 0) or torch.any(critical_labels_tensor > 1):
                print(f"⚠️ 标签值超出范围 [0,1]: {critical_labels_tensor.unique()}")
                # 将所有非0值转换为1
                critical_labels_tensor = torch.clamp(critical_labels_tensor, 0, 1)
            
            return critical_labels_tensor
            
        except Exception as e:
            print(f"⚠️ 关键时间段标注失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 🔧 降级方案：返回安全的默认标签
            print("🔧 使用安全的默认标签")
            critical_labels = np.zeros(action_horizon, dtype=np.int64)
            
            # 简单策略：中间30%为关键时间段
            start_idx = int(action_horizon * 0.35)
            end_idx = int(action_horizon * 0.65)
            critical_labels[start_idx:end_idx] = 1
            
            return torch.from_numpy(critical_labels).long()

    def __getitem__(self, index):
        # For robustness, we will try to load the data until we succeed
        while True:
            data_dict = None
            try:
                if self.use_hdf5:
                    res = self.hdf5_dataset.get_item()
                    content = res["meta"]
                    states = res["state"]
                    actions = res["actions"]
                    state_elem_mask = res["state_indicator"]
                    image_metas = [
                        res["cam_high"],
                        res["cam_high_mask"],
                        res["cam_right_wrist"],
                        res["cam_right_wrist_mask"],
                        res["cam_left_wrist"],
                        res["cam_left_wrist_mask"],
                    ]
                    state_std = res["state_std"]
                    state_mean = res["state_mean"]
                    state_norm = res["state_norm"]
                    
                    # 🆕 获取qpos轨迹用于关键时间段分析
                    qpos_trajectory = res.get("qpos_trajectory", None)
                else:
                    (
                        content,
                        _,
                        states,
                        _,
                        actions,
                        _,
                        state_elem_mask,
                        *image_metas,
                        state_std,
                        state_mean,
                        state_norm,
                    ) = self._safe_load(index)
                    qpos_trajectory = None  # 非HDF5模式暂不支持

                data_dict = {}
                data_dict["dataset_name"] = content["dataset_name"]
                data_dict["data_idx"] = self.dataset_name2id[data_dict["dataset_name"]]
                data_dict["ctrl_freq"] = (self.control_freq[data_dict["dataset_name"]]
                                          if random.random() > self.cond_mask_prob else 0)

                if self.state_noise_snr is not None:
                    states += np.random.normal(
                        0.0,
                        state_std / np.sqrt(10**(self.state_noise_snr / 10)),
                        states.shape,
                    )
                ds_state_mean = np.array(self.dataset_stat[data_dict["dataset_name"]]["state_mean"])
                ds_state_mean = np.tile(ds_state_mean[None], (states.shape[0], 1))
                
                data_dict["states"] = (states if random.random() > self.cond_mask_prob else ds_state_mean)
                data_dict["actions"] = actions
                data_dict["state_elem_mask"] = (state_elem_mask if random.random() > self.cond_mask_prob else
                                                np.zeros_like(state_elem_mask))
                data_dict["state_norm"] = state_norm

                # # 🆕 生成关键时间段标签 - 修复版本
                # action_horizon = actions.shape[0]  # 动作序列长度
                
                # if self.enable_critical_annotation:
                #     try:
                #         critical_labels = self._generate_critical_labels(qpos_trajectory, action_horizon)
                        
                #         # 🔧 额外验证
                #         if critical_labels.shape[0] != action_horizon:
                #             print(f"⚠️ 样本 {index}: 标签形状不匹配 {critical_labels.shape[0]} vs {action_horizon}")
                #             # 重新生成正确形状的标签
                #             critical_labels = torch.zeros(action_horizon, dtype=torch.long)
                #             critical_labels[action_horizon//3:2*action_horizon//3] = 1
                        
                #         data_dict["critical_labels"] = critical_labels
                        
                #         # 可选：记录标注统计信息（用于调试）
                #         if hasattr(self.critical_annotator, 'verbose') and self.critical_annotator and self.critical_annotator.verbose:
                #             critical_ratio = critical_labels.float().mean().item()
                #             print(f"📊 样本 {index}: 关键时间段比例 = {critical_ratio:.3f}, 形状 = {critical_labels.shape}")
                            
                #     except Exception as e:
                #         print(f"⚠️ 样本 {index} 关键时间段标注失败: {e}")
                #         # 🔧 失败时使用安全的默认标签
                #         critical_labels = torch.zeros(action_horizon, dtype=torch.long)
                #         # 简单模式：每4个时间步有1个关键时间段
                #         critical_labels[::4] = 1
                #         data_dict["critical_labels"] = critical_labels
                # else:
                #     # 如果没有启用标注，使用简单的默认模式
                #     critical_labels = torch.zeros(action_horizon, dtype=torch.long)
                #     # 使用启发式：中间50%为关键时间段
                #     start_critical = action_horizon // 4
                #     end_critical = 3 * action_horizon // 4
                #     critical_labels[start_critical:end_critical] = 1
                #     data_dict["critical_labels"] = critical_labels

                # # 🔧 最终验证critical_labels
                # final_critical_labels = data_dict["critical_labels"]
                # if not isinstance(final_critical_labels, torch.Tensor):
                #     final_critical_labels = torch.tensor(final_critical_labels, dtype=torch.long)
                #     data_dict["critical_labels"] = final_critical_labels
                
                # # 验证形状和类型
                # assert final_critical_labels.shape == (action_horizon,), f"标签形状错误: {final_critical_labels.shape} vs ({action_horizon},)"
                # assert final_critical_labels.dtype == torch.long, f"标签类型错误: {final_critical_labels.dtype}"
                # assert torch.all(final_critical_labels >= 0) and torch.all(final_critical_labels <= 1), f"标签值超出范围: {final_critical_labels.unique()}"

                # ... 继续处理其他数据（图像等）
                # Background image for padding/masking
                background_color = np.array(
                    [int(x * 255) for x in self.image_processor.image_mean],
                    dtype=np.uint8,
                ).reshape(1, 1, 3)
                background_image = (np.ones(
                    (
                        self.image_processor.size["height"],
                        self.image_processor.size["width"],
                        3,
                    ),
                    dtype=np.uint8,
                ) * background_color)

                image_metas = list(self.pairwise(image_metas))
                mask_probs = [self.cond_mask_prob] * self.num_cameras
                if self.cam_ext_mask_prob >= 0.0:
                    mask_probs[0] = self.cam_ext_mask_prob
                
                rearranged_images = []
                for i in range(self.img_history_size):
                    for j in range(self.num_cameras):
                        images, image_mask = image_metas[j]
                        image, valid = images[i], image_mask[i]
                        if (valid and (math.prod(image.shape) > 0) and (random.random() > mask_probs[j])):
                            rearranged_images.append((image, True))
                        else:
                            rearranged_images.append((background_image.copy(), False))

                # 为DINOv2准备单独的图像
                if self.use_dinov2_features:
                    camera_idx = 0
                    frame_idx = self.img_history_size - 1
                    
                    if camera_idx >= len(image_metas):
                        raise ValueError(f"相机索引 {camera_idx} 超出范围")
                    
                    images, image_mask = image_metas[camera_idx]
                    image, valid = images[frame_idx], image_mask[frame_idx]
                    
                    if not valid or math.prod(image.shape) <= 0:
                        raise ValueError(f"DINOv2图像无效")
                    
                    # 预处理DINOv2图像
                    pil_image = Image.fromarray(image)
                    pil_image = pil_image.resize((self.dinov2_image_size, self.dinov2_image_size), Image.BILINEAR)
                    image_array = np.array(pil_image).astype(np.float32) / 255.0
                    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
                    
                    # ImageNet标准化
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    image_tensor = (image_tensor - mean) / std
                    
                    data_dict["dinov2_images"] = image_tensor.unsqueeze(0)

                # 为DepthAnythingV2准备深度图像
                if self.use_depth_features:
                    camera_idx = 0
                    frame_idx = self.img_history_size - 1
                    
                    if camera_idx >= len(image_metas):
                        raise ValueError(f"深度编码器相机索引 {camera_idx} 超出范围")
                    
                    images, image_mask = image_metas[camera_idx]
                    image, valid = images[frame_idx], image_mask[frame_idx]
                    
                    if not valid or math.prod(image.shape) <= 0:
                        raise ValueError(f"深度编码器图像无效")
                    
                    # DepthAnythingV2预处理
                    pil_image = Image.fromarray(image)
                    pil_image = pil_image.resize((self.depth_image_size, self.depth_image_size), Image.BILINEAR)
                    image_array = np.array(pil_image).astype(np.float32) / 255.0
                    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
                    
                    # ImageNet标准化
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    image_tensor = (image_tensor - mean) / std
                    
                    data_dict["depth_images"] = image_tensor.unsqueeze(0)

                # 处理原始图像（SigLIP编码用）
                preprocessed_images = []
                processor = self.image_processor
                for image, valid in rearranged_images:
                    image = Image.fromarray(image)
                    if self.image_size is not None:
                        image = transforms.Resize(self.image_size)(image)

                    if valid and self.auto_adjust_image_brightness:
                        pixel_values = list(image.getdata())
                        average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                        if average_brightness <= 0.15:
                            image = transforms.ColorJitter(brightness=(1.75, 1.75))(image)

                    # 图像增强
                    if valid and self.image_aug and (random.random() > 0.5):
                        aug_type = random.choice(["corrput_only", "color_only", "both"])
                        if aug_type != "corrput_only":
                            image = transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5,
                                                           hue=0.03)(image)
                        if aug_type != "color_only":
                            image = image_corrupt(image)

                    if self.image_aspect_ratio == "pad":
                        def expand2square(pil_img, background_color):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result

                        image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
                    image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
                    preprocessed_images.append(image)
                data_dict["images"] = preprocessed_images

                # 处理语言输入
                if self.use_precomp_lang_embed:
                    if content["instruction"][-1] == ".":
                        content["instruction"] = content["instruction"][:-1]
                    data_dict["lang_embed"] = (torch.load(content["instruction"])
                                               if random.random() > self.cond_mask_prob else self.empty_lang_embed)
                else:
                    instruction = (content["instruction"] if random.random() > self.cond_mask_prob else "")
                    data_dict["input_ids"] = self.tokenizer(
                        instruction,
                        return_tensors="pt",
                        padding="longest",
                        truncation=False,
                    ).input_ids[0]

                    assert (
                        len(data_dict["input_ids"]) <= self.tokenizer_max_length
                    ), f"Instruction length {len(data_dict['input_ids'])} exceeds the maximum length {self.tokenizer_max_length}."

                # 转换numpy数组为tensor
                for k, v in data_dict.items():
                    if isinstance(v, np.ndarray):
                        data_dict[k] = torch.from_numpy(v)

                # 🔧 最终检查所有tensor
                for k, v in data_dict.items():
                    assert not isinstance(v, np.ndarray), f"key: {k}, value: {v}"

                # 🔧 最终验证critical_labels（再次检查）
                if "critical_labels" in data_dict:
                    labels = data_dict["critical_labels"]
                    if not isinstance(labels, torch.Tensor):
                        print(f"⚠️ critical_labels不是tensor: {type(labels)}")
                        data_dict["critical_labels"] = torch.tensor(labels, dtype=torch.long)
                    elif labels.dtype != torch.long:
                        print(f"⚠️ critical_labels类型错误: {labels.dtype}")
                        data_dict["critical_labels"] = labels.long()

                return data_dict
                
            except BaseException as e:
                if data_dict is not None:
                    print(f"Error catched when processing sample from {data_dict.get('dataset_name')}: {e}")
                else:
                    print(f"Error catched when processing sample: {e}")
                traceback.print_exc()
                index = (index + 1) % len(self)

# 修复 train/dataset.py 中的 DataCollatorForVLAConsumerDataset 部分

class DataCollatorForVLAConsumerDataset(object):
    """Collate examples for supervised training.
    🆕 支持关键时间段标签的数据收集
    🔧 修复版本：增强错误处理和形状验证
    """

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = {
            "states": [],
            "actions": [],
            "state_elem_mask": [],
            "state_norm": [],
            "images": [],
            "data_indices": [],
            "ctrl_freqs": [],
        }
        
        # 视觉特征收集
        dinov2_images = []
        depth_images = []
        
        # # 🆕 关键时间段标签收集 - 修复版本
        # critical_labels = []
        
        # 语言特征收集
        input_ids = []
        lang_embeds = []
        lang_embed_lens = []

        for idx, instance in enumerate(instances):
            try:
                # 收集基础数据
                keys_to_check = ["states", "actions", "state_elem_mask", "state_norm"]
                for key in keys_to_check:
                    if isinstance(instance[key], torch.Tensor):
                        item = instance[key]
                    else:
                        item = torch.from_numpy(instance[key])
                    batch[key].append(item)

                # 收集语言数据
                if "input_ids" in instance:
                    input_ids.append(instance["input_ids"])
                else:
                    lang_embeds.append(instance["lang_embed"])
                    lang_embed_lens.append(instance["lang_embed"].shape[0])

                # 收集图像和其他数据
                batch["images"].append(torch.stack(instance["images"], dim=0))
                batch["data_indices"].append(instance["data_idx"])
                batch["ctrl_freqs"].append(instance["ctrl_freq"])
                
                # 收集DINOv2图像
                if "dinov2_images" in instance:
                    dinov2_images.append(instance["dinov2_images"])
                
                # 收集DepthAnythingV2图像
                if "depth_images" in instance:
                    depth_images.append(instance["depth_images"])
                
                # # 🆕 收集关键时间段标签 - 修复版本
                # if "critical_labels" in instance:
                #     labels = instance["critical_labels"]
                    
                #     # 🔧 验证和转换标签
                #     if not isinstance(labels, torch.Tensor):
                #         labels = torch.tensor(labels, dtype=torch.long)
                #     elif labels.dtype != torch.long:
                #         labels = labels.long()
                    
                #     # 🔧 验证标签值范围
                #     if torch.any(labels < 0) or torch.any(labels > 1):
                #         print(f"⚠️ 实例 {idx}: 标签值超出范围 [0,1]: {labels.unique()}")
                #         labels = torch.clamp(labels, 0, 1)
                    
                #     # 🔧 验证形状
                #     if labels.dim() != 1:
                #         print(f"⚠️ 实例 {idx}: 标签维度错误: {labels.shape}，应该是1D")
                #         if labels.numel() > 0:
                #             labels = labels.flatten()
                #         else:
                #             # 如果为空，创建默认标签
                #             action_len = instance["actions"].shape[0]
                #             labels = torch.zeros(action_len, dtype=torch.long)
                    
                #     critical_labels.append(labels)
                # else:
                #     # 如果没有标签，创建默认标签
                #     action_len = instance["actions"].shape[0]
                #     default_labels = torch.zeros(action_len, dtype=torch.long)
                #     # 简单启发式：中间部分为关键时间段
                #     start_critical = action_len // 4
                #     end_critical = 3 * action_len // 4
                #     default_labels[start_critical:end_critical] = 1
                #     critical_labels.append(default_labels)
                    
            except Exception as e:
                print(f"❌ 处理实例 {idx} 时出错: {e}")
                import traceback
                traceback.print_exc()
                
                # 🔧 创建降级数据
                action_len = 64  # 默认长度
                if "actions" in instance:
                    try:
                        action_len = instance["actions"].shape[0]
                    except:
                        pass
                
                # 添加默认的critical_labels
                default_labels = torch.zeros(action_len, dtype=torch.long)
                default_labels[action_len//3:2*action_len//3] = 1  # 中间1/3为关键
                critical_labels.append(default_labels)
                
                # 继续处理其他必要字段...
                continue

        # 🔧 验证所有实例都有标签
        # if len(critical_labels)  # 🔴 视觉融合模式不需要 != len(instances):
#            print(f"⚠️ 标签数量不匹配: {len(critical_labels)} vs {len(instances)}")
            # 补充缺失的标签
#            while len(critical_labels) < len(instances):
#                default_labels = torch.zeros(64, dtype=torch.long)
#                critical_labels.append(default_labels)

        # 堆叠基础数据
        keys_to_stack = ["states", "actions", "state_elem_mask", "state_norm", "images"]
        for key in keys_to_stack:
            batch[key] = torch.stack(batch[key], dim=0)

        batch["ctrl_freqs"] = torch.tensor(batch["ctrl_freqs"])

        # 处理语言数据
        if len(input_ids) > 0:
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids,
                                                        batch_first=True,
                                                        padding_value=self.tokenizer.pad_token_id)
            batch["input_ids"] = input_ids
            batch["lang_attn_mask"] = input_ids.ne(self.tokenizer.pad_token_id)
        else:
            lang_embeds = torch.nn.utils.rnn.pad_sequence(lang_embeds, batch_first=True, padding_value=0)
            input_lang_attn_mask = torch.zeros(lang_embeds.shape[0], lang_embeds.shape[1], dtype=torch.bool)
            for i, l in enumerate(lang_embed_lens):
                input_lang_attn_mask[i, :l] = True
            batch["lang_embeds"] = lang_embeds
            batch["lang_attn_mask"] = input_lang_attn_mask
        
        # 堆叠DINOv2图像
        if len(dinov2_images) > 0:
            batch["dinov2_images"] = torch.stack(dinov2_images, dim=0)
        
        # 堆叠DepthAnythingV2图像
        if len(depth_images) > 0:
            batch["depth_images"] = torch.stack(depth_images, dim=0)
        
        # # 🆕 堆叠关键时间段标签 - 修复版本
        # if len(critical_labels) > 0:
        #     try:
        #         # 🔧 找到最大长度进行填充
        #         max_len = max(labels.shape[0] for labels in critical_labels)
                
        #         # 🔧 验证所有标签都是1D
        #         validated_labels = []
        #         for i, labels in enumerate(critical_labels):
        #             if labels.dim() != 1:
        #                 print(f"⚠️ 标签 {i} 维度错误: {labels.shape}")
        #                 labels = labels.flatten()
                    
        #             # 🔧 填充到统一长度
        #             if labels.shape[0] < max_len:
        #                 padding_len = max_len - labels.shape[0]
        #                 # 使用0填充（非关键时间段）
        #                 padding = torch.zeros(padding_len, dtype=labels.dtype)
        #                 labels = torch.cat([labels, padding], dim=0)
        #             elif labels.shape[0] > max_len:
        #                 # 截断到最大长度
        #                 labels = labels[:max_len]
                    
        #             validated_labels.append(labels)
                
        #         # 🔧 堆叠标签
        #         batch["critical_labels"] = torch.stack(validated_labels, dim=0)
                
        #         # 🔧 最终验证
        #         final_shape = batch["critical_labels"].shape
        #         expected_shape = (len(instances), max_len)
                
        #         if final_shape != expected_shape:
        #             print(f"⚠️ 最终标签形状不匹配: {final_shape} vs 期望的 {expected_shape}")
        #             # 创建正确形状的默认标签
        #             batch["critical_labels"] = torch.zeros(len(instances), max_len, dtype=torch.long)
        #             # 使用简单模式填充
        #             for i in range(len(instances)):
        #                 # 中间40%为关键时间段
        #                 start_idx = int(max_len * 0.3)
        #                 end_idx = int(max_len * 0.7)
        #                 batch["critical_labels"][i, start_idx:end_idx] = 1
                
        #         # 🔧 验证数据类型和值范围
        #         if batch["critical_labels"].dtype != torch.long:
        #             batch["critical_labels"] = batch["critical_labels"].long()
                
        #         # 验证值在[0,1]范围内
        #         if torch.any(batch["critical_labels"] < 0) or torch.any(batch["critical_labels"] > 1):
        #             print(f"⚠️ 批次中有标签值超出范围: {batch['critical_labels'].unique()}")
        #             batch["critical_labels"] = torch.clamp(batch["critical_labels"], 0, 1)
        #     except Exception as e:
        #         print(f"❌ 处理关键时间段标签时出错: {e}")
        #         import traceback
        #         traceback.print_exc()
                
        #         # 🔧 创建安全的默认标签
        #         print("🔧 使用安全的默认标签")
        #         batch_size = len(instances)
        #         default_seq_len = 64  # 默认序列长度
                
        #         # 尝试从actions获取真实长度
        #         if "actions" in batch and len(batch["actions"]) > 0:
        #             try:
        #                 default_seq_len = batch["actions"].shape[1]
        #             except:
        #                 pass
                
        #         # 创建默认标签张量
        #         batch["critical_labels"] = torch.zeros(batch_size, default_seq_len, dtype=torch.long)
                
        #         # 为每个序列设置简单的关键时间段模式
        #         for i in range(batch_size):
        #             # 策略：开始25%和结束25%为关键时间段
        #             quarter_len = default_seq_len // 4
        #             batch["critical_labels"][i, :quarter_len] = 1          # 开始阶段
        #             batch["critical_labels"][i, -quarter_len:] = 1         # 结束阶段
        
        # else:
        #     # 🔧 如果完全没有标签，创建默认批次
        #     print("⚠️ 没有任何关键时间段标签，创建默认批次")
        #     batch_size = len(instances)
        #     default_seq_len = 64
            
        #     if "actions" in batch and len(batch["actions"]) > 0:
        #         try:
        #             default_seq_len = batch["actions"].shape[1]
        #         except:
        #             pass
            
        #     batch["critical_labels"] = torch.zeros(batch_size, default_seq_len, dtype=torch.long)
        #     # 简单策略：中间50%为关键时间段
        #     start_critical = default_seq_len // 4
        #     end_critical = 3 * default_seq_len // 4
        #     batch["critical_labels"][:, start_critical:end_critical] = 1

        return batch