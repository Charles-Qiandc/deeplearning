#!/usr/bin/env python3
"""
检查HDF5数据文件结构的脚本
"""
import h5py
import numpy as np
import os
import fnmatch

def inspect_hdf5_file(file_path):
    """检查单个HDF5文件的结构"""
    print(f"检查文件: {file_path}")
    print("=" * 80)
    
    try:
        with h5py.File(file_path, 'r') as f:
            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"数据集: {name}")
                    print(f"  形状: {obj.shape}")
                    print(f"  数据类型: {obj.dtype}")
                    if len(obj.shape) <= 2 and obj.size < 100:  # 只打印小数据集的内容
                        print(f"  数据: {obj[...]}")
                    print()
                elif isinstance(obj, h5py.Group):
                    print(f"组: {name}")
            
            print("HDF5文件结构:")
            print("-" * 40)
            f.visititems(print_structure)
            
            # 重点检查qpos数据
            if 'observations' in f and 'qpos' in f['observations']:
                qpos = f['observations']['qpos']
                print("\n重点分析 qpos:")
                print(f"  形状: {qpos.shape}")
                print(f"  数据类型: {qpos.dtype}")
                print(f"  第一步数据: {qpos[0] if len(qpos) > 0 else 'empty'}")
                print(f"  最后一步数据: {qpos[-1] if len(qpos) > 0 else 'empty'}")
                
                # 检查维度信息
                if 'left_arm_dim' in f['observations']:
                    left_dim = f['observations']['left_arm_dim'][...]
                    print(f"  左臂维度: {left_dim}")
                if 'right_arm_dim' in f['observations']:
                    right_dim = f['observations']['right_arm_dim'][...]
                    print(f"  右臂维度: {right_dim}")
                    
            # 检查action数据
            if 'action' in f:
                action = f['action']
                print(f"\naction数据:")
                print(f"  形状: {action.shape}")
                print(f"  数据类型: {action.dtype}")
                print(f"  第一步数据: {action[0] if len(action) > 0 else 'empty'}")
                
    except Exception as e:
        print(f"读取文件时出错: {e}")

def find_and_inspect_hdf5_files(data_dir, max_files=3):
    """在数据目录中查找并检查HDF5文件"""
    print(f"在目录 {data_dir} 中查找HDF5文件...")
    
    hdf5_files = []
    for root, dirs, files in os.walk(data_dir):
        for filename in fnmatch.filter(files, "*.hdf5"):
            hdf5_files.append(os.path.join(root, filename))
            if len(hdf5_files) >= max_files:
                break
        if len(hdf5_files) >= max_files:
            break
    
    if not hdf5_files:
        print("未找到HDF5文件!")
        return
    
    print(f"找到 {len(hdf5_files)} 个HDF5文件，检查前 {min(max_files, len(hdf5_files))} 个:")
    print()
    
    for i, file_path in enumerate(hdf5_files[:max_files]):
        print(f"\n{'='*20} 文件 {i+1} {'='*20}")
        inspect_hdf5_file(file_path)

if __name__ == "__main__":
    # 根据你的配置文件路径调整
    # 你需要修改这个路径为你的实际数据路径
    data_directories = ["/home/deng_xiang/qian_daichao/RoboTwin/policy/RDT_repa/processed_data/adjust_bottle"

    ]
    
    found_data = False
    for data_dir in data_directories:
        if os.path.exists(data_dir):
            print(f"使用数据目录: {data_dir}")
            find_and_inspect_hdf5_files(data_dir, max_files=2)
            found_data = True
            break
    
    if not found_data:
        print("未找到数据目录，请手动指定HDF5文件路径:")
        print("python inspect_hdf5_structure.py")
        print("然后修改脚本中的文件路径")
        
        # 如果你知道具体的HDF5文件路径，可以直接检查
        # inspect_hdf5_file("/path/to/your/file.hdf5")