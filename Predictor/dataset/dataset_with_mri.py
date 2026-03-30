"""
扩展数据集：支持加载原始MRI用于pixel loss
"""
import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
from pathlib import Path
from .dataset_glioma_all_pairs_text import GliomaAllPairsTextDataset, Config


class GliomaDatasetWithMRI(GliomaAllPairsTextDataset):
    """
    扩展数据集，支持加载原始MRI
    
    Args:
        cfg: 原有的Config
        mri_data_root: MRI数据根目录，如 "./datasets/MU-Glioma-Post"
        load_mri: 是否加载MRI（训练pixel loss时True，否则False）
        target_size: MRI目标尺寸 (D, H, W)，用于resize
        normalize: 是否归一化到[-1, 1]
    """
    def __init__(
        self,
        cfg: Config,
        mri_data_root: str = "./datasets/MU-Glioma-Post",
        load_mri: bool = False,
        target_size: tuple = (64, 64, 64),
        normalize: bool = True,
        **kwargs
    ):
        super().__init__(cfg, **kwargs)
        
        self.mri_data_root = Path(mri_data_root)
        self.load_mri = load_mri
        self.target_size = target_size
        self.normalize = normalize
        self.modalities = ["t1c", "t1n", "t2w", "t2f"]
        
        if load_mri:
            print(f"✓ MRI loading enabled (target size: {target_size})")
    
    def _parse_timepoint_number(self, tp_str):
        """从'T1', 'T2'等提取数字"""
        import re
        match = re.search(r'(\d+)', str(tp_str))
        return match.group(1) if match else tp_str
    
    def load_single_mri_volume(self, patient_id, timepoint):
        """
        加载单个时间点的4个MRI模态
        
        Args:
            patient_id: 如 "PatientID_0038"
            timepoint: 如 "T1"
        
        Returns:
            mri_volume: [4, D, H, W] tensor，已归一化到[-1,1]
        """
        tp_num = self._parse_timepoint_number(timepoint)
        tp_dir = self.mri_data_root / patient_id / f"Timepoint_{tp_num}"
        
        volumes = []
        for mod in self.modalities:
            nii_path = tp_dir / f"{patient_id}_Timepoint_{tp_num}_brain_{mod}.nii.gz"
            
            if not nii_path.exists():
                # 如果文件不存在，返回全零
                volumes.append(np.zeros(self.target_size, dtype=np.float32))
                continue
            
            try:
                # 加载NIfTI
                img = nib.load(str(nii_path))
                volume = img.get_fdata().astype(np.float32)
                
                # Resize到目标尺寸
                volume_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
                volume_resized = F.interpolate(
                    volume_tensor,
                    size=self.target_size,
                    mode='trilinear',
                    align_corners=False
                )
                volume_resized = volume_resized.squeeze(0).squeeze(0).numpy()
                
                # 归一化到[-1, 1]
                if self.normalize:
                    # 使用0.5和99.5分位数归一化（剔除异常值）
                    valid_mask = volume_resized > 0
                    if valid_mask.sum() > 0:
                        p_low, p_high = np.percentile(
                            volume_resized[valid_mask], [0.5, 99.5]
                        )
                        volume_resized = np.clip(volume_resized, p_low, p_high)
                        volume_resized = (volume_resized - p_low) / (p_high - p_low + 1e-8)  # [0,1]
                        volume_resized = 2 * volume_resized - 1  # [-1,1]
                
                volumes.append(volume_resized)
                
            except Exception as e:
                print(f"Warning: Failed to load {nii_path}: {e}")
                volumes.append(np.zeros(self.target_size, dtype=np.float32))
        
        # 堆叠为[4, D, H, W]
        mri_tensor = torch.from_numpy(np.stack(volumes, axis=0))
        return mri_tensor
    
    def __getitem__(self, idx):
        # 先调用父类获取基础数据
        sample = super().__getitem__(idx)
        
        # 如果需要加载MRI
        if self.load_mri:
            try:
                # 获取patient ID和timepoint
                meta = sample['meta']
                patient_id = meta['pid']
                pre_tp = meta['pre_tp']
                post_tp = meta['post_tp']
                
                # 加载pre和post的MRI
                pre_mri = self.load_single_mri_volume(patient_id, pre_tp)
                post_mri = self.load_single_mri_volume(patient_id, post_tp)
                
                sample['pre_mri'] = pre_mri
                sample['post_mri'] = post_mri
                
            except Exception as e:
                print(f"Warning: Failed to load MRI for sample {idx}: {e}")
                # 如果加载失败，返回全零tensor
                sample['pre_mri'] = torch.zeros(4, *self.target_size)
                sample['post_mri'] = torch.zeros(4, *self.target_size)
        
        return sample
    
    @staticmethod
    def collate_fn_with_mri(batch):
        """扩展的collate函数，支持MRI tensor"""
        from typing import List, Dict, Any
        
        keys = batch[0].keys()
        collated_batch = {k: [] for k in keys}
        
        for item in batch:
            for key, value in item.items():
                collated_batch[key].append(value)
        
        # 堆叠所有tensor类型的字段
        tensor_keys = ['pre_latent', 'post_latent', 'survival_time', 
                      'event_indicator', 'time_delta']
        
        # 如果有MRI，也堆叠它们
        if 'pre_mri' in collated_batch:
            tensor_keys.extend(['pre_mri', 'post_mri'])
        
        for key in tensor_keys:
            if key in collated_batch:
                collated_batch[key] = torch.stack(collated_batch[key])
        
        return collated_batch


# 测试代码
if __name__ == "__main__":
    print("=" * 80)
    print("Testing Dataset with MRI Loading")
    print("=" * 80)
    
    cfg = Config(
        timeline_json="./dataset/MU_Glioma_Post/clinical_latest.json",
        features_csv="./dataset/MU_Glioma_Post/features_output.csv",
        take_dims=767
    )
    
    # 测试不加载MRI
    print("\n1. Testing without MRI loading...")
    ds_no_mri = GliomaDatasetWithMRI(cfg, load_mri=False)
    sample = ds_no_mri[0]
    print(f"   Sample keys: {sample.keys()}")
    print(f"   pre_latent shape: {sample['pre_latent'].shape}")
    assert 'pre_mri' not in sample, "MRI should not be loaded"
    print("   ✓ No MRI mode works!")
    
    # 测试加载MRI
    print("\n2. Testing with MRI loading...")
    ds_with_mri = GliomaDatasetWithMRI(
        cfg, 
        load_mri=True,
        target_size=(64, 64, 64),
        mri_data_root="./datasets/MU-Glioma-Post"
    )
    
    if len(ds_with_mri) > 0:
        sample = ds_with_mri[0]
        print(f"   Sample keys: {sample.keys()}")
        
        if 'pre_mri' in sample:
            print(f"   pre_mri shape: {sample['pre_mri'].shape}")
            print(f"   post_mri shape: {sample['post_mri'].shape}")
            print(f"   pre_mri range: [{sample['pre_mri'].min():.3f}, {sample['pre_mri'].max():.3f}]")
            print("   ✓ MRI loading works!")
        else:
            print("   ⚠ MRI not found (check data path)")
    
    # 测试DataLoader
    print("\n3. Testing DataLoader with MRI...")
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        ds_with_mri,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=GliomaDatasetWithMRI.collate_fn_with_mri
    )
    
    batch = next(iter(loader))
    print(f"   Batch keys: {batch.keys()}")
    print(f"   pre_latent batch shape: {batch['pre_latent'].shape}")
    if 'pre_mri' in batch:
        print(f"   pre_mri batch shape: {batch['pre_mri'].shape}")
        print("   ✓ Batching works!")
    
    print("\n" + "=" * 80)
    print("✅ Dataset extension works!")
    print("=" * 80)
