# utils/data_preprocessing.py
"""
数据预处理和诊断工具
用于检查和修复foundation model特征
"""
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm


def diagnose_dataset(dataset, dataloader, device='cuda'):
    """
    诊断数据集的各种统计信息
    """
    print("=" * 80)
    print("数据集诊断报告")
    print("=" * 80)
    
    # 1. 基本统计
    print(f"\n[1] 基本信息")
    print(f"  样本数量: {len(dataset)}")
    
    # 2. 收集所有数据
    all_pre = []
    all_post = []
    all_survival = []
    all_events = []
    
    print(f"\n[2] 收集数据中...")
    for batch in tqdm(dataloader):
        all_pre.append(batch['pre_latent'])
        all_post.append(batch['post_latent'])
        all_survival.append(batch['survival_time'])
        all_events.append(batch['event_indicator'])
    
    all_pre = torch.cat(all_pre, dim=0)      # [N, 4, 767]
    all_post = torch.cat(all_post, dim=0)    # [N, 4, 767]
    all_survival = torch.cat(all_survival)   # [N]
    all_events = torch.cat(all_events)       # [N]
    
    # 3. Latent特征统计
    print(f"\n[3] Latent特征统计")
    print(f"  Pre-latent:")
    print(f"    均值: {all_pre.mean():.6f}")
    print(f"    标准差: {all_pre.std():.6f}")
    print(f"    最小值: {all_pre.min():.6f}")
    print(f"    最大值: {all_pre.max():.6f}")
    print(f"    范围: [{all_pre.min():.3f}, {all_pre.max():.3f}]")
    
    print(f"\n  Post-latent:")
    print(f"    均值: {all_post.mean():.6f}")
    print(f"    标准差: {all_post.std():.6f}")
    print(f"    最小值: {all_post.min():.6f}")
    print(f"    最大值: {all_post.max():.6f}")
    
    print(f"\n  Pre vs Post差异:")
    diff = (all_post - all_pre).abs()
    print(f"    平均L1差异: {diff.mean():.6f}")
    print(f"    最大差异: {diff.max():.6f}")
    print(f"    相似度: {1 - diff.mean()/all_pre.abs().mean():.2%}")
    
    # 4. 生存数据统计
    print(f"\n[4] 生存数据统计")
    print(f"  生存时间:")
    print(f"    均值: {all_survival.mean():.2f} 天")
    print(f"    中位数: {all_survival.median():.2f} 天")
    print(f"    范围: [{all_survival.min():.1f}, {all_survival.max():.1f}]")
    
    print(f"\n  事件指示器:")
    print(f"    事件发生率: {all_events.float().mean():.2%}")
    print(f"    事件数量: {all_events.sum().item()}")
    print(f"    删失数量: {(1-all_events).sum().item()}")
    
    # 5. 5年生存统计
    five_year_survived = ((all_survival > 1825) & (all_events == 0))
    print(f"\n  5年生存:")
    print(f"    5年生存人数: {five_year_survived.sum().item()}")
    print(f"    5年生存率: {five_year_survived.float().mean():.2%}")
    print(f"    ⚠️  5年内死亡: {((all_survival <= 1825) & (all_events == 1)).sum().item()}")
    
    # 6. 潜在问题检测
    print(f"\n[5] 潜在问题检测")
    issues = []
    
    if all_pre.std() < 0.01:
        issues.append("⚠️  Pre-latent标准差过小，特征可能未归一化")
    
    if all_pre.abs().max() > 100:
        issues.append("⚠️  Pre-latent数值范围过大，建议归一化")
    
    if diff.mean() < 0.01:
        issues.append("⚠️  Pre和Post几乎相同，模型难以学习")
    
    if all_events.float().mean() < 0.1 or all_events.float().mean() > 0.9:
        issues.append(f"⚠️  事件发生率极端 ({all_events.float().mean():.2%})，可能导致训练不稳定")
    
    if five_year_survived.sum() < 10:
        issues.append("⚠️  5年生存样本过少，AUC可能无法计算")
    
    if len(issues) == 0:
        print("  ✓ 未发现明显问题")
    else:
        for issue in issues:
            print(f"  {issue}")
    
    print("\n" + "=" * 80)
    
    return {
        'pre_mean': all_pre.mean().item(),
        'pre_std': all_pre.std().item(),
        'post_mean': all_post.mean().item(),
        'post_std': all_post.std().item(),
        'event_rate': all_events.float().mean().item(),
        'five_year_survival_rate': five_year_survived.float().mean().item()
    }


def normalize_latent_features(dataset, method='standardize'):
    """
    归一化latent特征
    
    Args:
        dataset: GliomaAllPairsTextDataset
        method: 'standardize' (z-score) or 'minmax' (0-1)
    
    Returns:
        mean, std (for standardize) or min, max (for minmax)
    """
    print(f"计算归一化参数（方法: {method}）...")
    
    all_features = []
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        all_features.append(sample['pre_latent'])
        all_features.append(sample['post_latent'])
    
    all_features = torch.stack(all_features, dim=0)  # [2N, 4, 767]
    
    if method == 'standardize':
        mean = all_features.mean(dim=(0, 1), keepdim=True)  # [1, 1, 767]
        std = all_features.std(dim=(0, 1), keepdim=True)    # [1, 1, 767]
        std = torch.clamp(std, min=1e-6)  # 避免除0
        
        print(f"均值范围: [{mean.min():.4f}, {mean.max():.4f}]")
        print(f"标准差范围: [{std.min():.4f}, {std.max():.4f}]")
        
        return mean.squeeze(), std.squeeze()
    
    elif method == 'minmax':
        min_val = all_features.min(dim=0, keepdim=True)[0].min(dim=1, keepdim=True)[0]
        max_val = all_features.max(dim=0, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        
        print(f"最小值: {min_val.min():.4f}")
        print(f"最大值: {max_val.max():.4f}")
        
        return min_val.squeeze(), max_val.squeeze()


class NormalizedDataset(torch.utils.data.Dataset):
    """
    带归一化的数据集包装器
    """
    def __init__(self, base_dataset, mean, std, method='standardize'):
        self.base_dataset = base_dataset
        self.mean = mean
        self.std = std
        self.method = method
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        
        # 归一化
        if self.method == 'standardize':
            sample['pre_latent'] = (sample['pre_latent'] - self.mean) / self.std
            sample['post_latent'] = (sample['post_latent'] - self.mean) / self.std
        elif self.method == 'minmax':
            sample['pre_latent'] = (sample['pre_latent'] - self.mean) / (self.std - self.mean + 1e-6)
            sample['post_latent'] = (sample['post_latent'] - self.mean) / (self.std - self.mean + 1e-6)
        
        return sample


# 使用示例
if __name__ == "__main__":
    from dataset.dataset_glioma_all_pairs_text import GliomaAllPairsTextDataset, Config
    
    # 加载数据
    cfg = Config(
        timeline_json="dataset/MU-Glioma-Post/mu_glioma_post_timeline_with_survival.json",
        features_csv="./dataset/features_output.csv",
        take_dims=767
    )
    dataset = GliomaAllPairsTextDataset(cfg)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=0)
    
    # 诊断
    stats = diagnose_dataset(dataset, dataloader)
    
    # 归一化
    mean, std = normalize_latent_features(dataset, method='standardize')
    normalized_dataset = NormalizedDataset(dataset, mean, std)
    
    print("\n✓ 归一化完成！")