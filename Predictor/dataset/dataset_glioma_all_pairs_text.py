# dataset_glioma_all_pairs_text.py
import re, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset, DataLoader

def parse_tp_num(tp: str) -> int:
    import re
    m = re.search(r'(\d+)', str(tp))
    return int(m.group(1)) if m else -1

def pid_tp_to_id(pid: str, tp: str) -> str:
    return f"{pid}_Timepoint_{parse_tp_num(tp)}"

class FeatureStore:
    """
    Supports two backends:

    1. CSV  (BrainIAC / legacy):
       csv_path ends with .csv  → loads entire table, returns [4, take_dims]

    2. NPZ directory  (mri_foundation):
       csv_path is a directory containing PatientID_*_Timepoint_*.npz files.
       Each .npz has keys t1c/t2w/t1n/t2f, each shape [D, 256].
       get() returns [4*D, 256] — all slices across modalities as token sequence.
    """
    def __init__(self, csv_path: str, take_dims: int = 767):
        self.modalities = ["t1c", "t2w", "t1n", "t2f"]
        path = Path(csv_path)

        if path.is_dir():
            # mri_foundation NPZ backend
            self._backend = "npz"
            self._npz_dir = path
            self._index = {
                f.stem: f for f in path.glob("*.npz")
            }
        else:
            # CSV backend (BrainIAC)
            self._backend = "csv"
            self.take_dims = take_dims
            df = pd.read_csv(csv_path)
            assert "pat_timepoint_id" in df.columns
            self.cols_by_mod = {}
            for mod in self.modalities:
                cols = [c for c in df.columns if c.startswith(f"{mod}_Feature_")]
                cols = sorted(cols, key=lambda c: int(re.search(r"(\d+)$", c).group(1)))
                self.cols_by_mod[mod] = cols[:take_dims]
            keep = ["pat_timepoint_id"] + sum([self.cols_by_mod[m] for m in self.modalities], [])
            self.df = df[keep].copy().set_index("pat_timepoint_id")

    def has(self, id_: str) -> bool:
        if self._backend == "npz":
            return id_ in self._index
        return id_ in self.df.index

    def get(self, id_: str) -> np.ndarray:
        if self._backend == "npz":
            data = np.load(self._index[id_])
            # [4, D, 256] → [4*D, 256] — treat every slice as an independent token
            mats = [data[mod].astype(np.float32) for mod in self.modalities]
            return np.concatenate(mats, axis=0)   # [4*D, 256]
        else:
            row = self.df.loc[id_]
            mats = [row[self.cols_by_mod[m]].to_numpy(dtype=np.float32) for m in self.modalities]
            return np.stack(mats, axis=0)          # [4, take_dims]


@dataclass
class Config:
    timeline_json: str
    features_csv: Optional[str] = None   # path to .csv file OR npz directory
    take_dims: int = 767
    # Online MRI loading (used with SAMMRIEncoder end-to-end training)
    mri_data_dir: Optional[str] = None   # e.g. "datasets/MU-Glioma-Post"
    mri_modalities: List[str] = field(default_factory=lambda: ["t1c", "t2w", "t1n", "t2f"])

class MRIVolumeLoader:
    """
    Loads raw NIfTI MRI volumes for a patient-timepoint.
    Returns [4, H, W, D] float32 tensor (4 modalities stacked).
    Used for online SAMMRIEncoder inference (plan B end-to-end training).
    """
    MODALITIES = ["t1c", "t2w", "t1n", "t2f"]

    def __init__(self, data_dir: str, modalities: Optional[List[str]] = None):
        self.data_dir = Path(data_dir)
        self.modalities = modalities or self.MODALITIES

    def _pid_tp(self, id_: str):
        # id_ format: "PatientID_XXXX_Timepoint_Y"
        parts = id_.split("_Timepoint_")
        pid = parts[0]           # "PatientID_XXXX"
        tp_num = parts[1]        # "Y"
        return pid, f"Timepoint_{tp_num}"

    def has(self, id_: str) -> bool:
        pid, tp = self._pid_tp(id_)
        tp_dir = self.data_dir / pid / tp
        return all(
            (tp_dir / f"{pid}_Timepoint_{tp.split('_')[1]}_brain_{m}.nii.gz").exists()
            for m in self.modalities
        )

    def load(self, id_: str) -> torch.Tensor:
        """Returns [4, H, W, D] float32."""
        import nibabel as nib
        pid, tp = self._pid_tp(id_)
        tp_num = tp.split("_")[1]
        tp_dir = self.data_dir / pid / tp
        vols = []
        for m in self.modalities:
            path = tp_dir / f"{pid}_Timepoint_{tp_num}_brain_{m}.nii.gz"
            arr = np.nan_to_num(nib.load(str(path)).get_fdata(dtype=np.float32))
            vols.append(arr)  # [H, W, D]
        return torch.from_numpy(np.stack(vols, axis=0))  # [4, H, W, D]


class GliomaAllPairsTextDataset(Dataset):
    """
    ALL pairs (T_k -> T_{k+n}), 返回 drugs_text 为 JSON 字符串，
    包含 pre/post 两端的 therapies（可选也能包含中间段）。
    """
    def __init__(self, cfg: Config, include_between: bool = False):
        super().__init__()
        self.cfg = cfg
        self.feat = FeatureStore(cfg.features_csv, take_dims=cfg.take_dims) if cfg.features_csv else None
        self.mri_loader = MRIVolumeLoader(cfg.mri_data_dir, cfg.mri_modalities) if cfg.mri_data_dir else None
        with open(cfg.timeline_json, "r", encoding="utf-8") as f:
            self.patients: List[Dict[str, Any]] = json.load(f)

        self.include_between = include_between  # 是否把 k..n 间所有 timepoint 的 therapies 也拼进去

        # 构建索引：所有有效的 (i<j) 且两端有 feature
        self.index = []
        # 新的JSON格式是 "patients": {"PID": {...}}
        for pid, p_data in self.patients.get("patients", {}).items():
            pid = str(pid)
            tl = p_data.get("timeline", [])
            if len(tl) < 2: continue

            enriched = []
            for tp in tl:
                tp_name = str(tp["tp_id"])   # 'T3'
                id_ = pid_tp_to_id(pid, tp_name)
                enriched.append({
                    "tp": tp_name,
                    "tp_num": parse_tp_num(tp_name),
                    "id": id_,
                    "actions": tp.get("actions", {}), # 使用 'actions' 字段，并确保它是一个字典
                    # 修正：生存信息在 survival 字典中
                    "survival": tp.get("survival", {}),
                    "mri_day": tp.get("mri_day", 0), # 确保有mri_day
                    "context_static": p_data.get("context_static", {}) # 携带context_static
                })
            enriched.sort(key=lambda x: x["tp_num"])

            L = len(enriched)
            for i in range(L-1):
                pre = enriched[i]
                if self.feat is not None and not self.feat.has(pre["id"]):
                    continue
                for j in range(i+1, L):
                    post = enriched[j]
                    if self.feat is not None and not self.feat.has(post["id"]):
                        continue
                    pre_day = pre["mri_day"]
                    post_day = post["mri_day"]
                    if post_day <= pre_day:
                        continue  # 跳过时间顺序异常的样本对
                    survival_time = float(post["survival"].get("survival_from_tp_days", -1.0))
                    event_indicator = int(post["survival"].get("event_indicator", 0))
                    if survival_time < 0:
                        continue  # 跳过没有生存标签的样本
                    # 临时止血：去掉删失且生存时间为0/非正的样本，避免污染 survival supervision
                    if event_indicator == 0 and survival_time <= 0:
                        continue
                    item = {
                        "pid": pid, "i": i, "j": j,
                        "pre_id": pre["id"], "post_id": post["id"],
                        "pre_tp": pre["tp"], "post_tp": post["tp"],
                        "pre_actions": pre["actions"],
                        "post_actions": post["actions"],
                        "pre_mri_day": pre_day,
                        "post_mri_day": post_day,
                        "survival_time": survival_time,
                        "event_indicator": event_indicator,
                        "context_static": pre["context_static"]
                    }
                    if include_between:
                        mid = [enriched[k]["actions"] for k in range(i+1, j)]
                        item["between_actions"] = mid
                    self.index.append(item)

    def __len__(self): return len(self.index)

    @staticmethod
    def _pack_drugs_json(item: Dict[str, Any]) -> str:
        # 紧凑 JSON：保留 name/dose/q_days/cycles/interval
        def shrink_action(action_item: dict) -> dict:
            """保留action字典中的核心字段"""
            shrunk = {}
            # 通用字段
            if 'agent' in action_item: shrunk['agent'] = action_item['agent']
            if 'type' in action_item: shrunk['type'] = action_item['type']
            # 化疗/其他药物字段
            if 'dose' in action_item: shrunk['dose'] = action_item['dose']
            if 'cycle_length_days' in action_item: shrunk['cycle_length_days'] = action_item['cycle_length_days']
            if 'num_cycles' in action_item: shrunk['num_cycles'] = action_item['num_cycles']
            # 放疗字段
            if 'dose_gy' in action_item: shrunk['dose_gy'] = action_item['dose_gy']
            if 'fractions' in action_item: shrunk['fractions'] = action_item['fractions']
            return shrunk

        def shrink_actions_dict(actions: dict) -> dict:
            """处理包含多个治疗类别的actions字典"""
            if not actions: return {}
            shrunk_dict = {}
            for category, items in actions.items():
                if isinstance(items, list):
                    shrunk_dict[category] = [shrink_action(item) for item in items]
            return shrunk_dict

        payload = {
            "pre":   {"tp": item["pre_tp"],  "actions": shrink_actions_dict(item["pre_actions"])},
            "post":  {"tp": item["post_tp"], "actions": shrink_actions_dict(item["post_actions"])},
        }
        if "between_actions" in item:
            payload["between"] = [shrink_actions_dict(actions) for actions in item["between_actions"]]
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))  # 恢复紧凑模式

    @staticmethod
    def _format_clinical_text(context_static: dict) -> str:
        """将 context_static 字典转换为一段自然语言描述"""
        if not context_static:
            return ""
        
        parts = []
        age = context_static.get('age_at_diagnosis_years')
        sex = context_static.get('sex_at_birth')
        if age and sex:
            parts.append(f"A {int(age)}-year-old {sex}.")
        
        race = context_static.get('race')
        if race:
            parts.append(f"Race: {race}.")
            
        grade = context_static.get('who_grade')
        if grade:
            parts.append(f"WHO grade {int(grade)}.")
            
        # 假设genomics在demographics下一层
        genomics = context_static.get('genomics', {})
        markers = []
        for marker, status in genomics.items():
            if status == 1: markers.append(f"{marker} positive/mutated")
            elif status == 0: markers.append(f"{marker} negative/wild-type")
        if markers:
            parts.append("Genomic markers: " + ", ".join(markers) + ".")
            
        return " ".join(parts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.index[idx]

        # (新增) 计算时间差
        time_delta = torch.tensor(item['post_mri_day'] - item['pre_mri_day'], dtype=torch.float32)
        
        # (新增) 将临床静态信息格式化为文本
        clinical_text = self._format_clinical_text(item['context_static'])

        # 获取生存信息
        survival_time = torch.tensor(item.get('survival_time', -1.0), dtype=torch.float32)
        event_indicator = torch.tensor(item.get('event_indicator', 0), dtype=torch.float32)

        drugs_text = self._pack_drugs_json(item)

        sample = {
            "drugs_text": drugs_text,
            "survival_time": survival_time,
            "event_indicator": event_indicator,
            "time_delta": time_delta,
            "clinical_text": clinical_text,
            "meta": {
                "pid": item['pid'], "pre_tp": item['pre_tp'], "post_tp": item['post_tp']
            },
            "pre_mri_day": item['pre_mri_day'],
            "post_mri_day": item['post_mri_day'],
        }
        if self.feat is not None:
            sample["pre_latent"] = torch.from_numpy(self.feat.get(item["pre_id"]))
            sample["post_latent"] = torch.from_numpy(self.feat.get(item["post_id"]))
        if self.mri_loader is not None:
            sample["pre_mri"]  = self.mri_loader.load(item["pre_id"])   # [4, H, W, D]
            sample["post_mri"] = self.mri_loader.load(item["post_id"])  # [4, H, W, D]
        return sample

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        keys = batch[0].keys()
        collated_batch = {k: [] for k in keys}
        
        for item in batch:
            for key, value in item.items():
                collated_batch[key].append(value)
        
        # 堆叠Tensor, 其他保持为列表
        tensor_keys = ['survival_time', 'event_indicator', 'time_delta']
        if 'pre_latent' in collated_batch:
            tensor_keys.extend(['pre_latent', 'post_latent'])
        if 'pre_mri' in collated_batch:
            tensor_keys.extend(['pre_mri', 'post_mri'])
        for key in tensor_keys:
            collated_batch[key] = torch.stack(collated_batch[key])
            
        return collated_batch

# 用法示例（外部做 tokenizer）
if __name__ == "__main__":
    # 当直接运行此脚本时，修复相对导入问题
    import sys
    from pathlib import Path
    # 将项目根目录（MeWM-main）添加到sys.path
    project_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(project_root))

    cfg = Config(
        timeline_json="./dataset/MU_Glioma_Post/clinical_latest.json",
        features_csv="./dataset/MU_Glioma_Post/features_output.csv",
        take_dims=767
    )
    ds = GliomaAllPairsTextDataset(cfg, include_between=True)
    if len(ds) > 0:
        dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0,
                        collate_fn=lambda b: GliomaAllPairsTextDataset.collate_fn(b))
        batch = next(iter(dl))
        print(batch["pre_latent"].shape, batch["post_latent"].shape)  # [B,4,767] each
        print(batch["drugs_text"][0][:400], "...")
        print(batch["clinical_text"][0][0:1000], "...")
        print(batch["survival_time"][0:10])
        print(batch["event_indicator"][0:10])
        print(batch["time_delta"][0:10])
    else:
        print("Dataset is empty. Check paths and data format.")
 
