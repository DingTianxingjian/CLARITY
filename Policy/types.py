# Policy/types.py
from dataclasses import dataclass, field
from typing import List, Dict
import json

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json


@dataclass
class GenomicProfile:
    """基因组学特征（处理0/1/2编码）"""
    idh1: int = 2  # 0=wildtype, 1=mutated, 2=unknown
    idh2: int = 2
    atrx: int = 2
    mgmt_methylation: int = 2
    braf_v600e: int = 2
    tert_promoter: int = 2
    chr7_gain_chr10_loss: int = 2
    h3_3a: int = 2
    egfr_amp: int = 2
    pten: int = 2
    cdkn2ab_deletion: int = 2
    tp53_alteration: int = 2
    codeletion_1p19q_detail: str = "0"
    other_mutations_text: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'GenomicProfile':
        """从JSON数据创建"""
        return cls(
            idh1=int(d.get('idh1', 2)),
            idh2=int(d.get('idh2', 2)),
            atrx=int(d.get('atrx', 2)),
            mgmt_methylation=int(d.get('mgmt_methylation', 2)),
            braf_v600e=int(d.get('braf_v600e', 2)),
            tert_promoter=int(d.get('tert_promoter', 2)),
            chr7_gain_chr10_loss=int(d.get('chr7_gain_chr10_loss', 2)),
            h3_3a=int(d.get('h3_3a', 2)),
            egfr_amp=int(d.get('egfr_amp', 2)),
            pten=int(d.get('pten', 2)),
            cdkn2ab_deletion=int(d.get('cdkn2ab_deletion', 2)),
            tp53_alteration=int(d.get('tp53_alteration', 2)),
            codeletion_1p19q_detail=str(d.get('codeletion_1p19q_detail', '0')),
            other_mutations_text=d.get('other_mutations_text')
        )

    def to_text(self) -> str:
        """转换为临床可读文本"""
        def _status(val: int) -> str:
            return {0: "wildtype", 1: "mutated", 2: "unknown"}[val]
        
        parts = []
        
        # IDH状态（最关键）
        if self.idh1 != 2 or self.idh2 != 2:
            idh_status = []
            if self.idh1 == 1:
                idh_status.append("IDH1-mutant")
            elif self.idh1 == 0:
                idh_status.append("IDH1-wildtype")
            if self.idh2 == 1:
                idh_status.append("IDH2-mutant")
            
            if idh_status:
                parts.append(", ".join(idh_status))
            elif self.idh1 == 0 and self.idh2 == 0:
                parts.append("IDH-wildtype")
        
        # MGMT甲基化（预后标志）
        if self.mgmt_methylation == 1:
            parts.append("MGMT promoter methylated")
        elif self.mgmt_methylation == 0:
            parts.append("MGMT promoter unmethylated")
        
        # 1p/19q共缺失（寡突胶质瘤标志）
        if self.codeletion_1p19q_detail == "1":
            parts.append("1p/19q codeleted")
        elif self.codeletion_1p19q_detail == "0":
            parts.append("1p/19q intact")
        
        # 其他关键突变
        key_mutations = []
        if self.egfr_amp == 1:
            key_mutations.append("EGFR amplification")
        if self.tp53_alteration == 1:
            key_mutations.append("TP53 altered")
        if self.pten == 1:
            key_mutations.append("PTEN mutated")
        if self.atrx == 1:
            key_mutations.append("ATRX loss")
        if self.cdkn2ab_deletion == 1:
            key_mutations.append("CDKN2A/B deletion")
        if self.braf_v600e == 1:
            key_mutations.append("BRAF V600E")
        if self.tert_promoter == 1:
            key_mutations.append("TERT promoter mutated")
        if self.chr7_gain_chr10_loss == 1:
            key_mutations.append("chr7 gain/chr10 loss")
        if self.h3_3a == 1:
            key_mutations.append("H3.3A mutated")
        
        if key_mutations:
            parts.append("; ".join(key_mutations))
        
        # 其他文本注释
        if self.other_mutations_text:
            parts.append(f"Additional notes: {self.other_mutations_text}")
        
        return ". ".join(parts) + "." if parts else "Genomic profile not fully characterized."

    def to_dict(self) -> Dict[str, Any]:
        return {
            'idh1': self.idh1,
            'idh2': self.idh2,
            'atrx': self.atrx,
            'mgmt_methylation': self.mgmt_methylation,
            'braf_v600e': self.braf_v600e,
            'tert_promoter': self.tert_promoter,
            'chr7_gain_chr10_loss': self.chr7_gain_chr10_loss,
            'h3_3a': self.h3_3a,
            'egfr_amp': self.egfr_amp,
            'pten': self.pten,
            'cdkn2ab_deletion': self.cdkn2ab_deletion,
            'tp53_alteration': self.tp53_alteration,
            'codeletion_1p19q_detail': self.codeletion_1p19q_detail,
            'other_mutations_text': self.other_mutations_text
        }


@dataclass
class ClinicalProfile:
    """患者临床特征（对齐实际数据格式）"""
    patient_id: str
    sex_at_birth: str  # "male" / "female"
    race: str  # "white" / "black" / "asian" / "hispanic" / etc.
    age_at_diagnosis_years: float
    primary_diagnosis: str  # "gbm" / "oligodendroglioma" / etc.
    who_grade: int  # 2/3/4
    genomics: GenomicProfile
    
    # 可选字段
    kps: Optional[int] = None  # Karnofsky Performance Status (0-100)
    ecog: Optional[int] = None  # ECOG Performance Status (0-5)
    comorbidities: List[str] = field(default_factory=list)
    prior_treatments: List[str] = field(default_factory=list)
    disease_status: str = "newly_diagnosed"  # "newly_diagnosed" / "recurrent" / "progressive"
    days_from_diagnosis: int = 0
    extent_of_resection: Optional[str] = None  # "GTR" / "STR" / "biopsy"

    @classmethod
    def from_context_static(cls, patient_id: str, context_static: Dict[str, Any], **kwargs) -> 'ClinicalProfile':
        """从元数据的context_static创建"""
        genomics = GenomicProfile.from_dict(context_static.get('genomics', {}))
        
        return cls(
            patient_id=patient_id,
            sex_at_birth=context_static.get('sex_at_birth', 'unknown'),
            race=context_static.get('race', 'unknown'),
            age_at_diagnosis_years=float(context_static.get('age_at_diagnosis_years', 0)),
            primary_diagnosis=context_static.get('primary_diagnosis', 'glioma'),
            who_grade=int(context_static.get('who_grade', 4)),
            genomics=genomics,
            **kwargs  # kps, comorbidities等可选字段
        )

    def get_molecular_classification(self) -> str:
        """根据WHO 2021分类返回分子分型"""
        if self.primary_diagnosis.lower() == 'gbm':
            if self.genomics.idh1 == 1 or self.genomics.idh2 == 1:
                return "Astrocytoma, IDH-mutant, grade 4"
            elif self.genomics.idh1 == 0 and self.genomics.idh2 == 0:
                return "Glioblastoma, IDH-wildtype"
            else:
                return "Glioblastoma, IDH-status unknown"
        
        elif 'oligodendroglioma' in self.primary_diagnosis.lower():
            if self.genomics.codeletion_1p19q_detail == "1":
                return f"Oligodendroglioma, IDH-mutant and 1p/19q-codeleted, grade {self.who_grade}"
            else:
                return f"Oligodendroglioma, grade {self.who_grade}"
        
        return f"{self.primary_diagnosis}, WHO grade {self.who_grade}"

    def to_text(self) -> str:
        """生成临床摘要文本（供LLM使用）"""
        parts = []
        
        # 基本信息
        parts.append(
            f"A {int(self.age_at_diagnosis_years)}-year-old {self.sex_at_birth} patient "
            f"({self.race})."
        )
        
        # 诊断与分型
        mol_class = self.get_molecular_classification()
        parts.append(f"Diagnosis: {mol_class}.")
        
        # 基因组学特征
        genomics_text = self.genomics.to_text()
        if genomics_text:
            parts.append(f"Molecular profile: {genomics_text}")
        
        # 功能状态
        if self.kps is not None:
            parts.append(f"KPS: {self.kps}.")
        if self.ecog is not None:
            parts.append(f"ECOG: {self.ecog}.")
        
        # 手术情况
        if self.extent_of_resection:
            parts.append(f"Extent of resection: {self.extent_of_resection}.")
        
        # 既往治疗
        if self.prior_treatments:
            parts.append("Prior treatments: " + ", ".join(self.prior_treatments) + ".")
        
        # 合并症
        if self.comorbidities:
            parts.append("Comorbidities: " + ", ".join(self.comorbidities) + ".")
        
        # 疾病状态
        parts.append(
            f"Current status: {self.disease_status}, "
            f"{self.days_from_diagnosis} days from initial diagnosis."
        )
        
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于API调用）"""
        return {
            'patient_id': self.patient_id,
            'sex_at_birth': self.sex_at_birth,
            'race': self.race,
            'age_at_diagnosis_years': self.age_at_diagnosis_years,
            'primary_diagnosis': self.primary_diagnosis,
            'who_grade': self.who_grade,
            'genomics': self.genomics.to_dict(),
            'kps': self.kps,
            'ecog': self.ecog,
            'comorbidities': self.comorbidities,
            'prior_treatments': self.prior_treatments,
            'disease_status': self.disease_status,
            'days_from_diagnosis': self.days_from_diagnosis,
            'extent_of_resection': self.extent_of_resection
        }



@dataclass
class ImagingProfile:
    num_lesions: int
    total_volume_cc: float
    locations: List[str]
    min_distance_to_oar_mm: float
    enhancement_pattern: str
    edema_volume_cc: float
    mass_effect: bool
    multifocal: bool
    deep_seated: bool

    def to_dict(self):
        return self.__dict__

@dataclass
class TreatmentBlock:
    kind: str
    params: Dict
    start_day: int
    end_day: int
    rationale: str = ""

@dataclass
class TreatmentSequence:
    blocks: List[TreatmentBlock]
    total_score: float = float('inf')
    risk_score: float = 0.0
    toxicity_score: float = 0.0
    complexity_score: float = 0.0
    source: str = "LLM"  # "LLM", "LLM_variant", "Manual"

    def to_text(self) -> str:
        sequence_dict = {
            "sequence": [
                {"kind": b.kind, "params": b.params, "start_day": b.start_day, "end_day": b.end_day}
                for b in self.blocks
            ],
            "total_duration_days": self.blocks[-1].end_day if self.blocks else 0
        }
        return json.dumps(sequence_dict, sort_keys=True)

    def get_summary(self) -> str:
        return " → ".join([f"{b.kind}({b.start_day}d)" for b in self.blocks])

    def __hash__(self):
        canon = json.dumps(
            [{"k": b.kind, "p": b.params, "s": b.start_day, "e": b.end_day} for b in self.blocks],
            sort_keys=True
        )
        return hash(canon)


# Policy/types.py （在文件末尾新增/替换这些工具方法）

from typing import Dict, Any, List
VALID_ACTION_KEYS = ["radiation", "chemotherapy", "immunotherapy", "additional_1", "additional_2", "other_therapy"]

def _blocks_to_post_actions(blocks: List[TreatmentBlock]) -> Dict[str, List[Dict[str, Any]]]:
    """
    将内部统一的 blocks 映射为训练同构的 post.actions 六大类。
    约定各类字段最小 schema：
      - radiation: {technique, dose_gy, fractions}
      - chemo/immono/additional_*: {agent, cycle_length_days, num_cycles}
      - other_therapy: {name, notes?}
    """
    actions: Dict[str, List[Dict[str, Any]]] = {k: [] for k in VALID_ACTION_KEYS}
    for b in blocks:
        k = b.kind
        if k not in VALID_ACTION_KEYS:
            continue  # 忽略未知类别（或映射到 other_therapy）
        p = dict(b.params)

        if k == "radiation":
            entry = {
                "dose_gy": float(p.get("dose_gy", 60)),
                "fractions": int(p.get("fractions", 30))
            }
        elif k in VALID_ACTION_KEYS:
            entry = {
                "agent": p.get("agent", "TMZ"),
                "cycle_length_days": int(p.get("cycle_length_days", 28)),
                "num_cycles": int(p.get("num_cycles", 6))
            }
        else:  # other_therapy
            entry = {
                "agent": p.get("agent"),
            }
        actions[k].append(entry)
    # 去掉空类，保持跟你数据风格对齐
    return {k: v for k, v in actions.items() if v}

def _estimate_total_days_from_actions(actions: Dict[str, List[Dict[str, Any]]]) -> int:
    """粗略估计整个 post 的时间跨度：药物取 max(sum(cycle_len * num_cycles))，放疗取 fractions*2 或 42 天等近似。"""
    tot_drug = 0
    for key in VALID_ACTION_KEYS:
        if key in actions:
            for x in actions[key]:
                c = int(x.get("num_cycles", 0))
                l = int(x.get("cycle_length_days", 28))
                tot_drug = max(tot_drug, c * l)
    tot_rt = 0
    for rt in actions.get("radiation", []):
        fr = int(rt.get("fractions", 30))
        # 近似：分割治疗持续天数≈fractions * 1.4（含周末/机时），或直接 42 天
        tot_rt = max(tot_rt, max(42, int(fr * 1.4)))
    return max(tot_drug, tot_rt, 0)

# 给 TreatmentSequence 增加三个方法（若已有同名方法，请替换）
def to_post_actions(self) -> Dict[str, List[Dict[str, Any]]]:
    return _blocks_to_post_actions(self.blocks)

def to_model_triplet_json(self, pre_payload: Dict[str, Any], tp_post: str = "TP_post", between: List[Dict[str, Any]] = None) -> str:
    actions = self.to_post_actions()
    triplet = {
        "pre": pre_payload.get("pre", {}),
        "post": {"tp": tp_post, "actions": actions},
        "between": between if between is not None else []
    }
    import json
    return json.dumps(triplet, ensure_ascii=False, sort_keys=True)

def estimated_total_days(self) -> int:
    return _estimate_total_days_from_actions(self.to_post_actions())

# 将方法绑定到类（若你不想改类定义处）
TreatmentSequence.to_post_actions = to_post_actions
TreatmentSequence.to_model_triplet_json = to_model_triplet_json
TreatmentSequence.estimated_total_days = estimated_total_days
