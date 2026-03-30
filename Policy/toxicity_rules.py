# Policy/toxicity_rules.py
# -*- coding: utf-8 -*-
"""
治疗毒性评分规则
基于临床指南和真实数据结构
"""
from typing import Dict, Any
from Policy.types import TreatmentSequence, TreatmentBlock, ClinicalProfile


def compute_toxicity(sequence: TreatmentSequence, clinical: ClinicalProfile) -> float:
    """
    计算治疗序列的预期毒性评分
    
    考虑因素：
    1. 累计放疗剂量
    2. 化疗强度（TMZ, lomustine等）
    3. 治疗重叠（concurrent毒性）
    4. 患者因素（年龄、KPS、基因型）
    5. 多线治疗累积效应
    
    Args:
        sequence: 治疗序列
        clinical: 临床特征
    
    Returns:
        毒性评分（越高越差）
    """
    tox = 0.0

    # ========================================================================
    # 1. 放疗累计剂量（超过标准60Gy的惩罚）
    # ========================================================================
    total_rt_gy = 0.0
    num_rt_courses = 0
    for b in sequence.blocks:
        if b.kind == "radiation":
            try:
                dose = float(b.params.get("dose_gy", 0.0))
                total_rt_gy += dose
                num_rt_courses += 1
            except (ValueError, TypeError):
                pass
    
    # 超过60Gy的每Gy增加毒性
    if total_rt_gy > 60:
        tox += 0.05 * (total_rt_gy - 60)
    
    # 多疗程放疗（再照射）的额外惩罚
    if num_rt_courses > 1:
        tox += 0.15 * (num_rt_courses - 1)

    # ========================================================================
    # 2. 化疗毒性（根据新的六大类action）
    # ========================================================================
    # 2.1 TMZ累计周期
    tmz_cycles = 0
    for b in sequence.blocks:
        if b.kind == "chemo":
            agent = str(b.params.get("agent", "")).upper()
            if agent in {"TMZ", "TEMOZOLOMIDE"}:
                cycles = int(b.params.get("num_cycles", 0) or 0)
                tmz_cycles += cycles
    
    # TMZ标准6个周期，超过则增加毒性
    if tmz_cycles > 6:
        tox += 0.08 * (tmz_cycles - 6)
    
    # 2.2 其他化疗药物（additional_1, additional_2）
    # 常见的additional drugs: Lomustine (CCNU), Bevacizumab等
    additional_drugs_penalty = 0.0
    for b in sequence.blocks:
        if b.kind in ("additional_1", "additional_2"):
            agent = str(b.params.get("agent", "")).upper()
            cycles = int(b.params.get("num_cycles", 0) or 0)
            
            # Lomustine/CCNU: 骨髓抑制明显
            if agent in {"LOMUSTINE", "CCNU"}:
                additional_drugs_penalty += 0.12 * cycles
            # Carboplatin: 血小板减少
            elif agent in {"CARBOPLATIN"}:
                additional_drugs_penalty += 0.10 * cycles
            # 其他药物的通用惩罚
            else:
                additional_drugs_penalty += 0.06 * cycles
    
    tox += additional_drugs_penalty
    
    # 2.3 免疫治疗（immono）
    # Bevacizumab (anti-VEGF): 出血、伤口愈合、高血压风险
    immono_penalty = 0.0
    for b in sequence.blocks:
        if b.kind == "immono":
            agent = str(b.params.get("agent", "")).upper()
            cycles = int(b.params.get("num_cycles", 0) or 0)
            
            if agent in {"BEVACIZUMAB", "AVASTIN"}:
                immono_penalty += 0.08 * cycles  # 血管并发症风险
            else:
                immono_penalty += 0.05 * cycles  # 通用免疫治疗毒性
    
    tox += immono_penalty

    # ========================================================================
    # 3. 治疗重叠/并发惩罚
    # ========================================================================
    overlap_penalty = 0.0
    blocks = sequence.blocks
    
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            a, b = blocks[i], blocks[j]
            
            # 计算时间重叠（如果start_day/end_day有意义）
            if a.start_day >= 0 and a.end_day >= 0 and b.start_day >= 0 and b.end_day >= 0:
                overlap_days = max(0, min(a.end_day, b.end_day) - max(a.start_day, b.start_day))
                
                if overlap_days > 0:
                    # Concurrent RT + TMZ: 标准方案，轻度惩罚
                    is_rt_tmz = (
                        {a.kind, b.kind} == {"radiation", "chemo"} and
                        any(str(x.params.get("agent", "")).upper() in {"TMZ", "TEMOZOLOMIDE"} 
                            for x in [a, b] if x.kind == "chemo")
                    )
                    
                    if is_rt_tmz:
                        overlap_penalty += 0.02 * overlap_days  # 标准concurrent
                    else:
                        overlap_penalty += 0.05 * overlap_days  # 非标准重叠
    
    tox += overlap_penalty

    # ========================================================================
    # 4. 患者因素调制
    # ========================================================================
    age_factor = 1.0
    kps_factor = 1.0
    genomic_factor = 1.0
    
    # 4.1 年龄调制（>65岁毒性增加）
    age = clinical.age_at_diagnosis_years
    if age > 65:
        age_factor = 1.0 + (age - 65) / 25.0  # 65岁: 1.0, 75岁: 1.4, 90岁: 2.0
    
    # 4.2 KPS调制（<80分毒性增加）
    if clinical.kps is not None:
        if clinical.kps < 80:
            kps_factor = 1.0 + (80 - clinical.kps) / 50.0  # KPS 80: 1.0, 60: 1.4, 30: 2.0
    elif clinical.ecog is not None:
        # 如果没有KPS，用ECOG估算（ECOG 0~1 ≈ KPS 80-100）
        ecog_to_kps_penalty = {0: 0.0, 1: 0.2, 2: 0.5, 3: 0.8, 4: 1.2}
        kps_factor = 1.0 + ecog_to_kps_penalty.get(clinical.ecog, 0.5)
    
    # 4.3 基因组学调制
    genomics = clinical.genomics
    
    # MGMT甲基化: TMZ耐受性更好（毒性-10%）
    if genomics.mgmt_methylation == 1:  # methylated
        if tmz_cycles > 0:
            genomic_factor *= 0.9
    
    # IDH突变: 预后较好，可能耐受性更好（毒性-5%）
    if genomics.idh1 == 1 or genomics.idh2 == 1:
        genomic_factor *= 0.95
    
    # TP53突变 + 年龄>60: 血液毒性增加
    if genomics.tp53_alteration == 1 and age > 60:
        genomic_factor *= 1.15
    
    # 4.4 合并症调制
    comorbidity_factor = 1.0
    if clinical.comorbidities:
        comorbidities_lower = [c.lower() for c in clinical.comorbidities]
        
        # 心血管疾病: 放疗+bevacizumab风险增加
        if any(x in comorbidities_lower for x in ["hypertension", "cardiovascular", "stroke", "cad"]):
            if any(b.kind == "immono" for b in blocks):
                comorbidity_factor *= 1.2
        
        # 糖尿病: 伤口愈合、感染风险
        if any(x in comorbidities_lower for x in ["diabetes", "dm", "type_2_diabetes"]):
            comorbidity_factor *= 1.1
        
        # 肾功能不全: 化疗毒性增加
        if any(x in comorbidities_lower for x in ["ckd", "renal", "kidney"]):
            if any(b.kind in ("chemo", "additional_1", "additional_2") for b in blocks):
                comorbidity_factor *= 1.25
    
    # 综合调制
    tox *= age_factor * kps_factor * genomic_factor * comorbidity_factor

    # ========================================================================
    # 5. 复杂度惩罚（过多治疗线数）
    # ========================================================================
    # 统计不同类型的治疗
    treatment_types = set(b.kind for b in blocks)
    if len(treatment_types) > 3:  # 超过3种治疗类型
        tox += 0.1 * (len(treatment_types) - 3)

    return float(max(0.0, tox))  # 确保非负


# ============================================================================
# 辅助函数：详细毒性分析
# ============================================================================
def analyze_toxicity_breakdown(sequence: TreatmentSequence, clinical: ClinicalProfile) -> Dict[str, Any]:
    """
    返回毒性评分的详细分解，用于可解释性
    
    Returns:
        {
            "total_toxicity": float,
            "components": {
                "radiation": float,
                "chemotherapy": float,
                "overlap": float,
                "patient_factors": float
            },
            "risk_factors": List[str],
            "recommendations": List[str]
        }
    """
    components = {
        "radiation": 0.0,
        "chemotherapy": 0.0,
        "immunotherapy": 0.0,
        "overlap": 0.0,
        "patient_age": 0.0,
        "patient_kps": 0.0,
        "genomic": 0.0,
        "comorbidities": 0.0
    }
    risk_factors = []
    recommendations = []
    
    # Radiation
    total_rt_gy = sum(float(b.params.get("dose_gy", 0)) for b in sequence.blocks if b.kind == "radiation")
    if total_rt_gy > 60:
        components["radiation"] = 0.05 * (total_rt_gy - 60)
        risk_factors.append(f"High cumulative RT dose: {total_rt_gy:.1f} Gy")
        recommendations.append("Consider dose reduction or fractionation adjustment")
    
    # Chemotherapy
    tmz_cycles = sum(
        int(b.params.get("num_cycles", 0) or 0)
        for b in sequence.blocks
        if b.kind == "chemo" and str(b.params.get("agent", "")).upper() in {"TMZ", "TEMOZOLOMIDE"}
    )
    if tmz_cycles > 6:
        components["chemotherapy"] = 0.08 * (tmz_cycles - 6)
        risk_factors.append(f"Extended TMZ cycles: {tmz_cycles}")
        recommendations.append("Monitor CBC closely, consider dose interruption")
    
    # Patient factors
    age = clinical.age_at_diagnosis_years
    if age > 70:
        components["patient_age"] = (age - 65) / 25.0 - 0.2
        risk_factors.append(f"Advanced age: {int(age)} years")
        recommendations.append("Consider geriatric assessment and dose modifications")
    
    if clinical.kps and clinical.kps < 70:
        components["patient_kps"] = (80 - clinical.kps) / 50.0 - 0.2
        risk_factors.append(f"Reduced KPS: {clinical.kps}")
        recommendations.append("Consider supportive care and symptom management")
    
    # Genomics
    if clinical.genomics.mgmt_methylation == 0:  # unmethylated
        risk_factors.append("MGMT unmethylated (may affect TMZ response)")
    
    total_toxicity = compute_toxicity(sequence, clinical)
    
    return {
        "total_toxicity": float(total_toxicity),
        "components": {k: float(v) for k, v in components.items()},
        "risk_factors": risk_factors,
        "recommendations": recommendations,
        "toxicity_grade": _classify_toxicity_grade(total_toxicity)
    }


def _classify_toxicity_grade(tox: float) -> str:
    """将毒性分数映射到等级"""
    if tox < 0.3:
        return "Low (Grade 1-2)"
    elif tox < 0.6:
        return "Moderate (Grade 2-3)"
    elif tox < 1.0:
        return "High (Grade 3-4)"
    else:
        return "Severe (Grade 4-5)"


# ============================================================================
# 单元测试
# ============================================================================
if __name__ == "__main__":
    from Policy.types import TreatmentBlock, GenomicProfile
    
    # 创建测试患者
    genomics = GenomicProfile(
        idh1=0, idh2=0,
        mgmt_methylation=1,  # methylated
        tp53_alteration=1
    )
    
    clinical = ClinicalProfile(
        patient_id="TEST_001",
        sex_at_birth="female",
        race="white",
        age_at_diagnosis_years=72.0,
        primary_diagnosis="gbm",
        who_grade=4,
        genomics=genomics,
        kps=70,
        comorbidities=["hypertension", "type_2_diabetes"]
    )
    
    # 测试序列：标准concurrent + 长期adjuvant TMZ
    test_sequence = TreatmentSequence(blocks=[
        TreatmentBlock(
            kind="radiation",
            params={"technique": "IMRT", "dose_gy": 60, "fractions": 30},
            start_day=0,
            end_day=42
        ),
        TreatmentBlock(
            kind="chemo",
            params={"agent": "Temozolomide", "cycle_length_days": 28, "num_cycles": 12},
            start_day=0,
            end_day=336
        ),
        TreatmentBlock(
            kind="additional_1",
            params={"agent": "Lomustine", "cycle_length_days": 42, "num_cycles": 3},
            start_day=336,
            end_day=462
        )
    ])
    
    # 计算毒性
    tox = compute_toxicity(test_sequence, clinical)
    print(f"Total Toxicity: {tox:.4f}")
    
    # 详细分析
    breakdown = analyze_toxicity_breakdown(test_sequence, clinical)
    print("\nToxicity Breakdown:")
    for comp, val in breakdown["components"].items():
        if val > 0:
            print(f"  {comp}: {val:.4f}")
    
    print(f"\nToxicity Grade: {breakdown['toxicity_grade']}")
    
    if breakdown["risk_factors"]:
        print("\nRisk Factors:")
        for rf in breakdown["risk_factors"]:
            print(f"  - {rf}")
    
    if breakdown["recommendations"]:
        print("\nRecommendations:")
        for rec in breakdown["recommendations"]:
            print(f"  - {rec}")