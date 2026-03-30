# Policy/guardrails.py
from typing import List, Set
from .types import TreatmentBlock, TreatmentSequence

ALLOWED_KINDS: Set[str] = {"radiation", "chemotherapy", "immunotherapy", "additional_1", "additional_2", "other_therapy", "other"}

# 简单的 agent 规范化（可自行扩展）
AGENT_ALIASES = {
    "TEMOZOLOMIDE": "TMZ",
    "CCNU": "LOMUSTINE",
}
def _norm_agent(a: str) -> str:
    up = (a or "").upper()
    return AGENT_ALIASES.get(up, up)

class ClinicalGuardrails:
    def __init__(self, strict: bool = True):
        self.strict = strict

    def validate_block(self, b: TreatmentBlock) -> List[str]:
        errs: List[str] = []
        if b.kind not in ALLOWED_KINDS:
            errs.append(f"kind '{b.kind}' not allowed")
            return errs

        p = b.params
        if b.kind == "radiation":
            dose = p.get("dose_gy", None)
            fr   = p.get("fractions", None)
            if dose is None or fr is None:
                errs.append("radiation requires dose_gy & fractions")
            else:
                try:
                    dose = float(dose); fr = int(fr)
                    if not (40 <= dose <= 70): errs.append("radiation.dose_gy out of [40,70]")
                    if not (20 <= fr <= 35):   errs.append("radiation.fractions out of [20,35]")
                except Exception:
                    errs.append("radiation dose/fractions type invalid")

        elif b.kind in ALLOWED_KINDS:
            agent = _norm_agent(str(p.get("agent","")))
            if not agent:
                errs.append(f"{b.kind}.agent required")
            cld = p.get("cycle_length_days", 28)
            cyc = p.get("num_cycles", 1)
            try:
                cld = int(cld); cyc = int(cyc)
                if not (21 <= cld <= 56): errs.append(f"{b.kind}.cycle_length_days out of [21,56]")
                if not (1 <= cyc <= 12):  errs.append(f"{b.kind}.num_cycles out of [1,12]")
            except Exception:
                errs.append(f"{b.kind} cycles type invalid")

        elif b.kind == "other_therapy":
            # 最松：至少有 name
            if not str(p.get("agent","")).strip():
                errs.append("other_therapy.name required")
        return errs

    def validate_sequence(self, seq: TreatmentSequence) -> List[str]:
        errs: List[str] = []
        if not seq.blocks:
            return ["empty sequence"]
        for b in seq.blocks:
            errs.extend(self.validate_block(b))
        return errs

    def repair(self, seq: TreatmentSequence) -> TreatmentSequence:
        fixed: List[TreatmentBlock] = []
        for b in seq.blocks:
            if b.kind not in ALLOWED_KINDS:
                if self.strict: 
                    continue
                # 映射未知到 other_therapy
                fixed.append(TreatmentBlock("other_therapy", {"agent": b.kind}, b.start_day, b.end_day, b.rationale))
                continue

            p = dict(b.params)
            if b.kind == "radiation":
                # clamp
                p["dose_gy"] = max(40.0, min(70.0, float(p.get("dose_gy", 60))))
                p["fractions"] = max(20, min(35, int(p.get("fractions", 30))))
            elif b.kind in ALLOWED_KINDS:
                p["agent"] = _norm_agent(p.get("agent","") or "TMZ")
                p["cycle_length_days"] = max(21, min(56, int(p.get("cycle_length_days", 28))))
                p["num_cycles"] = max(1, min(12, int(p.get("num_cycles", 6))))
            else:  # other_therapy
                p["agent"] = p.get("agent")

            fixed.append(TreatmentBlock(b.kind, p, b.start_day, b.end_day, b.rationale))

        # 可按需排序；这里只保留输入顺序
        return TreatmentSequence(blocks=fixed, source=seq.source)
