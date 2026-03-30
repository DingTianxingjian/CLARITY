# main.py
# -*- coding: utf-8 -*-
import json
import math
import os
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np
import torch
from openai import OpenAI

# 你的世界模型
from Predictor.models.full_model import TimeAwareGliomaSurvivalPredictor

# 公共类型 & Policy 组件（请确保已放在 Policy/ 下）
from Policy.types import ClinicalProfile, ImagingProfile, TreatmentBlock, TreatmentSequence
from Policy.guardrails import ClinicalGuardrails
from Policy.toxicity_rules import compute_toxicity


# ============================================================================ 
# LLM Policy：直接生成“post.actions”（六大类），随后映射为内部 blocks 
# ============================================================================ 
class SequenceLLMPolicy:
    """
    只生成 POST 的 actions（六类: radiation/chemotherapy/immunotherapy/additional_1/additional_2/other_therapy）。
    注意：不让 LLM 产 pre/between，它们来自真实历史。
    """
    def __init__(self, api_key: str, constraints_path: str = "./treatment_constraints.json", model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

        # 加载训练数据约束
        try:
            with open(constraints_path, 'r') as f:
                self.constraints = json.load(f)
        except FileNotFoundError:
            print(f"⚠️  Constraints file not found at {constraints_path}, using defaults")
            self.constraints = {}

    def generate_post_sequences(
        self,
        clinical_text: str,
        imaging_dict: Dict[str, Any],
        pre_payload: Dict[str, Any],
        num_sequences: int = 5,
        feedback: Optional[str] = None
    ) -> List[TreatmentSequence]:

        feedback_note = (
            "If the user payload includes a \"feedback\" field, treat it as critique from the world "
            "model (lower total scores are better) and adjust the next proposals accordingly.\n"
        )

        system_prompt = """
You are a neuro-oncologist AI proposing POST treatment actions for glioblastoma.

BASE RECOMMENDATIONS ON TRAINING DATA (374 treatment events, 203 patients):
...

JSON SCHEMA EXAMPLE:
{{
  "candidates": [
    {{
      "post": {{
        "tp": "TP_post",
        "actions": {{
          "radiation": [{{"dose_gy": 60, "fractions": 30}}],
          "chemotherapy": [{{"agent": "Temozolomide", "cycle_length_days": 28, "num_cycles": 6}}],
          "immunotherapy": [],
          "additional_1": [{{"agent": "Temozolomide", "cycle_length_days": 28, "num_cycles": 6}}],
          "additional_2": [],
          "other_therapy": [{{"agent": "Optune TTF"}}]
        }}
      }},
      "rationale": "<brief justification>",
      "pattern": "training_pattern_b"
    }}
  ]
}}

CONSTRAINTS:
- radiation: dose_gy ∈ [40,60], fractions ∈ [10,33]
- drugs: cycle_length_days ∈ [14,56], num_cycles ∈ [1,12]
- Generate {num_sequences} candidates (60% match common patterns, 40% variations)
- Omit empty arrays

NO-POST OUTPUTS ARE VALID
You may encode "no active post therapy at this timepoint" in ANY ONE of these canonical ways:

Empty actions object:
  "post": {{ "tp": "TP_post", "actions": {{}} }}



IMPORTANT:
- Do NOT hallucinate therapies just to fill the schema.
- If you output (C) empty string, keep the rest of the JSON valid.
- If a patient is between cycles or in observation/washout, prefer (A)/(B) or (D).
- When you do include therapies, you MAY include "start_offset_days" (int, default 0).
{feedback_note}

CONTEXT MAPPING RULES:
- If PRE payload already includes "radiation" and "chemotherapy Temozolomide",
  then the POST phase should be adjuvant maintenance (usually Temozolomide alone, occasionally with Avastin).
- If PRE payload contains only "surgery" or no active therapy, use Pattern (b) or (d).
- If PRE payload already includes "additional_1 Temozolomide", 
  then consider supportive care or observation (no active therapy).
- If the PRE payload contains "chemotherapy", the POST plan is very likely to choose "additional_1".
- Never repeat identical radiation or chemo agents that already appear in PRE payload, 
  unless the context explicitly suggests recurrence.

You MUST always inspect the PRE payload content and avoid repeating identical modalities.

Generate {num_sequences} diverse candidates covering different post-treatment intensities:
- mild (supportive care / no therapy)
- moderate (Temozolomide maintenance)
- intensive (Temozolomide + Avastin)
- recurrence protocol (Avastin alone)
At least one candidate must represent "no therapy / observation".


5. OTHER THERAPY (non-drug adjunctive treatments):
   - Optune TTF (72.5 % of all cases with other therapy)
   - Dabrafenib and Trametinib (~15 %)
   - Avastin + Everolimus (~10 %)
   - Others rare (< 3 %)

When including "other_therapy", use the field:
  "other_therapy": [{{"agent": "Optune TTF"}}]

These should be used ONLY in addition to or following systemic therapies.
In most post-treatment phases, include **Optune TTF** as supportive or maintenance therapy if appropriate.
""".format(num_sequences=num_sequences, feedback_note=feedback_note)


        user_payload = {
            "clinical_text": clinical_text,
            # "imaging": imaging_dict,
            "pre": pre_payload.get("pre", {})
        }
        if feedback:
            user_payload["feedback"] = feedback

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
            ],
            temperature=0.7,
            
            response_format={"type": "json_object"}
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)

        seqs: List[TreatmentSequence] = []

        def _mk_block(kind: str, params: Dict[str, Any]) -> TreatmentBlock:
            # start/end_day 对 post.actions 不关键，这里统一 0
            return TreatmentBlock(kind=kind, params=params, start_day=0, end_day=0, rationale="LLM")


        for cand in data.get("candidates", []):
            post = cand.get("post", {})
            
            # --- A/B/C: 处理“无 post”的三种表达 ---
            if post is None or (isinstance(post, str) and post.strip() == ""):
                # 空字符串或 None → 无治疗
                seqs.append(TreatmentSequence(blocks=[], source="LLM"))
                continue

            if not isinstance(post, dict):
                # 非法类型兜底：当作无治疗
                seqs.append(TreatmentSequence(blocks=[], source="LLM"))
                continue

            actions = post.get("actions", {})
            if actions in (None, "", {}) or (isinstance(actions, dict) and len(actions) == 0):
                # 空对象/缺省/空串 → 无治疗
                seqs.append(TreatmentSequence(blocks=[], source="LLM"))
                continue

            blocks: List[TreatmentBlock] = []

            # radiation
            for rt in actions.get("radiation", []):
                dose = float(rt.get("dose_gy", 60))
                fractions = int(rt.get("fractions", 30))
                dose = max(40.0, min(60.0, dose))
                fractions = max(10, min(33, fractions))
                params = {
                    "dose_gy": dose,
                    "fractions": fractions,
                }
                # 可选：支持偏移
                if "start_offset_days" in rt:
                    params["start_offset_days"] = int(rt["start_offset_days"])
                blocks.append(_mk_block("radiation", params))

            # chemotherapy / immunotherapy / additional_*
            for key in ("chemotherapy", "immunotherapy", "additional_1", "additional_2"):
                for x in actions.get(key, []):
                    agent = str(x.get("agent", "Temozolomide"))
                    cycle_days = int(x.get("cycle_length_days", 28))
                    num_cycles = int(x.get("num_cycles", 6))

                    ck = key
                    if ck in self.constraints:
                        c = self.constraints[ck]
                        cycle_range = c.get("cycle_length_range", [14, 56])
                        cycles_range = c.get("num_cycles_range", [1, 12])
                        cycle_days = max(cycle_range[0], min(cycle_range[1], cycle_days))
                        num_cycles = max(cycles_range[0], min(cycles_range[1], num_cycles))

                    params = {
                        "agent": agent,
                        "cycle_length_days": cycle_days,
                        "num_cycles": num_cycles,
                    }
                    if "start_offset_days" in x:
                        params["start_offset_days"] = int(x["start_offset_days"])
                    blocks.append(_mk_block(key, params))

            # other_therapy
            for ot in actions.get("other_therapy", []):
                if isinstance(ot, str):
                    # GPT 输出直接是字符串
                    agent_name = ot.strip()
                elif isinstance(ot, dict):
                    # 优先取 agent，如果没有则取 name
                    agent_name = str(ot.get("agent") or ot.get("name") or "").strip()
                else:
                    continue

                if not agent_name:
                    continue

                params = {"agent": agent_name}
                if isinstance(ot, dict) and "start_offset_days" in ot:
                    params["start_offset_days"] = int(ot["start_offset_days"])
                blocks.append(_mk_block("other_therapy", params))
            seqs.append(TreatmentSequence(blocks=blocks, source="LLM"))

        return seqs


# ============================================================================ 
# 参数变体生成器：六类参数的局部扰动（保持物理/临床范围）
# ============================================================================ 
class ParameterVariantGenerator:
    def __init__(self, rng: Optional[np.random.Generator] = None):
        self._rng = rng or np.random.default_rng()

    def set_rng(self, rng: np.random.Generator):
        self._rng = rng

    def _rng_choice(self, options: List[int]) -> int:
        return int(self._rng.choice(options))

    def generate_variants(self, base_sequence: TreatmentSequence, num_variants: int = 10) -> List[TreatmentSequence]:
        variants: List[TreatmentSequence] = []

        for _ in range(num_variants):
            new_blocks = []
            for b in base_sequence.blocks:
                p = dict(b.params)
                if b.kind == "radiation":
                    if "dose_gy" in p:
                        p["dose_gy"] = float(np.clip(
                            p["dose_gy"] + self._rng.integers(-5, 6), 40, 60  # 40–60
                        ))
                    if "fractions" in p:
                        p["fractions"] = int(np.clip(
                            int(p["fractions"]) + self._rng_choice([-2, 0, 2]), 10, 33  # 10–33
                        ))
                elif b.kind in ("chemotherapy", "immunotherapy", "additional_1", "additional_2"):
                    if "num_cycles" in p:
                        p["num_cycles"] = int(np.clip(
                            int(p["num_cycles"]) + self._rng_choice([-1, 0, 1]), 1, 12
                        ))
                    if "cycle_length_days" in p:
                        p["cycle_length_days"] = int(np.clip(
                            int(p["cycle_length_days"]) + self._rng_choice([-7, 0, 7]), 14, 56
                        ))
                new_blocks.append(TreatmentBlock(b.kind, p, b.start_day, b.end_day, b.rationale))
            variants.append(TreatmentSequence(blocks=new_blocks, source="LLM_variant"))
        return variants


# ============================================================================ 
# 世界模型：以“训练同构三元组 JSON”作为 drugs_text 输入进行整序列评估
# ============================================================================ 
class SequenceWorldModel:
    def __init__(self, model_path: str, device: str = 'cuda:0'):
        self.device = device
        self.model = self._load_model(model_path)
        if self.model:
            self.model.to(self.device)
            self.model.eval()

    def _load_model(self, model_path: str):
        print(f"[WorldModel] Loading from {model_path}")
        try:
            from pathlib import Path
            import os
            import torch
            from peft import PeftModel
            from safetensors.torch import load_file, save_file

            base_path = Path(model_path)
            lora_dir = str(base_path) + "_lora"
            nonlora_file = str(base_path) + "_nonlora.pt"
            meta_file = str(base_path) + "_meta.pt"

            # 判断是否为 LoRA 混合模型
            is_lora_hybrid = (
                Path(lora_dir).exists() and
                Path(nonlora_file).exists() and
                Path(meta_file).exists()
            )

            # ====== 1️⃣ 创建模型（必须与训练时参数一致）======
            model = TimeAwareGliomaSurvivalPredictor(
                text_encoder_name="google/medgemma-4b-it",
                freeze_text_encoder=False,
                text_output_dim=768,
                time_dim=128,
                time_encoding_type='fourier',
                latent_dim=767,
                num_modalities=4,
                predictor_hidden_dim=512,
                predictor_num_layers=4,
                predictor_num_heads=8,
                survival_hidden_dim=128,
                lambda_l1=5.0,
                lambda_cox=1,
                lambda_bce=1,
                dropout=0.2
            )

            # ====== 2️⃣ 若是 LoRA 混合模型，执行加载逻辑 ======
            if is_lora_hybrid:
                print(f"[WorldModel] Detected LoRA hybrid checkpoint")

                # --- 2.1 检测 adapter 文件类型 ---
                adapter_path_st = os.path.join(lora_dir, "adapter_model.safetensors")
                adapter_path_bin = os.path.join(lora_dir, "adapter_model.bin")
                adapter_path = None
                use_safetensors = False

                if os.path.exists(adapter_path_st):
                    adapter_path = adapter_path_st
                    use_safetensors = True
                elif os.path.exists(adapter_path_bin):
                    adapter_path = adapter_path_bin
                    use_safetensors = False

                # --- 2.2 修复 adapter key prefix mismatch ---
                if adapter_path is not None:
                    print(f"[Check] Inspecting adapter file: {adapter_path}")
                    if use_safetensors:
                        state = load_file(adapter_path)
                    else:
                        state = torch.load(adapter_path, map_location="cpu")

                    sample_key = next(iter(state.keys()))
                    print("Adapter keys look like:", sample_key)

                    if sample_key.startswith("base_model.model.base_model.model."):
                        print("[Fix] Removing redundant 'base_model.model.' prefix from adapter keys...")
                        fixed = {}
                        for k, v in state.items():
                            fixed[k.replace("base_model.model.base_model.model.", "base_model.model.")] = v
                        if use_safetensors:
                            save_file(fixed, adapter_path)
                        else:
                            torch.save(fixed, adapter_path)
                        print("[Fix] Adapter key prefix corrected ✅")

                # --- 2.3 加载非 LoRA 参数 ---
                print(f"  Loading non-LoRA parameters from {nonlora_file}")
                non_lora_state = torch.load(nonlora_file, map_location=self.device, weights_only=False)
                model.load_state_dict(non_lora_state, strict=False)

                # --- 2.4 加载 LoRA adapter ---
                print(f"  Loading LoRA adapter from {lora_dir}")
                if isinstance(model.shared_text_encoder.model, PeftModel):
                    print("  [Skip] Already PEFT-wrapped, loading adapter only.")
                    model.shared_text_encoder.model.load_adapter(lora_dir, adapter_name="default", is_trainable=False)
                else:
                    model.shared_text_encoder.model = PeftModel.from_pretrained(
                        model.shared_text_encoder.model,
                        lora_dir
                    )

                # --- 2.5 推理时合并权重 ---
                model.shared_text_encoder.model = model.shared_text_encoder.model.merge_and_unload()
                print(f"  ✓ Successfully loaded LoRA hybrid model")

            # ====== 3️⃣ 否则加载常规 checkpoint ======
            else:
                print(f"[WorldModel] Loading regular checkpoint")
                ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
                model.load_state_dict(ckpt['model_state_dict'])
                print(f"  ✓ Successfully loaded checkpoint")

            return model

        # ====== 4️⃣ 捕获异常 ======
        except Exception as e:
            import traceback
            print(f"[WorldModel Error] Failed to load model: {e}")
            traceback.print_exc()
            print(f"Running in mock mode.")
            return None

    def _enable_mc_dropout_only(self):
        # 冻结BN（若有），仅启用Dropout；简单实现，找不到则忽略
        for m in self.model.modules():
            if m.__class__.__name__.startswith("BatchNorm"):
                m.eval()
            if m.__class__.__name__ == "Dropout":
                m.train()

    def evaluate_sequence(
        self,
        pre_latent: torch.Tensor,
        clinical_text: str,
        sequence_json: str,        # ← {"pre":...,"post":...,"between":...} 的字符串
        time_delta_days: float,    # 估计总时长（用于你的模型，如果不需要可传 0）
        num_samples: int = 5
    ) -> Dict[str, float]:
        if self.model is None:
            return {"risk_score": float(np.random.rand()),
                    "survival_prob": float(np.random.rand()),
                    "uncertainty": float(np.random.rand() * 0.1)}

        risk_scores, surv_probs = [], []
        with torch.no_grad():
            self._enable_mc_dropout_only()
            pre_latent = pre_latent.to(self.device)
            td = torch.tensor([float(time_delta_days)], device=self.device, dtype=torch.float)

            for _ in range(num_samples):
                pred_latent, risk, surv_logit = self.model(
                    pre_latent=pre_latent,
                    drugs_text=[sequence_json],
                    time_delta=td,
                    clinical_text=[clinical_text]
                )
                risk_scores.append(float(risk))
                # 注意：若你的head输出不是logit，请相应修改
                surv_probs.append(torch.sigmoid(surv_logit).item())

        return {
            "risk_score": float(np.mean(risk_scores)),
            "survival_prob": float(np.mean(surv_probs)),
            "uncertainty": float(np.std(risk_scores))
        }

    def rollout_trajectory(
        self,
        pre_latent: torch.Tensor,
        clinical_text: str,
        sequence_json: str,
        time_delta_days: float,
        horizon: int = 3,
        discount_factor: float = 0.95,
        num_samples: int = 5,
    ) -> Dict[str, Any]:
        """
        Approximate long-horizon actor rollout by repeatedly applying the same
        candidate plan over imagined latent transitions.
        """
        if self.model is None:
            mock_risks = [float(np.random.rand()) for _ in range(max(horizon, 1))]
            discounted = sum((discount_factor ** step) * risk for step, risk in enumerate(mock_risks))
            return {
                "risk_score": float(np.mean(mock_risks)),
                "survival_prob": float(np.random.rand()),
                "uncertainty": float(np.random.rand() * 0.1),
                "step_risks": mock_risks,
                "step_survival_probs": [float(np.random.rand()) for _ in mock_risks],
                "discounted_risk": float(discounted),
                "horizon": len(mock_risks),
            }

        step_risks: List[float] = []
        step_survival_probs: List[float] = []
        step_uncertainties: List[float] = []

        with torch.no_grad():
            self._enable_mc_dropout_only()
            current_latent = pre_latent.to(self.device)

            for step in range(max(horizon, 1)):
                td = torch.tensor(
                    [float(max(time_delta_days, 1.0) * (step + 1))],
                    device=self.device,
                    dtype=torch.float,
                )

                sample_risks: List[float] = []
                sample_survival_probs: List[float] = []
                sample_next_latents: List[torch.Tensor] = []

                for _ in range(num_samples):
                    pred_latent, risk, surv_logit = self.model(
                        pre_latent=current_latent,
                        drugs_text=[sequence_json],
                        time_delta=td,
                        clinical_text=[clinical_text],
                    )
                    sample_risks.append(float(risk))
                    sample_survival_probs.append(float(torch.sigmoid(surv_logit).item()))
                    sample_next_latents.append(pred_latent.detach())

                mean_risk = float(np.mean(sample_risks))
                mean_survival_prob = float(np.mean(sample_survival_probs))
                uncertainty = float(np.std(sample_risks))

                step_risks.append(mean_risk)
                step_survival_probs.append(mean_survival_prob)
                step_uncertainties.append(uncertainty)
                current_latent = torch.stack(sample_next_latents, dim=0).mean(dim=0)

        discounted_risk = float(
            sum((discount_factor ** step) * risk for step, risk in enumerate(step_risks))
        )
        return {
            "risk_score": float(np.mean(step_risks)),
            "survival_prob": float(np.mean(step_survival_probs)),
            "uncertainty": float(np.mean(step_uncertainties)) if step_uncertainties else 0.0,
            "step_risks": step_risks,
            "step_survival_probs": step_survival_probs,
            "discounted_risk": discounted_risk,
            "horizon": len(step_risks),
        }


# ============================================================================ 
# 评分器：接入 Policy/toxicity_rules.py，简单复杂度项 & 不确定性惩罚
# ============================================================================ 
class SequenceScorer:
    def __init__(self, w_risk=1.0, w_tox=0.2, w_comp=0.1, w_unc=0.15):
        self.w_risk = w_risk
        self.w_tox = w_tox
        self.w_comp = w_comp
        self.w_unc = w_unc

    def score(
        self,
        seq: TreatmentSequence,
        rollout: Dict[str, float],
        clinical: ClinicalProfile,
        entropy_bonus: float = 0.0,
        entropy_temperature: float = 0.0,
    ) -> TreatmentSequence:
        risk = float(rollout.get("discounted_risk", rollout["risk_score"]))
        surv = float(rollout.get("survival_prob", 0.0))
        tox  = float(compute_toxicity(seq, clinical))
        comp = float(len(seq.blocks) * 0.1)
        unc  = float(rollout["uncertainty"])
        total = (
            self.w_risk * risk
            + self.w_tox * tox
            + self.w_comp * comp
            + self.w_unc * unc
            - entropy_temperature * entropy_bonus
        )

        seq.risk_score = risk
        seq.survival_prob = surv              # ← 新增
        seq.toxicity_score = tox
        seq.complexity_score = comp
        seq.total_score = total
        seq.entropy_bonus = float(entropy_bonus)
        seq.discounted_risk = float(rollout.get("discounted_risk", risk))
        seq.rollout_horizon = int(rollout.get("horizon", 1))
        seq.step_risks = [float(x) for x in rollout.get("step_risks", [])]
        seq.step_survival_probs = [float(x) for x in rollout.get("step_survival_probs", [])]
        return seq


# ============================================================================ 
# 探索器：LLM→Guardrails→Variants→同构JSON评估→排序（Top-K）
# ============================================================================ 
class SequenceExplorer:
    def __init__(self, world: SequenceWorldModel, scorer: SequenceScorer,
                 variant_gen: ParameterVariantGenerator, guard: ClinicalGuardrails,
                 pre_payload: Dict[str, Any], between_payload: List[Dict[str, Any]],
                 rng: Optional[np.random.Generator] = None):
        self.world = world
        self.scorer = scorer
        self.variant_gen = variant_gen
        self.guard = guard
        self.pre_payload = pre_payload
        self.between_payload = between_payload
        self.rng = rng or np.random.default_rng()
        self._rollout_cache: Dict[Tuple, Dict[str, float]] = {}
        if hasattr(self.variant_gen, "set_rng"):
            self.variant_gen.set_rng(self.rng)

    def reset_rng(self, seed: Optional[int] = None):
        """Reset RNG state for reproducibility."""
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)
        if hasattr(self.variant_gen, "set_rng"):
            self.variant_gen.set_rng(self.rng)

    def _freeze_value(self, value: Any):
        if isinstance(value, dict):
            return tuple(sorted((k, self._freeze_value(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple, set)):
            return tuple(self._freeze_value(v) for v in value)
        return value

    def _sequence_key(self, sequence: TreatmentSequence) -> Tuple:
        blocks_repr = []
        for block in sequence.blocks:
            params_tuple = tuple(sorted((k, self._freeze_value(v)) for k, v in block.params.items()))
            blocks_repr.append((block.kind, params_tuple))
        return tuple(blocks_repr)

    def build_feedback_summary(self, sequences: List[TreatmentSequence], top_n: int = 3) -> str:
        if not sequences:
            return "No valid sequences were found in the previous round. Generate fresh diverse options."

        lines = [
            "Previous round trajectory evaluations (lower total score and lower discounted risk are better):"
        ]
        for idx, seq in enumerate(sequences[:top_n], 1):
            actions = seq.to_post_actions()
            actions_text = json.dumps(actions, ensure_ascii=False)
            step_risks = getattr(seq, "step_risks", [])
            step_risk_text = ", ".join(f"{x:.3f}" for x in step_risks) if step_risks else "n/a"
            lines.append(
                f"{idx}. total={seq.total_score:.3f}, discounted_risk={getattr(seq, 'discounted_risk', seq.risk_score):.3f}, "
                f"mean_risk={seq.risk_score:.3f}, tox={seq.toxicity_score:.3f}, comp={seq.complexity_score:.3f}, "
                f"entropy_bonus={getattr(seq, 'entropy_bonus', 0.0):.3f}, step_risks=[{step_risk_text}]; plan={actions_text}"
            )
        return "\n".join(lines)

    def _count_drugs(self, sequence: TreatmentSequence) -> int:
        """计算一个序列中的药物数量"""
        count = 0
        for block in sequence.blocks:
            if block.kind in ("chemotherapy", "immunotherapy", "additional_1", "additional_2", "other_therapy"):
                count += 1
        return count

    

    def search(
        self,
        initial_state: torch.Tensor,
        llm_sequences: List[TreatmentSequence],
        clinical: ClinicalProfile,
        imaging: ImagingProfile,
        num_variants_per_llm: int = 10,
        top_k: int = 10,
        drug_number: int = None,
        planning_horizon: int = 3,
        discount_factor: float = 0.95,
        entropy_temperature: float = 0.15,
        rollout_samples: int = 5,
    ) -> List[TreatmentSequence]:

        print(f"\n{'='*80}\nSequence Exploration")
        print(f"LLM Sequences: {len(llm_sequences)} | Variants/seq: {num_variants_per_llm}")
        if drug_number is not None:
            print(f"Target drug count constraint: {drug_number}")
        print(
            f"Planning horizon: {planning_horizon} | Discount: {discount_factor:.2f} | "
            f"Entropy temperature: {entropy_temperature:.2f}"
        )
        print(f"{'='*80}\n")

        # ----------------------------------------------------------------------
        # Case 0: drug_number == 0 → 仅生成一个空序列 (no-therapy)
        # ----------------------------------------------------------------------
        if drug_number == 0:
            print(f"\nDetected drug_number=0 → Generating a null (no-therapy) candidate only.")
            null_seq = TreatmentSequence(blocks=[], source="LLM_null")
            rollout = self.world.evaluate_sequence(
                pre_latent=initial_state,
                clinical_text=clinical.to_text(),
                sequence_json=null_seq.to_model_triplet_json(self.pre_payload, tp_post="TP_post", between=self.between_payload),
                time_delta_days=0.0,
                num_samples=5
            )
            scored = self.scorer.score(null_seq, rollout, clinical)
            print(f"   Total:{scored.total_score:.4f} | Risk:{scored.risk_score:.4f} | Tox:{scored.toxicity_score:.4f}")
            return [scored]

        # ----------------------------------------------------------------------
        # 1) Guardrails 修复 + 验证
        # ----------------------------------------------------------------------
        seeds: List[TreatmentSequence] = []
        for i, s in enumerate(llm_sequences, 1):
            fixed = self.guard.repair(s)
            errs = self.guard.validate_sequence(fixed)
            seeds.append(fixed)

        # ----------------------------------------------------------------------
        # 内部辅助函数：强制匹配药物数量
        # ----------------------------------------------------------------------
        def _match_drugs(sequence: TreatmentSequence, target_drug_num: int) -> TreatmentSequence:
            """
            调整序列中药物 blocks 的数量：
            - 药物类别不重复（每种 kind 仅保留一个）
            - 总数量等于 target_drug_num
            """
            allowed_kinds = ("chemotherapy", "immunotherapy", "additional_1", "additional_2", "other_therapy")

            # 1. 去除重复类别（保留每类第一个）
            unique_blocks: List[TreatmentBlock] = []
            seen_kinds = set()
            for b in sequence.blocks:
                if b.kind in allowed_kinds:
                    if b.kind not in seen_kinds:
                        unique_blocks.append(b)
                        seen_kinds.add(b.kind)
                else:
                    unique_blocks.append(b)  # 非药物类保留

            # 2. 只保留药物 blocks
            drug_blocks: List[TreatmentBlock] = [b for b in unique_blocks if b.kind in allowed_kinds]

            # 3. 若药物数量太多 → 截断
            if len(drug_blocks) > target_drug_num:
                drug_blocks = drug_blocks[:target_drug_num]

            # 4. 若药物数量太少 → 补足默认药物
            while len(drug_blocks) < target_drug_num:
                current_kinds = [b.kind for b in drug_blocks]
                remaining_kinds = [k for k in allowed_kinds if k not in current_kinds]
                if not remaining_kinds:
                    remaining_kinds = list(allowed_kinds)
                default_kind = str(self.rng.choice(remaining_kinds))
                default_params = {"agent": "Temozolomide", "cycle_length_days": 28, "num_cycles": 6}
                drug_blocks.append(TreatmentBlock(default_kind, default_params, 0, 0, "auto_added"))

            # 5. 非药物类保持原样 + 调整后药物
            non_drug_blocks = [b for b in unique_blocks if b.kind not in allowed_kinds]
            sequence.blocks = non_drug_blocks + drug_blocks
            return sequence

        # ----------------------------------------------------------------------
        # 2) 生成变体 + 修复 + 药物数量匹配
        # ----------------------------------------------------------------------
        candidates: List[TreatmentSequence] = []
        seen_sequences: Set[Tuple] = set()

        def _register(seq: TreatmentSequence) -> bool:
            key = self._sequence_key(seq)
            if key in seen_sequences:
                return False
            seen_sequences.add(key)
            candidates.append(seq)
            return True

        for i, seed in enumerate(seeds, 1):
            print(f"Seed {i}/{len(seeds)}: {seed.get_summary()}")

            if drug_number is not None:
                seed = _match_drugs(seed, drug_number)

            _register(seed)
            if hasattr(self.variant_gen, "set_rng"):
                self.variant_gen.set_rng(self.rng)
            variants = self.variant_gen.generate_variants(seed, num_variants=num_variants_per_llm)
            for v in variants:
                vf = self.guard.repair(v)
                if drug_number is not None:
                    vf = _match_drugs(vf, drug_number)
                if not self.guard.validate_sequence(vf):
                    _register(vf)

        print(f"\nTotal candidates (after repair and matching): {len(candidates)}")
        print("Evaluating...\n")

        action_counts: Dict[str, int] = {}
        for seq in candidates:
            key = json.dumps(seq.to_post_actions(), sort_keys=True, ensure_ascii=False)
            action_counts[key] = action_counts.get(key, 0) + 1

        # ----------------------------------------------------------------------
        # 3) 世界模型长程评估 + 熵正则打分
        # ----------------------------------------------------------------------
        results: List[TreatmentSequence] = []
        for seq in candidates:
            key = self._sequence_key(seq)
            rollout = self._rollout_cache.get(key)
            if rollout is None:
                seq_json = seq.to_model_triplet_json(self.pre_payload, tp_post="TP_post", between=self.between_payload)
                td = float(seq.estimated_total_days()) if hasattr(seq, "estimated_total_days") else 0.0
                rollout = self.world.rollout_trajectory(
                    pre_latent=initial_state,
                    clinical_text=clinical.to_text(),
                    sequence_json=seq_json,
                    time_delta_days=td,
                    horizon=planning_horizon,
                    discount_factor=discount_factor,
                    num_samples=rollout_samples,
                )
                self._rollout_cache[key] = rollout
            action_key = json.dumps(seq.to_post_actions(), sort_keys=True, ensure_ascii=False)
            novelty = 1.0 / max(action_counts.get(action_key, 1), 1)
            entropy_bonus = -novelty * math.log(max(novelty, 1e-8))
            results.append(
                self.scorer.score(
                    seq,
                    rollout,
                    clinical,
                    entropy_bonus=entropy_bonus,
                    entropy_temperature=entropy_temperature,
                )
            )

        # ----------------------------------------------------------------------
        # 4) 排序 + 输出 Top-K
        # ----------------------------------------------------------------------
        results.sort(key=lambda s: s.total_score)
        print(f"\n{'='*80}\nTop {top_k} Sequences\n{'='*80}")
        for i, s in enumerate(results[:top_k], 1):
            print(f"{i}. {s}")
            print(
                f"   Total:{s.total_score:.4f} | DiscRisk:{getattr(s, 'discounted_risk', s.risk_score):.4f} | "
                f"MeanRisk:{s.risk_score:.4f} | Entropy:{getattr(s, 'entropy_bonus', 0.0):.4f} | "
                f"Tox:{s.toxicity_score:.4f} | Comp:{s.complexity_score:.4f} | Src:{s.source}"
            )

        return results


# ============================================================================ 
# 主流程：给定历史 pre/between、临床/影像，探索 post（Top-K）
# ============================================================================ 
def glioma_sequence_exploration(
    patient_id: str,
    pre_treatment_latent: torch.Tensor,
    clinical: ClinicalProfile,
    imaging: ImagingProfile,
    pre_payload: Dict[str, Any],                 # ← 历史 (与训练同构里的 "pre" 对齐)
    between_payload: List[Dict[str, Any]],       # ← 可为空列表
    api_key: str,
    world_model_path: str,
    num_llm_sequences: int = 5,
    num_variants: int = 10,
    top_k: int = 10,
    device: str = 'cuda:0',
    gt_drug_number: int = None,
    feedback_rounds: int = 0,
    feedback_top_k: int = 3,
    rng_seed: Optional[int] = None,
    planning_horizon: int = 3,
    discount_factor: float = 0.95,
    entropy_temperature: float = 0.15,
    rollout_samples: int = 5,
) -> Dict[str, Any]:

    print(f"\n{'#'*80}\n# Glioma Sequence Exploration (Triplet-Compatible)\n# Patient: {patient_id}\n{'#'*80}\n")

    # Step 1: LLM 生成 post.actions（六大类）
    print("Step 1: Generating POST candidates via LLM...")
    policy = SequenceLLMPolicy(api_key=api_key)
    llm_sequences = policy.generate_post_sequences(
        clinical_text=clinical.to_text(),
        imaging_dict=imaging.to_dict(),
        pre_payload=pre_payload,
        num_sequences=num_llm_sequences
    )
    print(f"  ✓ LLM candidates: {len(llm_sequences)}\n")

    # Step 2: 初始化组件
    print("Step 2: Initializing components...")
    world = SequenceWorldModel(world_model_path, device)
    scorer = SequenceScorer(w_risk=1.0, w_tox=0.2, w_comp=0.1, w_unc=0.15)
    variant_gen = ParameterVariantGenerator()
    guard = ClinicalGuardrails(strict=True)
    explorer_rng = np.random.default_rng(rng_seed) if rng_seed is not None else None

    # Step 3: 探索与评估
    print("\nStep 3: Exploring...")
    explorer = SequenceExplorer(world, scorer, variant_gen, guard, pre_payload, between_payload, rng=explorer_rng)
    ranked = explorer.search(
        initial_state=pre_treatment_latent,
        llm_sequences=llm_sequences,
        clinical=clinical,
        imaging=imaging,
        num_variants_per_llm=num_variants,
        top_k=top_k,
        drug_number=gt_drug_number,
        planning_horizon=planning_horizon,
        discount_factor=discount_factor,
        entropy_temperature=entropy_temperature,
        rollout_samples=rollout_samples,
    )
    all_ranked = list(ranked)

    prev_ranked = ranked
    for round_idx in range(min(feedback_rounds, 5)):
        if not prev_ranked:
            break
        feedback_summary = explorer.build_feedback_summary(prev_ranked, top_n=feedback_top_k)
        print(f"\n[Feedback] Round {round_idx + 1}: Sending summary to LLM:\n{feedback_summary}\n")

        llm_sequences_fb = policy.generate_post_sequences(
            clinical_text=clinical.to_text(),
            imaging_dict=imaging.to_dict(),
            pre_payload=pre_payload,
            num_sequences=num_llm_sequences,
            feedback=feedback_summary
        )
        print(f"  ✓ Feedback round {round_idx + 1} candidates: {len(llm_sequences_fb)}")

        prev_ranked = explorer.search(
            initial_state=pre_treatment_latent,
            llm_sequences=llm_sequences_fb,
            clinical=clinical,
            imaging=imaging,
            num_variants_per_llm=num_variants,
            top_k=top_k,
            drug_number=gt_drug_number,
            planning_horizon=planning_horizon,
            discount_factor=discount_factor,
            entropy_temperature=entropy_temperature,
            rollout_samples=rollout_samples,
        )
        all_ranked.extend(prev_ranked)

    # Merge & deduplicate across rounds
    deduped_results: List[TreatmentSequence] = []
    seen_overall: Set[Tuple] = set()
    for seq in all_ranked:
        key = explorer._sequence_key(seq)
        if key in seen_overall:
            continue
        seen_overall.add(key)
        deduped_results.append(seq)

    deduped_results.sort(key=lambda s: s.total_score)
    ranked = deduped_results

    return {
        "patient_id": patient_id,
        "best_sequence": ranked[0] if ranked else None,
        "top_sequences": ranked[:top_k],
        "num_evaluated": len(ranked)
    }

# ============================================================================ 
# 批量处理多个患者的辅助函数
# ============================================================================ 
def batch_exploration(
    patient_data_list: List[Dict[str, Any]],
    api_key: str,
    world_model_path: str,
    output_dir: str = "./exploration_results",
    device: str = 'cuda:0'
):
    """
    批量处理多个患者的治疗序列探索
    
    Args:
        patient_data_list: 每个元素包含 {
            "patient_id": str,
            "context_static": dict,
            "pre_latent": torch.Tensor,
            "pre_payload": dict,
            "between_payload": list,
            "imaging": ImagingProfile (optional)
        }
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    results_summary = []
    
    for i, patient_data in enumerate(patient_data_list, 1):
        print(f"\n{'#'*80}")
        print(f"Processing Patient {i}/{len(patient_data_list)}: {patient_data['patient_id']}")
        print(f"{ '#' * 80}\n")
        
        # 创建临床特征
        clinical = ClinicalProfile.from_context_static(
            patient_id=patient_data["patient_id"],
            context_static=patient_data["context_static"],
            **patient_data.get("clinical_extras", {})
        )
        
        # 执行探索
        results = glioma_sequence_exploration(
            patient_id=clinical.patient_id,
            pre_treatment_latent=patient_data["pre_latent"],
            clinical=clinical,
            imaging=patient_data.get("imaging"),
            pre_payload=patient_data["pre_payload"],
            between_payload=patient_data["between_payload"],
            api_key=api_key,
            world_model_path=world_model_path,
            device=device
        )
        
        # 保存结果
        output_path = os.path.join(output_dir, f"{clinical.patient_id}_exploration.json")
        best = results["best_sequence"]
        out = {
            "patient_id": results["patient_id"],
            "best_sequence": {
                "blocks": [{"kind": b.kind, "params": b.params} for b in (best.blocks if best else [])],
                "scores": {
                    "risk_score": float(best.risk_score) if best else None,
                    "discounted_risk": float(getattr(best, "discounted_risk", 0.0)) if best else None,
                    "total_score": float(best.total_score) if best else None
                }
            }
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        
        results_summary.append({
            "patient_id": clinical.patient_id,
            "best_risk_score": float(best.risk_score) if best else None,
            "output_file": output_path
        })
    
    # 保存批量摘要
    summary_path = os.path.join(output_dir, "batch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Batch processing complete. Summary saved to: {summary_path}")
    return results_summary


# ============================================================================ 
# 示例运行（请改为你的真实 pre/between）
# ============================================================================ 
# main.py (continued)
# ============================================================================ 
# 示例运行（使用真实数据格式）
# ============================================================================ 
if __name__ == "__main__":
    API_KEY = os.environ.get("OPENAI_API_KEY")
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required to run main.py")
    WORLD_MODEL_PATH = "./Predictor/checkpoints/context/best_c_index.pth"
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    context_static = {
        "sex_at_birth": "female",
        "race": "white",
        "age_at_diagnosis_years": 57.0,
        "primary_diagnosis": "gbm",
        "who_grade": 4,
        "genomics": {
            "idh1": 0,              # IDH1 wildtype
            "idh2": 0,              # IDH2 wildtype
            "atrx": 2,              # unknown
            "mgmt_methylation": 1,  # methylated (预后较好)
            "braf_v600e": 2,
            "tert_promoter": 2,
            "chr7_gain_chr10_loss": 1,  # 典型GBM特征
            "h3_3a": 2,
            "egfr_amp": 1,          # EGFR扩增
            "pten": 0,              # PTEN wildtype
            "cdkn2ab_deletion": 1,  # CDKN2A/B缺失
            "tp53_alteration": 1,   # TP53突变
            "codeletion_1p19q_detail": "0",  # 非寡突胶质瘤
            "other_mutations_text": None
        }
    }
    
    clinical = ClinicalProfile.from_context_static(
        patient_id="GBM_001",
        context_static=context_static,
        # 可选附加字段
        kps=80,
        ecog=1,
        extent_of_resection="GTR",  # Gross Total Resection
        prior_treatments=["maximal_safe_resection"],
        comorbidities=["hypertension", "type_2_diabetes"],
        disease_status="newly_diagnosed",
        days_from_diagnosis=14
    )
    
    print("\n" + "="*80)
    print("Patient Clinical Summary")
    print("="*80)
    print(clinical.to_text())
    print("\nMolecular Classification:", clinical.get_molecular_classification())
    print("="*80 + "\n")

    # ======================================================================== 
    # 2. 影像特征（如果你有ImagingProfile，保持原样）
    # ======================================================================== 
    imaging = ImagingProfile(
        num_lesions=1,
        total_volume_cc=35.5,
        locations=["frontal_right"],
        min_distance_to_oar_mm=8.0,
        enhancement_pattern="ring",
        edema_volume_cc=65.0,
        mass_effect=True,
        multifocal=False,
        deep_seated=False
    )

    # ======================================================================== 
    # 3. 历史治疗输入（与训练同构）
    # ======================================================================== 
    # pre_payload: 患者在此次探索前已经接受的治疗
    # 例如：术后已完成的concurrent chemotherapyRT
    pre_payload = {
        "pre": {
            "tp": "TP_baseline",  # baseline时间点（术后2周）
            "actions": {
                "radiation": [
                    {
                        "technique": "IMRT",
                        "dose_gy": 60.0,
                        "fractions": 30
                    }
                ],
                "chemotherapy": [
                    {
                        "agent": "Temozolomide",
                        "cycle_length_days": 28,
                        "num_cycles": 6  # concurrent期间的TMZ
                    }
                ]
            }
        }
    }
    
    # between_payload: 如果有中间时间点的治疗，可以添加
    # 例如：在pre和post之间的维持治疗
    between_payload: List[Dict[str, Any]] = [
        # 示例：如果患者在concurrent RT+TMZ后有gap期的supportive care
        # {
        #     "additional_2": [
        #         {"agent": "Dexamethasone", "cycle_length_days": 28, "num_cycles": 2}
        #     ]
        # }
    ]

    # ======================================================================== 
    # 4. Pre-treatment latent（真实场景下从模型编码器获取）
    # ======================================================================== 
    # 实际使用时应该从你的影像数据通过encoder得到
    # 这里用随机tensor模拟
    pre_latent = torch.randn(1, 4, 767).to(DEVICE)
    
    # 如果你有真实的影像特征提取器：
    # from your_image_encoder import extract_latent
    # pre_latent = extract_latent(mri_data).to(DEVICE)  # [1, 4, 767]

    # ======================================================================== 
    # 5. 执行治疗序列探索
    # ======================================================================== 
    print("Starting treatment sequence exploration...")
    print(f"Device: {DEVICE}")
    print(f"World Model: {WORLD_MODEL_PATH}")
    print(f"LLM Sequences: 5 | Variants per sequence: 10 | Top-K: 10\n")
    
    results = glioma_sequence_exploration(
        patient_id=clinical.patient_id,
        pre_treatment_latent=pre_latent,
        clinical=clinical,
        imaging=imaging,
        pre_payload=pre_payload,
        between_payload=between_payload,
        api_key=API_KEY,
        world_model_path=WORLD_MODEL_PATH,
        num_llm_sequences=5,      # LLM生成5个初始候选
        num_variants=10,          # 每个候选生成10个参数变体
        top_k=10,                 # 返回Top-10最优方案
        device=DEVICE
    )

    # ======================================================================== 
    # 6. 保存结果
    # ======================================================================== 
    best = results["best_sequence"]
    
    # 构建输出JSON
    out = {
        "patient_id": results["patient_id"],
        "clinical_summary": clinical.to_text(),
        "molecular_classification": clinical.get_molecular_classification(),
        "num_evaluated": results["num_evaluated"],
        "best_sequence": {
            "blocks": [
                {
                    "kind": b.kind,
                    "params": b.params,
                    "start_day": b.start_day,
                    "end_day": b.end_day,
                    "rationale": b.rationale
                }
                for b in (best.blocks if best else [])
            ],
            "scores": {
                "risk_score": float(best.risk_score) if best else None,
                "discounted_risk": float(getattr(best, 'discounted_risk', 0.0)) if best else None,
                "survival_prob": float(getattr(best, 'survival_prob', 0.0)) if best else None,
                "toxicity_score": float(best.toxicity_score) if best else None,
                "complexity_score": float(best.complexity_score) if best else None,
                "entropy_bonus": float(getattr(best, 'entropy_bonus', 0.0)) if best else None,
                "step_risks": [float(x) for x in getattr(best, 'step_risks', [])] if best else [],
                "total_score": float(best.total_score) if best else None
            },
            "source": best.source if best else None,
            "summary": best.get_summary() if best else None
        },
        "top_10_sequences": [
            {
                "rank": i + 1,
                "total_score": float(seq.total_score),
                "risk_score": float(seq.risk_score),
                "discounted_risk": float(getattr(seq, 'discounted_risk', 0.0)),
                "toxicity_score": float(seq.toxicity_score),
                "entropy_bonus": float(getattr(seq, 'entropy_bonus', 0.0)),
                "summary": seq.get_summary()
            }
            for i, seq in enumerate(results["top_sequences"][:10])
        ],
        "metadata": {
            "timestamp": str(torch.cuda.Event(enable_timing=False)),
            "device": DEVICE,
            "model_path": WORLD_MODEL_PATH,
            "genomic_profile": clinical.genomics.to_dict(),
            "imaging_profile": imaging.to_dict() if hasattr(imaging, 'to_dict') else None
        }
    }
    
    # 保存JSON
    output_filename = f"sequence_exploration_{clinical.patient_id}.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"✅ Results saved to: {output_filename}")
    print(f"{ '='*80}\n")
    
    # 打印最优方案摘要
    if best:
        print("🏆 Best Treatment Sequence:")
        print(f"   {best.get_summary()}")
        print(f"\n   Risk Score: {best.risk_score:.4f}")
        print(f"   Toxicity Score: {best.toxicity_score:.4f}")
        print(f"   Total Score: {best.total_score:.4f}")
        print(f"\n   Composition:")
        for i, block in enumerate(best.blocks, 1):
            print(f"      {i}. {block.kind.upper()}: {block.params}")
    
    print("\n" + "="*80)
    print("Exploration Complete!")
    print("="*80 + "\n")
