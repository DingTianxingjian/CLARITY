"""
ise_llm.py  —  LLM-powered candidate generator for the ISE pipeline.

Two-stage ISE design:
  Stage 1 : LLM generates 3-5 patient-specific treatment candidates (JSON)
  Stage 2 : World model evaluates / ranks those candidates
  Optional : Feedback loop — world model scores → LLM refines proposals → re-rank

Usage:
    from ise_llm import ISECandidateGenerator
    gen = ISECandidateGenerator(api_key="...", model="claude-opus-4-8")
    candidates = gen.propose(patient_context, pre_actions)
    # ... world model scores them ...
    refined   = gen.propose(patient_context, pre_actions, feedback=scores)
"""
import json, os, re
from typing import Optional

from Policy.types import TreatmentBlock, TreatmentSequence

# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a neuro-oncologist AI proposing POST treatment actions for glioma patients.

### Task
1. Analyze the patient's pre-treatment clinical context provided by the user.
2. Output 3-5 potential POST treatment therapy sequences as JSON candidates.
   Candidates should cover a clinically meaningful spectrum — from standard-of-care
   to reasonable alternatives — but must all be *appropriate* for this patient's
   specific situation (phase, molecular profile, prior therapy).

### Constraints & Safety Rules

#### Dosage & Cycles
- radiation:   dose_gy ∈ [40, 60],  fractions ∈ [10, 33]
- drugs:       cycle_length_days ∈ [14, 56],  num_cycles ∈ [1, 12]

#### Clinical Mapping Rules
- Maintain temporal and clinical consistency with the PRE treatment payload.
- Do NOT repeat identical radiation fields already in the PRE payload unless
  explicitly indicating salvage re-irradiation at recurrence.
- Do NOT suggest bevacizumab (BEV) as standalone adjuvant therapy for patients
  who have NOT yet experienced radiographic progression — BEV is salvage, not adjuvant.
- For MGMT-methylated patients, prefer extended TMZ (12 cycles) over 6.
- For oligodendroglioma (1p/19q co-deleted), prefer PCV-based regimens.
- If the patient is still in adjuvant phase (post-CRT, no progression): candidates
  should be adjuvant variants, not salvage.
- If progression is documented: candidates should be salvage regimens.
- Include "no active therapy / observation" ONLY if clinically appropriate
  (e.g., elderly/frail patient, stable low-grade glioma under surveillance).

#### Feedback Loop
If the user message includes a "feedback" section, it contains risk scores from the
world model (lower total_score = better predicted survival). You MUST adjust your
next proposals to reduce the risk score — e.g., de-escalate if high-toxicity plans
scored poorly, or escalate if low-intensity plans scored poorly relative to alternatives.

### Output Schema (strict JSON, no prose outside the JSON block)
```json
{
  "candidates": [
    {
      "post": {
        "tp": "TP_post",
        "actions": {
          "radiation": [{"dose_gy": 60, "fractions": 30}],
          "chemotherapy": [{"agent": "Temozolomide", "cycle_length_days": 28, "num_cycles": 6}],
          "additional_1": [{"agent": "Bevacizumab", "cycle_length_days": 14, "num_cycles": 8}],
          "other_therapy": [{"agent": "Optune TTF"}]
        }
      },
      "rationale": "<one-line clinical justification>"
    }
  ]
}
```
Omit empty arrays. "actions": {} is valid for observation-only candidates.
Valid action keys: radiation, chemotherapy, immunotherapy, additional_1, additional_2, other_therapy.
"""

# ── User prompt builder ───────────────────────────────────────────────────────

def _build_user_prompt(patient_context: dict, pre_actions: dict,
                       feedback: Optional[list] = None) -> str:
    """
    Build the user-turn message:
      - Patient clinical context (WHO grade, IDH, MGMT, 1p19q, ...)
      - Pre-treatment actions
      - Optional: world model feedback from the previous round
    """
    g     = patient_context.get("genomics", {})
    grade = patient_context.get("who_grade", "unknown")
    idh   = "mutant" if g.get("idh1", 2) == 1 or g.get("idh2", 2) == 1 else "wildtype"
    mgmt  = "methylated" if g.get("mgmt_methylation", 2) == 1 else "unmethylated/unknown"
    del1  = "yes" if str(g.get("codeletion_1p19q_detail", "0")) == "1" else "no"
    kps   = patient_context.get("kps", "unknown")
    age   = patient_context.get("age_at_diagnosis", "unknown")

    lines = [
        "### Patient Clinical Context",
        f"- WHO grade: {grade}",
        f"- IDH status: {idh}",
        f"- MGMT promoter methylation: {mgmt}",
        f"- 1p/19q co-deletion: {del1}",
        f"- KPS: {kps}",
        f"- Age at diagnosis: {age}",
        "",
        "### PRE Treatment Actions (what this patient has already received)",
        json.dumps(pre_actions, indent=2),
        "",
        "### Request",
        "Generate 3-5 clinically appropriate POST treatment candidates for this patient.",
    ]

    if feedback:
        lines += [
            "",
            "### World Model Feedback (from previous round)",
            "The following candidates were evaluated. Lower total_score = better survival.",
            json.dumps(feedback, indent=2),
            "",
            "Revise your proposals to reduce the risk score while maintaining clinical validity.",
        ]

    return "\n".join(lines)


# ── JSON parser ───────────────────────────────────────────────────────────────

def _parse_candidates(raw_text: str) -> list:
    """Extract the JSON block from LLM output and parse candidates."""
    # Try to find ```json ... ``` block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if m:
        raw_text = m.group(1)
    # Fall back: first { ... } block
    m = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON found in LLM output:\n{raw_text[:500]}")
    data = json.loads(m.group(0))
    return data.get("candidates", [])


def _candidate_to_sequence(cand: dict) -> TreatmentSequence:
    """Convert a single parsed candidate dict → TreatmentSequence."""
    actions = cand.get("post", {}).get("actions", {})
    blocks  = []
    VALID_KEYS = ["radiation", "chemotherapy", "immunotherapy",
                  "additional_1", "additional_2", "other_therapy"]
    for key in VALID_KEYS:
        for item in actions.get(key, []):
            blocks.append(TreatmentBlock(kind=key, params=dict(item),
                                         start_day=0, end_day=0))
    seq         = TreatmentSequence(blocks=blocks, source="llm_api")
    seq.rationale = cand.get("rationale", "")
    return seq


# ── Main class ────────────────────────────────────────────────────────────────

class ISECandidateGenerator:
    """
    Wraps an LLM API to generate patient-specific treatment candidates.
    Supports Claude (Anthropic) and OpenAI-compatible endpoints.
    """

    def __init__(self, api_key: Optional[str] = None,
                 model: str = "claude-opus-4-8",
                 provider: str = "anthropic",
                 max_tokens: int = 2048,
                 temperature: float = 0.3):
        self.model       = model
        self.provider    = provider
        self.max_tokens  = max_tokens
        self.temperature = temperature

        key = api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("Set ANTHROPIC_API_KEY or OPENAI_API_KEY, or pass api_key=")

        if provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=key)
        else:
            import openai
            self._client = openai.OpenAI(api_key=key)

    # ── Low-level call ──────────────────────────────────────────────────

    def _call(self, user_message: str) -> str:
        if self.provider == "anthropic":
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            return resp.content[0].text
        else:  # openai-compatible
            resp = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "system",  "content": SYSTEM_PROMPT},
                    {"role": "user",    "content": user_message},
                ],
            )
            return resp.choices[0].message.content

    # ── Public API ──────────────────────────────────────────────────────

    def propose(self,
                patient_context: dict,
                pre_actions: dict,
                feedback: Optional[list] = None) -> list:
        """
        Generate treatment candidates.

        Args:
            patient_context : dict from full_ds.index (genomics, who_grade, kps, ...)
            pre_actions     : pre-treatment actions dict (from drugs_text["pre"]["actions"])
            feedback        : list of dicts from world model:
                              [{"label": str, "total_score": float,
                                "risk": float, "surv": float, "tox": float}, ...]

        Returns:
            list of (label: str, seq: TreatmentSequence)
        """
        user_msg   = _build_user_prompt(patient_context, pre_actions, feedback)
        raw        = self._call(user_msg)
        parsed     = _parse_candidates(raw)

        results = []
        for cand in parsed:
            try:
                seq   = _candidate_to_sequence(cand)
                label = cand.get("rationale", f"candidate_{len(results)+1}")[:60]
                results.append((label, seq))
            except Exception as e:
                print(f"  [ISECandidateGenerator] skipping malformed candidate: {e}")
        return results

    def propose_with_feedback_loop(self,
                                   patient_context: dict,
                                   pre_actions: dict,
                                   world_model,
                                   pre_latent,
                                   clinical_text: str,
                                   pre_payload: dict,
                                   time_delta_days: float,
                                   clinical_profile,
                                   scorer,
                                   n_rounds: int = 2,
                                   horizon: int = 3,
                                   n_samples: int = 5) -> list:
        """
        Full ISE feedback loop:
          Round 1: LLM proposes candidates → world model scores
          Round 2: scores fed back to LLM → refined candidates → world model re-scores
          Returns scored + ranked TreatmentSequences from the final round.

        Args:
            world_model    : SequenceWorldModel instance
            pre_latent     : torch.Tensor [1, 32, 768]
            clinical_profile: ClinicalProfile instance
        """
        feedback = None
        last_scored = []

        for rnd in range(n_rounds):
            candidates = self.propose(patient_context, pre_actions, feedback)
            if not candidates:
                print(f"  [ISE round {rnd+1}] LLM returned no valid candidates.")
                break

            # World model evaluation
            scored = []
            for label, seq in candidates:
                seq_json = seq.to_model_triplet_json(
                    pre_payload, tp_post="TP_post", between=[]
                )
                rollout = world_model.rollout_trajectory(
                    pre_latent=pre_latent,
                    clinical_text=clinical_text,
                    sequence_json=seq_json,
                    time_delta_days=time_delta_days,
                    horizon=horizon,
                    num_samples=n_samples,
                )
                s = scorer.score(seq, rollout, clinical_profile)
                s.source = label
                scored.append(s)

            scored.sort(key=lambda x: x.total_score)
            last_scored = scored

            if rnd < n_rounds - 1:
                # Build feedback for next round
                feedback = [
                    {
                        "label":       s.source,
                        "total_score": round(s.total_score, 4),
                        "risk":        round(s.discounted_risk, 4),
                        "surv_prob":   round(s.survival_prob, 4),
                        "toxicity":    round(s.toxicity_score, 4),
                    }
                    for s in scored
                ]
                print(f"  [ISE round {rnd+1}] best={scored[0].source}  "
                      f"score={scored[0].total_score:.4f} → feeding back to LLM")

        return last_scored
