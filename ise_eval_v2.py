"""
ise_eval_v2.py  —  ISE F1 Evaluation (LLM + World-Model Pipeline)

Two-stage ISE:
  1. LLM generates 3-5 patient-specific, phase-appropriate candidates.
     - With API key (ANTHROPIC_API_KEY / OPENAI_API_KEY): calls real LLM via ise_llm.py
     - Without key: falls back to clinical rule engine (phase-locked, same logic)
  2. World model evaluates the candidates and picks Top-1.
  3. F1(ISE Top-1, GT)  vs  F1(LLM Top-1, GT)  vs  Oracle
"""
import sys, os, json, re, copy
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, "Predictor")
for _d in (".", ""):
    if _d not in sys.path:
        sys.path.append(_d)

import numpy as np
import torch
from torch.utils.data import DataLoader

from main import SequenceWorldModel, SequenceScorer
from Policy.types import ClinicalProfile, TreatmentBlock, TreatmentSequence

from models.full_model import TimeAwareGliomaSurvivalPredictor
from dataset.dataset_glioma_all_pairs_text import Config, GliomaAllPairsTextDataset

# ═══════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════
CKPT         = "experiments/exp012_cf_diversity/checkpoints/best_c_index.pth"
ARGS_JSON    = "experiments/exp012_cf_diversity/checkpoints/args.json"
MAX_PATIENTS = 30
HORIZON      = 3
DISCOUNT     = 0.95
ROLLOUT_N    = 5
DEVICE       = torch.device("cuda")

# Set ANTHROPIC_API_KEY or OPENAI_API_KEY to use real LLM candidates.
# Without a key the rule engine fallback runs automatically.
_API_KEY = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
_LLM_MODEL = os.environ.get("ISE_LLM_MODEL", "claude-opus-4-8")
_LLM_PROVIDER = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openai"

try:
    from ise_llm import ISECandidateGenerator
    _llm_gen = ISECandidateGenerator(
        api_key=_API_KEY, model=_LLM_MODEL, provider=_LLM_PROVIDER
    ) if _API_KEY else None
except Exception as _e:
    print(f"[ise_llm] not available ({_e}), using rule engine fallback.")
    _llm_gen = None

print(f"LLM backend: {'API (' + _LLM_MODEL + ')' if _llm_gen else 'rule engine (no API key)'}")

# ═══════════════════════════════════════════════════════════════════
# Treatment fingerprint (for F1)
# ═══════════════════════════════════════════════════════════════════
def fingerprint(actions: dict) -> frozenset:
    tags = set()
    if actions.get("radiation"):
        tags.add("RT")
    for cat in ("chemotherapy", "immunotherapy", "additional_1", "additional_2"):
        for item in actions.get(cat, []):
            a = str(item.get("agent", "")).lower()
            if "temozolomide" in a:            tags.add("TMZ")
            elif "bevacizumab" in a or "avastin" in a: tags.add("BEV")
            elif "pcv" in a or "lomustine" in a or "ccnu" in a: tags.add("PCV")
            elif a:                            tags.add("OTHER_CHEMO")
    for item in actions.get("other_therapy", []):
        a = str(item.get("agent") or item.get("name") or "").lower()
        if "optune" in a or "ttf" in a: tags.add("TTF")
        elif a:                          tags.add("OTHER_DEVICE")
    return frozenset(tags) if tags else frozenset({"NO_THERAPY"})

def f1_score(pred: frozenset, gt: frozenset):
    if not pred and not gt:  return 1.0, 1.0, 1.0
    if not pred or not gt:   return 0.0, 0.0, 0.0
    tp   = len(pred & gt)
    prec = tp / len(pred)
    rec  = tp / len(gt)
    f1   = (2*prec*rec / (prec+rec)) if (prec+rec) > 0 else 0.0
    return prec, rec, f1

# ═══════════════════════════════════════════════════════════════════
# LLM-style candidate generator
# Generates 3-5 CLINICALLY COHESIVE variants for a specific treatment
# decision point — the LLM role in the ISE pipeline.
# ═══════════════════════════════════════════════════════════════════
def _blk(kind, agent=None, cycles=6, cycle_days=28, dose_gy=None, fractions=None):
    if kind == "radiation":
        return TreatmentBlock("radiation",
                              {"dose_gy": float(dose_gy or 60), "fractions": int(fractions or 30)},
                              0, 0)
    return TreatmentBlock(kind,
                          {"agent": agent, "cycle_length_days": int(cycle_days),
                           "num_cycles": int(cycles)},
                          0, 0)

def _has_agent(actions, *keywords):
    for cat in ("chemotherapy", "additional_1", "additional_2", "immunotherapy"):
        for item in actions.get(cat, []):
            a = str(item.get("agent", "")).lower()
            if any(k in a for k in keywords):
                return True
    return False

def _determine_phase(context_static, pre_actions):
    """Identify the clinical decision phase from context + prior treatment."""
    g          = context_static.get("genomics", {})
    grade      = int(context_static.get("who_grade", 4))
    codeleted  = str(g.get("codeletion_1p19q_detail", "0")) == "1"
    idh_mut    = g.get("idh1", 2) == 1 or g.get("idh2", 2) == 1
    has_rt     = bool(pre_actions.get("radiation"))
    has_tmz    = _has_agent(pre_actions, "temozolomide")
    has_bev    = _has_agent(pre_actions, "bevacizumab", "avastin")
    has_pcv    = _has_agent(pre_actions, "pcv", "lomustine", "ccnu")

    if codeleted:
        if not has_rt and not has_pcv:  return "oligo_chemonaive"
        if has_rt and not has_pcv:      return "oligo_post_rt"
        return "oligo_on_pcv"

    if has_bev:                         return "post_bev"          # BEV-refractory/progressive
    if has_rt and has_tmz:              return "post_crt"          # Adjuvant or recurrence phase
    if has_rt and not has_tmz:          return "post_rt_only"
    if grade == 4:                      return "gbm_chemonaive"    # Newly diagnosed GBM
    if grade in (2, 3) and idh_mut:     return "lgg_idh_mut"
    return "other"


def llm_candidates(context_static: dict, pre_actions: dict) -> list:
    """
    Simulate the LLM step: generate 3-5 clinically plausible treatment
    variants for the patient's current decision point.
    All candidates belong to the same treatment phase (no wildcard options).

    Returns list of (label, TreatmentSequence).
    """
    g         = context_static.get("genomics", {})
    mgmt_meth = g.get("mgmt_methylation", 2) == 1
    has_bev   = _has_agent(pre_actions, "bevacizumab", "avastin")

    phase = _determine_phase(context_static, pre_actions)
    cands = []

    # ── Phase: Newly-diagnosed GBM (standard Stupp variants) ───────────
    if phase == "gbm_chemonaive":
        cands = [
            ("RT 60Gy + TMZ x6 (Stupp)",
             [_blk("radiation"), _blk("chemotherapy", "Temozolomide", cycles=6)]),
            ("RT 60Gy + TMZ x6 + Optune",
             [_blk("radiation"), _blk("chemotherapy", "Temozolomide", cycles=6),
              TreatmentBlock("other_therapy", {"agent": "Optune TTF"}, 0, 0)]),
            ("RT 60Gy + TMZ x12 (extended, if MGMT-met)",
             [_blk("radiation"), _blk("chemotherapy", "Temozolomide", cycles=12)]),
            ("Hypofractionated RT 40Gy + TMZ x6",
             [_blk("radiation", dose_gy=40, fractions=15),
              _blk("chemotherapy", "Temozolomide", cycles=6)]),
        ]

    # ── Phase: Post-CRT (adjuvant variants only; BEV-alone is recurrence-phase
    #    therapy and should not appear for patients still in adjuvant setting) ──
    elif phase == "post_crt":
        n = 12 if mgmt_meth else 6
        cands = [
            (f"Adjuvant TMZ x{n}",
             [_blk("additional_1", "Temozolomide", cycles=n)]),
            (f"Adjuvant TMZ x{n} + Optune",
             [_blk("additional_1", "Temozolomide", cycles=n),
              TreatmentBlock("other_therapy", {"agent": "Optune TTF"}, 0, 0)]),
            ("Adjuvant TMZ + BEV (MGMT-unmethylated high-risk)",
             [_blk("additional_1", "Temozolomide", cycles=6),
              _blk("additional_2", "Bevacizumab", cycle_days=14, cycles=8)]),
            ("Adjuvant TMZ x6 (standard)",
             [_blk("additional_1", "Temozolomide", cycles=6)]),
        ]

    # ── Phase: Post-RT only (no TMZ yet) ───────────────────────────────
    elif phase == "post_rt_only":
        cands = [
            ("Adjuvant TMZ x6",
             [_blk("additional_1", "Temozolomide", cycles=6)]),
            ("Adjuvant TMZ x12 (if MGMT-met)",
             [_blk("additional_1", "Temozolomide", cycles=12)]),
            ("TMZ + BEV",
             [_blk("additional_1", "Temozolomide", cycles=6),
              _blk("additional_2", "Bevacizumab", cycle_days=14, cycles=8)]),
            ("TMZ + Optune",
             [_blk("additional_1", "Temozolomide", cycles=6),
              TreatmentBlock("other_therapy", {"agent": "Optune TTF"}, 0, 0)]),
        ]

    # ── Phase: BEV-refractory / post-BEV ───────────────────────────────
    elif phase == "post_bev":
        cands = [
            ("Lomustine (CCNU) salvage",
             [_blk("chemotherapy", "Lomustine", cycle_days=42, cycles=4)]),
            ("BEV + TMZ rechallenge",
             [_blk("chemotherapy", "Bevacizumab", cycle_days=14, cycles=6),
              _blk("additional_1", "Temozolomide", cycles=4)]),
            ("Lomustine + BEV (BELOB regimen, if MGMT-met)",
             [_blk("chemotherapy", "Lomustine", cycle_days=42, cycles=4),
              _blk("additional_1", "Bevacizumab", cycle_days=14, cycles=6)]),
            ("Continue BEV alone",
             [_blk("chemotherapy", "Bevacizumab", cycle_days=14, cycles=8)]),
        ]

    # ── Phase: Oligodendroglioma, chemo-naïve ──────────────────────────
    elif phase == "oligo_chemonaive":
        cands = [
            ("RT 54Gy + PCV x6 (RTOG9802)",
             [_blk("radiation", dose_gy=54, fractions=30),
              _blk("chemotherapy", "PCV", cycle_days=42, cycles=6)]),
            ("RT 54Gy + TMZ (if PCV not tolerated)",
             [_blk("radiation", dose_gy=54, fractions=30),
              _blk("chemotherapy", "Temozolomide", cycles=6)]),
            ("RT 54Gy alone (observation-preferred low-grade)",
             [_blk("radiation", dose_gy=54, fractions=30)]),
            ("RT 54Gy + PCV x4 (reduced for tolerability)",
             [_blk("radiation", dose_gy=54, fractions=30),
              _blk("chemotherapy", "PCV", cycle_days=42, cycles=4)]),
        ]

    # ── Phase: Post-RT oligo → adjuvant PCV ────────────────────────────
    elif phase == "oligo_post_rt":
        cands = [
            ("Adjuvant PCV x6 (RTOG9802)",
             [_blk("additional_1", "PCV", cycle_days=42, cycles=6)]),
            ("Adjuvant PCV x4 (tolerability)",
             [_blk("additional_1", "PCV", cycle_days=42, cycles=4)]),
            ("Adjuvant TMZ x6 (PCV substitute)",
             [_blk("additional_1", "Temozolomide", cycles=6)]),
            ("Adjuvant TMZ x12 (if MGMT-met)",
             [_blk("additional_1", "Temozolomide", cycles=12)]),
        ]

    # ── Phase: Ongoing PCV → continue or modify ────────────────────────
    elif phase == "oligo_on_pcv":
        cands = [
            ("Continue PCV x6",
             [_blk("additional_1", "PCV", cycle_days=42, cycles=6)]),
            ("Continue PCV x4 (tolerability)",
             [_blk("additional_1", "PCV", cycle_days=42, cycles=4)]),
            ("Switch to TMZ (PCV intolerance)",
             [_blk("additional_1", "Temozolomide", cycles=6)]),
        ]

    # ── Phase: IDH-mutant LGG, treatment-naïve ─────────────────────────
    elif phase == "lgg_idh_mut":
        cands = [
            ("RT 54Gy + TMZ x6",
             [_blk("radiation", dose_gy=54, fractions=30),
              _blk("chemotherapy", "Temozolomide", cycles=6)]),
            ("RT 54Gy + TMZ x12 (MGMT-met)",
             [_blk("radiation", dose_gy=54, fractions=30),
              _blk("chemotherapy", "Temozolomide", cycles=12)]),
            ("RT 54Gy alone (observation)",
             [_blk("radiation", dose_gy=54, fractions=30)]),
            ("RT 59.4Gy + TMZ x12 (grade 3 CATNON)",
             [_blk("radiation", dose_gy=59.4, fractions=33),
              _blk("chemotherapy", "Temozolomide", cycles=12)]),
        ]

    # ── Fallback ────────────────────────────────────────────────────────
    else:
        cands = [
            ("RT 60Gy + TMZ x6",
             [_blk("radiation"), _blk("chemotherapy", "Temozolomide", cycles=6)]),
            ("RT 60Gy + TMZ x6 + Optune",
             [_blk("radiation"), _blk("chemotherapy", "Temozolomide", cycles=6),
              TreatmentBlock("other_therapy", {"agent": "Optune TTF"}, 0, 0)]),
            ("TMZ x6 adjuvant",
             [_blk("additional_1", "Temozolomide", cycles=6)]),
        ]

    return [(label, TreatmentSequence(blocks=blocks, source="llm_sim"))
            for label, blocks in cands]


# ═══════════════════════════════════════════════════════════���═══════
# Load model + dataset
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("Loading exp012 model ...")
args = json.load(open(ARGS_JSON))

model = TimeAwareGliomaSurvivalPredictor(
    text_encoder_name=args["text_encoder_name"],
    latent_dim=args["latent_dim"],
    num_modalities=args["num_modalities"],
    predictor_hidden_dim=512,
    survival_hidden_dim=128,
    brainiac_ckpt=args.get("brainiac_ckpt"),
    brainiac_tokens_per_modality=args.get("brainiac_tokens", 8),
    brainiac_lora_r=args.get("brainiac_lora_r", 8),
).to(DEVICE)

ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["trainable_state_dict"], strict=False)
model.eval()
print(f"  epoch={ckpt['epoch']}  best_c_index={ckpt['best_c_index']:.4f}\n")

world  = object.__new__(SequenceWorldModel)
world.device = DEVICE
world.model  = model
W_SURV  = 0.5   # survival probability bonus weight (tune this)
scorer = SequenceScorer(w_risk=1.0, w_tox=0.2, w_comp=0.1, w_unc=0.15, w_surv=W_SURV)

cfg = Config(
    timeline_json=args["timeline_json"],
    features_csv=args["features_csv"],
    take_dims=args.get("take_dims", 767),
    mri_data_dir=args.get("mri_data_dir"),
)
full_ds = GliomaAllPairsTextDataset(cfg)

patient_to_indices = {}
for idx, item in enumerate(full_ds.index):
    patient_to_indices.setdefault(item["pid"], []).append(idx)
unique_pids = list(patient_to_indices.keys())
np.random.default_rng(42).shuffle(unique_pids)
target_train = int(0.8 * len(full_ds))
val_idx, cur = [], 0
for pid in unique_pids:
    idxs = patient_to_indices[pid]
    if cur < target_train:
        cur += len(idxs)
    else:
        val_idx.extend(idxs)

# pid → context_static lookup
pid_to_context = {}
for raw_idx in val_idx:
    item = full_ds.index[raw_idx]
    if item["pid"] not in pid_to_context:
        pid_to_context[item["pid"]] = item.get("context_static", {})

val_ds     = torch.utils.data.Subset(full_ds, val_idx)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                        collate_fn=GliomaAllPairsTextDataset.collate_fn)
print(f"Val: {len(val_idx)} samples  |  {len(pid_to_context)} patients\n")

# ═══════════════════════════════════════════════════════════════════
# Evaluation loop
# ═══════════════════════════════════════════════════════════════════
rng     = np.random.default_rng(0)
records = []
n_done  = 0

for batch in val_loader:
    if n_done >= MAX_PATIENTS:
        break

    pre_mri = batch.get("pre_mri")
    if pre_mri is None:
        continue

    pre_mri      = pre_mri.to(DEVICE)
    clinical_txt = batch["clinical_text"][0]
    time_delta   = batch["time_delta"].to(DEVICE)
    survival_t   = float(batch["survival_time"][0])
    event_ind    = int(batch["event_indicator"][0])
    meta         = batch["meta"][0] if "meta" in batch else {}
    pid          = meta.get("pid", f"pat_{n_done}")
    drugs_json   = json.loads(batch["drugs_text"][0])
    pre_actions  = drugs_json.get("pre",  {}).get("actions", {})
    post_actions = drugs_json.get("post", {}).get("actions", {})
    context_st   = pid_to_context.get(pid, {})

    gt_fp = fingerprint(post_actions)
    if gt_fp == frozenset({"NO_THERAPY"}):
        continue   # nothing clinically meaningful to compare

    # ── encode pre MRI ──────────────────────────────────────────────
    with torch.no_grad():
        pre_latent = model.mri_encoder(pre_mri)  # [1, 32, 768]

    # ── Stage 1: LLM generates candidates ──────────────────────────
    # Real API if key is set, rule engine otherwise.
    phase       = _determine_phase(context_st, pre_actions)
    pre_payload = {"pre": {"tp": "TP_pre", "actions": pre_actions}}
    if _llm_gen is not None:
        try:
            candidates = _llm_gen.propose(context_st, pre_actions)
        except Exception as _e:
            print(f"  [LLM API error] {_e} — falling back to rule engine")
            candidates = llm_candidates(context_st, pre_actions)
    else:
        candidates = llm_candidates(context_st, pre_actions)
    if not candidates:
        continue
    cand_jsons = [seq.to_model_triplet_json(pre_payload, tp_post="TP_post", between=[])
                  for _, seq in candidates]

    clinical = ClinicalProfile.from_context_static(
        patient_id=pid, context_static=context_st
    )
    td_days = float(time_delta.item())

    # ── world model rollout for each LLM candidate ──────────────────
    scored = []
    for (label, seq), seq_json in zip(candidates, cand_jsons):
        rollout = world.rollout_trajectory(
            pre_latent=pre_latent,
            clinical_text=clinical_txt,
            sequence_json=seq_json,
            time_delta_days=td_days,
            horizon=HORIZON,
            discount_factor=DISCOUNT,
            num_samples=ROLLOUT_N,
        )
        s       = scorer.score(seq, rollout, clinical)
        s.source = label
        s._fp   = fingerprint(seq.to_post_actions())
        scored.append(s)

    scored.sort(key=lambda x: x.total_score)

    ise_fp    = scored[0]._fp
    llm_fp    = candidates[0][1].to_post_actions()   # LLM primary (first candidate)
    llm_fp    = fingerprint(llm_fp)
    rand_fp   = fingerprint(candidates[int(rng.integers(len(candidates)))][1].to_post_actions())
    oracle_fp = max((s._fp for s in scored), key=lambda fp: f1_score(fp, gt_fp)[2])

    _, _, f1_ise    = f1_score(ise_fp,    gt_fp)
    _, _, f1_llm    = f1_score(llm_fp,    gt_fp)
    _, _, f1_rand   = f1_score(rand_fp,   gt_fp)
    _, _, f1_oracle = f1_score(oracle_fp, gt_fp)

    records.append({
        "pid":          pid,
        "phase":        phase,
        "n_cands":      len(candidates),
        "gt":           sorted(gt_fp),
        "ise_pred":     sorted(ise_fp),
        "llm_pred":     sorted(llm_fp),
        "f1_ise":       f1_ise,
        "f1_llm":       f1_llm,
        "f1_rand":      f1_rand,
        "f1_oracle":    f1_oracle,
        "ise_label":    scored[0].source,
        "llm_label":    candidates[0][0],
        "surv_days":    survival_t,
        "event":        event_ind,
    })

    ise_wins = "✓" if f1_ise > f1_llm else ("=" if f1_ise == f1_llm else "✗")
    print(f"[{n_done+1:2d}] {pid}  phase={phase}")
    print(f"     GT:  {sorted(gt_fp)}")
    print(f"     LLM: {sorted(llm_fp)}  ({candidates[0][0]})")
    print(f"     ISE: {sorted(ise_fp)}  ({scored[0].source})")
    print(f"     F1  LLM={f1_llm:.3f}  ISE={f1_ise:.3f}  Oracle={f1_oracle:.3f}  ISE{ise_wins}LLM")
    print()
    n_done += 1

# ═══════════════════════════════════════════════════════════════════
# Aggregate results
# ═══════════════════════════════════════════════════════════════════
n = len(records)
print("=" * 70)
print(f"ISE vs LLM F1  (N={n} patients,  w_surv={W_SURV})")
print("=" * 70)
print(f"{'Method':<20}  {'F1 mean':>8}  {'F1 std':>8}  {'F1 median':>10}")
print("-" * 55)
for key, label in [("f1_rand",   "Random"),
                   ("f1_llm",    "LLM primary"),
                   ("f1_ise",    "ISE (WM+LLM)"),
                   ("f1_oracle", "Oracle (UB)")]:
    vals = [r[key] for r in records]
    print(f"  {label:<18}  {np.mean(vals):>8.4f}  {np.std(vals):>8.4f}  {np.median(vals):>10.4f}")

exact_ise = sum(1 for r in records if set(r["ise_pred"]) == set(r["gt"])) / n
exact_llm = sum(1 for r in records if set(r["llm_pred"]) == set(r["gt"])) / n
print(f"\n  Exact match  ISE: {exact_ise:.3f}  |  LLM: {exact_llm:.3f}")

ise_better = sum(1 for r in records if r["f1_ise"] > r["f1_llm"])
ise_equal  = sum(1 for r in records if r["f1_ise"] == r["f1_llm"])
ise_worse  = sum(1 for r in records if r["f1_ise"] < r["f1_llm"])
print(f"\n  ISE > LLM: {ise_better} | ISE = LLM: {ise_equal} | ISE < LLM: {ise_worse}")
print(f"  ISE Δ F1 vs LLM: {np.mean([r['f1_ise']-r['f1_llm'] for r in records]):+.4f}")

# Per-phase breakdown
from collections import defaultdict
phase_stats = defaultdict(list)
for r in records:
    phase_stats[r["phase"]].append((r["f1_ise"], r["f1_llm"]))
print(f"\n  Per-phase F1 (ISE / LLM):")
for ph, vals in sorted(phase_stats.items()):
    ise_m = np.mean([v[0] for v in vals])
    llm_m = np.mean([v[1] for v in vals])
    print(f"    {ph:<22}  n={len(vals):2d}  ISE={ise_m:.3f}  LLM={llm_m:.3f}  Δ={ise_m-llm_m:+.3f}")

# GT distribution
from collections import Counter
gt_flat = [tag for r in records for tag in r["gt"]]
print(f"\n  GT distribution: {Counter(gt_flat).most_common()}")

# Save
out = {"n": n, "results": records,
       "summary": {m: {"mean": float(np.mean([r[f"f1_{m}"] for r in records])),
                       "std":  float(np.std( [r[f"f1_{m}"] for r in records]))}
                   for m in ("rand", "llm", "ise", "oracle")}}
with open("ise_eval_v2_results.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nResults saved to ise_eval_v2_results.json")
