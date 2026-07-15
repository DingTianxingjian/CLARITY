"""
ise_surv_sweep.py  —  sweep w_surv in {0.0, 0.3, 0.5, 1.0, 2.0}
Collects rollouts ONCE (GPU), then re-scores all values on CPU.
"""
import sys, os, json
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
from Policy.toxicity_rules import compute_toxicity

CKPT         = "experiments/exp012_cf_diversity/checkpoints/best_c_index.pth"
ARGS_JSON    = "experiments/exp012_cf_diversity/checkpoints/args.json"
MAX_PATIENTS = 30
HORIZON      = 3
DISCOUNT     = 0.95
ROLLOUT_N    = 5
DEVICE       = torch.device("cuda")

# ── helpers (copied from ise_eval_v2 to avoid import-time side effects) ──
VALID_ACTION_KEYS = ["radiation", "chemotherapy", "immunotherapy",
                     "additional_1", "additional_2", "other_therapy"]

def fingerprint(actions: dict) -> frozenset:
    tags = set()
    if actions.get("radiation"):
        tags.add("RT")
    for cat in ("chemotherapy", "immunotherapy", "additional_1", "additional_2"):
        for item in actions.get(cat, []):
            a = str(item.get("agent", "")).lower()
            if "temozolomide" in a:                            tags.add("TMZ")
            elif "bevacizumab" in a or "avastin" in a:        tags.add("BEV")
            elif "pcv" in a or "lomustine" in a or "ccnu" in a: tags.add("PCV")
            elif a:                                            tags.add("OTHER_CHEMO")
    for item in actions.get("other_therapy", []):
        a = str(item.get("agent") or item.get("name") or "").lower()
        if "optune" in a or "ttf" in a: tags.add("TTF")
        elif a:                          tags.add("OTHER_DEVICE")
    return frozenset(tags) if tags else frozenset({"NO_THERAPY"})

def f1_score(pred, gt):
    if not pred and not gt: return 1.0, 1.0, 1.0
    if not pred or  not gt: return 0.0, 0.0, 0.0
    tp   = len(pred & gt)
    prec = tp / len(pred)
    rec  = tp / len(gt)
    f1   = (2*prec*rec / (prec+rec)) if (prec+rec) > 0 else 0.0
    return prec, rec, f1

def _blk(kind, agent=None, cycles=6, cycle_days=28, dose_gy=None, fractions=None):
    if kind == "radiation":
        return TreatmentBlock("radiation",
            {"dose_gy": float(dose_gy or 60), "fractions": int(fractions or 30)}, 0, 0)
    return TreatmentBlock(kind,
        {"agent": agent, "cycle_length_days": int(cycle_days), "num_cycles": int(cycles)}, 0, 0)

def _has_agent(actions, *keywords):
    for cat in ("chemotherapy", "additional_1", "additional_2", "immunotherapy"):
        for item in actions.get(cat, []):
            if any(k in str(item.get("agent", "")).lower() for k in keywords):
                return True
    return False

def _determine_phase(context_static, pre_actions):
    g         = context_static.get("genomics", {})
    grade     = int(context_static.get("who_grade", 4))
    codeleted = str(g.get("codeletion_1p19q_detail", "0")) == "1"
    idh_mut   = g.get("idh1", 2) == 1 or g.get("idh2", 2) == 1
    has_rt    = bool(pre_actions.get("radiation"))
    has_tmz   = _has_agent(pre_actions, "temozolomide")
    has_bev   = _has_agent(pre_actions, "bevacizumab", "avastin")
    has_pcv   = _has_agent(pre_actions, "pcv", "lomustine", "ccnu")
    if codeleted:
        if not has_rt and not has_pcv: return "oligo_chemonaive"
        if has_rt and not has_pcv:     return "oligo_post_rt"
        return "oligo_on_pcv"
    if has_bev:                        return "post_bev"
    if has_rt and has_tmz:             return "post_crt"
    if has_rt and not has_tmz:         return "post_rt_only"
    if grade == 4:                     return "gbm_chemonaive"
    if grade in (2, 3) and idh_mut:    return "lgg_idh_mut"
    return "other"

def llm_candidates(context_static, pre_actions):
    g        = context_static.get("genomics", {})
    mgmt_met = g.get("mgmt_methylation", 2) == 1
    phase    = _determine_phase(context_static, pre_actions)
    cands    = []

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
    elif phase == "post_crt":
        n = 12 if mgmt_met else 6
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
    elif phase == "post_rt_only":
        cands = [
            ("Adjuvant TMZ x6",
             [_blk("additional_1", "Temozolomide", cycles=6)]),
            ("Adjuvant TMZ x12",
             [_blk("additional_1", "Temozolomide", cycles=12)]),
            ("TMZ + BEV",
             [_blk("additional_1", "Temozolomide", cycles=6),
              _blk("additional_2", "Bevacizumab", cycle_days=14, cycles=8)]),
            ("TMZ + Optune",
             [_blk("additional_1", "Temozolomide", cycles=6),
              TreatmentBlock("other_therapy", {"agent": "Optune TTF"}, 0, 0)]),
        ]
    elif phase == "post_bev":
        cands = [
            ("Lomustine (CCNU) salvage",
             [_blk("chemotherapy", "Lomustine", cycle_days=42, cycles=4)]),
            ("BEV + TMZ rechallenge",
             [_blk("chemotherapy", "Bevacizumab", cycle_days=14, cycles=6),
              _blk("additional_1", "Temozolomide", cycles=4)]),
            ("Lomustine + BEV",
             [_blk("chemotherapy", "Lomustine", cycle_days=42, cycles=4),
              _blk("additional_1", "Bevacizumab", cycle_days=14, cycles=6)]),
            ("Continue BEV alone",
             [_blk("chemotherapy", "Bevacizumab", cycle_days=14, cycles=8)]),
        ]
    elif phase == "oligo_chemonaive":
        cands = [
            ("RT 54Gy + PCV x6",
             [_blk("radiation", dose_gy=54, fractions=30),
              _blk("chemotherapy", "PCV", cycle_days=42, cycles=6)]),
            ("RT 54Gy + TMZ x6",
             [_blk("radiation", dose_gy=54, fractions=30),
              _blk("chemotherapy", "Temozolomide", cycles=6)]),
            ("RT 54Gy alone",
             [_blk("radiation", dose_gy=54, fractions=30)]),
            ("RT 54Gy + PCV x4",
             [_blk("radiation", dose_gy=54, fractions=30),
              _blk("chemotherapy", "PCV", cycle_days=42, cycles=4)]),
        ]
    elif phase == "oligo_post_rt":
        cands = [
            ("Adjuvant PCV x6", [_blk("additional_1", "PCV", cycle_days=42, cycles=6)]),
            ("Adjuvant PCV x4", [_blk("additional_1", "PCV", cycle_days=42, cycles=4)]),
            ("Adjuvant TMZ x6", [_blk("additional_1", "Temozolomide", cycles=6)]),
            ("Adjuvant TMZ x12",[_blk("additional_1", "Temozolomide", cycles=12)]),
        ]
    elif phase == "oligo_on_pcv":
        cands = [
            ("Continue PCV x6", [_blk("additional_1", "PCV", cycle_days=42, cycles=6)]),
            ("Continue PCV x4", [_blk("additional_1", "PCV", cycle_days=42, cycles=4)]),
            ("Switch to TMZ",   [_blk("additional_1", "Temozolomide", cycles=6)]),
        ]
    elif phase == "lgg_idh_mut":
        cands = [
            ("RT 54Gy + TMZ x6",
             [_blk("radiation", dose_gy=54, fractions=30),
              _blk("chemotherapy", "Temozolomide", cycles=6)]),
            ("RT 54Gy + TMZ x12",
             [_blk("radiation", dose_gy=54, fractions=30),
              _blk("chemotherapy", "Temozolomide", cycles=12)]),
            ("RT 54Gy alone",  [_blk("radiation", dose_gy=54, fractions=30)]),
            ("RT 59.4Gy + TMZ x12",
             [_blk("radiation", dose_gy=59.4, fractions=33),
              _blk("chemotherapy", "Temozolomide", cycles=12)]),
        ]
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

# ── load model ────────────────────────────────────────────────────────
print("Loading model ...", flush=True)
args  = json.load(open(ARGS_JSON))
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
print(f"  epoch={ckpt['epoch']}  c_index={ckpt['best_c_index']:.4f}", flush=True)

world        = object.__new__(SequenceWorldModel)
world.device = DEVICE
world.model  = model

cfg    = Config(timeline_json=args["timeline_json"], features_csv=args["features_csv"],
                take_dims=args.get("take_dims", 767), mri_data_dir=args.get("mri_data_dir"))
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
    if cur < target_train: cur += len(idxs)
    else: val_idx.extend(idxs)

pid_to_context = {}
for raw_idx in val_idx:
    item = full_ds.index[raw_idx]
    if item["pid"] not in pid_to_context:
        pid_to_context[item["pid"]] = item.get("context_static", {})

val_loader = DataLoader(torch.utils.data.Subset(full_ds, val_idx),
                        batch_size=1, shuffle=False,
                        collate_fn=GliomaAllPairsTextDataset.collate_fn)
print(f"Val: {len(val_idx)} samples\n", flush=True)

# ── collect rollouts once (GPU) ───────────────────────────────────────
# Each entry: dict with gt_fp, per-candidate (label, seq, rollout, fp), phase, etc.
raw_data = []
n_done   = 0
rng      = np.random.default_rng(0)

for batch in val_loader:
    if n_done >= MAX_PATIENTS: break
    pre_mri = batch.get("pre_mri")
    if pre_mri is None: continue

    pre_mri      = pre_mri.to(DEVICE)
    clinical_txt = batch["clinical_text"][0]
    time_delta   = batch["time_delta"].to(DEVICE)
    meta         = batch["meta"][0] if "meta" in batch else {}
    pid          = meta.get("pid", f"pat_{n_done}")
    drugs_json   = json.loads(batch["drugs_text"][0])
    pre_actions  = drugs_json.get("pre",  {}).get("actions", {})
    post_actions = drugs_json.get("post", {}).get("actions", {})
    context_st   = pid_to_context.get(pid, {})
    gt_fp        = fingerprint(post_actions)
    if gt_fp == frozenset({"NO_THERAPY"}): continue

    with torch.no_grad():
        pre_latent = model.mri_encoder(pre_mri)

    candidates  = llm_candidates(context_st, pre_actions)
    phase       = _determine_phase(context_st, pre_actions)
    pre_payload = {"pre": {"tp": "TP_pre", "actions": pre_actions}}
    cand_jsons  = [seq.to_model_triplet_json(pre_payload, tp_post="TP_post", between=[])
                   for _, seq in candidates]
    clinical    = ClinicalProfile.from_context_static(patient_id=pid, context_static=context_st)
    td_days     = float(time_delta.item())

    rollout_cache = []
    for (label, seq), seq_json in zip(candidates, cand_jsons):
        ro = world.rollout_trajectory(
            pre_latent=pre_latent, clinical_text=clinical_txt,
            sequence_json=seq_json, time_delta_days=td_days,
            horizon=HORIZON, discount_factor=DISCOUNT, num_samples=ROLLOUT_N,
        )
        fp  = fingerprint(seq.to_post_actions())
        tox = float(compute_toxicity(seq, clinical))
        rollout_cache.append({
            "label": label, "fp": fp,
            "risk":  float(ro.get("discounted_risk", ro["risk_score"])),
            "surv":  float(ro.get("survival_prob", 0.0)),
            "tox":   tox,
            "comp":  float(len(seq.blocks) * 0.1),
            "unc":   float(ro["uncertainty"]),
        })

    llm_fp  = fingerprint(candidates[0][1].to_post_actions())
    rand_fp = fingerprint(candidates[int(rng.integers(len(candidates)))][1].to_post_actions())
    raw_data.append({"gt_fp": gt_fp, "phase": phase, "pid": pid,
                     "llm_fp": llm_fp, "rand_fp": rand_fp,
                     "rollouts": rollout_cache})
    print(f"  [{n_done+1:2d}] {pid}  phase={phase}  {len(candidates)} cands", flush=True)
    n_done += 1

print(f"\nRollouts collected for {len(raw_data)} patients.\n")

# ── sweep w_surv ──────────────────────────────────────────────────────
SWEEPS = [0.0, 0.3, 0.5, 1.0, 2.0]

print(f"{'w_surv':>8}  {'F1_ISE':>8}  {'F1_LLM':>8}  {'Δ':>8}  "
      f"{'ISE>LLM':>8}  {'post_crt ISE':>14}  {'post_bev ISE':>14}")
print("-" * 85)

best_cfg = None
for w_surv in SWEEPS:
    records = []
    for entry in raw_data:
        gt_fp  = entry["gt_fp"]
        llm_fp = entry["llm_fp"]
        phase  = entry["phase"]

        best_total, best_fp = float("inf"), None
        for ro in entry["rollouts"]:
            total = (ro["risk"] + 0.2*ro["tox"] + 0.1*ro["comp"]
                     + 0.15*ro["unc"] - w_surv*ro["surv"])
            if total < best_total:
                best_total, best_fp = total, ro["fp"]

        _, _, f1_ise = f1_score(best_fp,  gt_fp)
        _, _, f1_llm = f1_score(llm_fp,   gt_fp)
        records.append({"f1_ise": f1_ise, "f1_llm": f1_llm, "phase": phase})

    mean_ise   = np.mean([r["f1_ise"] for r in records])
    mean_llm   = np.mean([r["f1_llm"] for r in records])
    n_better   = sum(1 for r in records if r["f1_ise"] > r["f1_llm"])
    post_crt_v = [r["f1_ise"] for r in records if r["phase"] == "post_crt"]
    post_bev_v = [r["f1_ise"] for r in records if r["phase"] == "post_bev"]
    pc_m  = np.mean(post_crt_v) if post_crt_v else float("nan")
    pb_m  = np.mean(post_bev_v) if post_bev_v else float("nan")

    print(f"{w_surv:>8.1f}  {mean_ise:>8.4f}  {mean_llm:>8.4f}  "
          f"{mean_ise-mean_llm:>+8.4f}  {n_better:>8d}  {pc_m:>14.4f}  {pb_m:>14.4f}")

    if best_cfg is None or mean_ise > best_cfg[0]:
        best_cfg = (mean_ise, w_surv)

print(f"\nBest w_surv = {best_cfg[1]}  (ISE F1 = {best_cfg[0]:.4f})")
print("\nSweep complete.")
