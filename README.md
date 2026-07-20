# CLARITY

CLARITY is the official code release for our paper:

> **CLARITY: Medical World Model for Guiding Treatment Decisions by Modeling Context-Aware Disease Trajectories in Latent Space**
> Tianxingjian Ding, Yuanhao Zou, Chen Chen, Mubarak Shah, Yu Tian
> arXiv:2512.08029 (2025)
> ### Official code release for our paper:
![ECCV 2025](https://img.shields.io/badge/ECCV-2025-blue?style=for-the-badge)

The repository includes:

- **World Model** — MRI-conditioned latent transition predictor with BrainIAC vision encoder + MedGemma text encoder
- **Survival Module** — Cox PH + BCE survival prediction head
- **ISE (Individual Survival Estimation)** — two-stage treatment planning: LLM proposes candidates → world model ranks them

---

## Installation

Python 3.10 required.

```bash
pip install -r requirements.txt
```

---

## Data Setup

The following external files are required but not distributed in this repository.
Place them at the paths shown (relative to the CLARITY root):

**Required:**
```
BrainIAC-main/src/checkpoints/BrainIAC.ckpt           # BrainIAC ViT-B weights
Predictor/dataset/MU_Glioma_Post/clinical_latest.json  # clinical timeline (JSON)
datasets/MU-Glioma-Post/                               # raw MRI NIfTI files (from TCIA)
```

**Optional:**
```
Predictor/dataset/MU_Glioma_Post/features_output.csv   # pre-extracted BrainIAC features
```
If present, acts as a data-availability filter (only patients listed in the CSV are trained on).
If absent, all patients in `clinical_latest.json` with corresponding MRI files are used.

Raw MRI data (MU-Glioma-Post) can be requested from [TCIA](https://www.cancerimagingarchive.net/).

---

## Training

Use the provided script to train exp012 (BrainIAC online encoder + MedGemma):

```bash
cd CLARITY
bash run_training.sh
```

To resume from a checkpoint:

```bash
bash run_training.sh --resume path/to/checkpoint.pth
```

Or run the training script directly with custom arguments:

```bash
export PYTHONPATH=Predictor:.
python Predictor/train.py \
  --exp_name          exp012_cf_diversity \
  --features_csv      Predictor/dataset/MU_Glioma_Post/features_output.csv \
  --timeline_json     Predictor/dataset/MU_Glioma_Post/clinical_latest.json \
  --mri_data_dir      datasets/MU-Glioma-Post \
  --brainiac_ckpt     BrainIAC-main/src/checkpoints/BrainIAC.ckpt \
  --brainiac_tokens   8 \
  --brainiac_lora_r   8 \
  --latent_dim        768 \
  --num_modalities    32 \
  --text_encoder_name google/medgemma-4b-it \
  --num_epochs        100 \
  --batch_size        16
```

The model trains:

- **Vision backbone** (pluggable — BrainIAC ViT-B by default, LoRA fine-tuned)
- MedGemma-4B text encoder (LoRA fine-tuned)
- Time-aware latent transition predictor
- Cox PH + BCE survival prediction head
- Counterfactual diversity loss (encourages treatment-discriminative latent space)

### Using a different vision backbone

The world model accepts any backbone that implements `VisionBackbone`:

```python
from models.vision_backbone import VisionBackbone, MultiModalVisionBackbone

class MyBackbone(VisionBackbone):
    @property
    def tokens_per_modality(self): return 8
    @property
    def hidden_dim(self): return 768
    def forward(self, x):   # x: [B, 1, D, H, W]
        return self.my_vit(x)[:, :8, :]

backbone = MultiModalVisionBackbone(MyBackbone(), num_modalities=4)
model = TimeAwareGliomaSurvivalPredictor(..., vision_backbone=backbone)
```

The BrainIAC adapter (`Predictor/models/brainiac_adapter.py`) is provided as the reference implementation — it requires the BrainIAC checkpoint (not redistributed).

---

## ISE Evaluation

ISE (Individual Survival Estimation) is a two-stage pipeline:

1. **LLM proposes 3–5 clinically plausible treatment candidates** for the patient's specific treatment phase (adjuvant, salvage, newly diagnosed, etc.)
2. **World model evaluates and ranks** those candidates via multi-step rollout

```bash
export PYTHONPATH=Predictor:.
python ise_eval_v2.py
```

### With a real LLM (Claude or GPT-4)

Set your API key before running:

```bash
# Claude (recommended)
export ANTHROPIC_API_KEY=your_key_here
export ISE_LLM_MODEL=claude-opus-4-8   # default

# or GPT-4
export OPENAI_API_KEY=your_key_here
export ISE_LLM_MODEL=gpt-4o
```

Without an API key the script falls back automatically to the built-in clinical rule engine (no LLM call, fully offline).

### w_surv sweep

To explore the effect of the survival-probability bonus weight on ISE ranking:

```bash
python ise_surv_sweep.py
```

This collects rollouts once (GPU) and re-scores all `w_surv` values on CPU — much faster than re-running the full evaluation per value.

---

## ISE Module API

```python
from ise_llm import ISECandidateGenerator
from main import SequenceWorldModel, SequenceScorer

# Stage 1: LLM generates candidates
gen = ISECandidateGenerator(api_key="...", model="claude-opus-4-8")
candidates = gen.propose(patient_context, pre_actions)

# Stage 2: world model scores and ranks
world  = SequenceWorldModel(model_path="path/to/checkpoint.pth")
scorer = SequenceScorer(w_risk=1.0, w_tox=0.2, w_comp=0.1, w_unc=0.15)

for label, seq in candidates:
    rollout = world.rollout_trajectory(pre_latent, clinical_text, seq_json, ...)
    score   = scorer.score(seq, rollout, clinical_profile)

# Full feedback loop (LLM ↔ world model, 2 rounds)
final_ranking = gen.propose_with_feedback_loop(
    patient_context, pre_actions,
    world_model=world, pre_latent=pre_latent, ...
    n_rounds=2
)
```

---

## Repository Structure

```
CLARITY/
├── main.py                  # ISE framework (SequenceWorldModel, SequenceScorer, ...)
├── ise_llm.py               # LLM candidate generator (Claude / GPT-4 / rule engine)
├── ise_eval_v2.py           # F1 evaluation: ISE vs LLM vs Oracle
├── ise_surv_sweep.py        # w_surv hyperparameter sweep
├── run_training.sh          # training launch script
├── requirements.txt
├── treatment_constraints.json
├── Policy/
│   ├── types.py             # TreatmentBlock, TreatmentSequence, ClinicalProfile
│   ├── guardrails.py        # clinical safety constraints
│   └── toxicity_rules.py
├── Predictor/
│   ├── train.py             # training entry point
│   ├── dataset/             # GliomaAllPairsTextDataset, context vectorizer
│   ├── models/
│   │   ├── full_model.py    # TimeAwareGliomaSurvivalPredictor (BrainIAC + MedGemma)
│   │   ├── survival_module.py
│   │   └── latent_predictor.py
│   ├── losses/              # Cox PH, Breslow, MTLR
│   └── utils/
├── mri_foundation/          # SAM ViT-B source (weights downloaded separately)
└── expert_eval/             # neuro-oncologist expert evaluation cases
```

---

## Citation

```bibtex
@article{ding2025clarity,
  title={CLARITY: Medical World Model for Guiding Treatment Decisions by
         Modeling Context-Aware Disease Trajectories in Latent Space},
  author={Ding, Tianxingjian and Zou, Yuanhao and Chen, Chen and Shah, Mubarak and Tian, Yu},
  journal={arXiv preprint arXiv:2512.08029},
  year={2025}
}
```

## Acknowledgements

- [BrainIAC](https://github.com/...) — 3D MRI ViT-B foundation model
- [MedGemma](https://huggingface.co/google/medgemma-4b-it) — medical vision-language model

## License

MIT License. See `LICENSE`.
