# CLARITY

CLARITY is a compact public release of our current treatment-planning codebase for glioma longitudinal modeling and inverse survival evaluation.

This repo contains two active parts:

- `Predictor/`: MRI-conditioned survival/world-model training
- `Policy/` + `main.py`: inverse therapy search with guardrails, toxicity heuristics, and trajectory-level risk evaluation

This release is intentionally code-only. It does not include patient data, MRI volumes, pretrained checkpoints, or experiment outputs.

## Repository Layout

```text
CLARITY/
├── Predictor/
│   ├── train.py
│   ├── dataset/
│   ├── losses/
│   ├── models/
│   └── utils/
├── Policy/
├── main.py
├── run_training.sh
├── treatment_constraints.json
├── requirements.txt
└── .env.example
```

## What Is Included

- MRI-based survival training entrypoint: `Predictor/train.py`
- BrainIAC-style MRI vision backbone integration
- MedGemma text encoder with LoRA fine-tuning
- time-aware latent prediction + survival supervision
- inverse survival evaluation in `main.py`
- single-step and long-horizon trajectory search

## What Is Not Included

- clinical timeline files
- MRI volumes
- BrainIAC checkpoints
- trained Predictor checkpoints
- external foundation-model repositories cloned locally during development
- legacy diffusion / pixel-loss code paths
- analysis, plotting, and dev-only test scripts

## Environment

Recommended:

- Python 3.10
- CUDA-enabled PyTorch
- `transformers`, `peft`, `bitsandbytes`
- `monai`
- `nibabel`

Install dependencies:

```bash
pip install -r requirements.txt
```

For inverse planning, set your OpenAI key:

```bash
export OPENAI_API_KEY=your_api_key_here
```

## External Assets You Must Provide

This repository will not run end-to-end unless you supply the following resources locally.

1. Clinical timeline JSON
2. MRI root directory
3. BrainIAC checkpoint
4. Trained Predictor checkpoint

The current code expects paths like:

```text
Predictor/dataset/MU_Glioma_Post/clinical_latest.json
datasets/MU-Glioma-Post/
BrainIAC-main/src/checkpoints/BrainIAC.ckpt
```

These assets are intentionally excluded from version control.

## Training

Primary training entrypoint:

```bash
python Predictor/train.py
```

The current training setup:

- reads pre/post MRI volumes directly
- uses BrainIAC-style vision encoding
- uses MedGemma for treatment and clinical text
- removes the old pixel-loss / diffusion branch
- trains survival objectives plus contrastive regularization

Convenience launcher:

```bash
bash run_training.sh
```

Note: `run_training.sh` assumes the external assets above already exist at the expected local paths.

## Inference And Inverse Planning

Main planning entrypoint:

```bash
python main.py
```

`main.py` implements:

- LLM-based post-treatment proposal generation
- guardrail repair and toxicity-aware scoring
- world-model rollout over imagined disease trajectories
- discounted cumulative risk scoring
- entropy-regularized candidate refinement

It supports both:

- single-step inverse survival evaluation with `planning_horizon=1`
- long-horizon planning with `planning_horizon>1`

## Current Public Scope

This repository is the first code-focused public snapshot, not a polished benchmark release. The current emphasis is:

- trainability
- inference/planning logic
- reproducible code structure

It is not yet packaged as a turnkey public benchmark because dataset preprocessing, checkpoint packaging, and example assets are not distributed here.

## Security Note

- `main.py` reads the OpenAI API key from `OPENAI_API_KEY`
- no hard-coded API key is kept in this public repo

## Next Cleanup Targets

If we turn this into a fuller open-source release later, the next steps should be:

- add config files for training and inference
- document checkpoint formats
- provide a small synthetic example input bundle
- clean up path assumptions in `run_training.sh` and `main.py`
