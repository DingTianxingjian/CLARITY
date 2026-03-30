# MeWM Public Release

This repository contains the public code for the current treatment planning stack built around:

- `Predictor/`: MRI-conditioned survival/world model training and evaluation
- `Policy/`: treatment representation, guardrails, and toxicity heuristics
- `main.py`: inverse survival evaluation and trajectory-level therapy search

## Scope

This public release is intentionally minimal. It excludes:

- patient data and derived clinical timelines
- MRI volumes
- model checkpoints
- experiment outputs and cached reports
- large third-party foundation model repositories vendored locally during development
- legacy diffusion / pixel-loss code paths that are no longer part of the current training setup

## Current Training Path

The active training entrypoint is:

- `Predictor/train.py`

The current setup uses:

- MRI volumes as direct input
- BrainIAC-style MRI vision backbone
- MedGemma text encoder for treatment/clinical text
- survival supervision without pixel reconstruction or diffusion loss

## Inverse Planning Path

The main inference/planning entrypoint is:

- `main.py`

It now supports:

- single-step inverse survival evaluation with `planning_horizon=1`
- long-horizon trajectory rollout with discounted cumulative risk
- entropy-regularized candidate search with iterative LLM feedback

## External Assets Required

This repo will not run end-to-end without external assets. You need to provide:

1. Clinical timeline JSON
2. MRI data directory
3. BrainIAC checkpoint
4. Trained Predictor checkpoint
5. OpenAI API key via environment variable

Example environment variable:

```bash
export OPENAI_API_KEY=...
```

## Suggested Public Repo Layout

Kept in version control:

- `Predictor/`
- `Policy/`
- `main.py`
- `run_training.sh`
- `requirements.txt`
- `treatment_constraints.json`

Ignored by default:

- `datasets/`
- `checkpoints/`
- `output/`
- `BrainIAC-main/`
- `mri_foundation/`
- local reports, caches, and generated figures

## Notes

- `main.py` no longer contains a hard-coded API key.
- If you want to publish a more polished open-source release, the next step should be splitting dev-only scripts from the stable API surface and adding a small example config directory.
