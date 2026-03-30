# CLARITY

CLARITY is the current public code release for our MRI-conditioned survival modeling and inverse treatment planning pipeline for glioma longitudinal data.


## Installation

Use Python 3.10 and install dependencies with:

```bash
pip install -r requirements.txt
```

For inverse planning with `main.py`, set:

```bash
export OPENAI_API_KEY=your_api_key_here
```

## Data

The raw public dataset can be obtained from TCIA.

## Training

The active training entrypoint is:

```bash
python Predictor/train.py
```

This version trains:

- MRI vision backbone
- MedGemma text encoder
- time-aware latent transition predictor
- survival latent predictor


### Example Training Command

```bash
python Predictor/train.py \
  --exp_name mri_backbone_survival \
  --timeline_json ./Predictor/dataset/MU_Glioma_Post/clinical_latest.json \
  --mri_root ./datasets/MU-Glioma-Post \
  --vision_checkpoint ./BrainIAC-main/src/checkpoints/BrainIAC.ckpt \
  --mri_size 96 \
  --batch_size 4 \
  --val_batch_size 4 \
  --num_epochs 40
```

You can also use:

```bash
bash run_training.sh
```

## Inference

Example of inverse treatment planning inference:

```bash
python main.py
```

`main.py` currently runs through the example block under `if __name__ == "__main__":`.
Before running it, edit the following fields in `main.py` to match your case:

- `WORLD_MODEL_PATH`
- `pre_treatment_latent`
- patient clinical profile
- `pre_payload`
- `between_payload`

## Inverse Survival Evaluation

The core function is:

```python
glioma_sequence_exploration(...)
```

defined in `main.py`.

The current inverse pipeline works as follows:

1. Generate candidate post-treatment actions with the LLM policy
2. Repair candidates using clinical guardrails
3. Evaluate candidates with the survival/world model
4. Rank sequences by predicted risk, toxicity, complexity, and uncertainty
5. Return the best treatment sequence and top-ranked alternatives

It supports:

- single-step inverse evaluation with `planning_horizon=1`
- long-horizon inverse planning with `planning_horizon>1`

Important arguments in `glioma_sequence_exploration(...)`:

- `pre_treatment_latent`
- `clinical`
- `imaging`
- `pre_payload`
- `between_payload`
- `world_model_path`
- `num_llm_sequences`
- `num_variants`
- `top_k`
- `planning_horizon`
- `discount_factor`
- `entropy_temperature`

## Citation

If you find this repository helpful, please cite:

```bibtex
@article{ding2025clarity,
  title={CLARITY: Medical World Model for Guiding Treatment Decisions by Modeling Context-Aware Disease Trajectories in Latent Space},
  author={Ding, Tianxingjian and Zou, Yuanhao and Chen, Chen and Shah, Mubarak and Tian, Yu},
  journal={arXiv preprint arXiv:2512.08029},
  year={2025}
}
```

## Acknowledgement

This codebase acknowledges prior and related foundation-model components including:

- MRI-CORE
- BrainIAC

## License

This repository is released under the MIT License. See `LICENSE`.
