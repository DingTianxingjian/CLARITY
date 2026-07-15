"""Train the latent-space TimeAwareGliomaSurvivalPredictor from features_output.csv."""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset_glioma_all_pairs_text import Config, GliomaAllPairsTextDataset
from models.full_model import TimeAwareGliomaSurvivalPredictor, extract_drug_category
from utils.metrics import (
    concordance_index,
    compute_auc,
    one_year_survival_targets_torch,
)


TEXT_ENCODER_NAME = "google/medgemma-4b-it"


class EarlyStopping:
    """Stop training when Val C-index stops improving."""
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = 0.0
        self.wait = 0

    def step(self, val_c_index: float) -> bool:
        if val_c_index > self.best + self.min_delta:
            self.best = val_c_index
            self.wait = 0
            return False
        self.wait += 1
        return self.wait >= self.patience


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        save_dir,
        contrastive_weight: float = 0.1,
        contrastive_temperature: float = 0.1,
        variance_weight: float = 0.0,
        cf_weight: float = 0.0,
        cf_cos_margin: float = 0.9,
        log_interval: int = 10,
        grad_clip_norm: float = 1.0,
        warmup_epochs: int = 10,
        lam_l1: float = 3.0,
        lam_cox: float = 2.0,
        lam_bce: float = 1.0,
        early_stopping_patience: int = 0,
        ema_momentum: float = 0.0,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        self.variance_weight = variance_weight
        self.cf_weight = cf_weight
        self.cf_cos_margin = cf_cos_margin
        self.log_interval = log_interval
        self.grad_clip_norm = grad_clip_norm
        self.warmup_epochs = warmup_epochs
        self.lam_l1  = lam_l1
        self.lam_cox = lam_cox
        self.lam_bce = lam_bce
        self.early_stopper = (
            EarlyStopping(patience=early_stopping_patience)
            if early_stopping_patience > 0 else None
        )
        self.ema_momentum = ema_momentum
        # Build EMA target encoder when requested (momentum > 0 and backbone exists)
        self.ema_mri_encoder = None
        if ema_momentum > 0 and model.mri_encoder is not None:
            self.ema_mri_encoder = copy.deepcopy(model.mri_encoder).to(device)
            for p in self.ema_mri_encoder.parameters():
                p.requires_grad_(False)
            print(f"  EMA target encoder initialised  (momentum={ema_momentum})")

        self.best_val_loss = float("inf")
        self.best_c_index = 0.0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_c_index": [],
            "val_c_index": [],
            "train_auc": [],
            "val_auc": [],
            "train_contrastive_loss": [],
            "train_cf_loss": [],
        }

    def _shared_step(self, batch, compute_contrastive: bool):
        pre_latent = batch["pre_latent"].to(self.device, non_blocking=True)
        post_latent = batch["post_latent"].to(self.device, non_blocking=True)
        survival_time = batch["survival_time"].to(self.device, non_blocking=True)
        event_indicator = batch["event_indicator"].to(self.device, non_blocking=True)
        time_delta = batch["time_delta"].to(self.device, non_blocking=True)
        drugs_text = batch["drugs_text"]
        clinical_text = batch["clinical_text"]

        # Online MRI encoding: bypass frozen NPZ, run ViT+LoRA end-to-end
        if self.model.mri_encoder is not None and "pre_mri" in batch:
            pre_mri  = batch["pre_mri"].to(self.device, non_blocking=True)
            post_mri = batch["post_mri"].to(self.device, non_blocking=True)
            pre_latent = self.model.mri_encoder(pre_mri)
            if self.ema_mri_encoder is not None:
                # JEPA/BYOL-style: post_latent from a slowly evolving EMA encoder.
                # EMA encoder is never directly gradient-updated → stable target.
                with torch.no_grad():
                    post_latent = self.ema_mri_encoder(post_mri)
            else:
                # Fallback: stop-gradient on the shared encoder.
                post_latent = self.model.mri_encoder(post_mri).detach()

        pred_latent, risk_score, survival_logit, condition_emb = self.model(
            pre_latent,
            drugs_text,
            time_delta,
            clinical_text,
        )

        # SupCon + drug-swap CF diversity loss
        contrastive_loss = torch.zeros((), device=self.device)
        cf_loss = torch.zeros((), device=self.device)
        needs_contra = compute_contrastive and (
            self.contrastive_weight > 0 or self.cf_weight > 0
        )
        if needs_contra:
            drug_categories = [extract_drug_category(t) for t in drugs_text]
            if self.contrastive_weight > 0:
                contrastive_loss = self.model.drug_category_contrastive_loss(
                    drug_categories, pred_latent, temperature=self.contrastive_temperature
                )
            if self.cf_weight > 0:
                cf_loss = self.model.drug_swap_diversity_loss(
                    predictor=self.model.latent_predictor,
                    pre_latent=pre_latent,
                    condition_emb=condition_emb,
                    time_delta=time_delta,
                    pred_latent=pred_latent,
                    drug_categories=drug_categories,
                    cos_margin=self.cf_cos_margin,
                )

        # Variance regularization: penalize pred_latent collapse (VICReg-style)
        var_loss = torch.zeros((), device=self.device)
        if self.variance_weight > 0:
            var_loss = self.model.variance_loss(pred_latent, gamma=0.05)

        main_loss, loss_dict = self.model.compute_loss(
            pred_latent,
            risk_score,
            survival_logit,
            post_latent,
            survival_time,
            event_indicator,
        )
        total_loss = (main_loss
                      + self.contrastive_weight * contrastive_loss
                      + self.cf_weight * cf_loss
                      + self.variance_weight * var_loss)
        loss_dict["contrastive"] = contrastive_loss.item()
        loss_dict["cf"] = cf_loss.item()
        loss_dict["var"] = var_loss.item()
        loss_dict["total"] = total_loss.item()

        return {
            "total_loss": total_loss,
            "main_loss": main_loss,
            "loss_dict": loss_dict,
            "risk_score": risk_score,
            "survival_logit": survival_logit,
            "survival_time": survival_time,
            "event_indicator": event_indicator,
            "time_delta": time_delta,
        }

    def train_epoch(self, epoch):
        self.model.train()
        epoch_losses = {"total": [], "l1": [], "cox": [], "bce": [], "contrastive": [], "cf": [], "var": []}
        all_risks = []
        all_survival_times = []
        all_events = []
        all_survival_probs = []
        all_one_year_labels = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, batch in enumerate(pbar):
            outputs = self._shared_step(
                batch,
                compute_contrastive=(self.contrastive_weight > 0 or self.cf_weight > 0),
            )

            self.optimizer.zero_grad(set_to_none=True)
            outputs["total_loss"].backward()
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
            self.optimizer.step()

            # EMA update: θ_ema = m·θ_ema + (1-m)·θ_online
            if self.ema_mri_encoder is not None:
                m = self.ema_momentum
                with torch.no_grad():
                    for p_ema, p_online in zip(
                        self.ema_mri_encoder.parameters(),
                        self.model.mri_encoder.parameters()
                    ):
                        p_ema.data.mul_(m).add_(p_online.data, alpha=1.0 - m)

            for key in epoch_losses:
                epoch_losses[key].append(outputs["loss_dict"][key])

            all_risks.append(outputs["risk_score"].detach().cpu())
            all_survival_times.append(outputs["survival_time"].detach().cpu())
            all_events.append(outputs["event_indicator"].detach().cpu())
            one_year_labels, one_year_mask = one_year_survival_targets_torch(
                outputs["survival_time"], outputs["event_indicator"]
            )
            one_year_mask = one_year_mask.bool()
            if one_year_mask.any():
                all_survival_probs.append(
                    torch.sigmoid(outputs["survival_logit"]).detach().cpu()[one_year_mask.cpu()]
                )
                all_one_year_labels.append(one_year_labels.detach().cpu()[one_year_mask.cpu()])

            if batch_idx % self.log_interval == 0:
                pbar.set_postfix(
                    loss=f"{outputs['loss_dict']['total']:.4f}",
                    main=f"{outputs['main_loss'].item():.4f}",
                    cf=f"{outputs['loss_dict']['cf']:.4f}",
                    contra=f"{outputs['loss_dict']['contrastive']:.4f}",
                )
           

        avg_losses = {k: float(np.mean(v)) for k, v in epoch_losses.items()}
        c_index, auc = self._compute_epoch_metrics(
            all_risks,
            all_survival_times,
            all_events,
            all_survival_probs,
            all_one_year_labels,
        )

        # --- 诊断：梯度范数 + risk score 分布 ---
        surv_grad_norm = 0.0
        for p in self.model.survival_module.parameters():
            if p.grad is not None:
                surv_grad_norm += p.grad.norm().item() ** 2
        surv_grad_norm = surv_grad_norm ** 0.5
        _risks = torch.cat(all_risks).squeeze().numpy()
        _surv  = torch.cat(all_survival_times).numpy()
        _corr  = float(np.corrcoef(_risks, _surv)[0, 1]) if _risks.std() > 1e-6 else float('nan')
        print(
            f"  [diag-train] risk: std={_risks.std():.4f}  corr(risk,surv)={_corr:.3f}  "
            f"survival_module grad_norm={surv_grad_norm:.4f}"
        )

        return avg_losses, c_index, auc

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        epoch_losses = {"total": [], "l1": [], "cox": [], "bce": [], "contrastive": [], "cf": [], "var": []}
        all_risks = []
        all_survival_times = []
        all_events = []
        all_survival_probs = []
        all_one_year_labels = []

        # --- One-shot forward-hook diagnostics (runs on first val batch only) ---
        _debug = {}
        def _store(name, tensor):
            tf = tensor.detach().float().reshape(tensor.shape[0], -1)  # [B, *]
            _debug[name] = (tf.std(dim=0).mean().item(),   # cross-sample diversity
                            tf.std(dim=1).mean().item(),   # within-sample spread
                            tf.mean().item())
        def _make_hook(name):
            def _h(module, inp, out):
                t = out[0] if isinstance(out, (tuple, list)) else out
                if isinstance(t, torch.Tensor):
                    _store(name, t)
            return _h
        def _make_hook_in(name):
            """Hook that captures the INPUT tensor (for measuring combined vector)."""
            def _h(module, inp, out):
                if inp and isinstance(inp[0], torch.Tensor):
                    _store(name, inp[0])
            return _h
        sm = self.model.survival_module
        _hooks = [
            self.model.shared_text_encoder.final_proj.register_forward_hook(
                _make_hook("text_final_proj_out")),
            self.model.latent_predictor.output_proj.register_forward_hook(
                _make_hook("latent_delta")),
            sm.input_proj.register_forward_hook(
                _make_hook("surv_input_proj")),
            sm.two_way_attn.register_forward_hook(
                _make_hook("two_way_attn_seq1_out")),
            sm.modality_fusion[0].register_forward_hook(
                _make_hook_in("combined_input")),
            sm.modality_fusion[0].register_forward_hook(
                _make_hook("surv_fusion_fc1")),
            sm.modality_fusion[-1].register_forward_hook(
                _make_hook("surv_fusion_last")),
            sm.risk_head[0].register_forward_hook(
                _make_hook("risk_head_fc1")),
            sm.risk_head[-1].register_forward_hook(
                _make_hook("risk_head_out")),
            sm.survival_head[0].register_forward_hook(
                _make_hook("surv_head_fc1")),
            sm.survival_head[-1].register_forward_hook(
                _make_hook("surv_head_out")),
        ]
        _debug_done = False

        pbar = tqdm(self.val_loader, desc="Validation")
        for batch in pbar:
            outputs = self._shared_step(batch, compute_contrastive=False)

            if not _debug_done:
                for h in _hooks:
                    h.remove()
                _hooks.clear()
                _debug_done = True
                _pl = outputs["risk_score"].float()
                _pre = batch["pre_latent"].float()
                print(f"\n  [DEBUG-VAL first batch]  (batch_std=cross-sample, feat_std=within-sample)")
                print(f"    {'layer':<30}  batch_std   feat_std    mean")
                print(f"    {'pre_latent':<30}  {_pre.reshape(_pre.shape[0],-1).std(dim=0).mean():.4f}      {_pre.reshape(_pre.shape[0],-1).std(dim=1).mean():.4f}      {_pre.mean():.4f}")
                for name, (bstd, fstd, mean) in sorted(_debug.items()):
                    print(f"    {name:<30}  {bstd:.4f}      {fstd:.4f}      {mean:.4f}")
                print(f"    risk_score (batch):             std={_pl.std():.6f}  mean={_pl.mean():.4f}")
                # Weight norms for final heads
                print(f"    risk_head[-1].weight  norm={sm.risk_head[-1].weight.norm():.4f}  |bias|={sm.risk_head[-1].bias.abs().mean():.4f}")
                print(f"    surv_head[-1].weight  norm={sm.survival_head[-1].weight.norm():.4f}  |bias|={sm.survival_head[-1].bias.abs().mean():.4f}")

            for key in epoch_losses:
                epoch_losses[key].append(outputs["loss_dict"][key])

            all_risks.append(outputs["risk_score"].cpu())
            all_survival_times.append(outputs["survival_time"].cpu())
            all_events.append(outputs["event_indicator"].cpu())
            one_year_labels, one_year_mask = one_year_survival_targets_torch(
                outputs["survival_time"], outputs["event_indicator"]
            )
            one_year_mask = one_year_mask.bool()
            if one_year_mask.any():
                all_survival_probs.append(
                    torch.sigmoid(outputs["survival_logit"]).cpu()[one_year_mask.cpu()]
                )
                all_one_year_labels.append(one_year_labels.cpu()[one_year_mask.cpu()])

        avg_losses = {k: float(np.mean(v)) for k, v in epoch_losses.items()}
        c_index, auc = self._compute_epoch_metrics(
            all_risks,
            all_survival_times,
            all_events,
            all_survival_probs,
            all_one_year_labels,
        )

        # --- 诊断：val risk score 分布 ---
        _risks = torch.cat(all_risks).squeeze().numpy()
        _surv  = torch.cat(all_survival_times).numpy()
        _corr  = float(np.corrcoef(_risks, _surv)[0, 1]) if _risks.std() > 1e-6 else float('nan')
        print(
            f"  [diag-val]   risk: std={_risks.std():.4f}  corr(risk,surv)={_corr:.3f}"
        )

        return avg_losses, c_index, auc

    @staticmethod
    def _compute_epoch_metrics(
        all_risks,
        all_survival_times,
        all_events,
        all_survival_probs,
        all_one_year_labels,
    ):
        all_risks = torch.cat(all_risks).squeeze().numpy()
        all_survival_times = torch.cat(all_survival_times).numpy()
        all_events = torch.cat(all_events).numpy()
        if all_survival_probs:
            all_survival_probs = torch.cat(all_survival_probs).squeeze().numpy()
            all_one_year_labels = torch.cat(all_one_year_labels)
        else:
            all_survival_probs = np.array([])
            all_one_year_labels = torch.tensor([])

        c_index = concordance_index(all_risks, all_survival_times, all_events)
        if (
            len(all_one_year_labels) == 0
            or all_one_year_labels.sum() == 0
            or all_one_year_labels.sum() == len(all_one_year_labels)
        ):
            auc = 0.5
        else:
            auc = compute_auc(all_survival_probs, all_one_year_labels.numpy())
        return c_index, auc

    def train(self, num_epochs, resume_from=None):
        start_epoch = 1
        if resume_from is not None:
            print(f"Loading checkpoint from {resume_from}...")
            checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
            # Support both full state_dict and trainable-only state_dict
            if "trainable_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["trainable_state_dict"], strict=False)
            else:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except (ValueError, KeyError) as e:
                print(f"  [Resume] Skipping optimizer/scheduler state ({e}); "
                      "starting fresh optimizer from loaded model weights.")
            start_epoch = checkpoint["epoch"] + 1
            self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            self.best_c_index = checkpoint.get("best_c_index", 0.0)
            loaded_history = checkpoint.get("history", {})
            # Merge: keep loaded values for existing keys, add missing keys from self.history
            for k in self.history:
                if k in loaded_history:
                    self.history[k] = loaded_history[k]
                # else: keep self.history[k] = [] (new keys start empty)
            print(f"Resumed from epoch {checkpoint['epoch']}")
            print(f"Best C-index so far: {self.best_c_index:.4f}")

        print(f"Starting training from epoch {start_epoch} to {num_epochs}...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Contrastive loss weight: {self.contrastive_weight}")
        print("-" * 80)

        warmup_epochs = self.warmup_epochs

        for epoch in range(start_epoch, num_epochs + 1):
            # --- Phase switching ---
            if warmup_epochs > 0 and epoch <= warmup_epochs:
                if epoch == start_epoch or epoch == 1:
                    print(f"[Phase 1] Epochs 1-{warmup_epochs}: L1-only warmup (λ=1,0,0)")
                self.model.lambda_l1.fill_(1.0)
                self.model.lambda_cox.fill_(0.0)
                self.model.lambda_bce.fill_(0.0)
            else:
                if epoch == (warmup_epochs + 1) or (warmup_epochs == 0 and epoch == start_epoch):
                    print(f"[Phase 2] Epoch {epoch}+: survival training "
                          f"(λ_l1={self.lam_l1}, λ_cox={self.lam_cox}, λ_bce={self.lam_bce})")
                self.model.lambda_l1.fill_(self.lam_l1)
                self.model.lambda_cox.fill_(self.lam_cox)
                self.model.lambda_bce.fill_(self.lam_bce)

            train_losses, train_c_index, train_auc = self.train_epoch(epoch)
            val_losses, val_c_index, val_auc = self.validate()
            self.scheduler.step()

            self.history["train_loss"].append(train_losses["total"])
            self.history["val_loss"].append(val_losses["total"])
            self.history["train_c_index"].append(train_c_index)
            self.history["val_c_index"].append(val_c_index)
            self.history["train_auc"].append(train_auc)
            self.history["val_auc"].append(val_auc)
            self.history["train_contrastive_loss"].append(train_losses["contrastive"])
            self.history["train_cf_loss"].append(train_losses["cf"])

            print(f"\nEpoch {epoch}/{num_epochs}")
            print(
                "Train - Loss: "
                f"{train_losses['total']:.4f} "
                f"(L1: {train_losses['l1']:.4f}, Cox: {train_losses['cox']:.4f}, "
                f"BCE: {train_losses['bce']:.4f}, Contra: {train_losses['contrastive']:.4f}, "
                f"CF: {train_losses['cf']:.4f}), "
                f"C-index: {train_c_index:.4f}, 1-year AUC: {train_auc:.4f}"
            )
            print(
                "Val   - Loss: "
                f"{val_losses['total']:.4f} "
                f"(L1: {val_losses['l1']:.4f}, Cox: {val_losses['cox']:.4f}, "
                f"BCE: {val_losses['bce']:.4f}), "
                f"C-index: {val_c_index:.4f}, 1-year AUC: {val_auc:.4f}"
            )
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            if epoch > warmup_epochs and val_c_index > self.best_c_index:
                self.best_c_index = val_c_index
                self.save_checkpoint(epoch, "best_c_index.pth")
                print(f"Saved best C-index model ({val_c_index:.4f})")

            if val_losses["total"] < self.best_val_loss:
                self.best_val_loss = val_losses["total"]
                self.save_checkpoint(epoch, "best_loss.pth")
                print(f"Saved best loss model ({val_losses['total']:.4f})")

            if epoch % 10 == 0:
                self.save_checkpoint(epoch, f"epoch_{epoch}.pth")

            print("-" * 80)

            if epoch > warmup_epochs and self.early_stopper is not None:
                if self.early_stopper.step(val_c_index):
                    print(f"[EarlyStopping] No improvement for {self.early_stopper.patience} "
                          f"epochs. Best Val C-index: {self.early_stopper.best:.4f}. Stopping.")
                    break

        self.save_history()
        print("\nTraining completed!")
        print(f"Best C-index: {self.best_c_index:.4f}")
        print(f"Best val loss: {self.best_val_loss:.4f}")

    def save_history(self):
        history_path = self.save_dir / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as handle:
            json.dump(self.history, handle, indent=2)
        print(f"Training history saved to {history_path}")

    def save_checkpoint(self, epoch, filename):
        checkpoint_path = self.save_dir / filename
        # Only save trainable parameters to keep checkpoint small (~100MB vs 4.6GB).
        # Frozen MedGemma base weights are reloaded from HF cache at resume time.
        trainable_state = {
            k: v for k, v in self.model.state_dict().items()
            if any(k.startswith(prefix) for prefix in [
                "latent_predictor.", "survival_module.",
                "shared_text_encoder.final_proj.",
                "shared_text_encoder.model.base_model.model.",   # LoRA adapters via PEFT
            ]) and self.model.state_dict()[k].requires_grad or
            # always include non-frozen small modules
            k.startswith("lambda_")
        }
        # Fallback: if filter too aggressive, save all trainable
        trainable_keys = {n for n, p in self.model.named_parameters() if p.requires_grad}
        trainable_state = {k: v for k, v in self.model.state_dict().items()
                           if k in trainable_keys or k.startswith("lambda_")}
        torch.save(
            {
                "epoch": epoch,
                "trainable_state_dict": trainable_state,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_loss": self.best_val_loss,
                "best_c_index": self.best_c_index,
                "history": self.history,
            },
            checkpoint_path,
        )
        size_mb = checkpoint_path.stat().st_size / 1e6
        print(f"Checkpoint saved at {checkpoint_path} ({size_mb:.0f} MB)")


def build_dataloaders(args):
    mri_data_dir = getattr(args, "mri_data_dir", None)
    features_csv = getattr(args, "features_csv", None)
    # In online mode (sam_ckpt set + mri_data_dir set), features_csv is optional
    data_cfg = Config(
        timeline_json=args.timeline_json,
        features_csv=features_csv,
        take_dims=args.take_dims,
        mri_data_dir=mri_data_dir,
    )
    full_dataset = GliomaAllPairsTextDataset(data_cfg, include_between=False)

    print("Splitting dataset by patient ID (80/20 split)...")
    patient_to_indices = {}
    for index, item in enumerate(full_dataset.index):
        patient_to_indices.setdefault(item["pid"], []).append(index)

    unique_pids = list(patient_to_indices.keys())
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique_pids)

    target_train_size = int(0.8 * len(full_dataset))
    train_indices, val_indices = [], []
    train_pids, val_pids = set(), set()
    current_train_size = 0

    for pid in unique_pids:
        sample_indices = patient_to_indices[pid]
        if current_train_size < target_train_size:
            train_indices.extend(sample_indices)
            train_pids.add(pid)
            current_train_size += len(sample_indices)
        else:
            val_indices.extend(sample_indices)
            val_pids.add(pid)

    if train_pids:
        last_pid = next(iter(train_pids))
        indices_to_move = patient_to_indices[last_pid]
        err_before = abs(current_train_size - target_train_size)
        err_after = abs(current_train_size - len(indices_to_move) - target_train_size)
        if err_after < err_before:
            train_indices = [idx for idx in train_indices if idx not in indices_to_move]
            val_indices.extend(indices_to_move)
            train_pids.remove(last_pid)
            val_pids.add(last_pid)

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset   = torch.utils.data.Subset(full_dataset, val_indices)

    # Overfit mode: use train set as val to verify C-index can memorise training data
    overfit = getattr(args, "overfit", False)
    if overfit:
        print("[OVERFIT MODE] val = train set to verify C-index can memorise")
        val_dataset = train_dataset

    print("Split complete:")
    print(
        f"  Train: {len(train_pids)} patients, {len(train_dataset)} samples "
        f"({len(train_dataset) / len(full_dataset):.1%})"
    )
    print(
        f"  Val:   {'[=train]' if overfit else f'{len(val_pids)} patients'}, "
        f"{len(val_dataset)} samples"
    )

    collate_fn = GliomaAllPairsTextDataset.collate_fn
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def build_model(args, device):
    # Construct vision backbone externally so the core model stays backbone-agnostic.
    # To use a different backbone, construct it here and pass it as vision_backbone.
    vision_backbone = None
    brainiac_ckpt = getattr(args, "brainiac_ckpt", None)
    if brainiac_ckpt is not None:
        from models.brainiac_adapter import BrainIACAdapter
        from models.vision_backbone import MultiModalVisionBackbone
        _adapter = BrainIACAdapter(
            checkpoint_path=brainiac_ckpt,
            tokens_per_modality=getattr(args, "brainiac_tokens", 8),
            lora_r=getattr(args, "brainiac_lora_r", 8),
        )
        vision_backbone = MultiModalVisionBackbone(_adapter, num_modalities=4)

    model = TimeAwareGliomaSurvivalPredictor(
        text_encoder_name=args.text_encoder_name,
        freeze_text_encoder=args.freeze_text_encoder,
        text_output_dim=768,
        time_dim=128,
        time_encoding_type="fourier",
        latent_dim=args.latent_dim,
        num_modalities=args.num_modalities,
        predictor_hidden_dim=512,
        predictor_num_layers=4,
        predictor_num_heads=8,
        survival_hidden_dim=128,
        lambda_l1=args.lambda_l1,
        lambda_cox=args.lambda_cox,
        lambda_bce=args.lambda_bce,
        dropout=args.dropout,
        vision_backbone=vision_backbone,
    ).to(device)

    param_count = model.get_parameter_count()
    print("\nModel statistics:")
    print(f"  Total parameters:     {param_count['total']:,}")
    print(f"  Trainable parameters: {param_count['trainable']:,}")
    if "mri_encoder_trainable" in param_count:
        print(f"  MRI encoder LoRA:     {param_count['mri_encoder_trainable']:,}")
    return model


def build_optimizer(args, model):
    optimizer_groups = []

    def add_group(params, lr, weight_decay):
        params = [param for param in params if param.requires_grad]
        if params:
            optimizer_groups.append(
                {"params": params, "lr": lr, "weight_decay": weight_decay}
            )

    add_group(model.latent_predictor.time_proj.parameters(), args.lr, 5e-4)
    add_group(
        [p for n, p in model.latent_predictor.named_parameters() if "time_proj" not in n],
        args.lr * 0.1,
        1e-5,
    )
    # Survival module: high weight_decay to fight small-dataset overfitting
    survival_wd = getattr(args, "survival_wd", 1e-2)
    add_group(model.survival_module.parameters(), args.lr, survival_wd)

    text_lora_params = []
    text_proj_params = []
    text_other_params = []
    for name, param in model.shared_text_encoder.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            text_lora_params.append(param)
        elif "final_proj" in name:
            text_proj_params.append(param)
        else:
            text_other_params.append(param)
    add_group(text_lora_params, args.text_lr, 0.0)
    add_group(text_proj_params, args.text_proj_lr, 1e-4)
    add_group(text_other_params, args.text_lr, 0.0)

    # Vision encoder LoRA (only when sam_ckpt is set)
    if model.mri_encoder is not None:
        vision_lora_params = [
            p for _, p in model.mri_encoder.named_parameters() if p.requires_grad
        ]
        if vision_lora_params:
            add_group(vision_lora_params, args.vision_lr, 0.0)
            print(f"  Vision LoRA param group: {sum(p.numel() for p in vision_lora_params):,} params @ lr={args.vision_lr}")

    if not optimizer_groups:
        raise ValueError("No trainable parameter groups were created.")

    optimizer = AdamW(optimizer_groups)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    return optimizer, scheduler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="mri_backbone_survival",
                        help="Experiment name. Checkpoints saved to experiments/<exp_name>/checkpoints/")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_epochs", type=int, default=70)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--text_lr", type=float, default=1e-4)
    parser.add_argument("--text_proj_lr", type=float, default=5e-4)
    parser.add_argument("--vision_lr", type=float, default=5e-5)
    parser.add_argument("--vision_full_lr", type=float, default=1e-5)
    parser.add_argument("--brainiac_ckpt", type=str, default=None,
                        help="Path to BrainIAC.ckpt for online fine-tuning (overrides sam_ckpt)")
    parser.add_argument("--brainiac_tokens", type=int, default=8,
                        help="Patch tokens per modality from BrainIAC (default 8, total M=32)")
    parser.add_argument("--brainiac_lora_r", type=int, default=8,
                        help="LoRA rank for BrainIAC backbone (0=frozen)")
    parser.add_argument("--brainiac_ema_momentum", type=float, default=0.0,
                        help="EMA momentum for BrainIAC target encoder (0=disabled, use stop-grad; "
                             "0.996 recommended for BYOL/JEPA-style training)")
    parser.add_argument("--sam_ckpt", type=str, default=None,
                        help="SAM ViT-B checkpoint. Enables end-to-end MRI encoder LoRA (plan B).")
    parser.add_argument("--mri_data_dir", type=str, default=None,
                        help="Root dir of raw NIfTI MRI volumes (e.g. datasets/MU-Glioma-Post). "
                             "Required when --sam_ckpt is set for online inference.")
    parser.add_argument("--vision_lora_r", type=int, default=4)
    parser.add_argument("--vision_lora_alpha", type=int, default=16)
    parser.add_argument("--vision_lora_dropout", type=float, default=0.1)
    parser.add_argument("--num_slices", type=int, default=32,
                        help="Central axial slices per modality fed to SAM ViT (VRAM budget).")
    parser.add_argument("--contrastive_weight", type=float, default=0.0)
    parser.add_argument("--contrastive_temperature", type=float, default=0.1)
    parser.add_argument("--variance_weight", type=float, default=0.0,
                        help="VICReg variance loss weight to prevent pred_latent collapse")
    parser.add_argument("--cf_weight", type=float, default=0.0,
                        help="Drug-swap counterfactual diversity loss weight "
                             "(pushes pred_latent apart for different drugs)")
    parser.add_argument("--cf_cos_margin", type=float, default=0.9,
                        help="Cosine similarity margin for CF diversity loss "
                             "(penalise cos_sim > margin for different-drug pairs)")
    parser.add_argument("--early_stopping_patience", type=int, default=0,
                        help="Early stopping patience (0=disabled)")
    parser.add_argument("--survival_wd", type=float, default=1e-2,
                        help="Weight decay for survival module (default 1e-2)")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="L1-only warmup epochs before Cox+BCE activate (0=no warmup)")
    parser.add_argument("--lambda_l1",  type=float, default=3.0)
    parser.add_argument("--lambda_cox", type=float, default=2.0)
    parser.add_argument("--lambda_bce", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--overfit", action="store_true",
                        help="Use train set as val to verify C-index can overfit")
    parser.add_argument(
        "--features_csv",
        type=str,
        default="./Predictor/dataset/MU_Glioma_Post/features_output.csv",
    )
    parser.add_argument(
        "--timeline_json",
        type=str,
        default="./Predictor/dataset/MU_Glioma_Post/clinical_latest.json",
    )
    parser.add_argument("--text_encoder_name", type=str, default=TEXT_ENCODER_NAME)
    parser.add_argument("--freeze_text_encoder", action="store_true")
    # encoder-agnostic feature dims (767=BrainIAC, 256=mri_foundation SAM)
    parser.add_argument("--latent_dim",    type=int, default=767,
                        help="Feature dim per token from the visual encoder")
    parser.add_argument("--take_dims",     type=int, default=767,
                        help="Dims to take per modality from CSV (BrainIAC only)")
    parser.add_argument("--num_modalities", type=int, default=4,
                        help="Number of token slots: 4 for BrainIAC, 4*N_slices for mri_foundation")
    return parser.parse_args()


def main():
    args = parse_args()
    # features_csv is optional when mri_data_dir+sam_ckpt is provided (online mode)
    if args.features_csv:
        features_path = Path(args.features_csv)
        if not features_path.exists():
            raise FileNotFoundError(f"features_csv not found: {features_path}")
        print(f"Using precomputed latent features from {features_path}")
    else:
        print("Online MRI encoding mode (no pre-extracted features)")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_dataloaders(args)
    model = build_model(args, device)
    optimizer, scheduler = build_optimizer(args, model)

    save_dir = Path("experiments") / args.exp_name / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "args.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=save_dir,
        contrastive_weight=args.contrastive_weight,
        contrastive_temperature=args.contrastive_temperature,
        variance_weight=args.variance_weight,
        cf_weight=getattr(args, "cf_weight", 0.0),
        cf_cos_margin=getattr(args, "cf_cos_margin", 0.9),
        grad_clip_norm=args.grad_clip_norm,
        warmup_epochs=args.warmup_epochs,
        lam_l1=args.lambda_l1,
        lam_cox=args.lambda_cox,
        lam_bce=args.lambda_bce,
        early_stopping_patience=args.early_stopping_patience,
        ema_momentum=getattr(args, "brainiac_ema_momentum", 0.0),
    )

    print(f"\n{'=' * 80}")
    print("Starting latent-space survival training...")
    print(f"{'=' * 80}\n")
    trainer.train(num_epochs=args.num_epochs, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
