"""
Train the time-aware glioma survival model directly from MRI volumes.

This entrypoint removes the old pixel-loss / diffusion branch and instead uses
an MRI foundation vision backbone to extract pre/post MRI features online.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset_glioma_all_pairs_text import Config
from dataset.dataset_with_mri import GliomaDatasetWithMRI
from models.full_model import MRITimeAwareSurvivalPredictor
from utils.metrics import concordance_index, compute_auc


TEXT_ENCODER_NAME = "google/medgemma-4b-it"


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
        log_interval: int = 10,
        grad_clip_norm: float = 1.0,
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
        self.log_interval = log_interval
        self.grad_clip_norm = grad_clip_norm

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
        }

    @staticmethod
    def _compute_contrastive_loss(z_anchor, z_negative, margin=0.5):
        z_anchor = F.normalize(z_anchor.mean(dim=1), p=2, dim=-1)
        z_negative = F.normalize(z_negative.mean(dim=1), p=2, dim=-1)
        cosine_sim = F.cosine_similarity(z_anchor, z_negative)
        return F.relu(cosine_sim - margin).mean()

    @staticmethod
    def _one_year_labels(survival_time, event_indicator):
        return ((survival_time > 365) & (event_indicator == 0)).float()

    def _shared_step(self, batch, compute_contrastive: bool):
        pre_mri = batch["pre_mri"].to(self.device, non_blocking=True)
        post_mri = batch["post_mri"].to(self.device, non_blocking=True)
        survival_time = batch["survival_time"].to(self.device, non_blocking=True)
        event_indicator = batch["event_indicator"].to(self.device, non_blocking=True)
        time_delta = batch["time_delta"].to(self.device, non_blocking=True)
        drugs_text = batch["drugs_text"]
        clinical_text = batch["clinical_text"]

        pre_latent = self.model.encode_mri(pre_mri)
        post_latent = self.model.encode_mri(post_mri).detach()
        pred_latent, risk_score, survival_prob = self.model.predict_from_features(
            pre_latent,
            drugs_text,
            time_delta,
            clinical_text,
        )

        contrastive_loss = torch.zeros((), device=self.device)
        if compute_contrastive:
            negative_drugs_text = drugs_text[1:] + drugs_text[:1]
            pred_negative, _, _ = self.model.predict_from_features(
                pre_latent,
                negative_drugs_text,
                time_delta,
                clinical_text,
            )
            contrastive_loss = self._compute_contrastive_loss(pred_latent, pred_negative)

        main_loss, loss_dict = self.model.compute_loss(
            pred_latent,
            risk_score,
            survival_prob,
            post_latent,
            survival_time,
            event_indicator,
        )
        total_loss = main_loss + self.contrastive_weight * contrastive_loss
        loss_dict["contrastive"] = contrastive_loss.item()
        loss_dict["total"] = total_loss.item()

        return {
            "total_loss": total_loss,
            "main_loss": main_loss,
            "loss_dict": loss_dict,
            "risk_score": risk_score,
            "survival_prob": survival_prob,
            "survival_time": survival_time,
            "event_indicator": event_indicator,
            "time_delta": time_delta,
        }

    def train_epoch(self, epoch):
        self.model.train()
        epoch_losses = {"total": [], "l1": [], "cox": [], "bce": [], "contrastive": []}
        all_risks = []
        all_survival_times = []
        all_events = []
        all_survival_probs = []
        all_one_year_labels = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, batch in enumerate(pbar):
            outputs = self._shared_step(batch, compute_contrastive=True)

            self.optimizer.zero_grad(set_to_none=True)
            outputs["total_loss"].backward()
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
            self.optimizer.step()

            for key in epoch_losses:
                epoch_losses[key].append(outputs["loss_dict"][key])

            all_risks.append(outputs["risk_score"].detach().cpu())
            all_survival_times.append(outputs["survival_time"].detach().cpu())
            all_events.append(outputs["event_indicator"].detach().cpu())
            all_survival_probs.append(torch.sigmoid(outputs["survival_prob"]).detach().cpu())
            all_one_year_labels.append(
                self._one_year_labels(
                    outputs["survival_time"], outputs["event_indicator"]
                ).detach().cpu()
            )

            if batch_idx % self.log_interval == 0:
                pbar.set_postfix(
                    loss=f"{outputs['loss_dict']['total']:.4f}",
                    main=f"{outputs['main_loss'].item():.4f}",
                    contra=f"{outputs['loss_dict']['contrastive']:.4f}",
                )
            if batch_idx == 0:
                print(f"\n[Epoch {epoch} - First batch diagnostics]")
                print(
                    f"  Loss weights: l1={self.model.lambda_l1.item():.3f}, "
                    f"cox={self.model.lambda_cox.item():.3f}, "
                    f"bce={self.model.lambda_bce.item():.3f}, "
                    f"contrastive={self.contrastive_weight:.3f}"
                )
                print(
                    f"  Raw losses: l1={outputs['loss_dict']['l1']:.4f}, "
                    f"cox={outputs['loss_dict']['cox']:.4f}, "
                    f"bce={outputs['loss_dict']['bce']:.4f}, "
                    f"contrastive={outputs['loss_dict']['contrastive']:.4f}"
                )
                print(
                    f"  Main loss: {outputs['main_loss'].item():.4f}, "
                    f"Total loss: {outputs['loss_dict']['total']:.4f}"
                )
                print(
                    f"  Time delta range: "
                    f"[{outputs['time_delta'].min():.1f}, {outputs['time_delta'].max():.1f}] days"
                )

        avg_losses = {k: float(np.mean(v)) for k, v in epoch_losses.items()}
        c_index, auc = self._compute_epoch_metrics(
            all_risks,
            all_survival_times,
            all_events,
            all_survival_probs,
            all_one_year_labels,
        )
        return avg_losses, c_index, auc

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        epoch_losses = {"total": [], "l1": [], "cox": [], "bce": [], "contrastive": []}
        all_risks = []
        all_survival_times = []
        all_events = []
        all_survival_probs = []
        all_one_year_labels = []

        pbar = tqdm(self.val_loader, desc="Validation")
        for batch in pbar:
            outputs = self._shared_step(batch, compute_contrastive=False)
            for key in epoch_losses:
                epoch_losses[key].append(outputs["loss_dict"][key])

            all_risks.append(outputs["risk_score"].cpu())
            all_survival_times.append(outputs["survival_time"].cpu())
            all_events.append(outputs["event_indicator"].cpu())
            all_survival_probs.append(torch.sigmoid(outputs["survival_prob"]).cpu())
            all_one_year_labels.append(
                self._one_year_labels(
                    outputs["survival_time"], outputs["event_indicator"]
                ).cpu()
            )

        avg_losses = {k: float(np.mean(v)) for k, v in epoch_losses.items()}
        c_index, auc = self._compute_epoch_metrics(
            all_risks,
            all_survival_times,
            all_events,
            all_survival_probs,
            all_one_year_labels,
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
        all_survival_probs = torch.cat(all_survival_probs).squeeze().numpy()
        all_one_year_labels = torch.cat(all_one_year_labels)

        c_index = concordance_index(all_risks, all_survival_times, all_events)
        if (
            all_one_year_labels.sum() == 0
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
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            self.best_c_index = checkpoint.get("best_c_index", 0.0)
            self.history = checkpoint.get("history", self.history)
            print(f"Resumed from epoch {checkpoint['epoch']}")
            print(f"Best C-index so far: {self.best_c_index:.4f}")

        print(f"Starting training from epoch {start_epoch} to {num_epochs}...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Contrastive loss weight: {self.contrastive_weight}")
        print("-" * 80)

        for epoch in range(start_epoch, num_epochs + 1):
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

            print(f"\nEpoch {epoch}/{num_epochs}")
            print(
                "Train - Loss: "
                f"{train_losses['total']:.4f} "
                f"(L1: {train_losses['l1']:.4f}, Cox: {train_losses['cox']:.4f}, "
                f"BCE: {train_losses['bce']:.4f}, Contra: {train_losses['contrastive']:.4f}), "
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

            if val_c_index > self.best_c_index:
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
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_loss": self.best_val_loss,
                "best_c_index": self.best_c_index,
                "history": self.history,
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved at {checkpoint_path}")


def build_dataloaders(args):
    data_cfg = Config(
        timeline_json=args.timeline_json,
        features_csv=None,
    )
    full_dataset = GliomaDatasetWithMRI(
        data_cfg,
        mri_data_root=args.mri_root,
        load_mri=True,
        target_size=(args.mri_size, args.mri_size, args.mri_size),
        include_between=False,
    )

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
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    print("Split complete:")
    print(
        f"  Train: {len(train_pids)} patients, {len(train_dataset)} samples "
        f"({len(train_dataset) / len(full_dataset):.1%})"
    )
    print(
        f"  Val:   {len(val_pids)} patients, {len(val_dataset)} samples "
        f"({len(val_dataset) / len(full_dataset):.1%})"
    )

    collate_fn = GliomaDatasetWithMRI.collate_fn_with_mri
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
    model = MRITimeAwareSurvivalPredictor(
        vision_checkpoint_path=args.vision_checkpoint,
        mri_img_size=(args.mri_size, args.mri_size, args.mri_size),
        freeze_vision_backbone=args.freeze_vision_backbone,
        vision_lora_r=0 if args.vision_full_finetune else args.vision_lora_r,
        vision_lora_alpha=args.vision_lora_alpha,
        vision_lora_dropout=args.vision_lora_dropout,
        text_encoder_name=args.text_encoder_name,
        freeze_text_encoder=args.freeze_text_encoder,
        text_output_dim=768,
        time_dim=128,
        time_encoding_type="fourier",
        latent_dim=768,
        num_modalities=4,
        predictor_hidden_dim=512,
        predictor_num_layers=4,
        predictor_num_heads=8,
        survival_hidden_dim=128,
        lambda_l1=5.0,
        lambda_cox=2.0,
        lambda_bce=1.0,
        dropout=0.2,
    ).to(device)

    param_count = model.get_parameter_count()
    print("\nModel statistics:")
    print(f"  Total parameters: {param_count['total']:,}")
    print(f"  Trainable parameters: {param_count['trainable']:,}")
    print(f"  Vision backbone parameters: {param_count['vision_backbone']:,}")
    print(
        f"  Vision backbone trainable parameters: "
        f"{param_count['vision_backbone_trainable']:,}"
    )
    print(f"  Vision LoRA modules: {param_count['vision_lora_modules']}")
    return model


def build_optimizer(args, model):
    optimizer_groups = []

    def add_group(params, lr, weight_decay):
        params = [param for param in params if param.requires_grad]
        if params:
            optimizer_groups.append(
                {"params": params, "lr": lr, "weight_decay": weight_decay}
            )

    add_group(model.drug_projection.parameters(), args.lr, 1e-4)
    add_group(model.context_projection.parameters(), args.lr, 1e-4)
    add_group(model.latent_predictor.time_encoder.parameters(), args.lr, 1e-4)
    add_group(
        [
            param
            for name, param in model.latent_predictor.named_parameters()
            if "time_encoder" not in name
        ],
        args.lr * 0.1,
        1e-5,
    )
    add_group(model.survival_module.parameters(), args.lr * 0.1, 1e-3)

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

    vision_lora_params = []
    vision_other_params = []
    for name, param in model.vision_backbone.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            vision_lora_params.append(param)
        else:
            vision_other_params.append(param)

    add_group(text_lora_params, args.text_lr, 0.0)
    add_group(text_proj_params, args.text_proj_lr, 1e-4)
    add_group(text_other_params, args.text_lr, 0.0)
    add_group(vision_lora_params, args.vision_lr, 0.0)
    add_group(vision_other_params, args.vision_full_lr, 1e-5)

    if not optimizer_groups:
        raise ValueError("No trainable parameter groups were created.")

    optimizer = AdamW(optimizer_groups)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    return optimizer, scheduler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="mri_backbone_survival")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--text_lr", type=float, default=1e-4)
    parser.add_argument("--text_proj_lr", type=float, default=5e-4)
    parser.add_argument("--vision_lr", type=float, default=5e-5)
    parser.add_argument("--vision_full_lr", type=float, default=1e-5)
    parser.add_argument("--contrastive_weight", type=float, default=0.1)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--mri_size", type=int, default=96)
    parser.add_argument("--mri_root", type=str, default="./datasets/MU-Glioma-Post")
    parser.add_argument(
        "--timeline_json",
        type=str,
        default="./Predictor/dataset/MU_Glioma_Post/clinical_latest.json",
    )
    parser.add_argument(
        "--vision_checkpoint",
        type=str,
        default="./BrainIAC-main/src/checkpoints/BrainIAC.ckpt",
        help="Checkpoint for the MRI foundation/BrainIAC vision backbone.",
    )
    parser.add_argument("--text_encoder_name", type=str, default=TEXT_ENCODER_NAME)
    parser.add_argument("--freeze_text_encoder", action="store_true")
    parser.add_argument("--vision_lora_r", type=int, default=8)
    parser.add_argument("--vision_lora_alpha", type=int, default=16)
    parser.add_argument("--vision_lora_dropout", type=float, default=0.1)
    parser.add_argument(
        "--vision_full_finetune",
        action="store_true",
        help="Disable LoRA and full-finetune the entire BrainIAC backbone.",
    )
    parser.add_argument(
        "--freeze_vision_backbone",
        action="store_true",
        help="Freeze the MRI vision backbone and train only the temporal/survival heads.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_dataloaders(args)
    model = build_model(args, device)
    optimizer, scheduler = build_optimizer(args, model)

    save_dir = Path("checkpoints") / args.exp_name
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
        grad_clip_norm=args.grad_clip_norm,
    )

    print(f"\n{'=' * 80}")
    print("Starting MRI-backbone survival training...")
    print(f"{'=' * 80}\n")
    trainer.train(num_epochs=args.num_epochs, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
