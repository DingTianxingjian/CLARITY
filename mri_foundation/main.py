# -*- coding: utf-8 -*-
"""
Single-file main + integrated cfg.parse_args().

Notes / compatibility:
- Keeps original behavior: warmup schedule, power decay (0.9), eval every 10 epochs,
  early stop (no improvement for 200 epochs after epoch >= 1000), save to
  temp/<dataset>_<model>_best.pth, 2D+3D Dice/NSD, threshold (pred > 0).
- Accepts both -encoder_adapter_depths and -encoder-adapter-depths; normalizes to
  args.encoder_adapter_depths for internal use.
- Fixes a typo in your cfg where two add_arguments were on one line.
- Leaves your bool flags as-is (type=bool) to avoid behavior changes.
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import monai

from models.sam import sam_model_registry
from utils.dataset import Public_dataset
from utils.metrics import compute_dice, compute_nsd

# ---- Optional LoRA import (keeps old behavior if available) -----------------
try:
    from models.lora_sam import LoRA_Sam  # adjust if your LoRA wrapper lives elsewhere
except Exception:
    LoRA_Sam = None  # Will raise a helpful error if user selects finetune_type='lora

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-net', type=str, default='sam', help='net type')
    parser.add_argument('-arch', type=str, default='vit_b', help='net architecture, pick between vit_h, vit_b, vit_t')
    parser.add_argument('-baseline', type=str, default='unet', help='baseline net type')
    parser.add_argument('-dataset_name', type=str, default='MRI-Prostate', help='the name of dataset to be finetuned')
    parser.add_argument('-img_folder', type=str, default='./datasets/2D-slices/images', help='the folder putting images')
    parser.add_argument('-mask_folder', type=str, default='./datasets/2D-slices/masks', help='the folder putting masks')
    parser.add_argument('-train_img_list', type=str, default='./datasets/train.txt')
    parser.add_argument('-val_img_list', type=str, default='./datasets/val.txt')
    parser.add_argument('-test_img_list', type=str, default='./datasets/test.txt')
    parser.add_argument('-targets', type=str, default='combine_all')
    parser.add_argument('-cls', type=int, default=-1, help='cls to be segmented')
    parser.add_argument('-model', type=str, default='ours', help='which model to use')

    parser.add_argument('-finetune_type', type=str, default='adapter', help='pick among vanilla, adapter, lora')
    parser.add_argument('-normalize_type', type=str, default='sam', help='normalization type, pick between sam or medsam')

    parser.add_argument('-dir_checkpoint', type=str, default='checkpoints', help='the checkpoint folder to save final model')
    parser.add_argument('-num_cls', type=int, default=1, help='output channels (#target classes + 1)')
    parser.add_argument('-epochs', type=int, default=200, help='max epochs to train')
    parser.add_argument('-val_freq', type=int, default=10, help='validation frequency (epochs). Original code used 10.')
    parser.add_argument('-sam_ckpt', type=str, default='sam_vit_b_01ec64.pth', help='path to the SAM checkpoint')

    parser.add_argument('-image_size', type=int, default=1024, help='input image size')
    parser.add_argument('-out_size', type=int, default=256, help='output mask size')
    parser.add_argument('-if_warmup', type=bool, default=True, help='if warm up training phase')
    parser.add_argument('-warmup_period', type=int, default=200, help='warm up iterations')
    parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')

    parser.add_argument('-if_update_encoder', type=bool, default=True, help='if update_image_encoder')
    parser.add_argument('-if_encoder_adapter', type=bool, default=True, help='add adapter to encoder')
    # Provide both spellings; we normalize after parsing.
    parser.add_argument('-encoder_adapter_depths', nargs='*', type=int, default=[0, 1, 10, 11],
                        help='depths of blocks to add adapter')
    parser.add_argument('-if_mask_decoder_adapter', type=bool, default=True, help='add adapter to mask decoder')
    parser.add_argument('-decoder_adapt_depth', type=int, default=2, help='depth of the decoder adapter')

    parser.add_argument('-if_encoder_lora_layer', type=bool, default=False, help='add lora to encoder')
    parser.add_argument('-if_decoder_lora_layer', type=bool, default=False, help='add lora to decoder')
    parser.add_argument('-encoder_lora_layer', nargs='*', type=int, default=[0, 1, 10, 11],
                        help='depths to add lora; [] means every layer')

    opt = parser.parse_args()
    return opt


# ==========================
# Config & Globals
# ==========================
args = parse_args()
# Force original script’s overrides (you had these on top-level right after parse_args)
args.num_cls = 1
args.epochs = 3000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESIZE_TO_MASK = None  # set lazily on first use

# ==========================
# Utilities
# ==========================
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_args_json(save_dir: str, args_obj):
    ensure_dir(save_dir)
    path_to_json = os.path.join(save_dir, "args.json")
    with open(path_to_json, "w") as f:
        json.dump(vars(args_obj), f, indent=4)


def adjust_lr_with_warmup_and_decay(optimizer, base_lr, iter_num, max_iterations, warmup_period, use_warmup):
    """
    Matches original behavior:
      - Warmup: linear from (0 -> base_lr) over warmup_period iters
      - After warmup: decay as base_lr * (1 - shift_iter / max_iterations) ** 0.9
    """
    if use_warmup and iter_num < warmup_period:
        lr_ = base_lr * ((iter_num + 1) / warmup_period)
    else:
        if use_warmup:
            shift_iter = iter_num - warmup_period
            assert shift_iter >= 0, f"Shift iter is {shift_iter}, smaller than zero"
            lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9
        else:
            lr_ = base_lr
    for pg in optimizer.param_groups:
        pg["lr"] = lr_
    return lr_


def resize_masks(msks: torch.Tensor) -> torch.Tensor:
    global RESIZE_TO_MASK
    if RESIZE_TO_MASK is None:
        RESIZE_TO_MASK = torchvision.transforms.Resize((args.out_size, args.out_size))
    return RESIZE_TO_MASK(msks)


# ==========================
# Model Build / Freeze
# ==========================
def build_model(args):
    """Replicates your original selection and freezing logic exactly."""
    # Model variants
    if args.model == "random":
        sam = sam_model_registry[args.arch](
            args, checkpoint=None, num_classes=args.num_cls, image_size=args.image_size
        )
    elif args.model == "sam":
        sam = sam_model_registry[args.arch](
            args, checkpoint=args.sam_ckpt, num_classes=args.num_cls, image_size=args.image_size
        )
    elif args.model == "medsam":
        sam = sam_model_registry[args.arch](
            args,
            checkpoint="/data/humanBodyProject/mri_foundation_model/pretrained_weights/medsam_vit_b.pth",
            num_classes=args.num_cls,
            image_size=args.image_size,
        )
    elif args.model == "ours":
        sam = sam_model_registry[args.arch](
            args,
            checkpoint="/data/humanBodyProject/mri_foundation_model/dinov2/dino_vitb+sam_0429_nolayerscale_smallerlr/eval/training_47535/teacher_checkpoint.pth",
            num_classes=args.num_cls,
            image_size=args.image_size,
            #pretrained_sam=False,
            pretrained_sam=True,
        )
    else:
        raise ValueError(f"Unknown args.model: {args.model}")

    # Finetune modes & freezing (same prints/flags)
    if args.finetune_type == "adapter":
        for n, p in sam.named_parameters():
            if "Adapter" not in n:
                p.requires_grad = False
        print("if update encoder:", args.if_update_encoder)
        print("if image encoder adapter:", args.if_encoder_adapter)
        print("if mask decoder adapter:", args.if_mask_decoder_adapter)
        if args.if_encoder_adapter:
            print("added adapter layers:", args.encoder_adapter_depths)

    elif args.finetune_type == "vanilla" and args.if_update_encoder is False:
        print("if update encoder:", args.if_update_encoder)
        for _, p in sam.image_encoder.named_parameters():
            p.requires_grad = False

    elif args.finetune_type == "lora":
        print("if update encoder:", args.if_update_encoder)
        print("if image encoder lora:", getattr(args, "if_encoder_lora_layer", None))
        print("if mask decoder lora:", getattr(args, "if_decoder_lora_layer", None))
        if LoRA_Sam is None:
            raise ImportError(
                "LoRA_Sam is required for finetune_type='lora' but could not be imported."
            )
        sam = LoRA_Sam(args, sam, r=4).sam

    sam.to(DEVICE)

    # Optimizer
    base_lr_for_opt = (args.lr / args.warmup_period) if args.if_warmup else args.lr
    optimizer = optim.AdamW(
        sam.parameters(),
        lr=base_lr_for_opt,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.1,
        amsgrad=False,
    )

    # Criterion
    criterion = monai.losses.DiceCELoss(sigmoid=True)

    return sam, optimizer, criterion


# ==========================
# Forward / Predict
# ==========================
def forward_pass(sam, imgs, msks, criterion, update_encoder: bool, do_backward: bool, optimizer=None):
    """
    One pass that matches your original steps exactly:
      - image_encoder (with/without grad)
      - prompt_encoder default embeddings
      - mask_decoder with multimask_output=True
      - DiceCE loss (when msks provided)
    """
    imgs = imgs.to(DEVICE, non_blocking=True)
    msks = resize_masks(msks).to(DEVICE, non_blocking=True)

    if update_encoder:
        img_emb = sam.image_encoder(imgs)
    else:
        with torch.no_grad():
            img_emb = sam.image_encoder(imgs)

    sparse_emb, dense_emb = sam.prompt_encoder(points=None, boxes=None, masks=None)

    pred, _ = sam.mask_decoder(
        image_embeddings=img_emb,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=True,
    )

    loss = None
    if criterion is not None and msks is not None:
        loss = criterion(pred, msks)

    if do_backward and loss is not None:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return pred, loss


def binarize_pred(pred: torch.Tensor) -> torch.Tensor:
    # Matches (pred > 0)
    return (pred > 0).to(torch.long).cpu()


# ==========================
# Eval Helpers
# ==========================
@torch.no_grad()
def evaluate_epoch(sam, valloader, criterion):
    sam.eval()

    eval_loss = 0.0
    name2pred = {}
    name2mask = {}

    for i, data in enumerate(valloader):
        imgs = data["image"].to(DEVICE)
        msks = resize_masks(data["mask"]).to(DEVICE)

        img_emb = sam.image_encoder(imgs)
        sparse_emb, dense_emb = sam.prompt_encoder(points=None, boxes=None, masks=None)
        pred, _ = sam.mask_decoder(
            image_embeddings=img_emb,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=True,
        )

        loss = criterion(pred, msks)
        eval_loss += loss.item()

        prob = binarize_pred(pred)
        mask = msks.to(torch.long).cpu()

        for idx, name in enumerate(data["patient_name"]):
            if name not in name2pred:
                name2pred[name] = [prob[idx].unsqueeze(0)]
                name2mask[name] = [mask[idx].unsqueeze(0)]
            else:
                name2pred[name].append(prob[idx].unsqueeze(0))
                name2mask[name].append(mask[idx].unsqueeze(0))

    eval_loss /= (i + 1)

    # 3D Dice across cases
    val_dsc_3d_list = []
    for name in name2pred:
        pred_3d = torch.cat(name2pred[name], 0).squeeze()
        mask_3d = torch.cat(name2mask[name], 0).squeeze()
        tmp_dsc = compute_dice(pred_3d, mask_3d)
        val_dsc_3d_list.append(tmp_dsc)

    val_dsc_3d = float(np.mean(np.array(val_dsc_3d_list))) if val_dsc_3d_list else 0.0
    return eval_loss, val_dsc_3d


@torch.no_grad()
def test_full(sam, testloader):
    sam.eval()

    name2pred = {}
    name2mask = {}
    test_dsc_2d_list = []

    for _, data in enumerate(testloader):
        imgs = data["image"].to(DEVICE)
        msks = resize_masks(data["mask"]).to(DEVICE)

        img_emb = sam.image_encoder(imgs)
        sparse_emb, dense_emb = sam.prompt_encoder(points=None, boxes=None, masks=None)
        pred, _ = sam.mask_decoder(
            image_embeddings=img_emb,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=True,
        )

        prob = binarize_pred(pred)
        mask = msks.to(torch.long).cpu()

        # 2D DSC per batch
        tmp_dsc = compute_dice(prob, mask)
        test_dsc_2d_list.append(tmp_dsc)

        # Accumulate 3D stacks
        for idx, name in enumerate(data["patient_name"]):
            if name not in name2pred:
                name2pred[name] = [prob[idx].unsqueeze(0)]
                name2mask[name] = [mask[idx].unsqueeze(0)]
            else:
                name2pred[name].append(prob[idx].unsqueeze(0))
                name2mask[name].append(mask[idx].unsqueeze(0))

    # 3D metrics per case
    test_dsc_3d_list = []
    test_nsd_3d_list = []
    for name in name2pred:
        pred_3d = torch.cat(name2pred[name], 0).squeeze()
        mask_3d = torch.cat(name2mask[name], 0).squeeze()

        tmp_dsc = compute_dice(pred_3d, mask_3d)
        tmp_nsd = compute_nsd(pred_3d, mask_3d)
        test_dsc_3d_list.append(tmp_dsc)
        test_nsd_3d_list.append(tmp_nsd)

        print(name, tmp_dsc, tmp_nsd, pred_3d.shape)

    test_dsc_2d = float(np.mean(test_dsc_2d_list)) if test_dsc_2d_list else 0.0
    test_dsc_3d = float(np.mean(test_dsc_3d_list)) if test_dsc_3d_list else 0.0
    test_nsd_3d = float(np.mean(test_nsd_3d_list)) if test_nsd_3d_list else 0.0

    print(test_dsc_2d, test_dsc_3d, test_nsd_3d)
    return test_dsc_2d, test_dsc_3d, test_nsd_3d


# ==========================
# Train Loop
# ==========================
def train_model(trainloader, valloader, testloader, dir_checkpoint, epochs):
    # Build model/opt/loss
    sam, optimizer, criterion = build_model(args)

    iter_num = 0
    max_iterations = epochs * len(trainloader)

    best_val_dsc3d = 0.0
    last_update_epoch = 0

    ensure_dir("temp")
    best_ckpt_path = os.path.join("temp", f"{args.dataset_name}_{args.model}_best.pth")

    for epoch in range(epochs):
        sam.train()
        for _, data in enumerate(trainloader):
            imgs = data["image"]
            msks = data["mask"]

            # Train step (matches original behavior)
            _, loss = forward_pass(
                sam=sam,
                imgs=imgs,
                msks=msks,
                criterion=criterion,
                update_encoder=bool(args.if_update_encoder),
                do_backward=True,
                optimizer=optimizer,
            )

            # LR schedule update (same logic)
            _ = adjust_lr_with_warmup_and_decay(
                optimizer=optimizer,
                base_lr=args.lr,
                iter_num=iter_num,
                max_iterations=max_iterations,
                warmup_period=args.warmup_period,
                use_warmup=bool(args.if_warmup),
            )
            iter_num += 1

        # Eval cadence — original code used 10 epochs. We honor args.val_freq (default 10).
        if epoch % int(args.val_freq) == 0:
            eval_loss, val_dsc_3d = evaluate_epoch(sam, valloader, criterion)

            # Save best
            if val_dsc_3d > best_val_dsc3d:
                best_val_dsc3d = val_dsc_3d
                last_update_epoch = epoch
                torch.save(sam.state_dict(), best_ckpt_path)
            elif (epoch - last_update_epoch) >= 200 and epoch >= 1000:
                print("Training finished###########")
                break

            print(
                "Eval Epoch num %s | val loss %.4f | dsc %.4f | best %.4f"
                % (epoch, eval_loss, val_dsc_3d, best_val_dsc3d)
            )

    # Load best and test
    print("Load from", best_ckpt_path)
    sam.load_state_dict(torch.load(best_ckpt_path, map_location=DEVICE))
    sam.eval()

    return test_full(sam, testloader)


# ==========================
# Main
# ==========================
if __name__ == "__main__":
    dataset_name = args.dataset_name
    train_img_list = args.train_img_list
    val_img_list = args.val_img_list
    test_img_list = args.test_img_list

    # Prepare output dir + args.json
    ensure_dir(args.dir_checkpoint)
    #save_args_json(args.dir_checkpoint, args)

    # Mirror original knobs
    n_type = args.model
    args.n_type = n_type
    args.if_spatial = True
    args.b = 2

    # Datasets / loaders (same wiring & flags)
    delete_empty_masks = True

    test_dataset = Public_dataset(
        args,
        args.img_folder,
        args.mask_folder,
        test_img_list,
        phase="test",
        targets=[args.targets],
        normalize_type=n_type,
        if_prompt=False,
        crop_size=args.image_size,
        delete_empty_masks=delete_empty_masks,
        target_cls=args.cls,
    )
    testloader = DataLoader(test_dataset, batch_size=args.b, shuffle=False, num_workers=8)

    final_score_list = []
    num_workers = 0

    for repeat in range(10):
        print("Running", n_type, args, repeat)

        train_dataset = Public_dataset(
            args,
            args.img_folder,
            args.mask_folder,
            train_img_list,
            phase="train",
            targets=[args.targets],
            normalize_type=n_type,
            if_prompt=False,
            crop_size=args.image_size,
            few_shot=True,
            seed=repeat,
            delete_empty_masks=delete_empty_masks,
            target_cls=args.cls,
            if_spatial=args.if_spatial,
        )

        val_dataset = Public_dataset(
            args,
            args.img_folder,
            args.mask_folder,
            val_img_list,
            phase="val",
            targets=[args.targets],
            normalize_type=n_type,
            if_prompt=False,
            crop_size=args.image_size,
            few_shot=True,
            seed=repeat,
            delete_empty_masks=delete_empty_masks,
            target_cls=args.cls,
            if_spatial=args.if_spatial,
        )

        trainloader = DataLoader(train_dataset, batch_size=args.b, shuffle=True, num_workers=num_workers)
        valloader = DataLoader(val_dataset, batch_size=args.b, shuffle=False, num_workers=num_workers)

        final_score = train_model(trainloader, valloader, testloader, args.dir_checkpoint, args.epochs)
        final_score_list.append(final_score)

        print(final_score_list)
        print("****")

    print(final_score_list)

