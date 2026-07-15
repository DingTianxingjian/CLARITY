"""
BrainIAC Vision Backbone Adapter
=================================
Wraps MONAI ViT-B (96³ input) with BrainIAC pre-trained weights and LoRA fine-tuning,
implementing the VisionBackbone interface for plug-in use with the CLARITY world model.

Requirements
------------
- pip install monai peft
- BrainIAC.ckpt checkpoint  (not redistributed; obtain from the BrainIAC project)
  Place at: BrainIAC-main/src/checkpoints/BrainIAC.ckpt

Checkpoint format
-----------------
BrainIAC stores ViT weights under a ``backbone.`` prefix in the .ckpt file.
This adapter strips that prefix and loads directly into MonaiViT.

Usage
-----
    from models.brainiac_adapter import BrainIACAdapter
    from models.vision_backbone import MultiModalVisionBackbone

    backbone = BrainIACAdapter(
        checkpoint_path="BrainIAC-main/src/checkpoints/BrainIAC.ckpt",
        tokens_per_modality=8,   # → total 32 tokens for 4 modalities
        lora_r=8,
    )
    mri_encoder = MultiModalVisionBackbone(backbone, num_modalities=4)
    # mri_encoder(mri)  # mri: [B, 4, H, W, D] → [B, 32, 768]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vision_backbone import VisionBackbone

try:
    from monai.networks.nets import ViT as MonaiViT
except ImportError:
    MonaiViT = None

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = get_peft_model = None


class BrainIACAdapter(VisionBackbone):
    """
    MONAI ViT-B with BrainIAC weights and LoRA fine-tuning.

    Architecture:
      - 12 transformer layers, hidden_size=768, 12 heads
      - Input volume resized to 96³; patches 16³ → 216 patch tokens
      - Evenly samples `tokens_per_modality` tokens from the 216

    LoRA:
      - Backbone frozen; LoRA adapters on qkv and out_proj (via PEFT)
      - lora_r=0 → fully frozen backbone (feature extraction only)
    """

    def __init__(
        self,
        checkpoint_path: str,
        tokens_per_modality: int = 8,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        if MonaiViT is None:
            raise ImportError("MONAI is required: pip install monai")

        self._tokens_per_modality = tokens_per_modality
        self._hidden_dim = 768

        try:
            self.backbone = MonaiViT(
                in_channels=1,
                img_size=(96, 96, 96),
                patch_size=(16, 16, 16),
                hidden_size=768,
                mlp_dim=3072,
                num_layers=12,
                num_heads=12,
                pos_embed="conv",
                classification=False,
                dropout_rate=0.0,
            )
        except TypeError:
            self.backbone = MonaiViT(
                in_channels=1,
                img_size=(96, 96, 96),
                patch_size=(16, 16, 16),
                hidden_size=768,
                mlp_dim=3072,
                num_layers=12,
                num_heads=12,
            )

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        # BrainIAC stores encoder under "backbone." prefix
        backbone_sd = {k[9:]: v for k, v in state_dict.items() if k.startswith("backbone.")}
        if not backbone_sd:
            backbone_sd = state_dict
        self.backbone.load_state_dict(backbone_sd, strict=False)

        for p in self.backbone.parameters():
            p.requires_grad = False

        if lora_r > 0:
            if LoraConfig is None:
                raise ImportError("PEFT is required for LoRA: pip install peft")
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["qkv", "out_proj"],
                bias="none",
            )
            try:
                self.backbone = get_peft_model(self.backbone, lora_cfg)
            except Exception:
                pass  # no matching target modules — backbone stays frozen

    @property
    def tokens_per_modality(self) -> int:
        return self._tokens_per_modality

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, D, H, W]  →  [B, tokens_per_modality, 768]"""
        if x.shape[2:] != (96, 96, 96):
            x = F.interpolate(
                x.float(), size=(96, 96, 96), mode="trilinear", align_corners=False
            )
        feats = self.backbone(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        # feats: [B, 216, 768] for 96³/16³ patches
        num_patches = feats.shape[1]
        stride = max(1, num_patches // self._tokens_per_modality)
        indices = torch.arange(
            0, num_patches, stride, device=feats.device
        )[: self._tokens_per_modality]
        return feats[:, indices, :]
