"""
Abstract Vision Backbone interface for CLARITY.

Implement VisionBackbone to plug any 3D MRI encoder into the world model.
See brainiac_adapter.py for the reference BrainIAC implementation.
"""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class VisionBackbone(ABC, nn.Module):
    """
    Single-modality 3D MRI encoder interface.

    Contract:
      - Input:  [B, 1, D, H, W]  — single-channel 3D volume, arbitrary resolution
      - Output: [B, tokens_per_modality, hidden_dim]  — fixed-length token sequence

    Example::

        class MyBackbone(VisionBackbone):
            def __init__(self):
                super().__init__()
                self._tokens = 8
                self._dim = 768
                self.vit = SomeViT3D(...)

            @property
            def tokens_per_modality(self): return self._tokens

            @property
            def hidden_dim(self): return self._dim

            def forward(self, x):  # x: [B, 1, D, H, W]
                return self.vit(x)[:, :self._tokens, :]
    """

    @property
    @abstractmethod
    def tokens_per_modality(self) -> int:
        """Number of patch tokens emitted per input volume."""

    @property
    @abstractmethod
    def hidden_dim(self) -> int:
        """Embedding dimension of each token."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, D, H, W]  →  [B, tokens_per_modality, hidden_dim]"""


class MultiModalVisionBackbone(nn.Module):
    """
    Applies a VisionBackbone independently to each MRI modality, then concatenates.

    Input:  mri  [B, num_modalities, H, W, D]   (NIfTI axis order: H×W×D)
    Output:      [B, num_modalities × tokens_per_modality, hidden_dim]
    """

    def __init__(self, backbone: VisionBackbone, num_modalities: int = 4):
        super().__init__()
        self.backbone = backbone
        self.num_modalities = num_modalities

    @property
    def tokens_per_modality(self) -> int:
        return self.backbone.tokens_per_modality

    @property
    def hidden_dim(self) -> int:
        return self.backbone.hidden_dim

    def forward(self, mri: torch.Tensor) -> torch.Tensor:
        # NIfTI [B, M, H, W, D] → ViT-compatible [B, M, D, H, W]
        mri = mri.permute(0, 1, 4, 2, 3)
        tokens = [self.backbone(mri[:, i : i + 1]) for i in range(self.num_modalities)]
        return torch.cat(tokens, dim=1)   # [B, M * tokens_per_modality, hidden_dim]
