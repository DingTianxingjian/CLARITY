"""
优化版胶质瘤生存预测模型：共享文本编码器 + 时间编码
关键改进：正确处理时间差信息
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import math
from models.survival_module import SurvivalModule
from utils.metrics import one_year_survival_targets_torch



def extract_drug_category(drugs_text: str) -> str:
    """
    Parse the drug JSON string produced by GliomaAllPairsTextDataset into a
    canonical category string: e.g. "TMZ+RT", "TMZ", "no_treatment", "BEV".

    Combines agents from both pre and post action blocks so that any treatment
    active within the pair interval is captured.
    """
    import json
    try:
        d = json.loads(drugs_text)
    except Exception:
        return "unknown"

    agents: set = set()
    has_rt = False
    for side in ("pre", "post"):
        actions = d.get(side, {}).get("actions", {})
        if not isinstance(actions, dict):
            continue
        for cat, items in actions.items():
            if cat == "radiation":
                has_rt = True
            elif cat in ("chemotherapy", "additional_1", "additional_2"):
                for item in (items if isinstance(items, list) else []):
                    agent = item.get("agent", "").lower()
                    if "temozolomide" in agent:
                        agents.add("TMZ")
                    elif "bevacizumab" in agent or "avastin" in agent:
                        agents.add("BEV")
                    elif agent:
                        agents.add("OTHER")
    if has_rt:
        agents.add("RT")
    return "+".join(sorted(agents)) if agents else "no_treatment"

# Add mri_foundation to path for SAM encoder
_MRI_FOUNDATION_ROOT = Path(__file__).resolve().parents[2] / "mri_foundation"
if str(_MRI_FOUNDATION_ROOT) not in sys.path:
    sys.path.insert(0, str(_MRI_FOUNDATION_ROOT))


class LoRALinear(nn.Module):
    """Lightweight LoRA wrapper for arbitrary nn.Linear layers."""

    def __init__(self, base_layer: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base_layer)!r}")
        if r <= 0:
            raise ValueError("LoRA rank must be positive.")

        self.base_layer = base_layer
        self.r = r
        self.scaling = alpha / r
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(r, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    def forward(self, x):
        # base weights are frozen — run under no_grad to skip storing activations
        with torch.no_grad():
            base_out = self.base_layer(x)
        lora_hidden = F.linear(self.lora_dropout(x), self.lora_A)
        lora_out = F.linear(lora_hidden, self.lora_B)
        return base_out + lora_out * self.scaling


def _get_parent_module(root_module: nn.Module, module_name: str):
    parent = root_module
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora_modules(
    module: nn.Module,
    target_suffixes,
    r: int,
    alpha: int,
    dropout: float,
):
    replaced_modules = []
    for name, child in list(module.named_modules()):
        if not isinstance(child, nn.Linear):
            continue
        if not any(name.endswith(suffix) for suffix in target_suffixes):
            continue
        parent, attribute_name = _get_parent_module(module, name)
        setattr(parent, attribute_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
        replaced_modules.append(name)
    return replaced_modules




class SliceAttentionPooling(nn.Module):
    """
    Learnable attention pooling over spatial patch tokens.

    Input:  [B, H*W, embed_dim]  (ViT patch tokens from one slice, before neck)
    Output: [B, out_dim]

    Uses a single learnable query vector to attend over spatial tokens,
    learning WHICH patches matter (e.g. tumour region) rather than averaging all.
    Followed by a projection to out_dim.
    """
    def __init__(self, embed_dim: int = 768, out_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, out_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, N, embed_dim]  →  [B, out_dim]"""
        B = tokens.shape[0]
        q = self.query.expand(B, -1, -1)               # [B, 1, D]
        pooled, _ = self.attn(q, tokens, tokens)        # [B, 1, D]
        pooled = self.norm(pooled.squeeze(1))           # [B, D]
        return self.proj(pooled)                        # [B, out_dim]


class SAMMRIEncoder(nn.Module):
    """
    SAM ViT-B encoder (plan B): bypass the neck, use raw ViT patch tokens.

    Key differences from v1 (avg-pool on neck output):
      - Neck is bypassed entirely: ViT block outputs [B, 64, 64, 768] patch tokens
      - SliceAttentionPooling replaces global avg-pool: learns to attend tumour patches
      - LoRA injected into ViT qkv + proj: end-to-end fine-tuning
      - out_dim=256 preserved for compatibility with downstream LatentPredictor

    Input (online mode):  raw NIfTI volumes loaded per batch [B, 4, H, W, D]
    Output:               [B, 4*D_sel, 256]  (D_sel selected axial slices)

    D_sel < 155 because 155 slices × 4 modalities × full ViT forward is too slow.
    We sample the central `num_slices` slices per modality (tumour is usually central).
    """

    IMG_SIZE  = 256   # MRI native ~251×251; 256→16×16=256 patches, minimal upscaling
    EMBED_DIM = 768   # SAM ViT-B hidden dim
    OUT_DIM   = 256   # output per slice token

    def __init__(
        self,
        sam_ckpt: str,
        lora_r: int = 4,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        num_slices: int = 32,   # central slices sampled per modality (VRAM budget)
        out_dim: int = 256,
    ):
        super().__init__()
        self.num_slices = num_slices
        self.out_dim = out_dim

        # ── load SAM ViT-B ──────────────────────────────────────────────────
        import argparse
        ns = argparse.Namespace(
            if_encoder_adapter=False, if_mask_decoder_adapter=False,
            if_encoder_lora_layer=False, if_decoder_lora_layer=False,
            encoder_adapter_depths=[], encoder_lora_layer=[], decoder_adapt_depth=0,
        )
        from models.sam import sam_model_registry
        sam = sam_model_registry["vit_b"](
            ns, checkpoint=sam_ckpt, num_classes=1,
            image_size=1024, pretrained_sam=False,  # load original weights at 1024
        )
        self.vit = sam.image_encoder   # full ImageEncoderViT
        self.register_buffer("pixel_mean", sam.pixel_mean)
        self.register_buffer("pixel_std",  sam.pixel_std)

        # ── freeze everything first ─────────────────────────────────────────
        for p in self.vit.parameters():
            p.requires_grad = False

        # ── LoRA on ViT qkv + proj (NOT on neck) ───────────────────────────
        if lora_r > 0:
            # Only target blocks, not neck
            replaced = inject_lora_modules(
                self.vit.blocks,
                target_suffixes=("qkv", "proj"),
                r=lora_r, alpha=lora_alpha, dropout=lora_dropout,
            )
            print(f"[SAMMRIEncoder-B] LoRA on {len(replaced)} ViT layers (r={lora_r}, skip neck)")
        else:
            print("[SAMMRIEncoder-B] Frozen ViT (no LoRA)")

        # ── learnable attention pooling (fully trainable) ───────────────────
        self.slice_pool = SliceAttentionPooling(
            embed_dim=self.EMBED_DIM, out_dim=out_dim, num_heads=8
        )

    # ── preprocessing ────────────────────────────────────────────────────────

    def _preprocess_slice(self, slc: torch.Tensor) -> torch.Tensor:
        """slc: [B, H, W] float  →  [B, 3, 1024, 1024] SAM-normalised (no grad)."""
        with torch.no_grad():
            t = slc.unsqueeze(1).expand(-1, 3, -1, -1).float()
            t = F.interpolate(t, size=(self.IMG_SIZE, self.IMG_SIZE),
                              mode="bilinear", align_corners=False)
        pm = self.pixel_mean.view(1, 3, 1, 1)
        ps = self.pixel_std.view(1, 3, 1, 1)
        return (t - pm) / ps

    def _norm_volume(self, vol: torch.Tensor) -> torch.Tensor:
        """vol: [B, H, W, D] → percentile-normalised [0,255]."""
        B = vol.shape[0]
        flat = vol.reshape(B, -1).float()
        # 1st/99th percentile per volume
        lo = flat.kthvalue(max(1, int(0.01 * flat.shape[1])), dim=1).values.view(B,1,1,1)
        hi = flat.kthvalue(min(flat.shape[1], int(0.99 * flat.shape[1])), dim=1).values.view(B,1,1,1)
        return ((vol.float() - lo) / (hi - lo + 1e-6)).clamp(0, 1) * 255.0

    # ── ViT forward bypassing neck ────────────────────────────────────────────

    def _vit_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run SAM ViT blocks, return patch tokens BEFORE neck.
        x: [B, 3, IMG_SIZE, IMG_SIZE]  →  [B, N_patches, 768]
        IMG_SIZE=256 → 16×16=256 patches (native MRI resolution, no upscaling needed)
        """
        x = self.vit.patch_embed(x)           # [B, H', W', 768]
        if self.vit.pos_embed is not None:
            # pos_embed was trained for 64×64; interpolate to actual patch grid size
            pe = self.vit.pos_embed           # [1, 64, 64, 768]
            h, w = x.shape[1], x.shape[2]
            if pe.shape[1] != h or pe.shape[2] != w:
                pe = pe.permute(0, 3, 1, 2)  # [1, 768, 64, 64]
                pe = F.interpolate(pe, size=(h, w), mode="bilinear", align_corners=False)
                pe = pe.permute(0, 2, 3, 1)  # [1, h, w, 768]
            x = x + pe
        from torch.utils.checkpoint import checkpoint
        for blk in self.vit.blocks:
            x = checkpoint(blk, x, use_reentrant=False)
        return x.reshape(x.shape[0], -1, self.EMBED_DIM)  # [B, N_patches, 768]

    # ── encode one modality volume ────────────────────────────────────────────

    def _encode_volume(self, vol: torch.Tensor) -> torch.Tensor:
        """
        vol: [B, H, W, D]
        returns: [B, D_sel, out_dim]   D_sel = num_slices
        """
        D = vol.shape[3]
        vol = self._norm_volume(vol)           # [B, H, W, D]

        # Select central slices (tumour almost always within central 1/3 of brain)
        mid = D // 2
        half = self.num_slices // 2
        z_start = max(0, mid - half)
        z_end   = min(D, z_start + self.num_slices)
        z_start = max(0, z_end - self.num_slices)  # re-clamp if near edge
        z_indices = range(z_start, z_end)

        # Process one slice at a time across the full batch.
        # Peak VRAM = 1 ViT forward on [B, 3, 256, 256] with grad-checkpoint ≈ 3-4 GB.
        slice_feats = []
        for z in z_indices:
            t  = self._preprocess_slice(vol[:, :, :, z])  # [B, 3, 256, 256]
            tk = self._vit_tokens(t)                       # [B, 256, 768]
            f  = self.slice_pool(tk)                       # [B, out_dim]
            slice_feats.append(f)
        return torch.stack(slice_feats, dim=1)  # [B, D_sel, out_dim]

    # ── public API ────────────────────────────────────────────────────────────

    def forward(self, mri: torch.Tensor) -> torch.Tensor:
        """
        mri: [B, 4, H, W, D]
        returns: [B, 4*num_slices, out_dim]
        """
        assert mri.ndim == 5, f"Expected [B,4,H,W,D], got {tuple(mri.shape)}"
        mod_feats = []
        for m in range(mri.shape[1]):
            mod_feats.append(self._encode_volume(mri[:, m]))  # [B, D_sel, 256]
        return torch.cat(mod_feats, dim=1)   # [B, 4*D_sel, 256]


class PositionalTimeEncoding(nn.Module):
    """
    时间位置编码（类似Transformer的位置编码）
    将时间差编码为高维向量
    """
    def __init__(self, d_model=128, max_time=3650):
        """
        Args:
            d_model: 编码维度
            max_time: 最大时间（天），默认10年
        """
        super().__init__()
        self.d_model = d_model
        
        # 预计算位置编码
        position = torch.arange(0, max_time).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_time, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, time_delta):
        """
        Args:
            time_delta: [B] - 时间差（天）
        Returns:
            time_encoding: [B, d_model]
        """
        # 将时间差转换为整数索引（限制在max_time内）
        time_indices = torch.clamp(time_delta.long(), 0, self.pe.size(0) - 1)
        return self.pe[time_indices]


class LearnableTimeEmbedding(nn.Module):
    """
    可学习的时间嵌入（更灵活）
    将标量时间映射到高维空间
    """
    def __init__(self, d_model=128, max_time=3650):
        super().__init__()
        self.d_model = d_model
        self.max_time = max_time
        
        # 归一化层
        self.time_normalizer = nn.Parameter(torch.tensor(max_time / 2.0))
        
        # MLP编码器
        self.encoder = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(self, time_delta):
        """
        Args:
            time_delta: [B] - 时间差（天）
        Returns:
            time_encoding: [B, d_model]
        """
        # 归一化到[-1, 1]
        normalized_time = time_delta.unsqueeze(1) / self.time_normalizer
        normalized_time = torch.tanh(normalized_time)  # 软裁剪
        
        # MLP编码
        time_encoding = self.encoder(normalized_time)
        return time_encoding


class FourierTimeEmbedding(nn.Module):
    """
    Fourier特征编码（NERF风格）
    对时间进行多尺度周期性编码
    """
    def __init__(self, d_model=128, num_frequencies=32):
        super().__init__()
        self.d_model = d_model
        self.num_frequencies = num_frequencies
        
        # 频率参数（可学习或固定）
        frequencies = torch.logspace(0, 4, num_frequencies)  # 1天到10000天
        self.register_buffer('frequencies', frequencies)
        
        # 投影层
        self.projection = nn.Linear(num_frequencies * 2, d_model)
    
    def forward(self, time_delta):
        """
        Args:
            time_delta: [B] - 时间差（天）
        Returns:
            time_encoding: [B, d_model]
        """
        # 归一化
        time_normalized = time_delta.unsqueeze(1) / 365.0  # 转换为年
        
        # Fourier特征
        angles = 2 * math.pi * time_normalized * self.frequencies
        fourier_features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        
        # 投影
        time_encoding = self.projection(fourier_features)
        return time_encoding


class SharedTextEncoder(nn.Module):
    """适配 MedGemma-2B 的医学文本编码器，支持 4bit QLoRA + LoRA 微调"""
    def __init__(
        self,
        model_name: str = "google/medgemma-4b-it",
        lora_r: int = 4,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: list = ["q_proj", "v_proj"],
        use_mean_pooling: bool = True,
        load_in_4bit: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        self.use_mean_pooling = use_mean_pooling
        self._input_grad_hook = None

        # ✅ BitsAndBytes 量化配置（transformers 5.x: 必须用 BitsAndBytesConfig 对象）
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ) if load_in_4bit else None

        load_kwargs = dict(
            dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config

        # ✅ 1. 加载模型
        try:
            self.model = AutoModel.from_pretrained(model_name, **load_kwargs)
        except Exception:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.output_dim = getattr(self.model.config, "hidden_size", getattr(self.model.config, "dim", 2560))

        # ✅ 2. 为 QLoRA/LoRA 训练准备模型，并确保 gradient checkpointing 能反传到 adapter
        if load_in_4bit:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=False,
            )

        # ✅ 3. LoRA 注入（Feature Extraction 模式）
        self.lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        self.model = get_peft_model(self.model, self.lora_cfg)
        self._enable_gradient_checkpointing()
        self.model.print_trainable_parameters()

        # ✅ 4. 输出压缩映射到固定维度（768）
        self.final_proj = nn.Linear(self.output_dim, 768)

    def _enable_gradient_checkpointing(self):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
            return
        if self._input_grad_hook is not None:
            return
        input_embeddings = (
            self.model.get_input_embeddings()
            if hasattr(self.model, "get_input_embeddings")
            else None
        )
        if input_embeddings is None:
            return

        def _make_output_require_grad(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                output.requires_grad_(True)

        self._input_grad_hook = input_embeddings.register_forward_hook(
            _make_output_require_grad
        )

    def disable_gradient_checkpointing(self):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()

    def forward(self, texts):
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(device)

        outputs = self.model(**inputs)
        hidden_states = outputs.last_hidden_state  # [B, L, D]

        # 平均池化代替 CLS
        if self.use_mean_pooling:
            mask = inputs['attention_mask'].unsqueeze(-1)
            emb = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            emb = hidden_states[:, 0, :]



        emb = self.final_proj(emb.to(self.final_proj.weight.device))
        emb = F.normalize(emb, p=2, dim=-1)
        return emb


class TaskSpecificProjection(nn.Module):
    """任务特定投影层（与之前相同）"""
    def __init__(self, input_dim=768, output_dim=768, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, x):
        return self.projection(x)


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for Transformer sequences."""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LatentPredictor(nn.Module):
    """
    Drug- and time-conditioned latent predictor.
    Drug text embedding and time encoding are each prepended as condition tokens;
    modality tokens follow.  A Transformer encoder models cross-token interactions,
    and the output is added as a residual to pre_latent.
    """
    def __init__(
        self,
        latent_dim: int = 768,
        drug_dim: int = 768,
        time_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.drug_proj = nn.Linear(drug_dim, hidden_dim)
        # Fourier time encoding: time_delta → [B, time_dim]
        self.time_dim = time_dim
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        self.pos_encoder = SinusoidalPositionalEncoding(hidden_dim, max_len=5000, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        # Direct drug residual path: bypasses transformer attention so CF loss gradient
        # reaches drug_emb immediately (transformer attention too weak after L1-only pretraining).
        # std=0.01 → initial residual norm ~0.28, <1% of transformer_out norm → minimal L1 disruption.
        self.drug_residual_proj = nn.Linear(drug_dim, latent_dim, bias=False)
        nn.init.normal_(self.drug_residual_proj.weight, std=0.01)

    def _encode_time(self, time_delta: torch.Tensor) -> torch.Tensor:
        """Fourier time encoding: [B] → [B, time_dim]"""
        t = time_delta.float().unsqueeze(1)  # [B, 1]
        d = self.time_dim
        freqs = torch.exp(
            torch.arange(0, d, 2, device=t.device, dtype=t.dtype) *
            (-math.log(10000.0) / d)
        )  # [d/2]
        args = t * freqs.unsqueeze(0)  # [B, d/2]
        enc = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, d]
        return enc

    def forward(
        self,
        pre_latent: torch.Tensor,
        drug_emb: torch.Tensor,
        time_delta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pre_latent:  [B, M, latent_dim]
            drug_emb:    [B, drug_dim]
            time_delta:  [B]  (days between pre and post scan)
        Returns:
            predicted_latent: [B, M, latent_dim]
        """
        latent_tokens = self.latent_proj(pre_latent)              # [B, M, H]
        drug_token = self.drug_proj(drug_emb).unsqueeze(1)        # [B, 1, H]
        time_enc = self._encode_time(time_delta)                  # [B, time_dim]
        time_token = self.time_proj(time_enc).unsqueeze(1)        # [B, 1, H]
        # condition tokens first, then modality tokens
        tokens = torch.cat([drug_token, time_token, latent_tokens], dim=1)  # [B, M+2, H]
        tokens = self.pos_encoder(tokens)
        encoded = self.transformer(tokens)                         # [B, M+2, H]
        pred_latent = self.output_proj(encoded[:, 2:, :])          # [B, M, latent_dim]
        # Direct additive drug residual: CF-loss gradient flows here without
        # traversing 4 transformer layers.  Initialised at 0 → no L1 disruption on resume.
        drug_res = self.drug_residual_proj(drug_emb).unsqueeze(1)  # [B, 1, latent_dim]
        return pred_latent + drug_res                               # broadcast over M tokens




class TimeAwareGliomaSurvivalPredictor(nn.Module):
    """
    时间感知的胶质瘤生存预测模型
    
    关键改进：
    1. 共享文本编码器 + 任务特定投影
    2. 显式建模时间差的影响
    3. 时间相关的残差连接
    """
    def __init__(
        self,
        # 文本编码器参数
        text_encoder_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        freeze_text_encoder: bool = True,
        text_output_dim: int = 768,

        # 时间编码参数
        time_dim: int = 128,
        time_encoding_type: str = 'fourier',  # kept for compatibility, fourier used internally

        # 潜在预测器参数
        latent_dim: int = 767,
        num_modalities: int = 4,
        predictor_hidden_dim: int = 256,
        predictor_num_layers: int = 4,
        predictor_num_heads: int = 4,

        # 生存模块参数
        survival_hidden_dim: int = 128,

        # 损失权重
        lambda_l1: float = 1.0,
        lambda_cox: float = 0.5,
        lambda_bce: float = 0.5,
        lora_r: int = 8,
        lora_alpha: int = 16,
        dropout: float = 0.3,

        # Optional MRI encoder — pass any MultiModalVisionBackbone instance.
        # See Predictor/models/brainiac_adapter.py for the BrainIAC reference adapter.
        vision_backbone: nn.Module = None,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_modalities = num_modalities

        # 损失权重
        self.register_buffer('lambda_l1', torch.tensor(lambda_l1))
        self.register_buffer('lambda_cox', torch.tensor(lambda_cox))
        self.register_buffer('lambda_bce', torch.tensor(lambda_bce))

        # 0. Optional pluggable MRI encoder (any MultiModalVisionBackbone).
        self.mri_encoder = None
        if vision_backbone is not None:
            self.mri_encoder = vision_backbone
            if hasattr(vision_backbone, "tokens_per_modality"):
                self.num_modalities = 4 * vision_backbone.tokens_per_modality
            if hasattr(vision_backbone, "hidden_dim"):
                latent_dim = vision_backbone.hidden_dim
                self.latent_dim = latent_dim

        # 1. 共享文本编码器 (MedGemma)
        self.shared_text_encoder = SharedTextEncoder(
            model_name=text_encoder_name,
            lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            use_mean_pooling=True,
            load_in_4bit=True
        )
        if freeze_text_encoder:
            self.shared_text_encoder.disable_gradient_checkpointing()
            for param in self.shared_text_encoder.parameters():
                param.requires_grad = False

        # 2. Drug+clinical- and time-conditioned latent predictor (Transformer with residual)
        # drug_emb and clinical_emb are concatenated → condition_dim = 2 * text_output_dim
        self.latent_predictor = LatentPredictor(
            latent_dim=latent_dim,
            drug_dim=text_output_dim * 2,  # drug || clinical
            time_dim=time_dim,
            hidden_dim=predictor_hidden_dim,
            num_layers=predictor_num_layers,
            num_heads=predictor_num_heads,
            ffn_dim=predictor_hidden_dim * 4,
            dropout=dropout,
        )

        # 3. Two-way attention survival module
        # condition_dim=text_output_dim*2: clinical+drug emb fed into fusion (not a bypass shortcut)
        self.survival_module = SurvivalModule(
            latent_dim=latent_dim,
            num_modalities=num_modalities,
            hidden_dim=survival_hidden_dim,
            num_twoway_layers=2,
            num_heads=predictor_num_heads,
            dropout=dropout,
            condition_dim=text_output_dim * 2,
        )

    
    def encode_mri_raw(self, pre_mri: torch.Tensor, post_mri: torch.Tensor):
        """
        Encode raw MRI volumes through the SAM encoder (only valid when sam_ckpt is set).
        pre_mri / post_mri: [B, 4, H, W, D]
        Returns: pre_latent [B, 4*D, 256], post_latent [B, 4*D, 256]
        """
        assert self.mri_encoder is not None, "sam_ckpt must be set to use encode_mri_raw"
        return self.mri_encoder(pre_mri), self.mri_encoder(post_mri)

    def forward(self, pre_latent, drugs_text, time_delta, clinical_text=None,
                pre_mri=None):
        """
        Args:
            pre_latent:    [B, M, latent_dim]  — pre-computed features OR None
            drugs_text:    List[str]
            time_delta:    [B]  days between pre and post scan
            clinical_text: List[str] (optional; zeros used when absent)
            pre_mri:       [B, 4, H, W, D] raw MRI (replaces pre_latent when mri_encoder enabled)
        Returns:
            predicted_latent: [B, M, latent_dim]
            risk_score:       [B, 1]
            survival_logit:   [B, 1]
        """
        # 0. Optionally encode raw MRI on-the-fly
        if self.mri_encoder is not None and pre_mri is not None:
            pre_latent = self.mri_encoder(pre_mri)

        # 1. Encode drug text and clinical context through MedGemma
        drug_emb = self.shared_text_encoder(drugs_text)           # [B, D]
        if clinical_text and any(clinical_text):
            clinical_emb = self.shared_text_encoder(clinical_text)  # [B, D]
        else:
            clinical_emb = torch.zeros_like(drug_emb)
        condition_emb = torch.cat([drug_emb, clinical_emb], dim=-1)  # [B, 2D]

        # 2. Predict post-latent conditioned on drug, clinical context, and time
        predicted_latent = self.latent_predictor(pre_latent, condition_emb, time_delta)

        # 3. Survival prediction: MRI two-way attention + clinical condition fusion
        risk_score, survival_logit = self.survival_module(
            pre_latent, predicted_latent, condition_emb
        )

        return predicted_latent, risk_score, survival_logit, condition_emb

    def compute_loss(self, pred_latent, risk_score, survival_logit,
                     post_latent, survival_time, event_indicator):
        # 1. L1重建损失
        l1_loss = F.l1_loss(pred_latent, post_latent)
        
        # 2. Cox损失
        try:
            from losses.auton_survival_loss import NLLDeepSurvLoss
            cox_criterion = NLLDeepSurvLoss()
            cox_loss = cox_criterion(risk_score, survival_time, event_indicator)
        except ImportError:
            cox_loss = self._simple_cox_loss(risk_score, survival_time, event_indicator)
        
        # 3. 1-year survival BCE (exclude censored-before-1-year: uncertain label)
        one_year_label, one_year_mask = one_year_survival_targets_torch(
            survival_time, event_indicator
        )
        one_year_mask = one_year_mask.bool()
        if one_year_mask.any():
            bce_loss = F.binary_cross_entropy_with_logits(
                survival_logit.squeeze(-1)[one_year_mask],
                one_year_label[one_year_mask],
            )
        else:
            bce_loss = risk_score.new_tensor(0.0)
        
        # 4. 加权组合
        total_loss = (
            self.lambda_l1 * l1_loss +
            self.lambda_cox * cox_loss +
            self.lambda_bce * bce_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'l1': l1_loss.item(),
            'cox': cox_loss.item(),
            'bce': bce_loss.item()
        }
        
        return total_loss, loss_dict
    
    @staticmethod
    def drug_category_contrastive_loss(
        drug_categories: list,
        pred_latent: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """
        Supervised contrastive loss (SupCon) based on discrete drug categories.

        Same drug category  → positive pair (pull together in latent space)
        Different category  → negative pair (push apart)

        drug_categories: List[str] length B  — output of extract_drug_category()
        pred_latent:     [B, M, D]           — predicted post-treatment latents

        Returns 0 if no positive pair exists in this batch (all unique categories).
        """
        B = pred_latent.shape[0]
        if B < 2:
            return pred_latent.new_tensor(0.0)

        # Build positive mask: same category, exclude self  [B, B]
        same = torch.tensor(
            [[i != j and drug_categories[i] == drug_categories[j]
              for j in range(B)] for i in range(B)],
            dtype=torch.bool, device=pred_latent.device,
        )
        if not same.any():
            return pred_latent.new_tensor(0.0)

        # Student: mean-pool tokens → unit sphere  [B, D]
        z = F.normalize(pred_latent.float().mean(dim=1), p=2, dim=-1)

        # Pairwise cosine similarity; mask diagonal to -inf
        S = torch.mm(z, z.T) / temperature                          # [B, B]
        S = S.masked_fill(torch.eye(B, dtype=torch.bool, device=S.device), float('-inf'))

        # SupCon: for each anchor, average log-prob of its positives
        log_Q = F.log_softmax(S, dim=1)                     # [B, B]
        # Use masked_fill to avoid 0.0 * (-inf) = NaN on diagonal
        log_Q_pos = log_Q.masked_fill(~same, 0.0)           # zero out non-positives
        n_pos = same.float().sum(dim=1).clamp(min=1)        # [B]
        loss_per_sample = -log_Q_pos.sum(dim=1) / n_pos     # [B]

        has_pos = same.any(dim=1)
        return loss_per_sample[has_pos].mean()

    @staticmethod
    def variance_loss(pred_latent: torch.Tensor, gamma: float = 0.05) -> torch.Tensor:
        """VICReg-style variance regularization: penalize dims whose cross-sample
        std < gamma.  Prevents pred_latent from collapsing to a single vector.

        gamma calibrated to feature scale: pre_latent batch_std ≈ 0.12,
        so gamma=0.05 gives loss=0 for diverse latents, loss>0 only when collapsed.

        pred_latent: [B, M, D]
        Returns scalar loss (0 when every dim already has std >= gamma).
        """
        if pred_latent.shape[0] < 2:
            return pred_latent.new_tensor(0.0)
        z = pred_latent.float().mean(dim=1)          # [B, D] — pool over tokens
        std = z.std(dim=0)                            # [D]
        return F.relu(gamma - std).mean()

    @staticmethod
    def drug_swap_diversity_loss(
        predictor: nn.Module,
        pre_latent: torch.Tensor,      # [B, M, D]  (unused but kept for API compat)
        condition_emb: torch.Tensor,   # [B, 2*text_dim]  cat([drug_emb, clinical_emb])
        time_delta: torch.Tensor,      # [B]          (unused but kept for API compat)
        pred_latent: torch.Tensor,     # [B, M, D]   (unused but kept for API compat)
        drug_categories: list,         # List[str] length B
        cos_margin: float = 0.9,       # repurposed as L2 distance margin
    ) -> torch.Tensor:
        """
        Drug-swap diversity loss — operates directly on drug_residual_proj outputs.

        Operates on the drug_residual_proj layer rather than the full pred_latent so
        that the gradient signal is not drowned out by the much larger transformer_out
        component (which dominates cosine similarity in the full pred_latent space).

        For each pair (i, shuffled_j) with different drug categories, compute:
            drug_res_real = drug_residual_proj(condition_emb_i)
            drug_res_cf   = drug_residual_proj(condition_cf_i)  # drug part swapped to j
        and apply a hinge loss:
            max(0, l2_margin - ||drug_res_real - drug_res_cf||_2)

        This pushes drug residuals for different treatments apart by at least l2_margin,
        providing a direct, short-gradient path that converges much faster than
        pushing full pred_latents apart through the transformer.

        cos_margin is reinterpreted as l2_margin (target L2 distance).
        """
        B = condition_emb.shape[0]
        if B < 4:
            return condition_emb.new_tensor(0.0)

        # Shuffle indices within batch for drug-swap
        shuffled_idx = torch.randperm(B, device=condition_emb.device)

        # Mask: only penalise pairs whose drug categories actually differ
        different = torch.tensor(
            [drug_categories[i] != drug_categories[shuffled_idx[i].item()]
             for i in range(B)],
            dtype=torch.bool, device=condition_emb.device,
        )
        if not different.any():
            return condition_emb.new_tensor(0.0)

        # Counterfactual condition: swap drug_emb (first half), keep clinical_emb (second half)
        drug_dim = condition_emb.shape[-1] // 2   # text_output_dim (768)
        cf_condition = condition_emb.clone()
        cf_condition[:, :drug_dim] = condition_emb[shuffled_idx, :drug_dim]

        # Compute drug residuals directly — short gradient path, no full predictor pass
        drug_res_real = predictor.drug_residual_proj(condition_emb.float())   # [B, latent_dim]
        drug_res_cf   = predictor.drug_residual_proj(cf_condition.float())    # [B, latent_dim]

        # L2 distance between residuals for different-drug pairs
        dist = (drug_res_real - drug_res_cf).norm(dim=-1)  # [B]

        # Hinge: push residuals apart until L2 distance >= l2_margin
        l2_margin = cos_margin  # parameter reused (0.9 ≈ reasonable L2 target in 768-dim space)
        return F.relu(l2_margin - dist[different]).mean()

    def _simple_cox_loss(self, risk_score, survival_time, event_indicator):
        """
        Breslow-approximation Cox partial likelihood loss.

        Convention: higher risk_score → shorter survival (same as concordance_index).

        Numerically stable: uses log-sum-exp trick and clamps risk to [-10, 10]
        so exp() never overflows and gradients stay healthy.
        """
        lr = risk_score.view(-1).float()
        t  = survival_time.view(-1)
        e  = event_indicator.view(-1)

        # Clamp risk to a reasonable range to prevent exp overflow
        lr = torch.clamp(lr, -10.0, 10.0)

        # Sort ascending by time
        idx = torch.argsort(t)
        lr, t, e = lr[idx], t[idx], e[idx]

        n_events_total = e.sum()
        if n_events_total == 0:
            return lr.new_tensor(0.0)

        # Breslow approximation (no ties correction — simpler, more stable)
        # Loss = -1/N_events * Σ_{i: event} [ lr_i - log(Σ_{j: t_j>=t_i} exp(lr_j)) ]
        total_loss = lr.new_zeros(())
        n_events = 0

        for i in range(len(lr)):
            if e[i] == 0:
                continue
            # Risk set: all j with t_j >= t_i (since sorted ascending, these are i..end)
            risk_scores_at_risk = lr[i:]             # [R]
            # Numerically stable log-sum-exp
            max_r = risk_scores_at_risk.max()
            log_sum_exp = max_r + torch.log(
                torch.exp(risk_scores_at_risk - max_r).sum().clamp(min=1e-12)
            )
            total_loss = total_loss - (lr[i] - log_sum_exp)
            n_events += 1

        return total_loss / max(n_events, 1)
    
    def get_parameter_count(self):
        """统计参数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.shared_text_encoder.parameters())
        predictor_params = sum(p.numel() for p in self.latent_predictor.parameters())
        survival_params = sum(p.numel() for p in self.survival_module.parameters())
        counts = {
            'total': total,
            'trainable': trainable,
            'text_encoder': encoder_params,
            'predictor': predictor_params,
            'survival': survival_params,
        }
        if self.mri_encoder is not None:
            mri_total = sum(p.numel() for p in self.mri_encoder.parameters())
            mri_trainable = sum(p.numel() for p in self.mri_encoder.parameters() if p.requires_grad)
            counts['mri_encoder_total'] = mri_total
            counts['mri_encoder_trainable'] = mri_trainable
        return counts




# 测试代码
if __name__ == "__main__":
    print("=" * 80)
    print("时间感知的胶质瘤生存预测模型测试")
    print("=" * 80)
    
    # 创建模型（测试三种时间编码）
    for time_type in ['positional', 'learnable', 'fourier']:
        print(f"\n{'='*80}")
        print(f"测试时间编码类型: {time_type}")
        print(f"{'='*80}")
        
        model = TimeAwareGliomaSurvivalPredictor(
            freeze_text_encoder=True,
            time_encoding_type=time_type,
            latent_dim=767,
            num_modalities=4,
            predictor_hidden_dim=512,
            predictor_num_layers=4
        )
        
        # 统计参数
        param_count = model.get_parameter_count()
        print(f"\n📊 参数统计：")
        print(f"  总参数: {param_count['total']:,}")
        print(f"  可训练: {param_count['trainable']:,}")
        
        # 模拟数据
        B = 4
        pre_latent = torch.randn(B, 4, 767)
        drugs_text = [
            '{"pre":{"tp":"T1","therapies":["TMZ"]}}',
            '{"pre":{"tp":"T2","therapies":["RT","TMZ"]}}',
            '{"pre":{"tp":"T1","therapies":[]}}',
            '{"pre":{"tp":"T2","therapies":["RT"]}}'
        ]
        clinical_text = [
            "62 year old male, GBM grade 4, MGMT methylated",
            "55 year old female, GBM grade 3",
            "48 year old male, Astrocytoma grade 2",
            "70 year old female, GBM grade 4"
        ]
        
        # ✅ 关键：不同的时间差
        time_delta = torch.tensor([30.0, 90.0, 180.0, 365.0])  # 1个月到1年
        
        post_latent = torch.randn(B, 4, 767)
        survival_time = torch.tensor([500.0, 800.0, 1200.0, 300.0])
        event_indicator = torch.tensor([1.0, 1.0, 0.0, 1.0])
        
        # 前向传播
        print(f"\n🔄 前向传播测试（time_delta={time_delta.tolist()}）...")
        pred_latent, risk_score, survival_prob = model(
            pre_latent, drugs_text, time_delta, clinical_text
        )
        
        print(f"✓ 预测潜在表示: {pred_latent.shape}")
        print(f"✓ 风险评分: {risk_score.shape}")
        
        # 检查时间影响
        print(f"\n⏱️  时间影响分析：")
        with torch.no_grad():
            # 固定治疗，改变时间
            fixed_pre = pre_latent[0:1]
            fixed_drug = [drugs_text[0]]
            fixed_context = [clinical_text[0]]
            
            times = torch.tensor([30.0, 90.0, 180.0, 365.0])
            preds = []
            for t in times:
                pred, _, _ = model(fixed_pre, fixed_drug, torch.tensor([t.item()]), fixed_context)
                preds.append(pred)
            
            # 计算预测的变化量
            changes = [(pred - fixed_pre).abs().mean().item() for pred in preds]
            print(f"  时间30天：变化量 = {changes[0]:.4f}")
            print(f"  时间90天：变化量 = {changes[1]:.4f}")
            print(f"  时间180天：变化量 = {changes[2]:.4f}")
            print(f"  时间365天：变化量 = {changes[3]:.4f}")
            print(f"  ✓ 变化量随时间{' 递增' if changes[-1] > changes[0] else ' 不符合预期'}！")
        
        # 计算损失
        total_loss, loss_dict = model.compute_loss(
            pred_latent, risk_score, survival_prob,
            post_latent, survival_time, event_indicator
        )
        
        print(f"\n📉 损失：")
        for k, v in loss_dict.items():
            print(f"  {k}: {v:.4f}")
        
        # 反向传播测试
        total_loss.backward()
        print(f"✓ 梯度计算成功")
        
        if time_type != 'fourier':
            print(f"\n(跳过其他编码类型的详细测试...)")
            break
    
    print("\n" + "=" * 80)
    print("✅ 所有测试通过！时间编码工作正常。")
    print("=" * 80)
    
    print("\n💡 推荐配置：")
    print("  - 数据量<1000: 使用 'learnable' (最灵活)")
    print("  - 数据量>1000: 使用 'fourier' (泛化最好)")
    print("  - 快速实验: 使用 'positional' (无需训练)")
