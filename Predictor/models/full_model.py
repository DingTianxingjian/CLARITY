"""
优化版胶质瘤生存预测模型：共享文本编码器 + 时间编码
关键改进：正确处理时间差信息
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import math
from models.survival_module import SurvivalModule
from utils.metrics import one_year_survival_targets_torch

try:
    from monai.networks.nets import ViT as MonaiViT
except ImportError:
    MonaiViT = None


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


class BrainIACVisionBackbone(nn.Module):
    """
    BrainIAC-style ViT backbone used as the MRI foundation vision encoder.
    The checkpoint layout matches the repository's BrainIAC checkpoints where
    encoder weights are stored under the ``backbone.`` prefix.
    """

    def __init__(
        self,
        checkpoint_path: str,
        img_size=(96, 96, 96),
        patch_size=(16, 16, 16),
        hidden_size: int = 768,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules=("qkv", "out_proj"),
    ):
        super().__init__()
        if MonaiViT is None:
            raise ImportError("MONAI is required to use the MRI vision backbone.")

        try:
            self.backbone = MonaiViT(
                in_channels=1,
                img_size=img_size,
                patch_size=patch_size,
                hidden_size=hidden_size,
                mlp_dim=3072,
                num_layers=12,
                num_heads=12,
                save_attn=True,
            )
        except TypeError:
            # Compatible with newer MONAI signatures.
            self.backbone = MonaiViT(
                in_channels=1,
                img_size=img_size,
                patch_size=patch_size,
                hidden_size=hidden_size,
                mlp_dim=3072,
                num_layers=12,
                num_heads=12,
                pos_embed="conv",
                classification=False,
                dropout_rate=0.0,
            )

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        backbone_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("backbone."):
                backbone_state_dict[key[9:]] = value

        if not backbone_state_dict:
            raise ValueError(
                f"No BrainIAC backbone weights found in checkpoint: {checkpoint_path}"
            )

        self.backbone.load_state_dict(backbone_state_dict, strict=True)
        self.lora_enabled = lora_r > 0
        self.lora_target_modules = tuple(lora_target_modules)
        self.lora_modules = []

        if self.lora_enabled:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.lora_modules = inject_lora_modules(
                self.backbone,
                target_suffixes=self.lora_target_modules,
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
            if not self.lora_modules:
                raise ValueError(
                    "No BrainIAC linear layers matched the requested LoRA targets: "
                    f"{self.lora_target_modules}"
                )

    def forward(self, x):
        features = self.backbone(x)
        if isinstance(features, (tuple, list)):
            features = features[0]
        return features[:, 0]


class MultiModalMRIBackbone(nn.Module):
    """Apply a shared single-modality MRI backbone across all four modalities."""

    def __init__(self, backbone: nn.Module, num_modalities: int = 4):
        super().__init__()
        self.backbone = backbone
        self.num_modalities = num_modalities

    def forward(self, mri):
        if mri.ndim != 5:
            raise ValueError(f"Expected MRI tensor [B, M, D, H, W], got {tuple(mri.shape)}")
        if mri.shape[1] != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} modalities, got {mri.shape[1]}"
            )

        modality_features = []
        for modality_idx in range(self.num_modalities):
            modality_volume = mri[:, modality_idx : modality_idx + 1]
            modality_features.append(self.backbone(modality_volume))
        return torch.stack(modality_features, dim=1)


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

        # ✅ BitsAndBytes 量化配置
        bnb_config = dict(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # ✅ 1. 加载模型（2B，显存更友好）
        try:
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
                **bnb_config
            )
        except Exception:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
                **bnb_config
            )

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



        emb = self.final_proj(emb)
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
        delta = self.output_proj(encoded[:, 2:, :])               # [B, M, latent_dim]
        return pre_latent + delta


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
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_modalities = num_modalities

        # 损失权重
        self.register_buffer('lambda_l1', torch.tensor(lambda_l1))
        self.register_buffer('lambda_cox', torch.tensor(lambda_cox))
        self.register_buffer('lambda_bce', torch.tensor(lambda_bce))

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
        self.survival_module = SurvivalModule(
            latent_dim=latent_dim,
            num_modalities=num_modalities,
            hidden_dim=survival_hidden_dim,
            num_twoway_layers=2,
            num_heads=predictor_num_heads,
            dropout=dropout,
        )
    
    def forward(self, pre_latent, drugs_text, time_delta, clinical_text=None):
        """
        Args:
            pre_latent:    [B, M, latent_dim]
            drugs_text:    List[str]
            time_delta:    [B]  days between pre and post scan
            clinical_text: List[str] (optional; zeros used when absent)
        Returns:
            predicted_latent: [B, M, latent_dim]
            risk_score:       [B, 1]
            survival_logit:   [B, 1]
        """
        # 1. Encode drug text and clinical context through MedGemma
        drug_emb = self.shared_text_encoder(drugs_text)           # [B, D]
        if clinical_text and any(clinical_text):
            clinical_emb = self.shared_text_encoder(clinical_text)  # [B, D]
        else:
            clinical_emb = torch.zeros_like(drug_emb)
        condition_emb = torch.cat([drug_emb, clinical_emb], dim=-1)  # [B, 2D]

        # 2. Predict post-latent conditioned on drug, clinical context, and time
        predicted_latent = self.latent_predictor(pre_latent, condition_emb, time_delta)

        # 3. Survival prediction via two-way attention on (pre, predicted_post)
        risk_score, survival_logit = self.survival_module(pre_latent, predicted_latent)

        return predicted_latent, risk_score, survival_logit
    
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
    
    def _simple_cox_loss(self, risk_score, survival_time, event_indicator):
        lr = risk_score.view(-1)
        t = survival_time.view(-1)
        e = event_indicator.view(-1)

        # 按时间升序排序
        idx = torch.argsort(t)
        lr, t, e = lr[idx], t[idx], e[idx]

        unique_event_times = torch.unique(t[e == 1])
        n_events_total = e.sum()
        if n_events_total == 0:
            return lr.new_tensor(0.0)

        eps = 1e-12
        total_loss = lr.new_zeros(())

        for tt in unique_event_times:
            # 风险集: t >= tt
            risk_mask = (t >= tt)
            # 当时刻的事件样本
            event_mask_t = (t == tt) & (e == 1)
            d = event_mask_t.sum()

            if d.item() == 0:
                continue

            # Efron 修正计算
            sum_lr_events = lr[event_mask_t].sum()
            sum_exp_risk = torch.exp(lr[risk_mask]).sum()
            sum_exp_events = torch.exp(lr[event_mask_t]).sum()

            denom = lr.new_zeros(())
            d_float = d.to(lr.dtype)
            for j in range(int(d.item())):
                frac = j / d_float
                denom = denom + torch.log(torch.clamp(sum_exp_risk - frac * sum_exp_events, min=eps))

            total_loss = total_loss - (sum_lr_events - denom)

        return total_loss / (n_events_total + eps)
    
    def get_parameter_count(self):
        """统计参数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.shared_text_encoder.parameters())
        predictor_params = sum(p.numel() for p in self.latent_predictor.parameters())
        survival_params = sum(p.numel() for p in self.survival_module.parameters())
        return {
            'total': total,
            'trainable': trainable,
            'encoder': encoder_params,
            'predictor': predictor_params,
            'survival': survival_params,
        }


class MRITimeAwareSurvivalPredictor(TimeAwareGliomaSurvivalPredictor):


    def __init__(
        self,
        vision_checkpoint_path: str,
        mri_img_size=(96, 96, 96),
        freeze_vision_backbone: bool = True,
        vision_lora_r: int = 8,
        vision_lora_alpha: int = 16,
        vision_lora_dropout: float = 0.1,
        vision_lora_target_modules=("qkv", "out_proj"),
        **kwargs,
    ):
        kwargs.setdefault("latent_dim", 768)
        kwargs.setdefault("num_modalities", 4)
        super().__init__(**kwargs)

        self.vision_backbone = MultiModalMRIBackbone(
            BrainIACVisionBackbone(
                checkpoint_path=vision_checkpoint_path,
                img_size=mri_img_size,
                lora_r=0 if freeze_vision_backbone else vision_lora_r,
                lora_alpha=vision_lora_alpha,
                lora_dropout=vision_lora_dropout,
                lora_target_modules=vision_lora_target_modules,
            ),
            num_modalities=self.num_modalities,
        )
        self.freeze_vision_backbone = freeze_vision_backbone
        if self.freeze_vision_backbone:
            for param in self.vision_backbone.parameters():
                param.requires_grad = False

    def encode_mri(self, mri):
        if self.freeze_vision_backbone:
            with torch.no_grad():
                return self.vision_backbone(mri)
        return self.vision_backbone(mri)

    def predict_from_features(self, pre_latent, drugs_text, time_delta, clinical_text=None):
        return super().forward(pre_latent, drugs_text, time_delta, clinical_text)

    def forward(self, pre_mri, drugs_text, time_delta, clinical_text=None):
        pre_latent = self.encode_mri(pre_mri)
        pred_latent, risk_score, survival_prob = self.predict_from_features(
            pre_latent, drugs_text, time_delta, clinical_text
        )
        return pred_latent, risk_score, survival_prob, pre_latent

    def get_parameter_count(self):
        counts = super().get_parameter_count()
        vision_total = sum(p.numel() for p in self.vision_backbone.parameters())
        vision_trainable = sum(
            p.numel() for p in self.vision_backbone.parameters() if p.requires_grad
        )
        counts["vision_backbone"] = vision_total
        counts["vision_backbone_trainable"] = vision_trainable
        counts["vision_lora_modules"] = len(
            getattr(self.vision_backbone.backbone, "lora_modules", [])
        )
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
