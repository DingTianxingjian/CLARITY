# models/survival_module.py
"""
生存分析模块：使用TwoWayTransformer融合pre和pred特征，输出风险评分和生存概率
"""
import torch
import torch.nn as nn


class TwoWayCrossAttentionLayer(nn.Module):
    """
    单层双向交叉注意力 - Pre-norm 架构。

    与 post-norm（原版）不同，pre-norm 在 attention/FFN 之前对输入归一化，
    残差流保留原始尺度，跨样本多样性不会被层内 LayerNorm 抹掉。
    最终由 TwoWayTransformer 统一做一次 final_norm。
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()

        self.cross_attn_1to2 = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_2to1 = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        self.ffn1 = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

        # 4 个独立的 LayerNorm，每个用途都不共享权重
        self.norm_q1   = nn.LayerNorm(dim)   # seq1 作为 query 前的归一化
        self.norm_q2   = nn.LayerNorm(dim)   # seq2 作为 query 前的归一化
        self.norm_ffn1 = nn.LayerNorm(dim)   # seq1 进 FFN 前的归一化
        self.norm_ffn2 = nn.LayerNorm(dim)   # seq2 进 FFN 前的归一化

        self.dropout = nn.Dropout(dropout)

    def forward(self, seq1, seq2):
        # seq1 关注 seq2（seq2 作为 key/value，用 raw seq2 保留其多样性）
        attn_out1, _ = self.cross_attn_1to2(
            query=self.norm_q1(seq1), key=seq2, value=seq2
        )
        seq1 = seq1 + self.dropout(attn_out1)

        # seq2 关注 seq1
        attn_out2, _ = self.cross_attn_2to1(
            query=self.norm_q2(seq2), key=seq1, value=seq1
        )
        seq2 = seq2 + self.dropout(attn_out2)

        # Pre-norm FFN：各自用独立的 norm
        seq1 = seq1 + self.dropout(self.ffn1(self.norm_ffn1(seq1)))
        seq2 = seq2 + self.dropout(self.ffn2(self.norm_ffn2(seq2)))

        return seq1, seq2


class TwoWayTransformer(nn.Module):
    """
    双向注意力 Transformer。

    使用 pre-norm 层 + 最终统一 LayerNorm：
    - pre-norm 保证层内残差流多样性不被逐层抹掉
    - final_norm 限制输出量级，防止融合层激活爆炸
    """
    def __init__(self, dim, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            TwoWayCrossAttentionLayer(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # 最终归一化：稳定输出量级，同时保留跨样本方向多样性
        self.final_norm_seq1 = nn.LayerNorm(dim)
        self.final_norm_seq2 = nn.LayerNorm(dim)

    def forward(self, seq1, seq2):
        for layer in self.layers:
            seq1, seq2 = layer(seq1, seq2)
        return self.final_norm_seq1(seq1), self.final_norm_seq2(seq2)


class SurvivalModule(nn.Module):
    """
    生存分析模块：
        1. TwoWayTransformer 让 pre 和 pred 互相交互
        2. Mean-pool over tokens → 3 向量拼接 (pre, pred, delta)
        3. 轻量 MLP 融合（hidden_dim 小 + 强 dropout，防小数据集过拟合）
        4. 共享 fused 表征的 risk head 和 survival head

    v2 改动：
        - attention_dim 和 hidden_dim 缩小，减少参数量
        - fusion 只拼 3 向量（去掉 attn_delta，减少冗余）
        - dropout 从 0.3 → 0.5，强正则
        - risk_head/survival_head 去掉隐层，直接线性，避免在小数据上过拟合
    """
    def __init__(
        self,
        latent_dim: int = 767,
        num_modalities: int = 4,
        hidden_dim: int = 128,
        attention_dim: int = 128,
        num_twoway_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.5,
        condition_dim: int = 0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_modalities = num_modalities

        assert attention_dim % num_heads == 0, (
            f"attention_dim ({attention_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.input_proj = nn.Linear(latent_dim, attention_dim)

        self.two_way_attn = TwoWayTransformer(
            dim=attention_dim,
            num_heads=num_heads,
            num_layers=num_twoway_layers,
            dropout=dropout,
        )

        # Optional: project clinical/drug condition into the fusion vector
        if condition_dim > 0:
            self.condition_proj = nn.Linear(condition_dim, attention_dim)
            fusion_input_dim = attention_dim * 4  # pre, pred, delta, cond
        else:
            self.condition_proj = None
            fusion_input_dim = attention_dim * 3  # pre, pred, delta (original)

        self.modality_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Single linear head per task — fewer parameters, less overfit
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.survival_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, pre_latent, pred_latent, condition_emb=None):
        """
        Args:
            pre_latent:     [B, M, latent_dim]
            pred_latent:    [B, M, latent_dim]
            condition_emb:  [B, condition_dim] clinical+drug embedding (optional)
        Returns:
            risk_score:      [B, 1]
            survival_logit:  [B, 1]
        """
        # 0. 投影
        pre_proj  = self.input_proj(pre_latent)   # [B, M, D]
        pred_proj = self.input_proj(pred_latent)  # [B, M, D]

        # 1. 双向注意力
        pre_enhanced, pred_enhanced = self.two_way_attn(pre_proj, pred_proj)

        # 2. Mean-pool + cat (+ optional condition token)
        delta = pred_enhanced - pre_enhanced
        parts = [
            pre_enhanced.mean(dim=1),   # [B, D]
            pred_enhanced.mean(dim=1),  # [B, D]
            delta.mean(dim=1),          # [B, D]
        ]
        if self.condition_proj is not None and condition_emb is not None:
            parts.append(self.condition_proj(condition_emb))  # [B, D]
        combined = torch.cat(parts, dim=1)

        # 3. 融合 & 双头输出
        fused = self.modality_fusion(combined)
        risk_score     = self.risk_head(fused)
        survival_logit = self.survival_head(fused)
        return risk_score, survival_logit


# 测试代码
if __name__ == "__main__":
    module = SurvivalModule(latent_dim=767, num_modalities=4, hidden_dim=512)

    B = 8
    pre  = torch.randn(B, 4, 767)
    pred = torch.randn(B, 4, 767)

    risk, surv_logit = module(pre, pred)
    print(f"Input shape: {pre.shape}")
    print(f"Risk score shape: {risk.shape}")
    print(f"Survival logit shape: {surv_logit.shape}")
    print(f"\n✓ SurvivalModule test passed!")
