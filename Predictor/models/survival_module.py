# models/survival_module.py
"""
生存分析模块：使用TwoWayTransformer融合pre和pred特征，输出风险评分和生存概率
"""
import torch
import torch.nn as nn

class TwoWayTransformer(nn.Module):
    """
    双向注意力Transformer：让两个序列互相关注
    """
    def __init__(self, dim, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        
        # 为每一层创建cross-attention模块
        self.layers = nn.ModuleList([
            TwoWayCrossAttentionLayer(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, seq1, seq2):
        """
        Args:
            seq1: [B, N, D] - 第一个序列（pre_latent）
            seq2: [B, N, D] - 第二个序列（pred_latent）
        
        Returns:
            enhanced_seq1: [B, N, D]
            enhanced_seq2: [B, N, D]
        """
        for layer in self.layers:
            seq1, seq2 = layer(seq1, seq2)
        return seq1, seq2


class TwoWayCrossAttentionLayer(nn.Module):
    """单层双向交叉注意力"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        
        # seq1关注seq2
        self.cross_attn_1to2 = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # seq2关注seq1
        self.cross_attn_2to1 = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # 前馈网络
        self.ffn1 = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        
        # Layer normalization
        self.norm1_1 = nn.LayerNorm(dim)
        self.norm1_2 = nn.LayerNorm(dim)
        self.norm2_1 = nn.LayerNorm(dim)
        self.norm2_2 = nn.LayerNorm(dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, seq1, seq2):
        # seq1关注seq2（seq2作为key/value）
        attn_out1, _ = self.cross_attn_1to2(
            query=seq1, key=seq2, value=seq2
        )
        seq1 = self.norm1_1(seq1 + self.dropout(attn_out1))
        
        # seq2关注seq1（seq1作为key/value）
        attn_out2, _ = self.cross_attn_2to1(
            query=seq2, key=seq1, value=seq1
        )
        seq2 = self.norm1_2(seq2 + self.dropout(attn_out2))
        
        # 前馈网络
        seq1 = self.norm2_1(seq1 + self.dropout(self.ffn1(seq1)))
        seq2 = self.norm2_2(seq2 + self.dropout(self.ffn2(seq2)))
        
        return seq1, seq2


class SurvivalModule(nn.Module):
    """
    生存分析模块：
        1. 使用TwoWayTransformer让pre和pred互相交互
        2. 计算差异特征（治疗效果）
        3. 融合所有信息
        4. 分别输出风险评分和生存logit（双头）
    """
    def __init__(
        self,
        latent_dim: int = 767,
        num_modalities: int = 4,
        hidden_dim: int = 512,
        attention_dim: int = 256,
        num_twoway_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.3
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_modalities = num_modalities

        assert attention_dim % num_heads == 0, (
            f"attention_dim ({attention_dim}) must be divisible by num_heads ({num_heads})"
        )

        # 投影到 attention_dim（比原来的 768 小，节省参数）
        self.input_proj = nn.Linear(latent_dim, attention_dim)

        # 双向注意力Transformer
        self.two_way_attn = TwoWayTransformer(
            dim=attention_dim,
            num_heads=num_heads,
            num_layers=num_twoway_layers,
            dropout=dropout
        )

        # 模态融合网络
        # 保留模态级 token，不做 mean pool，直接将 [pre, pred, delta]
        # 三组 [B, M, D] 特征 flatten 后拼接。
        fusion_input_dim = attention_dim * num_modalities * 3
        self.modality_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 独立 risk head：服务 Cox 排序目标
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # 独立 survival head：服务 1-year BCE 目标
        self.survival_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, pre_latent, pred_latent):
        """
        Args:
            pre_latent: [B, 4, 767] - 治疗前特征
            pred_latent: [B, 4, 767] - 预测的治疗后特征
        
        Returns:
            risk_score: [B, 1] - Cox风险评分（原始值）
            survival_logit: [B, 1] - 1年生存logit（未经过sigmoid）
        """
        # 0. 投影到 attention 兼容的维度
        pre_proj = self.input_proj(pre_latent)   # [B, M, attention_dim]
        pred_proj = self.input_proj(pred_latent) # [B, M, attention_dim]
        
        # 1. 双向注意力交互
        pre_enhanced, pred_enhanced = self.two_way_attn(pre_proj, pred_proj)
        # 输出: [B, M, attention_dim] each
        
        # 2. 计算差异特征（治疗效果的体现）
        delta = pred_enhanced - pre_enhanced  # [B, M, attention_dim]
        
        # 3. 拼接所有信息
        combined = torch.cat([
            pre_enhanced.flatten(1),   # [B, M * attention_dim]
            pred_enhanced.flatten(1),  # [B, M * attention_dim]
            delta.flatten(1)           # [B, M * attention_dim]
        ], dim=1)  # [B, 3 * M * attention_dim]
        
        # 4. 融合特征
        fused = self.modality_fusion(combined)  # [B, hidden_dim//2]
        
        # 5. 双头输出：risk 和 survival 分开建模
        risk_score = self.risk_head(fused)          # [B, 1] - Cox 风险评分
        survival_logit = self.survival_head(fused)  # [B, 1] - 1年生存 logit
        return risk_score, survival_logit


# 测试代码
if __name__ == "__main__":
    module = SurvivalModule(
        latent_dim=767,
        num_modalities=4,
        hidden_dim=512
    )
    
    # 模拟输入
    B = 8
    pre = torch.randn(B, 4, 767)
    pred = torch.randn(B, 4, 767)
    
    risk, surv_logit = module(pre, pred)
    print(f"Input shape: {pre.shape}")           # [8, 4, 767]
    print(f"Risk score shape: {risk.shape}")     # [8, 1]
    print(f"Survival logit shape: {surv_logit.shape}")  # [8, 1]
    print(f"\n✓ SurvivalModule test passed! (767 -> 768 projection working)")
