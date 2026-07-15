# models/latent_predictor.py

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """????"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TimeEncoder(nn.Module):
    """将标量时间差编码为高维向量"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # 创建一个可学习的线性层来处理时间
        self.time_proj = nn.Linear(1, d_model)
        # 创建一个固定的sin/cos编码层
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, time_delta):
        # time_delta: [B] -> [B, 1]
        time_delta = time_delta.unsqueeze(1)
        # 固定的sin/cos编码
        pe = torch.zeros(time_delta.size(0), self.d_model, device=time_delta.device)
        pe[:, 0::2] = torch.sin(time_delta * self.div_term)
        pe[:, 1::2] = torch.cos(time_delta * self.div_term)
        # 结合可学习的线性映射
        return self.time_proj(time_delta) + pe

class TransformerLatentPredictor(nn.Module):
    
    def __init__(
        self,
        input_dim: int = 767,          # pre_latent?????
        condition_dim: int = 768,      # 条件向量的维度 (drug + clinical)
        num_modalities: int = 4,       # ????
        hidden_dim: int = 512,         # Transformer?????
        num_layers: int = 4,           # Transformer??
        num_heads: int = 8,            # ?????
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.num_modalities = num_modalities
        self.hidden_dim = hidden_dim
        
        # ?pre_latent???hidden_dim
        self.latent_proj = nn.Linear(input_dim, hidden_dim)
        
        # ????????hidden_dim
        self.condition_proj = nn.Linear(condition_dim, hidden_dim)

        # (新增) 时间编码器
        self.time_encoder = TimeEncoder(hidden_dim)
        
        # ????
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer???
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, pre_latent, condition_embedding, time_delta):
        """
        Args:
            pre_latent: [B, 4, 767] 
            condition_embedding: [B, condition_dim] (合并了药物和临床背景)
            time_delta: [B] - 时间差
        
        Returns:
            predicted_latent: [B, 4, 767] 
        """
        B = pre_latent.size(0)
        
        # 1. 投影pre_latent到hidden_dim
        pre_proj = self.latent_proj(pre_latent)  # [B, 4, hidden_dim]
        
        # 2. 投影条件向量 (药物+临床)
        condition_proj = self.condition_proj(condition_embedding)  # [B, hidden_dim]
        condition_token = condition_proj.unsqueeze(1)  # [B, 1, hidden_dim]

        # 3. (新增) 编码时间差
        time_token = self.time_encoder(time_delta) # [B, hidden_dim]
        time_token = time_token.unsqueeze(1) # [B, 1, hidden_dim]
        
        # 4. 组合序列: [条件, 时间, 模态1, 模态2, ...]
        sequence = torch.cat([condition_token, time_token, pre_proj], dim=1)  # [B, 1+1+4, hidden_dim]
        
        # 5. 添加位置编码
        sequence = self.pos_encoder(sequence)
        
        # 6. Transformer处理
        transformed = self.transformer(sequence)  # [B, 6, hidden_dim]
        
        # 7. 提取与影像模态对应的输出 (跳过前两个token: 条件和时间)
        modality_features = transformed[:, 2:, :]  # [B, 4, hidden_dim]
        
        # 8. 投影回原始维度
        predicted = self.output_proj(modality_features)  # [B, 4, 767]
        
        # 8. ?????????????????
        # predicted_latent = pre_latent + self.residual_weight * predicted
        # ?????????
        predicted_latent = predicted
        
        return predicted_latent


# ????
if __name__ == "__main__":
    predictor = TransformerLatentPredictor(
        input_dim=767,
        condition_dim=768 + 128, # drug_dim + clinical_embed_dim
        num_modalities=4,
        hidden_dim=512,
        num_layers=4,
        num_heads=8
    )
    
    # 模拟输入
    B = 8
    pre_latent = torch.randn(B, 4, 767)
    condition_emb = torch.randn(B, 768 + 128)
    time_delta = torch.randint(30, 365, (B,)).float()
    
    predicted = predictor(pre_latent, condition_emb, time_delta)
    print(f"Input shape: {pre_latent.shape}")
    print(f"Condition shape: {condition_emb.shape}")
    print(f"Time delta shape: {time_delta.shape}")
    print(f"Output shape: {predicted.shape}")