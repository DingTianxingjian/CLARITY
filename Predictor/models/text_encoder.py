# models/text_encoder.py
"""
文本编码器模块：使用PubMedBERT对药物治疗文本进行编码
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class DrugTextEncoder(nn.Module):
    """
    使用PubMedBERT编码药物治疗文本
    支持冻结/微调模式
    """
    def __init__(
        self, 
        model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        freeze: bool = False,
        output_dim: int = 768,
        max_length: int = 512,
        use_pooler_output: bool = True,
    ):
        super().__init__()
        self.max_length = max_length
        self.use_pooler_output = use_pooler_output
        
        # 加载预训练的tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # 是否冻结BERT参数
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # 投影层（可选，如果需要改变输出维度）
        self.projection = None
        if output_dim != 768:
            self.projection = nn.Linear(768, output_dim)
        
        self.output_dim = output_dim
    
    def forward(self, texts):
        """
        Args:
            texts: List[str] or str, 药物治疗的JSON字符串
        
        Returns:
            embeddings: [B, output_dim]
        """
        # 如果是单个字符串，转为列表
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenization
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 移到正确的设备
        device = next(self.bert.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # BERT编码
        outputs = self.bert(**encoded)
        
        # 旧版 checkpoint 更可能依赖 BERT pooler_output，而不是直接取 CLS hidden state
        if self.use_pooler_output and getattr(outputs, "pooler_output", None) is not None:
            cls_embeddings = outputs.pooler_output
        else:
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [B, 768]
        
        # 可选的投影
        if self.projection is not None:
            cls_embeddings = self.projection(cls_embeddings)
        
        return cls_embeddings
    
    def unfreeze(self):
        """解冻BERT参数，用于微调"""
        for param in self.bert.parameters():
            param.requires_grad = True
    
    def freeze(self):
        """冻结BERT参数"""
        for param in self.bert.parameters():
            param.requires_grad = False


# 测试代码
if __name__ == "__main__":
    encoder = DrugTextEncoder(freeze=True)
    
    # 模拟输入
    sample_text = '{"pre":{"tp":"T1","therapies":[{"type":"chemotherapy","agents":[{"name":"Temozolomide","dose":"75mg/m2"}]}]}}'
    
    embedding = encoder([sample_text, sample_text])
    print(f"Output shape: {embedding.shape}")  # [2, 768]
