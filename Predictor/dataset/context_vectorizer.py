import torch
import numpy as np

class ClinicalContextVectorizer:
    """
    将来自 clinical_latest.json 的 context_static 对象转换为一个扁平的数值向量。
    """
    def __init__(self, vector_dim: int, config: dict = None):
        self.vector_dim = vector_dim
        self.config = config or self._get_default_config()
        
        # 为分类特征构建映射
        self.feature_map = {}
        self.vector_layout = {}
        current_idx = 0
        
        for feature, details in self.config.items():
            if details['type'] == 'categorical':
                for value in details['values']:
                    self.feature_map[f"{feature}_{value}"] = current_idx
                    current_idx += 1
                self.vector_layout[feature] = {'start': self.feature_map[f"{feature}_{details['values'][0]}"] , 'end': current_idx}
            elif details['type'] == 'numerical':
                self.feature_map[feature] = current_idx
                self.vector_layout[feature] = {'start': current_idx, 'end': current_idx + 1}
                current_idx += 1
            elif details['type'] == 'binary_group':
                for sub_feature in details['values']:
                    self.feature_map[sub_feature] = current_idx
                    current_idx += 1
                self.vector_layout[feature] = {'start': self.feature_map[details['values'][0]], 'end': current_idx}

        # 确保最终维度匹配
        if current_idx != self.vector_dim:
            raise ValueError(f"Configured vector dimension ({current_idx}) does not match expected dimension ({self.vector_dim})")

    def _get_default_config(self):
        # 这个配置应该与你的模型定义中的 clinical_input_dim 匹配
        # 这里的维度是 3 (sex) + 4 (race) + 1 (age) + 1 (grade) + 12 (genomics) = 21
        return {
            "sex_at_birth": {"type": "categorical", "values": ["male", "female", "unknown"]},
            "race": {"type": "categorical", "values": ["white", "black or african american", "asian", "unknown"]},
            "age_at_diagnosis_years": {"type": "numerical", "norm_mean": 60, "norm_std": 15},
            "who_grade": {"type": "numerical", "norm_mean": 3.5, "norm_std": 0.5},
            "genomics": {
                "type": "binary_group",
                "values": [
                    "idh1", "idh2", "atrx", "mgmt_methylation", "braf_v600e", 
                    "tert_promoter", "chr7_gain_chr10_loss", "h3_3a", "egfr_amp", 
                    "pten", "cdkn2ab_deletion", "tp53_alteration"
                ]
            }
        }

    def vectorize(self, context_static: dict) -> torch.Tensor:
        """将单个 context_static 字典转换为 Tensor"""
        vec = torch.zeros(self.vector_dim)
        
        # 处理顶层特征
        for feature, details in self.config.items():
            if feature == "genomics":
                continue
            
            value = context_static.get(feature)
            
            if details['type'] == 'categorical':
                value = value if value in details['values'] else 'unknown'
                key = f"{feature}_{value}"
                if key in self.feature_map:
                    vec[self.feature_map[key]] = 1.0
            
            elif details['type'] == 'numerical':
                if value is not None:
                    # 归一化
                    norm_val = (value - details['norm_mean']) / details['norm_std']
                    vec[self.feature_map[feature]] = norm_val

        # 处理基因组学特征
        genomics_data = context_static.get("genomics", {})
        genomics_config = self.config.get("genomics", {})
        if genomics_config:
            for sub_feature in genomics_config['values']:
                value = genomics_data.get(sub_feature)
                if value is not None and sub_feature in self.feature_map:
                    # 0: no, 1: yes, 2: unknown. 我们将 unknown 映射为 0.5
                    if value == 0:
                        vec[self.feature_map[sub_feature]] = 0.0
                    elif value == 1:
                        vec[self.feature_map[sub_feature]] = 1.0
                    elif value == 2:
                        vec[self.feature_map[sub_feature]] = 0.5 # 代表未知
        
        return vec

if __name__ == '__main__':
    # 示例用法
    vectorizer = ClinicalContextVectorizer(vector_dim=20)
    sample_context = {"sex_at_birth": "female", "race": "white", "age_at_diagnosis_years": 57.0, "who_grade": 4, "genomics": {"idh1": 0, "idh2": 0, "atrx": 2, "mgmt_methylation": 2, "braf_v600e": 2, "tert_promoter": 2, "chr7_gain_chr10_loss": 2, "h3_3a": 2, "egfr_amp": 2, "pten": 0, "cdkn2ab_deletion": 0, "tp53_alteration": 0}}
    
    tensor_vec = vectorizer.vectorize(sample_context)
    print(tensor_vec)
    print(tensor_vec.shape)