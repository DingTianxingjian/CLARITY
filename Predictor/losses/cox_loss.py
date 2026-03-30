# losses/cox_loss.py
"""
Cox比例风险模型的损失函数实现
"""
import torch
import torch.nn as nn

def cox_partial_likelihood_loss(risk_scores, survival_times, event_indicators):
    """
    Cox部分似然损失
    
    Args:
        risk_scores: [B, 1] - 模型预测的风险评分（越高风险越大）
        survival_times: [B] - 生存时间（或删失时间）
        event_indicators: [B] - 事件指示器（1=事件发生，0=删失）
    
    Returns:
        loss: scalar - Cox损失值
    """
    # 确保维度正确
    risk_scores = risk_scores.squeeze(-1)  # [B]
    
    # 按生存时间排序（降序）
    sorted_indices = torch.argsort(survival_times, descending=True)
    risk_scores = risk_scores[sorted_indices]
    event_indicators = event_indicators[sorted_indices]
    
    # 计算风险集的log-sum-exp
    # 使用cumulative logsumexp来稳定计算
    max_risk = risk_scores.max()
    exp_risks = torch.exp(risk_scores - max_risk)
    cumsum_exp_risks = torch.cumsum(exp_risks.flip(0), dim=0).flip(0)
    log_risk_set = torch.log(cumsum_exp_risks) + max_risk
    
    # Cox部分似然：只计算发生事件的样本
    uncensored_likelihood = risk_scores - log_risk_set
    
    # 只对event_indicator=1的样本计算损失
    loss = -(uncensored_likelihood * event_indicators).sum() / (event_indicators.sum() + 1e-7)
    
    return loss


class CoxLoss(nn.Module):
    """Cox损失的nn.Module封装"""
    def __init__(self):
        super().__init__()
    
    def forward(self, risk_scores, survival_times, event_indicators):
        return cox_partial_likelihood_loss(risk_scores, survival_times, event_indicators)


# 测试代码
if __name__ == "__main__":
    # 模拟数据
    B = 10
    risk_scores = torch.randn(B, 1)
    survival_times = torch.rand(B) * 2000 + 100  # 100-2100天
    event_indicators = torch.randint(0, 2, (B,)).float()  # 0或1
    
    loss = cox_partial_likelihood_loss(risk_scores, survival_times, event_indicators)
    print(f"Cox loss: {loss.item():.4f}")
    
    # 测试梯度
    risk_scores.requires_grad = True
    loss.backward()
    print(f"Gradient exists: {risk_scores.grad is not None}")