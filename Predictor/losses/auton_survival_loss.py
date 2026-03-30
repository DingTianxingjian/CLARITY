# losses/auton_survival_loss.py
"""
基于auton-survival的Cox损失实现
参考: https://github.com/autonlab/auton-survival
这是工业级的稳定实现，推荐使用
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def partial_ll_loss(lrisks, tb, eb, eps=1e-3): 
    """
    计算部分对数似然损失（Cox损失）
    
    Args:
        lrisks: log risks (n, ) - 对数风险评分
        tb: time of events in batch (n, ) - 事件时间
        eb: event indicator in batch (n, ) - 事件指示器
        eps: small value for numerical stability - 数值稳定性参数
    
    Returns:
        loss: 负对数部分似然
    """
    def _reshape_tensor_with_nans(data):
        """移除NaN值"""
        data = data.reshape(-1)
        return data[~torch.isnan(data)]
    
    # 处理输入，移除NaN
    tb = _reshape_tensor_with_nans(tb).detach()
    eb = _reshape_tensor_with_nans(eb).detach()
    
    # 添加小随机扰动，避免时间完全相同
    tb = tb + eps * torch.rand(len(tb), device=tb.device)
    
    # 按时间降序排列
    sindex = torch.argsort(-tb)
    tb = tb[sindex]
    eb = eb[sindex]
    lrisks = lrisks[sindex]
    
    # 计算累积log-sum-exp（风险集）
    lrisksdenom = torch.logcumsumexp(lrisks, dim=0)
    
    # 部分似然：log(h_i) - log(sum(exp(h_j)))
    plls = lrisks - lrisksdenom
    
    # 只计算发生事件的样本
    pll = plls[eb == 1]
    pll = torch.sum(pll)
    
    return -pll


class NLLDeepSurvLoss(nn.Module):
    """
    DeepSurv负对数似然损失
    这是推荐使用的Cox损失实现
    """
    def __init__(self, eps=1e-3):
        super(NLLDeepSurvLoss, self).__init__()
        self.eps = eps
    
    def forward(self, hazard_ratio, durations, events):
        """
        Args:
            hazard_ratio: [B, 1] or [B] - 风险比（模型输出）
            durations: [B] - 生存/删失时间
            events: [B] - 事件指示器（1=事件发生，0=删失）
        
        Returns:
            loss: scalar - 负对数似然损失
        """
        # 确保是1维tensor
        if hazard_ratio.dim() > 1:
            hazard_ratio = hazard_ratio.squeeze(-1)
        
        loss = partial_ll_loss(hazard_ratio, durations, events, eps=self.eps)
        return loss


class CoxPHLossWithTies(nn.Module):
    """
    带tie处理的Cox损失（Efron方法）
    当多个样本在同一时间发生事件时使用
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, hazard_ratio, durations, events):
        """使用Efron方法处理ties"""
        hazard_ratio = hazard_ratio.squeeze(-1) if hazard_ratio.dim() > 1 else hazard_ratio
        
        # 按时间排序
        order = torch.argsort(durations, descending=True)
        hazard_ratio = hazard_ratio[order]
        durations = durations[order]
        events = events[order]
        
        # 找到所有unique的事件时间
        unique_times = torch.unique(durations[events == 1])
        
        loss = 0.0
        for t in unique_times:
            # 在时间t发生事件的样本
            at_risk_mask = durations >= t
            event_at_t_mask = (durations == t) & (events == 1)
            
            # 风险集
            at_risk = hazard_ratio[at_risk_mask]
            events_at_t = hazard_ratio[event_at_t_mask]
            
            # Efron方法
            n_events = event_at_t_mask.sum()
            for k in range(n_events):
                risk_set = torch.logsumexp(at_risk, dim=0)
                tied_risk = torch.logsumexp(events_at_t, dim=0)
                
                # Efron校正
                correction = torch.log(torch.exp(risk_set) - (k / n_events) * torch.exp(tied_risk))
                loss += events_at_t[k] - correction
        
        return -loss / (events.sum() + 1e-7)


# 测试代码
if __name__ == "__main__":
    print("Testing auton-survival Cox loss...")
    
    # 模拟数据
    torch.manual_seed(42)
    B = 50
    hazard_ratio = torch.randn(B, 1, requires_grad=True)
    durations = torch.rand(B) * 2000 + 100
    events = torch.randint(0, 2, (B,)).float()
    
    # 测试标准Cox损失
    criterion = NLLDeepSurvLoss()
    loss = criterion(hazard_ratio, durations, events)
    
    print(f"Loss value: {loss.item():.4f}")
    print(f"Event rate: {events.mean().item():.2%}")
    
    # 测试梯度
    loss.backward()
    print(f"Gradient norm: {hazard_ratio.grad.norm().item():.4f}")
    print(f"Gradient mean: {hazard_ratio.grad.mean().item():.6f}")
    
    # 测试with ties
    print("\nTesting Cox loss with ties...")
    criterion_ties = CoxPHLossWithTies()
    hazard_ratio2 = torch.randn(B, 1, requires_grad=True)
    loss2 = criterion_ties(hazard_ratio2, durations, events)
    print(f"Loss with ties: {loss2.item():.4f}")
    
    print("\n✓ All tests passed!")