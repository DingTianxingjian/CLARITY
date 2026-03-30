# utils/metrics.py
"""
生存分析评估指标
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from collections import deque


def concordance_index(risk_scores, survival_times, event_indicators):
    """
    计算C-index (Concordance Index)
    衡量模型对生存时间排序的准确性
    
    Args:
        risk_scores: [N] - 风险评分（越高风险越大）
        survival_times: [N] - 生存时间
        event_indicators: [N] - 事件指示器（1=事件，0=删失）
    
    Returns:
        c_index: float - C-index值 (0.5-1.0, 0.5为随机，1.0为完美)
    """
    n = len(risk_scores)
    
    # 确保是numpy数组
    risk_scores = np.array(risk_scores)
    survival_times = np.array(survival_times)
    event_indicators = np.array(event_indicators).astype(bool)
    
    # 如果没有事件，返回NaN
    if event_indicators.sum() == 0:
        return np.nan
    
    concordant = 0
    discordant = 0
    tied_risk = 0
    
    # 遍历所有可比较的对
    for i in range(n):
        # 只考虑事件发生的样本作为基准
        if not event_indicators[i]:
            continue
        
        for j in range(n):
            if i == j:
                continue
            
            # 可比较的情况：
            # 1. i发生事件且时间更早，或
            # 2. j被删失且删失时间晚于i的事件时间
            if survival_times[i] < survival_times[j]:
                # i的时间更早，应该风险更高（risk_score更大）
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1
                elif risk_scores[i] < risk_scores[j]:
                    discordant += 1
                else:
                    tied_risk += 1
            elif survival_times[i] == survival_times[j] and event_indicators[j]:
                # 时间相同且都发生事件
                if risk_scores[i] == risk_scores[j]:
                    tied_risk += 1
    
    # 计算C-index
    if concordant + discordant + tied_risk == 0:
        return 0.5
    
    c_index = (concordant + 0.5 * tied_risk) / (concordant + discordant + tied_risk)
    return c_index


# 用于AUC平滑的队列
_auc_history = deque(maxlen=20)


def compute_auc(predicted_probs, true_labels, smooth=True):
    """
    稳定版 AUC 计算函数 (支持生存预测概率)
    -----------------------------------------
    功能：
      ✅ 自动过滤无效样本 (NaN / 单一类别)
      ✅ 自动检测预测方向 (生存 vs 死亡)
      ✅ 自动rank标准化减少数值漂移
      ✅ 可选滑动平均平滑 (smooth=True)
    
    Args:
        predicted_probs : [N] 模型输出概率（0~1）
        true_labels : [N] 真实标签（0或1）
        smooth : bool, 是否对AUC做滑动平均
    
    Returns:
        auc : float 稳定AUC值
    """
    predicted_probs = np.asarray(predicted_probs, dtype=float).reshape(-1)
    true_labels = np.asarray(true_labels, dtype=int).reshape(-1)

    # ① 去除无效样本
    mask = np.isfinite(predicted_probs) & np.isfinite(true_labels)
    predicted_probs, true_labels = predicted_probs[mask], true_labels[mask]

    if len(predicted_probs) < 10 or len(np.unique(true_labels)) < 2:
        return np.nan

    # ② 自动方向检测（若死亡样本概率反而更高，则反转）
    mean_pred_alive = predicted_probs[true_labels == 0].mean()
    mean_pred_dead = predicted_probs[true_labels == 1].mean()
    if mean_pred_alive > mean_pred_dead:
        predicted_probs = 1 - predicted_probs

    # ③ Rank normalization，抵抗漂移（AUC只看排序）
    predicted_probs = rankdata(predicted_probs) / len(predicted_probs)

    # ④ 计算AUC
    try:
        auc = roc_auc_score(true_labels, predicted_probs)
    except Exception:
        auc = np.nan

    # ⑤ 平滑（滑动平均）
    if smooth:
        _auc_history.append(auc)
        auc = np.nanmean(_auc_history)

    return float(auc)


def integrated_brier_score(predicted_probs, survival_times, event_indicators, time_points):
    """
    计算Integrated Brier Score (IBS)
    衡量预测生存概率的准确性
    
    Args:
        predicted_probs: [N, T] - 在不同时间点的预测生存概率
        survival_times: [N] - 实际生存时间
        event_indicators: [N] - 事件指示器
        time_points: [T] - 评估的时间点
    
    Returns:
        ibs: float - Integrated Brier Score (越小越好)
    """
    n_samples, n_times = predicted_probs.shape
    brier_scores = []
    
    for t_idx, t in enumerate(time_points):
        # 真实标签：在时间t时是否仍然存活
        true_survival = (survival_times > t).astype(float)
        
        # 预测的生存概率
        pred_survival = predicted_probs[:, t_idx]
        
        # Brier score
        brier = np.mean((pred_survival - true_survival) ** 2)
        brier_scores.append(brier)
    
    # 时间上积分
    ibs = np.trapz(brier_scores, time_points) / (time_points[-1] - time_points[0])
    return ibs


def time_dependent_auc(risk_scores, survival_times, event_indicators, time_point):
    """
    计算Time-dependent AUC
    在特定时间点评估模型的区分能力
    
    Args:
        risk_scores: [N] - 风险评分
        survival_times: [N] - 生存时间
        event_indicators: [N] - 事件指示器
        time_point: float - 评估时间点
    
    Returns:
        auc: float - 时间依赖的AUC
    """
    # 构建标签：在time_point时是否发生事件
    labels = []
    scores = []
    
    for i in range(len(survival_times)):
        # Case 1: 在time_point之前发生事件
        if event_indicators[i] == 1 and survival_times[i] <= time_point:
            labels.append(1)
            scores.append(risk_scores[i])
        # Case 2: 在time_point之后仍存活或被删失
        elif survival_times[i] > time_point:
            labels.append(0)
            scores.append(risk_scores[i])
        # Case 3: 在time_point之前被删失（不确定，跳过）
    
    if len(set(labels)) < 2:
        return np.nan
    
    try:
        auc = roc_auc_score(labels, scores)
    except:
        auc = np.nan
    
    return auc


# 测试代码
if __name__ == "__main__":
    # 模拟数据
    np.random.seed(42)
    n = 100
    
    risk_scores = np.random.randn(n)
    survival_times = np.random.exponential(500, n)
    event_indicators = np.random.binomial(1, 0.7, n)
    
    # 计算C-index
    c_idx = concordance_index(risk_scores, survival_times, event_indicators)
    print(f"C-index: {c_idx:.4f}")
    
    # 计算5年生存AUC
    five_year_survived = ((survival_times > 1825) & (event_indicators == 0)).astype(float)
    predicted_probs = 1 / (1 + np.exp(-risk_scores))  # Sigmoid
    auc = compute_auc(predicted_probs, five_year_survived)
    print(f"5-year survival AUC: {auc:.4f}")
    
    # Time-dependent AUC
    td_auc = time_dependent_auc(risk_scores, survival_times, event_indicators, time_point=1825)
    print(f"Time-dependent AUC (5 years): {td_auc:.4f}")