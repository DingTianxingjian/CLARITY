# losses/breslow_estimator.py
"""
Breslow估计器：用于从Cox模型预测个体生存曲线
基于lifelines包的实现
"""
import numpy as np
import pandas as pd
from collections import namedtuple


# 生存函数结果
SurvivalFunction = namedtuple('SurvivalFunction', ['x', 'y'])


class BreslowEstimator:
    """
    Breslow基线风险估计器
    用于从Cox模型的风险评分预测生存曲线
    """
    def __init__(self):
        self.baseline_hazard_ = None
        self.baseline_survival_ = None
        self.unique_times_ = None
        self.cum_baseline_hazard_ = None
    
    def fit(self, risk_scores, durations, events):
        """
        拟合Breslow估计器
        
        Args:
            risk_scores: [N] - 风险评分（模型输出）
            durations: [N] - 生存/删失时间
            events: [N] - 事件指示器
        """
        # 转为numpy
        if not isinstance(risk_scores, np.ndarray):
            risk_scores = risk_scores.detach().cpu().numpy()
        if not isinstance(durations, np.ndarray):
            durations = durations.detach().cpu().numpy()
        if not isinstance(events, np.ndarray):
            events = events.detach().cpu().numpy()
        
        # 确保1维
        risk_scores = risk_scores.reshape(-1)
        durations = durations.reshape(-1)
        events = events.reshape(-1)
        
        # 按时间排序
        order = np.argsort(durations)
        risk_scores = risk_scores[order]
        durations = durations[order]
        events = events[order]
        
        # 计算风险集
        hazard_ratio = np.exp(risk_scores)
        unique_times = np.unique(durations[events == 1])
        
        # 计算基线风险
        baseline_hazard = np.zeros(len(unique_times))
        
        for i, t in enumerate(unique_times):
            # 在时间t发生事件的数量
            d_t = np.sum((durations == t) & (events == 1))
            
            # 在时间t的风险集
            at_risk = durations >= t
            risk_set = np.sum(hazard_ratio[at_risk])
            
            # Breslow估计
            baseline_hazard[i] = d_t / risk_set if risk_set > 0 else 0
        
        # 累积基线风险
        cum_baseline_hazard = np.cumsum(baseline_hazard)
        
        # 基线生存函数
        baseline_survival = np.exp(-cum_baseline_hazard)
        
        # 保存结果
        self.unique_times_ = unique_times
        self.baseline_hazard_ = baseline_hazard
        self.cum_baseline_hazard_ = cum_baseline_hazard
        self.baseline_survival_ = baseline_survival
        
        return self
    
    def get_survival_function(self, risk_scores):
        """
        预测个体生存曲线
        
        Args:
            risk_scores: [N] or scalar - 风险评分
        
        Returns:
            survival_functions: list of SurvivalFunction
        """
        if self.baseline_survival_ is None:
            raise ValueError("Must call fit() first!")
        
        # 转为numpy
        if not isinstance(risk_scores, np.ndarray):
            if hasattr(risk_scores, 'detach'):
                risk_scores = risk_scores.detach().cpu().numpy()
            else:
                risk_scores = np.array(risk_scores)
        
        risk_scores = risk_scores.reshape(-1)
        
        # 计算个体生存函数
        # S(t|x) = S_0(t)^exp(risk_score)
        survival_functions = []
        
        for score in risk_scores:
            hazard_ratio = np.exp(score)
            # 个体累积风险 = 基线累积风险 × 风险比
            individual_cum_hazard = self.cum_baseline_hazard_ * hazard_ratio
            # 生存概率
            individual_survival = np.exp(-individual_cum_hazard)
            
            # 添加时间0的生存概率（=1）
            times = np.concatenate([[0], self.unique_times_])
            survival = np.concatenate([[1.0], individual_survival])
            
            survival_functions.append(
                SurvivalFunction(x=times, y=survival)
            )
        
        return survival_functions
    
    def predict_survival_at_times(self, risk_scores, times):
        """
        预测在指定时间点的生存概率
        
        Args:
            risk_scores: [N] - 风险评分
            times: [T] or scalar - 时间点
        
        Returns:
            survival_probs: [N, T] - 在每个时间点的生存概率
        """
        if isinstance(times, (int, float)):
            times = [times]
        times = np.array(times)
        
        # 获取生存函数
        survival_functions = self.get_survival_function(risk_scores)
        
        # 插值到指定时间点
        survival_probs = []
        for sf in survival_functions:
            # 线性插值
            probs = np.interp(times, sf.x, sf.y, left=1.0, right=sf.y[-1])
            survival_probs.append(probs)
        
        return np.array(survival_probs)
    
    def get_cumulative_hazard_function(self, risk_scores):
        """获取累积风险函数"""
        if self.cum_baseline_hazard_ is None:
            raise ValueError("Must call fit() first!")
        
        risk_scores = np.array(risk_scores).reshape(-1)
        
        cumulative_hazards = []
        for score in risk_scores:
            hazard_ratio = np.exp(score)
            cum_hazard = self.cum_baseline_hazard_ * hazard_ratio
            
            times = np.concatenate([[0], self.unique_times_])
            hazards = np.concatenate([[0.0], cum_hazard])
            
            cumulative_hazards.append(
                SurvivalFunction(x=times, y=hazards)
            )
        
        return cumulative_hazards


# 测试代码
if __name__ == "__main__":
    print("Testing Breslow Estimator...")
    
    # 模拟数据
    np.random.seed(42)
    n = 100
    
    risk_scores = np.random.randn(n)
    durations = np.random.exponential(500, n)
    events = np.random.binomial(1, 0.7, n)
    
    # 拟合估计器
    breslow = BreslowEstimator()
    breslow.fit(risk_scores, durations, events)
    
    print(f"Unique event times: {len(breslow.unique_times_)}")
    print(f"Baseline survival range: [{breslow.baseline_survival_.min():.3f}, {breslow.baseline_survival_.max():.3f}]")
    
    # 预测新样本
    new_risks = np.array([-1.0, 0.0, 1.0])
    survival_funcs = breslow.get_survival_function(new_risks)
    
    print(f"\nPredicted {len(survival_funcs)} survival curves")
    for i, sf in enumerate(survival_funcs):
        print(f"  Sample {i} (risk={new_risks[i]:.1f}): {len(sf.x)} time points")
    
    # 预测5年生存率
    five_year_survival = breslow.predict_survival_at_times(new_risks, times=[1825])
    print(f"\n5-year survival probabilities:")
    for i, prob in enumerate(five_year_survival):
        print(f"  Risk {new_risks[i]:+.1f}: {prob[0]:.2%}")
    
    print("\n✓ Breslow estimator test passed!")