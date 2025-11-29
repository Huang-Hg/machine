"""
评估指标模块
对应论文 Section 5.1: 评估框架 (三维度九指标)

论文评估维度:
1. 收益维度: ROI, Sortino Ratio, Benchmark Relative Return
2. 风险控制维度: MDD, Volatility Ratio, VaR 95%
3. 稳定性维度: Signal Decay Rate, Win Rate, Profit/Loss Ratio
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config


class PerformanceMetrics:
    """
    性能指标计算器
    对应论文 Section 5.1 完整评估体系
    """
    
    def __init__(self, initial_balance: float = Config.INITIAL_BALANCE):
        """
        初始化指标计算器
        
        Args:
            initial_balance: 初始资金
        """
        self.initial_balance = initial_balance
    
    def calculate_all_metrics(self, 
                              net_worth_history: List[float],
                              returns: np.ndarray,
                              benchmark_returns: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        计算所有指标
        对应论文 Section 5.1: 三维度九指标
        
        Args:
            net_worth_history: 净值历史
            returns: 策略收益率序列
            benchmark_returns: 基准收益率序列（如Buy&Hold）
            
        Returns:
            指标字典
        """
        net_worth_array = np.array(net_worth_history)
        
        metrics = {}
        
        # ==================== 收益维度 (Return Dimension) ====================
        # 1. 绝对收益 (ROI)
        # 论文: calculated net return after deducting 0.2% transaction friction cost
        metrics['roi'] = self.calculate_roi(net_worth_array)
        
        # 2. 风险调整收益 (Sortino Ratio)
        # 论文: Sortino Ratio, 下行标准差作为风险度量
        metrics['sortino_ratio'] = self.calculate_sortino_ratio(returns)
        
        # 3. Sharpe Ratio
        metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(returns)
        
        # 4. 基准相对收益 (Benchmark Relative Return)
        if benchmark_returns is not None:
            metrics['benchmark_relative_return'] = self.calculate_relative_return(returns, benchmark_returns)
        
        # ==================== 风险控制维度 (Risk Control Dimension) ====================
        # 5. 最大回撤 (MDD)
        # 论文: Maximum Drawdown
        metrics['max_drawdown'] = self.calculate_max_drawdown(net_worth_array)
        
        # 6. 波动率比 (Volatility Ratio)
        # 论文: strategy volatility / market volatility
        if benchmark_returns is not None:
            metrics['volatility_ratio'] = self.calculate_volatility_ratio(returns, benchmark_returns)
        else:
            metrics['volatility'] = np.std(returns) * np.sqrt(252)
        
        # 7. 极端损失概率 (VaR 95%)
        # 论文: 95% Value at Risk
        metrics['var_95'] = self.calculate_var(returns, confidence=0.95)
        
        # 8. CVaR (条件VaR)
        metrics['cvar_95'] = self.calculate_cvar(returns, confidence=0.95)
        
        # ==================== 稳定性维度 (Stability Dimension) ====================
        # 9. 胜率 (Win Rate)
        # 论文: proportion of profitable trades
        metrics['win_rate'] = self.calculate_win_rate(returns)
        
        # 10. 盈亏比 (Profit/Loss Ratio)
        # 论文: average profit / average loss
        metrics['profit_loss_ratio'] = self.calculate_profit_loss_ratio(returns)
        
        # 11. Calmar Ratio (收益/最大回撤)
        metrics['calmar_ratio'] = metrics['roi'] / (metrics['max_drawdown'] + 1e-8)
        
        # ==================== 其他有用指标 ====================
        metrics['total_return'] = net_worth_array[-1] / self.initial_balance - 1
        metrics['annualized_return'] = self.calculate_annualized_return(returns)
        metrics['final_net_worth'] = net_worth_array[-1]
        metrics['total_periods'] = len(returns)
        
        return metrics
    
    def calculate_roi(self, net_worth_array: np.ndarray) -> float:
        """
        计算投资回报率
        
        Args:
            net_worth_array: 净值序列
            
        Returns:
            ROI
        """
        return (net_worth_array[-1] - self.initial_balance) / self.initial_balance
    
    def calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """
        计算Sharpe比率
        
        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率
            
        Returns:
            Sharpe Ratio (年化)
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate
        if np.std(returns) == 0:
            return 0.0
        
        # 年化 (假设日频数据，252个交易日)
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
    
    def calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """
        计算Sortino比率（只考虑下行风险）
        对应论文 Section 5.1: Sortino Ratio
        
        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率
            
        Returns:
            Sortino Ratio (年化)
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return 0.0
        
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0
        
        return np.mean(excess_returns) / downside_std * np.sqrt(252)
    
    def calculate_max_drawdown(self, net_worth_array: np.ndarray) -> float:
        """
        计算最大回撤
        对应论文 Section 5.1: Maximum Drawdown (MDD)
        
        Args:
            net_worth_array: 净值序列
            
        Returns:
            最大回撤 (0到1之间)
        """
        if len(net_worth_array) == 0:
            return 0.0
        
        peak = np.maximum.accumulate(net_worth_array)
        drawdown = (peak - net_worth_array) / peak
        return np.max(drawdown)
    
    def calculate_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        计算Value at Risk
        对应论文 Section 5.1: VaR 95%
        
        Args:
            returns: 收益率序列
            confidence: 置信度
            
        Returns:
            VaR值
        """
        if len(returns) == 0:
            return 0.0
        
        return np.percentile(returns, (1 - confidence) * 100)
    
    def calculate_cvar(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        计算Conditional VaR (Expected Shortfall)
        
        Args:
            returns: 收益率序列
            confidence: 置信度
            
        Returns:
            CVaR值
        """
        if len(returns) == 0:
            return 0.0
        
        var = self.calculate_var(returns, confidence)
        return np.mean(returns[returns <= var])
    
    def calculate_volatility_ratio(self, 
                                   strategy_returns: np.ndarray,
                                   market_returns: np.ndarray) -> float:
        """
        计算波动率比
        对应论文 Section 5.1: Volatility Ratio (strategy volatility / market volatility)
        
        Args:
            strategy_returns: 策略收益率
            market_returns: 市场收益率
            
        Returns:
            波动率比
        """
        strategy_vol = np.std(strategy_returns)
        market_vol = np.std(market_returns)
        
        if market_vol == 0:
            return 0.0
        
        return strategy_vol / market_vol
    
    def calculate_relative_return(self,
                                  strategy_returns: np.ndarray,
                                  benchmark_returns: np.ndarray) -> float:
        """
        计算相对收益
        对应论文 Section 5.1: Benchmark Relative Return
        
        Args:
            strategy_returns: 策略收益率
            benchmark_returns: 基准收益率
            
        Returns:
            相对收益
        """
        strategy_total = (1 + strategy_returns).prod() - 1
        benchmark_total = (1 + benchmark_returns).prod() - 1
        
        return strategy_total - benchmark_total
    
    def calculate_win_rate(self, returns: np.ndarray) -> float:
        """
        计算胜率
        对应论文 Section 5.1: Win Rate (proportion of profitable trades)
        
        Args:
            returns: 收益率序列
            
        Returns:
            胜率 (0到1之间)
        """
        if len(returns) == 0:
            return 0.0
        
        return np.sum(returns > 0) / len(returns)
    
    def calculate_profit_loss_ratio(self, returns: np.ndarray) -> float:
        """
        计算盈亏比
        对应论文 Section 5.1: Profit/Loss Ratio (average profit / average loss)
        
        Args:
            returns: 收益率序列
            
        Returns:
            盈亏比
        """
        profits = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(profits) == 0 or len(losses) == 0:
            return 0.0
        
        avg_profit = np.mean(profits)
        avg_loss = np.abs(np.mean(losses))
        
        if avg_loss == 0:
            return 0.0
        
        return avg_profit / avg_loss
    
    def calculate_annualized_return(self, returns: np.ndarray, periods_per_year: int = 252) -> float:
        """
        计算年化收益率
        
        Args:
            returns: 收益率序列
            periods_per_year: 每年的周期数（默认252个交易日）
            
        Returns:
            年化收益率
        """
        if len(returns) == 0:
            return 0.0
        
        total_return = (1 + returns).prod() - 1
        n_periods = len(returns)
        
        return (1 + total_return) ** (periods_per_year / n_periods) - 1
    
    def calculate_signal_decay_rate(self,
                                    in_sample_metrics: Dict[str, float],
                                    out_sample_metrics: Dict[str, float],
                                    metric_name: str = 'sharpe_ratio') -> float:
        """
        计算信号衰减率
        对应论文 Section 5.1: Signal Decay Rate
        
        衡量策略在样本外表现的衰减程度
        
        Args:
            in_sample_metrics: 样本内指标
            out_sample_metrics: 样本外指标
            metric_name: 要比较的指标名称
            
        Returns:
            衰减率
        """
        in_value = in_sample_metrics.get(metric_name, 0)
        out_value = out_sample_metrics.get(metric_name, 0)
        
        if in_value == 0:
            return 0.0
        
        return (in_value - out_value) / in_value
    
    def print_metrics(self, metrics: Dict[str, float]):
        """
        打印指标（格式化输出）
        
        Args:
            metrics: 指标字典
        """
        print("\n" + "=" * 60)
        print("性能指标报告 (Performance Metrics Report)")
        print("=" * 60)
        
        print("\n【收益维度 Return Dimension】")
        print(f"  ROI (绝对收益):           {metrics.get('roi', 0) * 100:>8.2f}%")
        print(f"  Annualized Return (年化):  {metrics.get('annualized_return', 0) * 100:>8.2f}%")
        print(f"  Sharpe Ratio:             {metrics.get('sharpe_ratio', 0):>8.2f}")
        print(f"  Sortino Ratio:            {metrics.get('sortino_ratio', 0):>8.2f}")
        if 'benchmark_relative_return' in metrics:
            print(f"  Benchmark Relative:       {metrics['benchmark_relative_return'] * 100:>8.2f}%")
        
        print("\n【风险控制维度 Risk Control Dimension】")
        print(f"  Max Drawdown (MDD):       {metrics.get('max_drawdown', 0) * 100:>8.2f}%")
        print(f"  VaR 95%:                  {metrics.get('var_95', 0) * 100:>8.2f}%")
        print(f"  CVaR 95%:                 {metrics.get('cvar_95', 0) * 100:>8.2f}%")
        if 'volatility_ratio' in metrics:
            print(f"  Volatility Ratio:         {metrics['volatility_ratio']:>8.2f}")
        print(f"  Calmar Ratio:             {metrics.get('calmar_ratio', 0):>8.2f}")
        
        print("\n【稳定性维度 Stability Dimension】")
        print(f"  Win Rate (胜率):          {metrics.get('win_rate', 0) * 100:>8.2f}%")
        print(f"  Profit/Loss Ratio:        {metrics.get('profit_loss_ratio', 0):>8.2f}")
        
        print("\n【其他指标 Other Metrics】")
        print(f"  Final Net Worth:          ${metrics.get('final_net_worth', 0):>,.2f}")
        print(f"  Total Periods:            {metrics.get('total_periods', 0):>8.0f}")
        
        print("=" * 60 + "\n")


def test_metrics():
    """测试指标计算"""
    print("=" * 60)
    print("测试评估指标 (Metrics Test)")
    print("=" * 60)
    
    # 生成模拟数据
    np.random.seed(42)
    n_periods = 1000
    
    # 模拟净值曲线
    returns = np.random.normal(0.001, 0.02, n_periods)  # 平均0.1%收益，2%波动
    net_worth = Config.INITIAL_BALANCE * np.cumprod(1 + returns)
    net_worth_history = net_worth.tolist()
    
    # 模拟基准收益
    benchmark_returns = np.random.normal(0.0005, 0.015, n_periods)
    
    # 创建指标计算器
    metrics_calc = PerformanceMetrics()
    
    # 计算所有指标
    metrics = metrics_calc.calculate_all_metrics(
        net_worth_history=net_worth_history,
        returns=returns,
        benchmark_returns=benchmark_returns
    )
    
    # 打印指标
    metrics_calc.print_metrics(metrics)
    
    # 测试信号衰减率
    print("\n测试信号衰减率:")
    in_sample = {'sharpe_ratio': 2.0, 'sortino_ratio': 2.5}
    out_sample = {'sharpe_ratio': 1.6, 'sortino_ratio': 2.0}
    
    decay_rate = metrics_calc.calculate_signal_decay_rate(in_sample, out_sample, 'sharpe_ratio')
    print(f"  Sharpe Ratio衰减率: {decay_rate * 100:.2f}%")


if __name__ == "__main__":
    test_metrics()

