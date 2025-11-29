"""
加密货币交易环境
对应论文 Section 5.1: Experimental setup

特性:
1. 0.2% 交易摩擦成本 (Transaction Fee)
2. 混合动作空间 (离散 + 连续)
3. 动态奖励设计 (ROI + Risk Penalty)
4. 市场状态追踪
5. 风险管理机制
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import gym
from gym import spaces

from config import Config


class CryptoTradingEnv(gym.Env):
    """
    加密货币交易环境
    对应论文 Section 5.1 & 6
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self,
                 data: pd.DataFrame,
                 features: np.ndarray,
                 initial_balance: float = Config.INITIAL_BALANCE,
                 transaction_fee: float = Config.TRANSACTION_FEE_RATE,
                 sequence_length: int = Config.SEQUENCE_LENGTH,
                 max_position: float = Config.MAX_POSITION_SIZE,
                 enable_defense_mode: bool = True):
        """
        初始化交易环境
        
        Args:
            data: 原始数据DataFrame（包含价格）
            features: 特征数组 [N, feature_dim]
            initial_balance: 初始资金
            transaction_fee: 交易费率 (论文: 0.2%)
            sequence_length: 状态序列长度
            max_position: 最大持仓比例
            enable_defense_mode: 是否启用防御模式（Section 6）
        """
        super(CryptoTradingEnv, self).__init__()
        
        self.data = data
        self.features = features
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.sequence_length = sequence_length
        self.max_position = max_position
        self.enable_defense_mode = enable_defense_mode
        
        # 验证数据长度
        assert len(data) == len(features), "数据和特征长度不匹配"
        
        # ==================== 动作空间 ====================
        # 混合动作空间: (离散, 连续)
        # 离散: 0=Hold, 1=Buy, 2=Sell
        # 连续: 持仓比例 [0, 1]
        self.action_space = spaces.Tuple((
            spaces.Discrete(Config.DISCRETE_ACTIONS),
            spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        ))
        
        # ==================== 观测空间 ====================
        # 状态序列: [sequence_length, feature_dim]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(sequence_length, features.shape[1]),
            dtype=np.float32
        )
        
        # ==================== 初始化状态 ====================
        self.reset()
    
    def reset(self) -> np.ndarray:
        """
        重置环境
        
        Returns:
            初始状态
        """
        # 账户状态
        self.balance = self.initial_balance  # USDT余额
        self.crypto_held = 0.0               # 持有的加密货币数量
        self.current_position = 0.0          # 当前持仓比例 [0, 1]
        
        # 时间步
        self.current_step = self.sequence_length
        self.max_steps = len(self.data) - 1
        
        # 性能追踪
        self.net_worth = self.initial_balance
        self.net_worth_history = [self.initial_balance]
        self.trades_history = []
        
        # 市场状态
        self.market_state = 'NORMAL'
        self.volatility_history = []
        
        # 风险追踪
        self.max_drawdown = 0.0
        self.peak_net_worth = self.initial_balance
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """
        获取当前观测
        
        Returns:
            状态序列 [sequence_length, feature_dim]
        """
        start_idx = self.current_step - self.sequence_length
        end_idx = self.current_step
        
        obs = self.features[start_idx:end_idx]
        return obs.astype(np.float32)
    
    def _get_current_price(self) -> float:
        """获取当前价格"""
        return self.data.iloc[self.current_step]['close']
    
    def _calculate_market_state(self) -> str:
        """
        计算市场状态
        对应论文 Section 6: Market state evaluation
        
        Returns:
            市场状态: 'NORMAL', 'HIGH_VOL', 'CRASH'
        """
        # 计算最近的波动率
        recent_returns = self.data['close'].pct_change().iloc[
            max(0, self.current_step-20):self.current_step
        ]
        volatility = recent_returns.std()
        
        self.volatility_history.append(volatility)
        
        # 检测闪崩
        recent_change = self.data['close'].pct_change(periods=10).iloc[self.current_step]
        
        if recent_change < -Config.FLASH_CRASH_THRESHOLD:
            return 'CRASH'
        elif volatility > Config.DEFENSE_MODE_VOLATILITY:
            return 'HIGH_VOL'
        else:
            return 'NORMAL'
    
    def _execute_trade(self, 
                      discrete_action: int, 
                      continuous_action: float) -> Tuple[float, Dict]:
        """
        执行交易
        对应论文 Section 5.1: 0.2% transaction friction cost
        
        Args:
            discrete_action: 0=Hold, 1=Buy, 2=Sell
            continuous_action: 持仓比例 [0, 1]
            
        Returns:
            (奖励, 交易信息字典)
        """
        current_price = self._get_current_price()
        prev_net_worth = self.net_worth
        
        # 防御模式检查（Section 6）
        if self.enable_defense_mode and self.market_state == 'CRASH':
            # 崩盘时强制减仓到30%
            continuous_action = min(continuous_action, Config.DEFENSE_MODE_POSITION_LIMIT)
        
        # 限制最大持仓
        continuous_action = np.clip(continuous_action, 0.0, self.max_position)
        
        trade_info = {
            'action': discrete_action,
            'position_size': continuous_action,
            'price': current_price,
            'cost': 0.0,
            'crypto_amount': 0.0
        }
        
        # ==================== 执行动作 ====================
        if discrete_action == 1:  # Buy
            # 计算买入金额
            available_balance = self.balance * continuous_action
            
            if available_balance > 0:
                # 扣除手续费
                total_cost = available_balance * (1 + self.transaction_fee)
                
                if self.balance >= total_cost:
                    # 执行买入
                    crypto_amount = available_balance / current_price
                    self.crypto_held += crypto_amount
                    self.balance -= total_cost
                    
                    trade_info['cost'] = total_cost
                    trade_info['crypto_amount'] = crypto_amount
                    
                    self.trades_history.append({
                        'step': self.current_step,
                        'action': 'BUY',
                        'price': current_price,
                        'amount': crypto_amount,
                        'cost': total_cost
                    })
        
        elif discrete_action == 2:  # Sell
            # 计算卖出数量
            crypto_to_sell = self.crypto_held * continuous_action
            
            if crypto_to_sell > 0:
                # 计算收入（扣除手续费）
                revenue = crypto_to_sell * current_price * (1 - self.transaction_fee)
                
                self.balance += revenue
                self.crypto_held -= crypto_to_sell
                
                trade_info['cost'] = -revenue  # 负数表示收入
                trade_info['crypto_amount'] = -crypto_to_sell
                
                self.trades_history.append({
                    'step': self.current_step,
                    'action': 'SELL',
                    'price': current_price,
                    'amount': crypto_to_sell,
                    'revenue': revenue
                })
        
        # discrete_action == 0 (Hold): 不执行交易
        
        # ==================== 更新净值 ====================
        self.net_worth = self.balance + self.crypto_held * current_price
        self.current_position = (self.crypto_held * current_price) / self.net_worth if self.net_worth > 0 else 0
        
        # ==================== 计算奖励 ====================
        # 论文 Section 5.1: ROI as reward
        reward = (self.net_worth - prev_net_worth) / prev_net_worth
        
        # 奖励缩放
        reward = reward * Config.REWARD_SCALING
        
        # 风险惩罚（可选）
        if self.enable_defense_mode:
            # 在高波动期持有大量仓位的惩罚
            if self.market_state == 'HIGH_VOL' and self.current_position > 0.7:
                reward -= Config.RISK_PENALTY_WEIGHT * self.current_position
        
        # ==================== 更新性能指标 ====================
        self.net_worth_history.append(self.net_worth)
        
        # 更新最大回撤
        if self.net_worth > self.peak_net_worth:
            self.peak_net_worth = self.net_worth
        
        current_drawdown = (self.peak_net_worth - self.net_worth) / self.peak_net_worth
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        return reward, trade_info
    
    def step(self, action: Tuple[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步
        
        Args:
            action: (离散动作, 连续动作)
            
        Returns:
            (observation, reward, done, info)
        """
        discrete_action, continuous_action = action
        
        # 如果continuous_action是数组，提取标量
        if isinstance(continuous_action, np.ndarray):
            continuous_action = continuous_action[0]
        
        # 评估市场状态
        self.market_state = self._calculate_market_state()
        
        # 执行交易
        reward, trade_info = self._execute_trade(discrete_action, continuous_action)
        
        # 前进一步
        self.current_step += 1
        
        # 检查是否结束
        done = self.current_step >= self.max_steps
        
        # 如果破产，提前结束
        if self.net_worth <= 0:
            done = True
            reward = -10.0  # 破产惩罚
        
        # 获取新状态
        obs = self._get_observation() if not done else np.zeros_like(self._get_observation())
        
        # 构建info字典
        info = {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'crypto_held': self.crypto_held,
            'position': self.current_position,
            'market_state': self.market_state,
            'max_drawdown': self.max_drawdown,
            'trade_info': trade_info,
            'step': self.current_step
        }
        
        return obs, reward, done, info
    
    def render(self, mode='human'):
        """渲染环境"""
        if mode == 'human':
            print(f"\n{'='*60}")
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Net Worth: ${self.net_worth:.2f} (初始: ${self.initial_balance:.2f})")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Crypto Held: {self.crypto_held:.6f}")
            print(f"Position: {self.current_position*100:.2f}%")
            print(f"Current Price: ${self._get_current_price():.2f}")
            print(f"Market State: {self.market_state}")
            print(f"Max Drawdown: {self.max_drawdown*100:.2f}%")
            print(f"ROI: {(self.net_worth/self.initial_balance - 1)*100:.2f}%")
            print(f"Total Trades: {len(self.trades_history)}")
            print(f"{'='*60}\n")
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        获取性能指标
        对应论文 Section 5.1: 评估维度
        
        Returns:
            指标字典
        """
        net_worth_array = np.array(self.net_worth_history)
        returns = np.diff(net_worth_array) / net_worth_array[:-1]
        
        # 论文评估指标
        metrics = {
            # 1. 绝对收益
            'roi': (self.net_worth - self.initial_balance) / self.initial_balance,
            
            # 2. 风险调整收益 (Sortino Ratio)
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            
            # 3. Sharpe Ratio
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252),
            
            # 4. 最大回撤
            'max_drawdown': self.max_drawdown,
            
            # 5. 波动率比
            'volatility_ratio': np.std(returns) / (np.std(self.data['close'].pct_change()) + 1e-8),
            
            # 6. 95% VaR
            'var_95': np.percentile(returns, 5),
            
            # 7. 胜率
            'win_rate': np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0,
            
            # 8. 盈亏比
            'profit_loss_ratio': self._calculate_profit_loss_ratio(returns),
            
            # 其他指标
            'total_trades': len(self.trades_history),
            'final_net_worth': self.net_worth,
            'total_return': self.net_worth / self.initial_balance - 1
        }
        
        return metrics
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """计算Sortino比率（只考虑下行风险）"""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return 0.0
        
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0
        
        return np.mean(returns) / downside_std * np.sqrt(252)
    
    def _calculate_profit_loss_ratio(self, returns: np.ndarray) -> float:
        """计算盈亏比"""
        profits = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(profits) == 0 or len(losses) == 0:
            return 0.0
        
        avg_profit = np.mean(profits)
        avg_loss = np.abs(np.mean(losses))
        
        return avg_profit / avg_loss if avg_loss > 0 else 0.0


def test_trading_env():
    """测试交易环境"""
    from data.dataloader import CryptoDataLoader
    from data.feature_engineer import FeatureEngineer
    
    print("=" * 60)
    print("测试交易环境 (Trading Environment Test)")
    print("=" * 60)
    
    # 加载数据
    loader = CryptoDataLoader()
    df = loader.merge_all_data('BTCUSDT')
    
    # 构建特征
    engineer = FeatureEngineer()
    features, _ = engineer.build_state_space(df)
    
    # 创建环境
    env = CryptoTradingEnv(df, features)
    
    print(f"\n环境配置:")
    print(f"  数据长度: {len(df)}")
    print(f"  特征维度: {features.shape}")
    print(f"  初始资金: ${env.initial_balance}")
    print(f"  交易费率: {env.transaction_fee*100}%")
    print(f"  序列长度: {env.sequence_length}")
    
    # 测试运行
    print(f"\n开始测试运行...")
    state = env.reset()
    print(f"初始状态形状: {state.shape}")
    
    total_reward = 0
    for i in range(10):
        # 随机动作
        discrete_action = np.random.randint(0, 3)
        continuous_action = np.random.random()
        action = (discrete_action, np.array([continuous_action]))
        
        state, reward, done, info = env.step(action)
        total_reward += reward
        
        if i % 5 == 0:
            env.render()
        
        if done:
            break
    
    print(f"\n测试完成!")
    print(f"  总步数: {i+1}")
    print(f"  累计奖励: {total_reward:.4f}")
    print(f"  最终净值: ${info['net_worth']:.2f}")
    
    # 获取性能指标
    metrics = env.get_performance_metrics()
    print(f"\n性能指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    test_trading_env()

