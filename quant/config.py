"""
全局配置文件
对应论文 Section 5.1: 实验设置
包含所有超参数、路径配置和模型参数
"""

import torch

class Config:
    # ==================== 数据配置 (Section 4) ====================
    # 数据时间范围：2018-2023（论文实验周期）
    DATA_START_DATE = '2018-01-01'
    DATA_END_DATE = '2023-12-31'
    
    # 数据频率：1分钟K线
    DATA_FREQUENCY = '1min'
    
    # 交易对
    TRADING_PAIRS = ['BTCUSDT', '1INCHUSDT', 'ETHUSDT']
    
    # 数据源配置
    DATA_DIR = './data/raw/'
    PROCESSED_DATA_DIR = './data/processed/'
    
    # ==================== 特征工程配置 (Section 3.2) ====================
    # 自适应滑动窗口参数
    VOLATILITY_THRESHOLD = 0.02  # 波动率阈值
    HIGH_VOL_WINDOW = 15         # 高波动期窗口（分钟）
    LOW_VOL_WINDOW = 60          # 平稳期窗口（分钟）
    
    # 技术指标参数
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    
    # 链上数据阈值（Section 3.2：>100 BTC触发警报）
    WHALE_ALERT_THRESHOLD = 100  # BTC
    
    # PCA降维参数（Section 3.3：127维->21维）
    PCA_N_COMPONENTS = 21
    PCA_VARIANCE_RETAINED = 0.85  # 保留85%方差
    
    # 序列长度
    SEQUENCE_LENGTH = 60  # 60个时间步
    
    # ==================== 模型架构配置 (Section 3.3 & 3.4) ====================
    # 共享编码器参数
    ENCODER_HIDDEN_DIM = 64      # 论文：64维潜在表示
    ENCODER_INPUT_DIM = 21       # PCA降维后维度
    
    # TCN参数（Section 3.3：膨胀因子[1,2,3]）
    TCN_CHANNELS = [32, 32, 32]
    TCN_KERNEL_SIZE = 3
    TCN_DILATION_RATES = [1, 2, 3]
    
    # GRU参数
    GRU_HIDDEN_DIM = 64
    GRU_NUM_LAYERS = 1
    
    # Attention参数
    ATTENTION_DIM = 1
    
    # ==================== Rainbow DQN配置 (Section 3.4) ====================
    # 离散动作空间：Buy/Sell/Hold
    DISCRETE_ACTIONS = 3
    
    # Rainbow DQN特性
    USE_DOUBLE_DQN = True
    USE_DUELING = True          # 论文使用Dueling架构
    USE_NOISY_NET = True        # 论文：参数噪声注入
    USE_PRIORITIZED_REPLAY = True  # 论文：优先经验回放
    USE_N_STEP = True           # 论文：n-step bootstrap (n=5)
    USE_DISTRIBUTIONAL = True   # 论文：51-atom支持
    
    # Noisy Net参数（论文：初始σ=0.17）
    NOISY_SIGMA_INIT = 0.17
    
    # N-step参数
    N_STEP = 5
    
    # Distributional RL参数
    V_MIN = -10
    V_MAX = 10
    N_ATOMS = 51
    
    # Prioritized Replay参数（论文：α=0.7, β=0.5）
    PRIORITY_ALPHA = 0.7
    PRIORITY_BETA_START = 0.5
    PRIORITY_BETA_FRAMES = 100000
    
    # ==================== TD3配置 (Section 3.4) ====================
    # 连续动作空间：持仓比例 0-100%
    CONTINUOUS_ACTION_DIM = 1
    CONTINUOUS_ACTION_MIN = 0.0
    CONTINUOUS_ACTION_MAX = 1.0
    
    # TD3特性
    POLICY_NOISE = 0.2          # 论文：ε~N(0,0.2)
    NOISE_CLIP = 0.5
    POLICY_FREQ = 2             # 论文：2:1策略-评论家更新比
    
    # Actor-Critic网络参数
    ACTOR_HIDDEN_DIM = 128      # 论文：128单元MLP
    CRITIC_HIDDEN_DIM = 128
    ACTOR_LR = 3e-4
    CRITIC_LR = 3e-4
    
    # ==================== SAC配置（可选，Section 5.2提到SAC） ====================
    SAC_ALPHA = 0.2             # 熵正则化系数
    SAC_AUTO_ENTROPY = True     # 自动调整熵
    SAC_TARGET_ENTROPY = -1.0
    
    # ==================== 训练配置 (Section 5.1) ====================
    # 基础训练参数
    BATCH_SIZE = 64
    BUFFER_SIZE = 100000
    GAMMA = 0.99                # 折扣因子
    TAU = 0.005                 # 软更新系数
    
    # 学习率
    DQN_LR = 1e-4
    
    # 探索策略
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 10000
    
    # 训练周期
    NUM_EPISODES = 1000
    MAX_STEPS_PER_EPISODE = 5000
    
    # 更新频率
    UPDATE_FREQUENCY = 4        # 每4步更新一次
    TARGET_UPDATE_FREQUENCY = 1000  # 每1000步更新目标网络
    
    # Early stopping
    PATIENCE = 50
    MIN_DELTA = 0.001
    
    # ==================== 环境配置 (Section 5.1 & 6) ====================
    # 初始资金
    INITIAL_BALANCE = 10000.0   # USDT
    
    # 交易成本（论文：0.2%摩擦成本）
    TRANSACTION_FEE_RATE = 0.002  # 0.2%
    
    # 风险管理
    MAX_POSITION_SIZE = 1.0     # 最大100%仓位
    MIN_POSITION_SIZE = 0.0     # 最小0%仓位
    
    # 奖励设计参数
    REWARD_SCALING = 0.01
    RISK_PENALTY_WEIGHT = 0.1   # 风险惩罚权重
    
    # ==================== 评估指标配置 (Section 5.1) ====================
    # 论文提到的评估维度
    METRICS = [
        'roi',              # 绝对收益率
        'sortino_ratio',    # Sortino比率
        'sharpe_ratio',     # Sharpe比率
        'max_drawdown',     # 最大回撤
        'volatility_ratio', # 波动率比
        'var_95',          # 95% VaR
        'win_rate',        # 胜率
        'profit_loss_ratio' # 盈亏比
    ]
    
    # 回测窗口
    BACKTEST_WINDOW = 90  # 3个月信号衰减测试
    
    # ==================== 异常检测配置 (Section 6) ====================
    # 闪崩前兆检测
    FLASH_CRASH_THRESHOLD = 0.05  # 5%快速下跌
    FLASH_CRASH_WINDOW = 10       # 10分钟内
    
    # 巨鲸地址监控
    WHALE_TX_THRESHOLD = 100      # BTC
    EXCHANGE_INFLOW_ALERT = 500   # BTC流入交易所
    
    # ==================== 市场状态自适应配置 (Section 6) ====================
    # Meta-RL市场状态评估
    MARKET_STATE_EVAL_INTERVAL = 6  # 小时
    
    # 市场状态类型
    MARKET_STATES = [
        'bull',         # 牛市
        'bear',         # 熊市
        'sideways',     # 震荡
        'high_vol',     # 高波动
        'crash'         # 崩盘
    ]
    
    # 防御模式阈值（Section 6：波动超阈值切换保守策略）
    DEFENSE_MODE_VOLATILITY = 0.05
    DEFENSE_MODE_POSITION_LIMIT = 0.3  # 压缩至30%仓位
    
    # ==================== 设备配置 ====================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42
    
    # ==================== 日志与保存 ====================
    LOG_DIR = './logs/'
    MODEL_SAVE_DIR = './saved_models/'
    RESULT_SAVE_DIR = './results/'
    
    LOG_INTERVAL = 100          # 每100步记录一次
    SAVE_INTERVAL = 1000        # 每1000步保存一次模型
    
    # ==================== 可视化配置 ====================
    PLOT_INTERVAL = 500
    PLOT_METRICS = ['portfolio_value', 'roi', 'sharpe_ratio']
    
    @staticmethod
    def print_config():
        """打印当前配置"""
        print("=" * 60)
        print("Configuration Summary (Based on Paper Section 5.1)")
        print("=" * 60)
        print(f"Data Period: {Config.DATA_START_DATE} to {Config.DATA_END_DATE}")
        print(f"Trading Pairs: {Config.TRADING_PAIRS}")
        print(f"Sequence Length: {Config.SEQUENCE_LENGTH}")
        print(f"PCA Components: {Config.PCA_N_COMPONENTS}")
        print(f"Encoder Hidden Dim: {Config.ENCODER_HIDDEN_DIM}")
        print(f"Discrete Actions: {Config.DISCRETE_ACTIONS}")
        print(f"Transaction Fee: {Config.TRANSACTION_FEE_RATE * 100}%")
        print(f"Initial Balance: ${Config.INITIAL_BALANCE}")
        print(f"Batch Size: {Config.BATCH_SIZE}")
        print(f"Learning Rate (DQN): {Config.DQN_LR}")
        print(f"Device: {Config.DEVICE}")
        print("=" * 60)

if __name__ == "__main__":
    Config.print_config()

