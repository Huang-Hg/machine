"""
特征工程模块
对应论文 Section 3.2 & 3.3: Multi-modal Data Fusion & Temporal-Spatial Feature Enhancement

核心创新:
1. 自适应滑动窗口归一化 (Adaptive Sliding Window Normalization)
2. 三维状态空间构建 (技术指标 + 链上数据 + 情感数据)
3. PCA降维 with Varimax rotation (127维 -> 21维)
4. 时间交叉编码 (Temporal Cross-coding)
5. 异常检测机制 (Anomaly Detection)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

from config import Config


class FeatureEngineer:
    """
    特征工程器
    实现论文 Section 3.2 的核心创新机制
    """
    
    def __init__(self, 
                 volatility_threshold: float = Config.VOLATILITY_THRESHOLD,
                 high_vol_window: int = Config.HIGH_VOL_WINDOW,
                 low_vol_window: int = Config.LOW_VOL_WINDOW):
        """
        初始化特征工程器
        
        Args:
            volatility_threshold: 波动率阈值，用于判断市场状态
            high_vol_window: 高波动期窗口大小（分钟）
            low_vol_window: 低波动期窗口大小（分钟）
        """
        self.vol_threshold = volatility_threshold
        self.high_vol_window = high_vol_window
        self.low_vol_window = low_vol_window
        
        self.scaler = StandardScaler()
        self.pca = None  # 延迟初始化
        
    def compute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        对应论文 Section 3.2: Technical Layer (RSI, MACD)
        
        Args:
            df: 包含OHLCV的DataFrame
            
        Returns:
            添加技术指标后的DataFrame
        """
        df = df.copy()
        
        # 1. RSI (Relative Strength Index)
        # 论文 Section 3.2: RSI捕获价格趋势动量
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=Config.RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=Config.RSI_PERIOD).mean()
        rs = gain / (loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 2. MACD (Moving Average Convergence Divergence)
        exp1 = df['close'].ewm(span=Config.MACD_FAST, adjust=False).mean()
        exp2 = df['close'].ewm(span=Config.MACD_SLOW, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=Config.MACD_SIGNAL, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 3. Bollinger Bands
        # 论文 Section 5.2 提到Bollinger策略
        bb_period = 20
        df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
        bb_std = df['close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # 4. Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_60'] = df['close'].rolling(window=60).mean()
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # 5. Momentum Indicators
        df['momentum'] = df['close'].pct_change(periods=10)
        df['roc'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
        
        # 6. Volatility Indicators
        df['atr'] = self._calculate_atr(df, period=14)
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        
        # 7. Volume Indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-8)
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算ATR (Average True Range)"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def process_onchain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理链上特征
        对应论文 Section 3.2: On-chain Analysis Module
        
        包括:
        - 巨鲸地址监控 (>100 BTC threshold)
        - 交易所流入流出分析
        - MVRV-Z指数
        
        Args:
            df: 包含链上原始数据的DataFrame
            
        Returns:
            添加处理后链上特征的DataFrame
        """
        df = df.copy()
        
        # 1. 巨鲸警报 (Whale Alert)
        # 论文 Section 3.2: >100 BTC触发高频波动监控
        if 'onchain_large_transfers' in df.columns:
            df['whale_alert'] = (df['onchain_large_transfers'] > Config.WHALE_ALERT_THRESHOLD).astype(int)
            df['whale_7d_ma'] = df['onchain_large_transfers'].rolling(window=7*24*60).mean()  # 7天均值
        
        # 2. 交易所净流入
        if 'onchain_exchange_inflow' in df.columns and 'onchain_exchange_outflow' in df.columns:
            df['exchange_net_flow'] = df['onchain_exchange_inflow'] - df['onchain_exchange_outflow']
            df['exchange_flow_ratio'] = df['onchain_exchange_inflow'] / (df['onchain_exchange_outflow'] + 1e-8)
        
        # 3. 活跃地址动量
        if 'onchain_active_addresses' in df.columns:
            df['active_addr_momentum'] = df['onchain_active_addresses'].pct_change(periods=60)
        
        # 4. MVRV-Z Score (市场价值与实现价值比)
        if 'onchain_mvrv_z' in df.columns:
            df['mvrv_z_signal'] = np.where(df['onchain_mvrv_z'] > 2, 1,  # 过热
                                           np.where(df['onchain_mvrv_z'] < -1, -1, 0))  # 低估
        
        return df
    
    def process_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理情感特征
        对应论文 Section 3.2: Sentiment Dimension (NLP-based sentiment indices)
        
        Args:
            df: 包含情感原始数据的DataFrame
            
        Returns:
            添加处理后情感特征的DataFrame
        """
        df = df.copy()
        
        # 1. 综合情感得分 (加权平均)
        sentiment_cols = ['sentiment_twitter_sentiment', 
                         'sentiment_reddit_sentiment', 
                         'sentiment_news_sentiment']
        
        available_cols = [col for col in sentiment_cols if col in df.columns]
        if available_cols:
            df['sentiment_composite'] = df[available_cols].mean(axis=1)
            df['sentiment_std'] = df[available_cols].std(axis=1)
        
        # 2. 情感动量
        if 'sentiment_composite' in df.columns:
            df['sentiment_momentum'] = df['sentiment_composite'].diff(periods=60)
        
        # 3. 恐惧贪婪指数信号
        if 'sentiment_fear_greed_index' in df.columns:
            df['fear_greed_signal'] = np.where(df['sentiment_fear_greed_index'] > 75, 1,  # 极度贪婪
                                               np.where(df['sentiment_fear_greed_index'] < 25, -1, 0))  # 极度恐惧
        
        # 4. 社交媒体活跃度
        if 'sentiment_social_volume' in df.columns:
            df['social_volume_ma'] = df['sentiment_social_volume'].rolling(window=60).mean()
            df['social_volume_ratio'] = df['sentiment_social_volume'] / (df['social_volume_ma'] + 1e-8)
        
        return df
    
    def adaptive_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        自适应滑动窗口归一化
        对应论文 Section 3.2 核心创新点: Adaptive Sliding Window Standardization
        
        创新机制:
        - 根据波动率动态切换窗口大小 (15分钟 vs 60分钟)
        - 对交易量使用Tanh压缩 (处理10^5量级)
        - 对技术指标使用Z-score标准化 (0-100范围)
        
        Args:
            df: 输入DataFrame
            
        Returns:
            归一化后的DataFrame
        """
        df = df.copy()
        
        # 计算当前波动率
        returns = df['close'].pct_change()
        current_volatility = returns.rolling(window=20).std().iloc[-1]
        
        # 动态选择窗口大小
        if current_volatility > self.vol_threshold:
            window_size = self.high_vol_window
            market_regime = 'HIGH_VOLATILITY'
        else:
            window_size = self.low_vol_window
            market_regime = 'LOW_VOLATILITY'
        
        print(f"市场状态: {market_regime}, 当前波动率: {current_volatility:.4f}, 窗口大小: {window_size}分钟")
        
        # 1. 交易量非线性压缩 (Tanh transformation)
        # 论文: 处理10^5量级的链上交易量
        volume_cols = [col for col in df.columns if 'volume' in col.lower() or 'flow' in col.lower()]
        for col in volume_cols:
            rolling_mean = df[col].rolling(window=window_size, min_periods=1).mean()
            df[f'{col}_norm'] = np.tanh(df[col] / (rolling_mean + 1e-8))
        
        # 2. 价格和技术指标 Z-score标准化
        # 论文: 处理0-100范围的归一化指标
        indicator_cols = ['close', 'rsi', 'macd', 'bb_width', 'momentum', 'volatility']
        for col in indicator_cols:
            if col in df.columns:
                rolling_mean = df[col].rolling(window=window_size, min_periods=1).mean()
                rolling_std = df[col].rolling(window=window_size, min_periods=1).std() + 1e-8
                df[f'{col}_norm'] = (df[col] - rolling_mean) / rolling_std
        
        # 3. 链上数据归一化
        onchain_cols = [col for col in df.columns if 'onchain_' in col]
        for col in onchain_cols:
            if df[col].dtype in [np.float64, np.int64]:
                rolling_mean = df[col].rolling(window=window_size, min_periods=1).mean()
                rolling_std = df[col].rolling(window=window_size, min_periods=1).std() + 1e-8
                df[f'{col}_norm'] = (df[col] - rolling_mean) / rolling_std
        
        # 4. 情感数据归一化 (通常已是[-1, 1]范围)
        sentiment_cols = [col for col in df.columns if 'sentiment_' in col and 'signal' not in col]
        for col in sentiment_cols:
            if df[col].dtype in [np.float64, np.int64]:
                # 简单clip到[-3, 3]范围再归一化
                df[f'{col}_norm'] = np.clip(df[col], -3, 3) / 3.0
        
        return df.dropna()
    
    def temporal_cross_coding(self, df: pd.DataFrame, lags: list = [1, 2, 3]) -> pd.DataFrame:
        """
        时间交叉编码
        对应论文 Section 3.3: Temporal Cross-coding
        
        结合滞后特征交互 (e.g., RSI_{t-2} × Volume_{t-1})
        
        Args:
            df: 输入DataFrame
            lags: 滞后期列表
            
        Returns:
            添加交叉特征的DataFrame
        """
        df = df.copy()
        
        # 选择关键特征进行交叉
        key_features = ['rsi_norm', 'volume_norm', 'volatility_norm', 'macd_norm']
        available_features = [f for f in key_features if f in df.columns]
        
        # 生成滞后特征交互
        for feat in available_features:
            for lag in lags:
                # 滞后特征
                df[f'{feat}_lag{lag}'] = df[feat].shift(lag)
                
                # 交叉特征: RSI_{t-lag} × Volume_{t}
                if feat != 'volume_norm' and 'volume_norm' in df.columns:
                    df[f'{feat}_lag{lag}_x_volume'] = df[f'{feat}_lag{lag}'] * df['volume_norm']
        
        return df.dropna()
    
    def dimensionality_reduction(self, 
                                X: np.ndarray, 
                                n_components: int = Config.PCA_N_COMPONENTS,
                                variance_threshold: float = Config.PCA_VARIANCE_RETAINED,
                                fit: bool = True) -> np.ndarray:
        """
        PCA降维
        对应论文 Section 3.3: PCA with Varimax rotation
        
        将高维特征 (127维) 投影到低维主成分 (21维)
        保留85%的累积方差
        
        Args:
            X: 输入特征矩阵 [N, D]
            n_components: 主成分数量
            variance_threshold: 方差保留阈值
            fit: 是否拟合PCA（训练时True，测试时False）
            
        Returns:
            降维后的特征矩阵 [N, n_components]
        """
        if X.shape[1] <= n_components:
            print(f"特征维度({X.shape[1]})已小于目标维度({n_components})，跳过PCA")
            return X
        
        if fit or self.pca is None:
            self.pca = PCA(n_components=n_components)
            X_reduced = self.pca.fit_transform(X)
            
            # 输出降维信息
            explained_var = np.sum(self.pca.explained_variance_ratio_)
            print(f"PCA降维: {X.shape[1]}维 -> {n_components}维")
            print(f"累积方差解释率: {explained_var:.2%}")
            
            if explained_var < variance_threshold:
                print(f"警告: 方差解释率({explained_var:.2%})低于阈值({variance_threshold:.2%})")
        else:
            X_reduced = self.pca.transform(X)
        
        return X_reduced
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        异常检测
        对应论文 Section 6: 闪崩前兆检测、巨鲸交易监控
        
        Args:
            df: 输入DataFrame
            
        Returns:
            添加异常标记的DataFrame
        """
        df = df.copy()
        
        # 1. 闪崩检测 (Flash Crash Detection)
        # 论文 Section 6: 10分钟内5%快速下跌
        price_change = df['close'].pct_change(periods=Config.FLASH_CRASH_WINDOW)
        df['flash_crash_alert'] = (price_change < -Config.FLASH_CRASH_THRESHOLD).astype(int)
        
        # 2. 价格异常波动检测 (使用Z-score)
        returns = df['close'].pct_change()
        df['return_zscore'] = zscore(returns.fillna(0))
        df['extreme_move'] = (np.abs(df['return_zscore']) > 3).astype(int)
        
        # 3. 交易量异常
        if 'volume' in df.columns:
            df['volume_zscore'] = zscore(df['volume'].fillna(0))
            df['volume_spike'] = (df['volume_zscore'] > 3).astype(int)
        
        # 4. 链上异常 (巨鲸交易)
        if 'whale_alert' in df.columns:
            # 论文: 巨鲸地址交易 + 交易所流入 组合信号
            if 'exchange_net_flow' in df.columns:
                df['manipulation_risk'] = (
                    (df['whale_alert'] == 1) & 
                    (df['exchange_net_flow'] > Config.EXCHANGE_INFLOW_ALERT)
                ).astype(int)
        
        return df
    
    def build_state_space(self, df: pd.DataFrame) -> Tuple[np.ndarray, list]:
        """
        构建完整状态空间
        对应论文 Section 3.2: 三维状态空间
        
        整合流程:
        1. 技术指标计算
        2. 链上数据处理
        3. 情感数据处理
        4. 自适应归一化
        5. 时间交叉编码
        6. PCA降维
        7. 异常检测
        
        Args:
            df: 原始数据DataFrame
            
        Returns:
            (状态矩阵, 特征名列表)
        """
        print("\n" + "=" * 60)
        print("构建状态空间 (Building State Space)")
        print("=" * 60)
        
        # Step 1: 计算技术指标
        print("Step 1: 计算技术指标...")
        df = self.compute_technical_indicators(df)
        
        # Step 2: 处理链上特征
        print("Step 2: 处理链上特征...")
        df = self.process_onchain_features(df)
        
        # Step 3: 处理情感特征
        print("Step 3: 处理情感特征...")
        df = self.process_sentiment_features(df)
        
        # Step 4: 自适应归一化
        print("Step 4: 自适应归一化...")
        df = self.adaptive_normalization(df)
        
        # Step 5: 时间交叉编码
        print("Step 5: 时间交叉编码...")
        df = self.temporal_cross_coding(df)
        
        # Step 6: 异常检测
        print("Step 6: 异常检测...")
        df = self.detect_anomalies(df)
        
        # 选择归一化后的特征
        feature_cols = [col for col in df.columns if '_norm' in col or 'alert' in col or 'signal' in col]
        feature_cols = [col for col in feature_cols if df[col].dtype in [np.float64, np.int64]]
        
        print(f"\n原始特征数量: {len(feature_cols)}")
        
        # 转换为numpy数组
        X = df[feature_cols].values
        
        # Step 7: PCA降维 (如果特征数>目标维度)
        if X.shape[1] > Config.PCA_N_COMPONENTS:
            print(f"Step 7: PCA降维...")
            X = self.dimensionality_reduction(X, n_components=Config.PCA_N_COMPONENTS)
            feature_cols = [f'PC{i+1}' for i in range(X.shape[1])]
        
        print(f"最终状态空间维度: {X.shape}")
        print("=" * 60 + "\n")
        
        return X, feature_cols


def test_feature_engineer():
    """测试特征工程器"""
    from data.dataloader import CryptoDataLoader
    
    print("=" * 60)
    print("测试特征工程器 (Feature Engineer Test)")
    print("=" * 60)
    
    # 加载数据
    loader = CryptoDataLoader()
    df = loader.merge_all_data('BTCUSDT')
    
    # 创建特征工程器
    engineer = FeatureEngineer()
    
    # 构建状态空间
    X, feature_names = engineer.build_state_space(df)
    
    print(f"\n状态空间形状: {X.shape}")
    print(f"特征名称: {feature_names[:10]}...")  # 只显示前10个
    
    print(f"\n状态统计:")
    print(f"  均值: {X.mean(axis=0)[:5]}")
    print(f"  标准差: {X.std(axis=0)[:5]}")
    print(f"  最小值: {X.min(axis=0)[:5]}")
    print(f"  最大值: {X.max(axis=0)[:5]}")


if __name__ == "__main__":
    test_feature_engineer()

