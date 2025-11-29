"""
数据加载模块
对应论文 Section 4: Dataset and Preprocessing Description

功能：
1. 加载多源加密货币交易数据（分钟级K线）
2. 整合技术指标、链上数据、情感数据
3. 数据清洗和缺失值处理
4. 时间序列分割（训练集/测试集）
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

from config import Config


class CryptoDataLoader:
    """
    加密货币数据加载器
    对应论文 Section 4: 整合多源数据流
    """
    
    def __init__(self, 
                 data_dir: str = Config.DATA_DIR,
                 trading_pairs: List[str] = None):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录
            trading_pairs: 交易对列表（默认使用配置文件）
        """
        self.data_dir = data_dir
        self.trading_pairs = trading_pairs or Config.TRADING_PAIRS
        self.imputer = SimpleImputer(strategy='median')  # 论文：中位数填充
        
    def load_price_data(self, symbol: str) -> pd.DataFrame:
        """
        加载价格数据（OHLCV）
        
        Args:
            symbol: 交易对符号，如 'BTCUSDT'
            
        Returns:
            包含OHLCV的DataFrame
        """
        file_path = os.path.join(self.data_dir, f'{symbol}_1m.csv')
        
        try:
            df = pd.read_csv(file_path, parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # 确保包含必需列
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")
            
            return df
            
        except FileNotFoundError:
            print(f"警告: 未找到 {symbol} 数据文件，生成模拟数据")
            return self._generate_mock_price_data(symbol)
    
    def load_onchain_data(self, symbol: str) -> pd.DataFrame:
        """
        加载链上数据
        对应论文 Section 3.2: on-chain transaction analytics
        
        包括：
        - 大额转账监控（>100 BTC）
        - 交易所流入流出
        - 矿工持仓变化
        - MVRZ-Z指数
        
        Args:
            symbol: 交易对符号
            
        Returns:
            链上指标DataFrame
        """
        file_path = os.path.join(self.data_dir, f'{symbol}_onchain.csv')
        
        try:
            df = pd.read_csv(file_path, parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
            return df
        except FileNotFoundError:
            print(f"警告: 未找到 {symbol} 链上数据，生成模拟数据")
            return self._generate_mock_onchain_data(symbol)
    
    def load_sentiment_data(self, symbol: str) -> pd.DataFrame:
        """
        加载市场情感数据
        对应论文 Section 3.2: social media sentiment indices
        
        使用FinBERT预训练模型提取Twitter/Reddit情感
        
        Args:
            symbol: 交易对符号
            
        Returns:
            情感指标DataFrame
        """
        file_path = os.path.join(self.data_dir, f'{symbol}_sentiment.csv')
        
        try:
            df = pd.read_csv(file_path, parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
            return df
        except FileNotFoundError:
            print(f"警告: 未找到 {symbol} 情感数据，生成模拟数据")
            return self._generate_mock_sentiment_data(symbol)
    
    def merge_all_data(self, symbol: str) -> pd.DataFrame:
        """
        整合所有数据源
        对应论文 Section 3.2: 三维状态空间构建
        
        Args:
            symbol: 交易对符号
            
        Returns:
            整合后的完整DataFrame
        """
        # 1. 加载价格数据
        price_df = self.load_price_data(symbol)
        
        # 2. 加载链上数据
        onchain_df = self.load_onchain_data(symbol)
        
        # 3. 加载情感数据
        sentiment_df = self.load_sentiment_data(symbol)
        
        # 4. 按时间索引合并
        merged_df = price_df.copy()
        
        # 合并链上数据
        for col in onchain_df.columns:
            merged_df[f'onchain_{col}'] = onchain_df[col]
        
        # 合并情感数据
        for col in sentiment_df.columns:
            merged_df[f'sentiment_{col}'] = sentiment_df[col]
        
        # 5. 处理缺失值（论文 Section 4：中位数填充）
        merged_df = self._handle_missing_values(merged_df)
        
        print(f"成功加载 {symbol} 数据: {len(merged_df)} 行, {len(merged_df.columns)} 列")
        return merged_df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理缺失值
        对应论文 Section 4: SimpleImputer with median strategy
        
        Args:
            df: 输入DataFrame
            
        Returns:
            处理后的DataFrame
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        return df
    
    def train_test_split(self, 
                        df: pd.DataFrame, 
                        train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        时间序列分割
        对应论文 Section 4: 80% training / 20% testing
        
        Args:
            df: 完整数据集
            train_ratio: 训练集比例
            
        Returns:
            (训练集, 测试集)
        """
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        print(f"数据分割完成:")
        print(f"  训练集: {len(train_df)} 条 ({train_df.index[0]} 至 {train_df.index[-1]})")
        print(f"  测试集: {len(test_df)} 条 ({test_df.index[0]} 至 {test_df.index[-1]})")
        
        return train_df, test_df
    
    # ==================== 模拟数据生成函数（用于演示） ====================
    
    def _generate_mock_price_data(self, symbol: str, n_samples: int = 10000) -> pd.DataFrame:
        """生成模拟价格数据"""
        np.random.seed(Config.SEED)
        
        dates = pd.date_range(
            start=Config.DATA_START_DATE, 
            periods=n_samples, 
            freq='1min'
        )
        
        # 模拟价格随机游走 + 趋势
        price_base = 30000 if 'BTC' in symbol else 2000 if 'ETH' in symbol else 100
        returns = np.random.normal(0.0001, 0.01, n_samples)
        prices = price_base * np.exp(np.cumsum(returns))
        
        # 生成OHLCV
        data = {
            'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
            'close': prices,
            'volume': np.random.exponential(1000, n_samples)
        }
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    def _generate_mock_onchain_data(self, symbol: str, n_samples: int = 10000) -> pd.DataFrame:
        """生成模拟链上数据"""
        np.random.seed(Config.SEED + 1)
        
        dates = pd.date_range(
            start=Config.DATA_START_DATE, 
            periods=n_samples, 
            freq='1min'
        )
        
        data = {
            'large_transfers': np.random.exponential(50, n_samples),  # 大额转账
            'exchange_inflow': np.random.exponential(30, n_samples),  # 交易所流入
            'exchange_outflow': np.random.exponential(30, n_samples), # 交易所流出
            'active_addresses': np.random.randint(5000, 20000, n_samples),  # 活跃地址
            'hash_rate': np.random.normal(200, 20, n_samples),  # 算力（仅BTC）
            'mvrv_z': np.random.normal(0, 1, n_samples)  # MVRV-Z指数
        }
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    def _generate_mock_sentiment_data(self, symbol: str, n_samples: int = 10000) -> pd.DataFrame:
        """生成模拟情感数据"""
        np.random.seed(Config.SEED + 2)
        
        dates = pd.date_range(
            start=Config.DATA_START_DATE, 
            periods=n_samples, 
            freq='1min'
        )
        
        # 模拟FinBERT情感得分
        data = {
            'twitter_sentiment': np.random.normal(0, 1, n_samples),  # Twitter情感
            'reddit_sentiment': np.random.normal(0, 1, n_samples),   # Reddit情感
            'news_sentiment': np.random.normal(0, 1, n_samples),     # 新闻情感
            'social_volume': np.random.exponential(1000, n_samples), # 社交媒体量
            'fear_greed_index': np.random.randint(0, 100, n_samples) # 恐惧贪婪指数
        }
        
        df = pd.DataFrame(data, index=dates)
        return df


class MultiAssetDataLoader:
    """
    多资产数据加载器
    用于加载多个交易对的数据并进行组合
    """
    
    def __init__(self, trading_pairs: List[str] = None):
        """
        初始化多资产加载器
        
        Args:
            trading_pairs: 交易对列表
        """
        self.trading_pairs = trading_pairs or Config.TRADING_PAIRS
        self.loaders = {
            pair: CryptoDataLoader(trading_pairs=[pair]) 
            for pair in self.trading_pairs
        }
    
    def load_all_pairs(self) -> Dict[str, pd.DataFrame]:
        """
        加载所有交易对的数据
        
        Returns:
            {symbol: DataFrame} 字典
        """
        all_data = {}
        for pair in self.trading_pairs:
            try:
                df = self.loaders[pair].merge_all_data(pair)
                all_data[pair] = df
            except Exception as e:
                print(f"加载 {pair} 失败: {e}")
        
        return all_data
    
    def create_portfolio_dataset(self, 
                                correlation_threshold: float = 0.7) -> pd.DataFrame:
        """
        创建组合数据集（用于多资产策略）
        
        Args:
            correlation_threshold: 相关性阈值，用于资产选择
            
        Returns:
            组合数据集
        """
        all_data = self.load_all_pairs()
        
        # 提取收益率计算相关性
        returns = {}
        for pair, df in all_data.items():
            returns[pair] = df['close'].pct_change()
        
        returns_df = pd.DataFrame(returns)
        corr_matrix = returns_df.corr()
        
        print("资产相关性矩阵:")
        print(corr_matrix)
        
        # 这里可以基于相关性进行资产筛选
        # 暂时返回第一个资产的数据作为示例
        return all_data[self.trading_pairs[0]]


def test_dataloader():
    """测试数据加载器"""
    print("=" * 60)
    print("测试数据加载器 (Data Loader Test)")
    print("=" * 60)
    
    # 单资产加载
    loader = CryptoDataLoader()
    df = loader.merge_all_data('BTCUSDT')
    
    print(f"\n数据形状: {df.shape}")
    print(f"\n前5行数据:")
    print(df.head())
    
    print(f"\n数据统计:")
    print(df.describe())
    
    # 训练测试分割
    train_df, test_df = loader.train_test_split(df)
    
    # 多资产加载
    print("\n" + "=" * 60)
    print("测试多资产加载")
    print("=" * 60)
    
    multi_loader = MultiAssetDataLoader()
    all_data = multi_loader.load_all_pairs()
    
    print(f"\n成功加载 {len(all_data)} 个交易对")
    for pair, df in all_data.items():
        print(f"  {pair}: {df.shape}")


if __name__ == "__main__":
    test_dataloader()

