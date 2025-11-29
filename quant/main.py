"""
主入口文件
提供统一的命令行接口用于训练、测试和评估

使用示例:
    python main.py --mode train --symbol BTCUSDT --episodes 1000
    python main.py --mode test --symbol BTCUSDT --model_path saved_models/BTCUSDT_best_model.pt
    python main.py --mode backtest --symbol BTCUSDT
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from trainer.trainer import Trainer
from data.dataloader import CryptoDataLoader
from data.feature_engineer import FeatureEngineer
from models.hybrid_agent import HybridAgent
from quant.trader import CryptoTradingEnv
from utils.metrics import PerformanceMetrics


def train_model(args):
    """
    训练模型
    
    Args:
        args: 命令行参数
    """
    print("\n" + "="*70)
    print("模式: 训练 (MODE: TRAINING)")
    print("="*70)
    
    # 创建训练器
    trainer = Trainer(
        symbol=args.symbol,
        train_ratio=args.train_ratio,
        device=Config.DEVICE
    )
    
    # 开始训练
    trainer.train(
        num_episodes=args.episodes,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        early_stopping_patience=args.patience
    )
    
    print("\n✓ 训练完成！")


def test_model(args):
    """
    测试模型
    
    Args:
        args: 命令行参数
    """
    print("\n" + "="*70)
    print("模式: 测试 (MODE: TESTING)")
    print("="*70)
    
    # 加载数据
    print("\n1. 加载数据...")
    data_loader = CryptoDataLoader()
    raw_data = data_loader.merge_all_data(args.symbol)
    
    # 特征工程
    print("\n2. 特征工程...")
    feature_engineer = FeatureEngineer()
    features, _ = feature_engineer.build_state_space(raw_data)
    
    # 创建测试环境
    print("\n3. 创建测试环境...")
    test_env = CryptoTradingEnv(raw_data, features)
    
    # 创建智能体并加载模型
    print(f"\n4. 加载模型: {args.model_path}")
    agent = HybridAgent(
        input_dim=features.shape[1],
        device=Config.DEVICE
    )
    agent.load(args.model_path)
    
    # 测试
    print("\n5. 开始测试...")
    state = test_env.reset()
    done = False
    episode_reward = 0
    step = 0
    
    while not done:
        # 选择动作（不探索）
        state_tensor = torch.FloatTensor(state)
        discrete_action, continuous_action = agent.select_hybrid_action(
            state_tensor,
            epsilon=0.0,
            noise=0.0
        )
        
        # 执行动作
        next_state, reward, done, info = test_env.step((discrete_action, continuous_action))
        
        episode_reward += reward
        state = next_state
        step += 1
        
        # 定期显示
        if step % 100 == 0:
            print(f"  Step {step}: Net Worth = ${info['net_worth']:,.2f}, Position = {info['position']*100:.1f}%")
    
    # 获取性能指标
    metrics = test_env.get_performance_metrics()
    
    # 打印结果
    print("\n" + "="*70)
    print("测试结果 (Test Results)")
    print("="*70)
    
    metrics_calc = PerformanceMetrics()
    metrics_calc.print_metrics(metrics)
    
    # 对比基准
    benchmark_return = (raw_data['close'].iloc[-1] / raw_data['close'].iloc[0]) - 1
    print(f"\n【基准策略 (Buy & Hold)】")
    print(f"  收益率: {benchmark_return*100:.2f}%")
    print(f"  策略相对表现: {(metrics['roi'] - benchmark_return)*100:+.2f}%")
    
    print("\n✓ 测试完成！")


def backtest_model(args):
    """
    回测模型（对历史数据的完整评估）
    
    Args:
        args: 命令行参数
    """
    print("\n" + "="*70)
    print("模式: 回测 (MODE: BACKTESTING)")
    print("="*70)
    
    # 加载数据
    print("\n1. 加载历史数据...")
    data_loader = CryptoDataLoader()
    raw_data = data_loader.merge_all_data(args.symbol)
    
    # 特征工程
    print("\n2. 特征工程...")
    feature_engineer = FeatureEngineer()
    features, _ = feature_engineer.build_state_space(raw_data)
    
    # 分割数据
    split_idx = int(len(raw_data) * args.train_ratio)
    train_data = raw_data.iloc[:split_idx]
    train_features = features[:split_idx]
    test_data = raw_data.iloc[split_idx:]
    test_features = features[split_idx:]
    
    print(f"  训练期: {train_data.index[0]} 至 {train_data.index[-1]}")
    print(f"  测试期: {test_data.index[0]} 至 {test_data.index[-1]}")
    
    # 创建环境
    train_env = CryptoTradingEnv(train_data, train_features)
    test_env = CryptoTradingEnv(test_data, test_features)
    
    # 创建智能体
    print("\n3. 创建智能体...")
    agent = HybridAgent(
        input_dim=features.shape[1],
        device=Config.DEVICE
    )
    
    # 如果提供了模型路径，加载模型
    if args.model_path:
        print(f"\n4. 加载模型: {args.model_path}")
        agent.load(args.model_path)
    else:
        print("\n4. 使用随机策略（未提供模型）")
    
    # 样本内回测
    print("\n5. 样本内回测...")
    state = train_env.reset()
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state)
        discrete_action, continuous_action = agent.select_hybrid_action(state_tensor, 0.0, 0.0)
        state, _, done, _ = train_env.step((discrete_action, continuous_action))
    
    train_metrics = train_env.get_performance_metrics()
    
    # 样本外回测
    print("\n6. 样本外回测...")
    state = test_env.reset()
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state)
        discrete_action, continuous_action = agent.select_hybrid_action(state_tensor, 0.0, 0.0)
        state, _, done, _ = test_env.step((discrete_action, continuous_action))
    
    test_metrics = test_env.get_performance_metrics()
    
    # 打印结果
    print("\n" + "="*70)
    print("回测结果 (Backtest Results)")
    print("="*70)
    
    metrics_calc = PerformanceMetrics()
    
    print("\n【样本内表现 (In-Sample)】")
    metrics_calc.print_metrics(train_metrics)
    
    print("\n【样本外表现 (Out-of-Sample)】")
    metrics_calc.print_metrics(test_metrics)
    
    # 计算信号衰减率
    decay_rate = metrics_calc.calculate_signal_decay_rate(train_metrics, test_metrics, 'sharpe_ratio')
    print(f"\n【信号衰减率 (Signal Decay Rate)】")
    print(f"  Sharpe Ratio衰减: {decay_rate*100:.2f}%")
    
    # 基准对比
    train_benchmark = (train_data['close'].iloc[-1] / train_data['close'].iloc[0]) - 1
    test_benchmark = (test_data['close'].iloc[-1] / test_data['close'].iloc[0]) - 1
    
    print(f"\n【基准对比 (Benchmark Comparison)】")
    print(f"  训练期 Buy&Hold: {train_benchmark*100:.2f}%")
    print(f"  测试期 Buy&Hold: {test_benchmark*100:.2f}%")
    print(f"  训练期超额收益: {(train_metrics['roi'] - train_benchmark)*100:+.2f}%")
    print(f"  测试期超额收益: {(test_metrics['roi'] - test_benchmark)*100:+.2f}%")
    
    print("\n✓ 回测完成！")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='加密货币混合RL交易系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  训练模型:
    python main.py --mode train --symbol BTCUSDT --episodes 1000
  
  测试模型:
    python main.py --mode test --symbol BTCUSDT --model_path saved_models/BTCUSDT_best_model.pt
  
  回测模型:
    python main.py --mode backtest --symbol BTCUSDT --model_path saved_models/BTCUSDT_best_model.pt
        """
    )
    
    # 基本参数
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'test', 'backtest'],
                       help='运行模式: train(训练), test(测试), backtest(回测)')
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                       help='交易对符号 (默认: BTCUSDT)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型路径（用于test和backtest模式）')
    
    # 训练参数
    parser.add_argument('--episodes', type=int, default=Config.NUM_EPISODES,
                       help=f'训练episode数 (默认: {Config.NUM_EPISODES})')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例 (默认: 0.8)')
    parser.add_argument('--eval_interval', type=int, default=10,
                       help='评估间隔 (默认: 10)')
    parser.add_argument('--save_interval', type=int, default=50,
                       help='保存间隔 (默认: 50)')
    parser.add_argument('--patience', type=int, default=Config.PATIENCE,
                       help=f'早停patience (默认: {Config.PATIENCE})')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='计算设备 (默认: auto)')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    Config.DEVICE = device
    
    # 设置随机种子
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.SEED)
    
    # 打印配置
    print("\n" + "="*70)
    print("加密货币混合RL交易系统")
    print("Cryptocurrency Hybrid RL Trading System")
    print("="*70)
    print(f"设备: {device}")
    print(f"模式: {args.mode.upper()}")
    print(f"交易对: {args.symbol}")
    print("="*70)
    
    # 根据模式执行
    try:
        if args.mode == 'train':
            train_model(args)
        elif args.mode == 'test':
            if not args.model_path:
                print("错误: test模式需要指定 --model_path")
                sys.exit(1)
            test_model(args)
        elif args.mode == 'backtest':
            backtest_model(args)
    except KeyboardInterrupt:
        print("\n\n⚠ 用户中断")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

