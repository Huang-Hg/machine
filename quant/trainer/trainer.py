"""
训练器模块
对应论文 Section 5 & 6: 实验设计与训练流程

整合所有组件:
1. 数据加载与特征工程
2. 混合智能体
3. 交易环境
4. 经验回放
5. 性能评估
"""

import numpy as np
import torch
import os
from typing import Dict, List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config
from data.dataloader import CryptoDataLoader
from data.feature_engineer import FeatureEngineer
from models.hybrid_agent import HybridAgent
from quant.trader import CryptoTradingEnv
from utils.replay_buffer import HybridReplayBuffer
from utils.metrics import PerformanceMetrics


class Trainer:
    """
    混合RL交易系统训练器
    对应论文 Section 5 & 6 完整训练流程
    """
    
    def __init__(self,
                 symbol: str = 'BTCUSDT',
                 train_ratio: float = 0.8,
                 device: torch.device = Config.DEVICE):
        """
        初始化训练器
        
        Args:
            symbol: 交易对符号
            train_ratio: 训练集比例
            device: 计算设备
        """
        self.symbol = symbol
        self.train_ratio = train_ratio
        self.device = device
        
        # ==================== 数据加载与处理 ====================
        print("=" * 60)
        print("初始化训练器 (Initializing Trainer)")
        print("=" * 60)
        
        print("\n1. 加载数据...")
        self.data_loader = CryptoDataLoader()
        self.raw_data = self.data_loader.merge_all_data(symbol)
        
        print("\n2. 特征工程...")
        self.feature_engineer = FeatureEngineer()
        self.features, self.feature_names = self.feature_engineer.build_state_space(self.raw_data)
        
        print("\n3. 数据分割...")
        split_idx = int(len(self.raw_data) * train_ratio)
        
        self.train_data = self.raw_data.iloc[:split_idx]
        self.train_features = self.features[:split_idx]
        
        self.test_data = self.raw_data.iloc[split_idx:]
        self.test_features = self.features[split_idx:]
        
        print(f"  训练集: {len(self.train_data)} 样本")
        print(f"  测试集: {len(self.test_data)} 样本")
        
        # ==================== 创建环境 ====================
        print("\n4. 创建交易环境...")
        self.train_env = CryptoTradingEnv(self.train_data, self.train_features)
        self.test_env = CryptoTradingEnv(self.test_data, self.test_features)
        
        # ==================== 创建智能体 ====================
        print("\n5. 创建混合智能体...")
        self.agent = HybridAgent(
            input_dim=self.features.shape[1],
            device=device
        )
        
        # ==================== 经验回放缓冲区 ====================
        print("\n6. 初始化经验回放缓冲区...")
        self.replay_buffer = HybridReplayBuffer(
            capacity=Config.BUFFER_SIZE,
            state_shape=(Config.SEQUENCE_LENGTH, self.features.shape[1]),
            use_prioritized=Config.USE_PRIORITIZED_REPLAY,
            use_n_step=Config.USE_N_STEP
        )
        
        # ==================== 性能评估器 ====================
        self.metrics_calc = PerformanceMetrics()
        
        # ==================== 训练状态 ====================
        self.episode = 0
        self.total_steps = 0
        self.best_test_return = -np.inf
        
        # 训练历史
        self.train_history = {
            'episode_rewards': [],
            'episode_returns': [],
            'episode_lengths': [],
            'losses': [],
            'test_returns': []
        }
        
        print("\n✓ 训练器初始化完成！")
        print("=" * 60 + "\n")
    
    def collect_experience(self, 
                          env: CryptoTradingEnv,
                          epsilon: float = 0.0,
                          noise: float = 0.0,
                          render: bool = False) -> Tuple[float, Dict]:
        """
        收集一个episode的经验
        
        Args:
            env: 交易环境
            epsilon: ε-greedy参数
            noise: 连续动作噪声
            render: 是否渲染
            
        Returns:
            (episode总奖励, 性能指标)
        """
        state = env.reset()
        episode_reward = 0
        step = 0
        
        done = False
        while not done:
            # 选择动作
            state_tensor = torch.FloatTensor(state)
            discrete_action, continuous_action = self.agent.select_hybrid_action(
                state_tensor,
                epsilon=epsilon,
                noise=noise
            )
            
            # 执行动作
            next_state, reward, done, info = env.step((discrete_action, continuous_action))
            
            # 存储经验
            self.replay_buffer.add(
                state=state,
                discrete_action=discrete_action,
                continuous_action=continuous_action,
                reward=reward,
                next_state=next_state,
                done=done
            )
            
            episode_reward += reward
            state = next_state
            step += 1
            self.total_steps += 1
            
            # 训练（如果缓冲区足够大）
            if len(self.replay_buffer) >= Config.BATCH_SIZE and self.total_steps % Config.UPDATE_FREQUENCY == 0:
                losses = self.agent.update(self.replay_buffer, Config.BATCH_SIZE)
                self.train_history['losses'].append(losses)
            
            if render:
                env.render()
        
        # 获取性能指标
        metrics = env.get_performance_metrics()
        
        return episode_reward, metrics
    
    def train(self,
             num_episodes: int = Config.NUM_EPISODES,
             eval_interval: int = 10,
             save_interval: int = 50,
             early_stopping_patience: int = Config.PATIENCE):
        """
        训练智能体
        对应论文 Section 5: Training Process
        
        Args:
            num_episodes: 训练episode数
            eval_interval: 评估间隔
            save_interval: 保存模型间隔
            early_stopping_patience: 早停耐心值
        """
        print("\n" + "=" * 60)
        print("开始训练 (Starting Training)")
        print("=" * 60 + "\n")
        
        patience_counter = 0
        
        for episode in tqdm(range(num_episodes), desc="训练进度"):
            self.episode = episode
            
            # ==================== 动态探索策略 ====================
            # ε-greedy衰减 (用于Rainbow DQN)
            epsilon = max(
                Config.EPSILON_END,
                Config.EPSILON_START - (Config.EPSILON_START - Config.EPSILON_END) * episode / Config.EPSILON_DECAY
            )
            
            # 连续动作噪声衰减 (用于TD3)
            noise = max(0.01, Config.POLICY_NOISE * (0.99 ** episode))
            
            # ==================== 收集训练经验 ====================
            episode_reward, train_metrics = self.collect_experience(
                env=self.train_env,
                epsilon=epsilon,
                noise=noise,
                render=False
            )
            
            # 记录训练指标
            self.train_history['episode_rewards'].append(episode_reward)
            self.train_history['episode_returns'].append(train_metrics['roi'])
            self.train_history['episode_lengths'].append(train_metrics.get('total_trades', 0))
            
            # ==================== 定期评估 ====================
            if (episode + 1) % eval_interval == 0:
                print(f"\n{'='*60}")
                print(f"Episode {episode + 1}/{num_episodes} 评估")
                print(f"{'='*60}")
                
                # 训练集性能
                print(f"\n【训练集性能】")
                print(f"  Episode Reward: {episode_reward:.4f}")
                print(f"  ROI: {train_metrics['roi']*100:.2f}%")
                print(f"  Sharpe: {train_metrics.get('sharpe_ratio', 0):.2f}")
                print(f"  Max Drawdown: {train_metrics['max_drawdown']*100:.2f}%")
                print(f"  Win Rate: {train_metrics['win_rate']*100:.2f}%")
                
                # 测试集评估
                print(f"\n【测试集评估】")
                _, test_metrics = self.collect_experience(
                    env=self.test_env,
                    epsilon=0.0,  # 不探索
                    noise=0.0,
                    render=False
                )
                
                test_return = test_metrics['roi']
                self.train_history['test_returns'].append(test_return)
                
                print(f"  ROI: {test_return*100:.2f}%")
                print(f"  Sharpe: {test_metrics.get('sharpe_ratio', 0):.2f}")
                print(f"  Max Drawdown: {test_metrics['max_drawdown']*100:.2f}%")
                print(f"  Win Rate: {test_metrics['win_rate']*100:.2f}%")
                
                # ==================== Early Stopping ====================
                if test_return > self.best_test_return:
                    self.best_test_return = test_return
                    patience_counter = 0
                    
                    # 保存最佳模型
                    save_path = os.path.join(Config.MODEL_SAVE_DIR, f'{self.symbol}_best_model.pt')
                    os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
                    self.agent.save(save_path)
                    print(f"\n✓ 保存最佳模型 (ROI: {test_return*100:.2f}%)")
                else:
                    patience_counter += 1
                    
                    if patience_counter >= early_stopping_patience:
                        print(f"\n⚠ 早停触发 (patience={patience_counter})")
                        break
                
                print(f"{'='*60}\n")
            
            # ==================== 定期保存 ====================
            if (episode + 1) % save_interval == 0:
                save_path = os.path.join(Config.MODEL_SAVE_DIR, f'{self.symbol}_episode_{episode+1}.pt')
                os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
                self.agent.save(save_path)
                print(f"✓ 保存检查点: episode {episode+1}")
        
        print("\n" + "=" * 60)
        print("训练完成 (Training Completed)")
        print("=" * 60 + "\n")
        
        # 最终评估
        self.final_evaluation()
    
    def final_evaluation(self):
        """最终评估"""
        print("\n" + "=" * 60)
        print("最终评估 (Final Evaluation)")
        print("=" * 60)
        
        # 加载最佳模型
        best_model_path = os.path.join(Config.MODEL_SAVE_DIR, f'{self.symbol}_best_model.pt')
        if os.path.exists(best_model_path):
            self.agent.load(best_model_path)
            print("✓ 加载最佳模型")
        
        # 测试集评估
        _, test_metrics = self.collect_experience(
            env=self.test_env,
            epsilon=0.0,
            noise=0.0,
            render=False
        )
        
        print("\n【测试集最终性能】")
        self.metrics_calc.print_metrics(test_metrics)
        
        # 计算基准性能 (Buy & Hold)
        benchmark_return = (self.test_data['close'].iloc[-1] / self.test_data['close'].iloc[0]) - 1
        print(f"\n【基准策略 (Buy & Hold)】")
        print(f"  收益率: {benchmark_return*100:.2f}%")
        print(f"  相对表现: {(test_metrics['roi'] - benchmark_return)*100:+.2f}%")
        
        # 绘制训练曲线
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Episode收益
        axes[0, 0].plot(self.train_history['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True)
        
        # 2. ROI
        axes[0, 1].plot(self.train_history['episode_returns'], label='Train')
        if self.train_history['test_returns']:
            test_episodes = np.arange(0, len(self.train_history['episode_returns']), 
                                     len(self.train_history['episode_returns'])//len(self.train_history['test_returns']))
            axes[0, 1].plot(test_episodes[:len(self.train_history['test_returns'])], 
                           self.train_history['test_returns'], label='Test', marker='o')
        axes[0, 1].set_title('ROI Over Episodes')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('ROI')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. Episode长度（交易次数）
        axes[1, 0].plot(self.train_history['episode_lengths'])
        axes[1, 0].set_title('Trades per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Number of Trades')
        axes[1, 0].grid(True)
        
        # 4. 损失
        if self.train_history['losses']:
            discrete_losses = [l.get('discrete_loss', 0) for l in self.train_history['losses']]
            critic_losses = [l.get('critic_loss', 0) for l in self.train_history['losses']]
            
            axes[1, 1].plot(discrete_losses, label='Discrete (DQN)', alpha=0.7)
            axes[1, 1].plot(critic_losses, label='Continuous (TD3)', alpha=0.7)
            axes[1, 1].set_title('Training Losses')
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(Config.RESULT_SAVE_DIR, f'{self.symbol}_training_curves.png')
        os.makedirs(Config.RESULT_SAVE_DIR, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ 训练曲线已保存: {save_path}")
        
        plt.close()


def main():
    """主函数"""
    # 设置随机种子
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.SEED)
    
    # 打印配置
    Config.print_config()
    
    # 创建训练器
    trainer = Trainer(symbol='BTCUSDT')
    
    # 开始训练
    trainer.train(
        num_episodes=Config.NUM_EPISODES,
        eval_interval=10,
        save_interval=50,
        early_stopping_patience=Config.PATIENCE
    )


if __name__ == "__main__":
    main()

