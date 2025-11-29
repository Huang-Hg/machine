"""
混合动作空间智能体
对应论文 Section 3.4: Hybrid Action Space Coordination

核心创新:
1. 双路径决策系统 (Dual-pathway Decision System)
   - Rainbow DQN处理离散动作 (Buy/Sell/Hold)
   - TD3处理连续动作 (Position Sizing 0-100%)
2. 共享时空特征编码器 (Shared Encoder)
3. 梯度停止层 (Gradient Stopping Layers)
4. 跨模态知识迁移 (Cross-modal Knowledge Transfer)

论文原文: "achieves 18.7% annualized return improvement over single-policy baselines 
during regime-switching periods (2020-2022 crypto cycles)"
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict

from config import Config
from models.encoder import SpatioTemporalEncoder
from models.rainbow_dqn import RainbowDQN
from models.td3 import TD3Agent


class HybridAgent(nn.Module):
    """
    混合动作空间智能体
    对应论文 Section 3.4 完整架构
    
    决策流程:
    1. 共享编码器提取64维潜在表示
    2. Rainbow DQN决定离散动作类型 (买/卖/持有)
    3. TD3决定连续动作大小 (持仓比例)
    4. 联合执行动作
    """
    
    def __init__(self,
                 input_dim: int = Config.ENCODER_INPUT_DIM,
                 hidden_dim: int = Config.ENCODER_HIDDEN_DIM,
                 discrete_actions: int = Config.DISCRETE_ACTIONS,
                 continuous_action_dim: int = Config.CONTINUOUS_ACTION_DIM,
                 device: torch.device = Config.DEVICE):
        """
        初始化混合智能体
        
        Args:
            input_dim: 输入维度（PCA后的21维）
            hidden_dim: 编码器输出维度（64维）
            discrete_actions: 离散动作数量
            continuous_action_dim: 连续动作维度
            device: 计算设备
        """
        super(HybridAgent, self).__init__()
        
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # ==================== 共享编码器 ====================
        # 论文 Section 3.4: Shared Spatial-Temporal Feature Encoder
        # 论文: initialized from Innovation 2's 64D GRU outputs
        self.shared_encoder = SpatioTemporalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim
        ).to(device)
        
        # ==================== 离散动作路径 (Rainbow DQN) ====================
        # 论文: discrete pathway (e.g., position sizing 0%/50%/100%)
        self.discrete_policy = RainbowDQN(
            input_dim=hidden_dim,
            num_actions=discrete_actions,
            use_distributional=Config.USE_DISTRIBUTIONAL,
            use_noisy=Config.USE_NOISY_NET
        ).to(device)
        
        # ==================== 连续动作路径 (TD3) ====================
        # 论文: continuous action pathway (e.g., dynamic hedging ratios)
        self.continuous_policy = TD3Agent(
            state_dim=hidden_dim,
            action_dim=continuous_action_dim,
            device=device
        )
        
        # ==================== 梯度停止开关 ====================
        # 论文 Section 3.4: gradient stopping layers
        # discrete/continuous policy heads maintain separate MLPs 
        # yet share base network parameters
        self.use_gradient_stopping = True
        
        # ==================== 优化器 ====================
        # 只为共享编码器和离散策略创建优化器
        # TD3内部已有自己的优化器
        self.encoder_optimizer = torch.optim.Adam(
            self.shared_encoder.parameters(), 
            lr=Config.DQN_LR
        )
        self.discrete_optimizer = torch.optim.Adam(
            self.discrete_policy.parameters(),
            lr=Config.DQN_LR
        )
        
    def encode_state(self, state: torch.Tensor, stop_gradient: bool = False) -> torch.Tensor:
        """
        编码状态（共享编码器）
        
        Args:
            state: 原始状态 [batch, seq_len, input_dim]
            stop_gradient: 是否停止梯度传播
            
        Returns:
            潜在表示 [batch, hidden_dim]
        """
        latent, _ = self.shared_encoder(state)
        
        if stop_gradient:
            latent = latent.detach()
        
        return latent
    
    def select_discrete_action(self, latent: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        选择离散动作 (Buy/Sell/Hold)
        
        Args:
            latent: 潜在表示 [hidden_dim]
            epsilon: ε-greedy参数
            
        Returns:
            动作索引 (0: Hold, 1: Buy, 2: Sell)
        """
        return self.discrete_policy.select_action(latent, epsilon)
    
    def select_continuous_action(self, latent: torch.Tensor, noise: float = 0.0) -> np.ndarray:
        """
        选择连续动作 (Position Size)
        
        Args:
            latent: 潜在表示 [hidden_dim]
            noise: 探索噪声
            
        Returns:
            持仓比例 [0, 1]
        """
        return self.continuous_policy.select_action(latent, noise)
    
    def select_hybrid_action(self, 
                           state: torch.Tensor,
                           epsilon: float = 0.0,
                           noise: float = 0.0) -> Tuple[int, np.ndarray]:
        """
        选择混合动作
        对应论文 Section 3.4: Joint optimization of tactical entries/exits 
        and strategic capital deployment
        
        Args:
            state: 状态序列 [seq_len, input_dim] 或 [batch, seq_len, input_dim]
            epsilon: 离散动作探索参数
            noise: 连续动作探索噪声
            
        Returns:
            (离散动作, 连续动作)
        """
        # 确保输入维度正确
        if state.dim() == 2:
            state = state.unsqueeze(0)  # [1, seq_len, input_dim]
        
        state = state.to(self.device)
        
        # 1. 通过共享编码器获取潜在表示
        with torch.no_grad():
            latent = self.encode_state(state, stop_gradient=False)
            latent = latent.squeeze(0)  # [hidden_dim]
        
        # 2. 离散策略选择动作类型
        discrete_action = self.select_discrete_action(latent, epsilon)
        
        # 3. 连续策略选择持仓比例
        # 论文: gradient stopping - 连续路径使用detach的特征
        if self.use_gradient_stopping:
            latent_continuous = latent.detach()
        else:
            latent_continuous = latent
        
        continuous_action = self.select_continuous_action(latent_continuous, noise)
        
        return discrete_action, continuous_action
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播（用于训练）
        
        Args:
            state: [batch, seq_len, input_dim]
            
        Returns:
            (离散Q值 [batch, num_actions], 连续动作 [batch, action_dim])
        """
        # 共享编码
        latent = self.encode_state(state, stop_gradient=False)
        
        # 离散路径
        q_values = self.discrete_policy(latent)
        
        # 连续路径（使用detached特征）
        if self.use_gradient_stopping:
            latent_continuous = latent.detach()
        else:
            latent_continuous = latent
        
        continuous_action = self.continuous_policy.actor(latent_continuous)
        
        return q_values, continuous_action
    
    def update_discrete(self, 
                       states: torch.Tensor,
                       actions: torch.Tensor,
                       rewards: torch.Tensor,
                       next_states: torch.Tensor,
                       dones: torch.Tensor) -> Dict[str, float]:
        """
        更新离散策略 (Rainbow DQN)
        
        Args:
            states: [batch, seq_len, input_dim]
            actions: [batch] (离散动作索引)
            rewards: [batch]
            next_states: [batch, seq_len, input_dim]
            dones: [batch]
            
        Returns:
            损失字典
        """
        # 编码当前状态
        latent = self.encode_state(states)
        q_values = self.discrete_policy(latent)
        
        # 编码下一状态
        with torch.no_grad():
            next_latent = self.encode_state(next_states)
            next_q_values = self.discrete_policy(next_latent)
        
        # 计算TD误差
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN目标
        next_actions = next_q_values.argmax(dim=1)
        next_q = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        
        target_q = rewards + Config.GAMMA * next_q * (1 - dones)
        
        # 损失
        loss = torch.nn.functional.smooth_l1_loss(current_q, target_q.detach())
        
        # 反向传播
        self.encoder_optimizer.zero_grad()
        self.discrete_optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.shared_encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.discrete_policy.parameters(), 1.0)
        
        self.encoder_optimizer.step()
        self.discrete_optimizer.step()
        
        return {
            'discrete_loss': loss.item(),
            'q_mean': current_q.mean().item(),
            'target_q_mean': target_q.mean().item()
        }
    
    def update_continuous(self,
                         states: torch.Tensor,
                         actions: torch.Tensor,
                         rewards: torch.Tensor,
                         next_states: torch.Tensor,
                         dones: torch.Tensor) -> Dict[str, float]:
        """
        更新连续策略 (TD3)
        
        Args:
            states: [batch, seq_len, input_dim]
            actions: [batch, action_dim] (连续动作)
            rewards: [batch]
            next_states: [batch, seq_len, input_dim]
            dones: [batch]
            
        Returns:
            损失字典
        """
        # 编码状态（使用detached特征，实现梯度停止）
        with torch.no_grad():
            latent = self.encode_state(states, stop_gradient=True)
            next_latent = self.encode_state(next_states, stop_gradient=True)
        
        # 准备TD3训练数据
        # 创建临时缓冲区（因为TD3期望从replay buffer采样）
        class TempBuffer:
            def sample(self, batch_size):
                return latent, actions, rewards, next_latent, dones
        
        temp_buffer = TempBuffer()
        
        # 使用TD3的训练方法
        losses = self.continuous_policy.train(temp_buffer, batch_size=latent.size(0))
        
        return losses
    
    def update(self,
              replay_buffer,
              batch_size: int = Config.BATCH_SIZE,
              update_discrete: bool = True,
              update_continuous: bool = True) -> Dict[str, float]:
        """
        联合更新策略
        对应论文 Section 3.4: Coordinated design ensures feature consistency 
        across action types while allowing specialized optimization
        
        Args:
            replay_buffer: 经验回放缓冲区
            batch_size: 批次大小
            update_discrete: 是否更新离散策略
            update_continuous: 是否更新连续策略
            
        Returns:
            所有损失的字典
        """
        losses = {}
        
        # 采样经验
        batch = replay_buffer.sample(batch_size)
        states, discrete_actions, continuous_actions, rewards, next_states, dones = batch
        
        # 转换为tensor
        states = states.to(self.device)
        discrete_actions = discrete_actions.to(self.device).long()
        continuous_actions = continuous_actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # 更新离散策略
        if update_discrete:
            discrete_losses = self.update_discrete(
                states, discrete_actions, rewards, next_states, dones
            )
            losses.update(discrete_losses)
        
        # 更新连续策略
        if update_continuous:
            continuous_losses = self.update_continuous(
                states, continuous_actions, rewards, next_states, dones
            )
            losses.update(continuous_losses)
        
        return losses
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'encoder': self.shared_encoder.state_dict(),
            'discrete_policy': self.discrete_policy.state_dict(),
            'encoder_optimizer': self.encoder_optimizer.state_dict(),
            'discrete_optimizer': self.discrete_optimizer.state_dict(),
        }, filepath)
        
        # TD3单独保存
        self.continuous_policy.save(filepath.replace('.pt', '_td3.pt'))
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.shared_encoder.load_state_dict(checkpoint['encoder'])
        self.discrete_policy.load_state_dict(checkpoint['discrete_policy'])
        self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
        self.discrete_optimizer.load_state_dict(checkpoint['discrete_optimizer'])
        
        # 加载TD3
        self.continuous_policy.load(filepath.replace('.pt', '_td3.pt'))
    
    def reset_noise(self):
        """重置噪声（用于Noisy Networks）"""
        if Config.USE_NOISY_NET:
            self.discrete_policy.reset_noise()


def test_hybrid_agent():
    """测试混合智能体"""
    print("=" * 60)
    print("测试混合智能体 (Hybrid Agent Test)")
    print("=" * 60)
    
    # 创建智能体
    agent = HybridAgent()
    
    print(f"\n智能体配置:")
    print(f"  输入维度: {Config.ENCODER_INPUT_DIM}")
    print(f"  编码器输出: {Config.ENCODER_HIDDEN_DIM}")
    print(f"  离散动作数: {Config.DISCRETE_ACTIONS}")
    print(f"  连续动作维度: {Config.CONTINUOUS_ACTION_DIM}")
    print(f"  使用梯度停止: {agent.use_gradient_stopping}")
    
    # 测试动作选择
    seq_len = Config.SEQUENCE_LENGTH
    input_dim = Config.ENCODER_INPUT_DIM
    state = torch.randn(seq_len, input_dim)
    
    print(f"\n动作选择测试:")
    print(f"  输入状态形状: {state.shape}")
    
    discrete_action, continuous_action = agent.select_hybrid_action(state, epsilon=0.1, noise=0.1)
    
    print(f"  离散动作: {discrete_action} (0=Hold, 1=Buy, 2=Sell)")
    print(f"  连续动作: {continuous_action} (持仓比例)")
    
    # 测试前向传播
    batch_size = 32
    state_batch = torch.randn(batch_size, seq_len, input_dim)
    
    q_values, position_sizes = agent(state_batch)
    
    print(f"\n前向传播测试:")
    print(f"  批次大小: {batch_size}")
    print(f"  Q值形状: {q_values.shape}")
    print(f"  持仓比例形状: {position_sizes.shape}")
    print(f"  Q值示例: {q_values[0]}")
    print(f"  持仓比例示例: {position_sizes[0]}")
    
    # 测试编码器
    latent = agent.encode_state(state_batch)
    print(f"\n编码器输出:")
    print(f"  潜在表示形状: {latent.shape}")
    print(f"  潜在表示均值: {latent.mean().item():.4f}")
    print(f"  潜在表示标准差: {latent.std().item():.4f}")
    
    # 参数统计
    encoder_params = sum(p.numel() for p in agent.shared_encoder.parameters())
    discrete_params = sum(p.numel() for p in agent.discrete_policy.parameters())
    actor_params = sum(p.numel() for p in agent.continuous_policy.actor.parameters())
    critic_params = sum(p.numel() for p in agent.continuous_policy.critic.parameters())
    
    print(f"\n模型参数统计:")
    print(f"  共享编码器: {encoder_params:,}")
    print(f"  离散策略: {discrete_params:,}")
    print(f"  连续策略Actor: {actor_params:,}")
    print(f"  连续策略Critic: {critic_params:,}")
    print(f"  总参数量: {encoder_params + discrete_params + actor_params + critic_params:,}")
    
    print(f"\n✓ 混合智能体测试通过！")


if __name__ == "__main__":
    test_hybrid_agent()

