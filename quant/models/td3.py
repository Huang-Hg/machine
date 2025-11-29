"""
TD3 (Twin Delayed Deep Deterministic Policy Gradient) 算法
对应论文 Section 3.4: TD3 concurrently optimizes continuous position sizing

核心特性:
1. Twin Critic Networks - 双评论家网络缓解过估计
2. Delayed Policy Updates - 延迟策略更新 (2:1 ratio)
3. Target Policy Smoothing - 目标策略平滑 (ε~N(0,0.2))

连续动作空间: 持仓比例 0-100% (Position Sizing)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

from config import Config


class Actor(nn.Module):
    """
    Actor网络 (策略网络)
    对应论文 Section 3.4: TD3 Actor
    
    输出连续动作: 持仓比例 [0, 1]
    """
    
    def __init__(self,
                 state_dim: int = Config.ENCODER_HIDDEN_DIM,
                 action_dim: int = Config.CONTINUOUS_ACTION_DIM,
                 hidden_dim: int = Config.ACTOR_HIDDEN_DIM,
                 max_action: float = Config.CONTINUOUS_ACTION_MAX):
        """
        初始化Actor
        
        Args:
            state_dim: 状态维度（来自编码器的64维潜在表示）
            action_dim: 动作维度（1维：持仓比例）
            hidden_dim: 隐藏层维度（论文：128单元MLP）
            max_action: 最大动作值（1.0）
        """
        super(Actor, self).__init__()
        
        self.max_action = max_action
        
        # 论文 Section 3.4: 128-unit MLPs with ELU activation
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 状态表示 [batch_size, state_dim]
            
        Returns:
            动作 [batch_size, action_dim]，范围[0, max_action]
        """
        x = F.elu(self.ln1(self.fc1(state)))
        x = F.elu(self.ln2(self.fc2(x)))
        
        # Sigmoid激活输出[0,1]范围的持仓比例
        # 论文: dynamic hedging ratios -1.5 to +1.5，这里简化为[0,1]
        action = torch.sigmoid(self.fc3(x)) * self.max_action
        
        return action


class Critic(nn.Module):
    """
    Critic网络 (价值网络)
    对应论文 Section 3.4: TD3 Twin Critics
    
    评估状态-动作对的价值
    """
    
    def __init__(self,
                 state_dim: int = Config.ENCODER_HIDDEN_DIM,
                 action_dim: int = Config.CONTINUOUS_ACTION_DIM,
                 hidden_dim: int = Config.CRITIC_HIDDEN_DIM):
        """
        初始化Critic
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super(Critic, self).__init__()
        
        # Q1网络
        self.fc1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q1 = nn.Linear(hidden_dim, 1)
        
        self.ln1_q1 = nn.LayerNorm(hidden_dim)
        self.ln2_q1 = nn.LayerNorm(hidden_dim)
        
        # Q2网络（Twin Critic）
        self.fc1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q2 = nn.Linear(hidden_dim, 1)
        
        self.ln1_q2 = nn.LayerNorm(hidden_dim)
        self.ln2_q2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播（双Q网络）
        
        Args:
            state: 状态 [batch_size, state_dim]
            action: 动作 [batch_size, action_dim]
            
        Returns:
            (Q1值, Q2值)
        """
        sa = torch.cat([state, action], dim=1)
        
        # Q1
        q1 = F.elu(self.ln1_q1(self.fc1_q1(sa)))
        q1 = F.elu(self.ln2_q1(self.fc2_q1(q1)))
        q1 = self.fc3_q1(q1)
        
        # Q2
        q2 = F.elu(self.ln1_q2(self.fc1_q2(sa)))
        q2 = F.elu(self.ln2_q2(self.fc2_q2(q2)))
        q2 = self.fc3_q2(q2)
        
        return q1, q2
    
    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """仅计算Q1值（用于Actor更新）"""
        sa = torch.cat([state, action], dim=1)
        
        q1 = F.elu(self.ln1_q1(self.fc1_q1(sa)))
        q1 = F.elu(self.ln2_q1(self.fc2_q1(q1)))
        q1 = self.fc3_q1(q1)
        
        return q1


class TD3Agent:
    """
    TD3智能体
    对应论文 Section 3.4: TD3 algorithm implementation
    
    实现完整的TD3训练逻辑
    """
    
    def __init__(self,
                 state_dim: int = Config.ENCODER_HIDDEN_DIM,
                 action_dim: int = Config.CONTINUOUS_ACTION_DIM,
                 max_action: float = Config.CONTINUOUS_ACTION_MAX,
                 device: torch.device = Config.DEVICE):
        """
        初始化TD3智能体
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            max_action: 最大动作值
            device: 计算设备
        """
        self.device = device
        self.max_action = max_action
        self.action_dim = action_dim
        
        # ==================== 创建网络 ====================
        # Actor网络
        self.actor = Actor(state_dim, action_dim, max_action=max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action=max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=Config.ACTOR_LR)
        
        # Critic网络
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=Config.CRITIC_LR)
        
        # ==================== TD3超参数 ====================
        self.gamma = Config.GAMMA
        self.tau = Config.TAU
        self.policy_noise = Config.POLICY_NOISE  # 论文: ε~N(0,0.2)
        self.noise_clip = Config.NOISE_CLIP
        self.policy_freq = Config.POLICY_FREQ    # 论文: 2:1 policy-to-critic ratio
        
        self.total_it = 0
    
    def select_action(self, state: torch.Tensor, noise: float = 0.0) -> np.ndarray:
        """
        选择动作
        
        Args:
            state: 状态 [state_dim]
            noise: 探索噪声标准差
            
        Returns:
            动作 [action_dim]
        """
        state = state.to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        
        # 添加探索噪声
        if noise > 0:
            action = action + np.random.normal(0, noise, size=self.action_dim)
            action = np.clip(action, 0, self.max_action)
        
        return action
    
    def train(self,
             replay_buffer,
             batch_size: int = Config.BATCH_SIZE) -> dict:
        """
        训练一步
        
        Args:
            replay_buffer: 经验回放缓冲区
            batch_size: 批次大小
            
        Returns:
            损失字典
        """
        self.total_it += 1
        
        # 从缓冲区采样
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)
        
        with torch.no_grad():
            # ==================== 目标策略平滑 ====================
            # 论文 Section 3.4: Target Policy Smoothing
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(0, self.max_action)
            
            # 计算目标Q值（取两个Critic的最小值）
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # ==================== 更新Critic ====================
        current_q1, current_q2 = self.critic(state, action)
        
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        actor_loss = torch.tensor(0.0)
        
        # ==================== 延迟更新Actor ====================
        # 论文 Section 3.4: Delayed Policy Updates (2:1 ratio)
        if self.total_it % self.policy_freq == 0:
            # Actor损失：最大化Q1值
            actor_loss = -self.critic.q1(state, self.actor(state)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            
            # 软更新目标网络
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'q1_mean': current_q1.mean().item(),
            'q2_mean': current_q2.mean().item()
        }
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """
        软更新目标网络
        
        Args:
            source: 源网络
            target: 目标网络
        """
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        # 同步目标网络
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())


def test_td3():
    """测试TD3"""
    print("=" * 60)
    print("测试TD3 (TD3 Test)")
    print("=" * 60)
    
    # 创建智能体
    agent = TD3Agent()
    
    print(f"\n智能体配置:")
    print(f"  状态维度: {Config.ENCODER_HIDDEN_DIM}")
    print(f"  动作维度: {Config.CONTINUOUS_ACTION_DIM}")
    print(f"  最大动作: {Config.CONTINUOUS_ACTION_MAX}")
    print(f"  Actor学习率: {Config.ACTOR_LR}")
    print(f"  Critic学习率: {Config.CRITIC_LR}")
    print(f"  策略噪声: {Config.POLICY_NOISE}")
    print(f"  更新频率比: 1:{Config.POLICY_FREQ}")
    
    # 测试动作选择
    state = torch.randn(Config.ENCODER_HIDDEN_DIM)
    action = agent.select_action(state, noise=0.1)
    
    print(f"\n动作选择测试:")
    print(f"  状态形状: {state.shape}")
    print(f"  动作值: {action}")
    print(f"  动作范围: [{action.min():.3f}, {action.max():.3f}]")
    
    # 测试前向传播
    state_batch = torch.randn(32, Config.ENCODER_HIDDEN_DIM).to(Config.DEVICE)
    action_batch = agent.actor(state_batch)
    q1, q2 = agent.critic(state_batch, action_batch)
    
    print(f"\n前向传播测试:")
    print(f"  批次大小: 32")
    print(f"  Actor输出形状: {action_batch.shape}")
    print(f"  Q1形状: {q1.shape}")
    print(f"  Q2形状: {q2.shape}")
    print(f"  Q1均值: {q1.mean().item():.3f}")
    print(f"  Q2均值: {q2.mean().item():.3f}")
    
    # 参数统计
    actor_params = sum(p.numel() for p in agent.actor.parameters())
    critic_params = sum(p.numel() for p in agent.critic.parameters())
    
    print(f"\n模型参数:")
    print(f"  Actor参数量: {actor_params:,}")
    print(f"  Critic参数量: {critic_params:,}")
    print(f"  总参数量: {actor_params + critic_params:,}")
    
    print(f"\n✓ TD3测试通过！")


if __name__ == "__main__":
    test_td3()

