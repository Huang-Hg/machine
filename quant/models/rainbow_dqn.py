"""
Rainbow DQN模型
对应论文 Section 3.4: Enhanced Rainbow DQN architecture for discrete action space

Rainbow DQN整合了7项改进 (论文提到):
1. Double DQN - 解耦动作选择和评估
2. Dueling DQN - 分离价值流和优势流
3. Noisy Networks - 参数噪声探索 (σ=0.17)
4. Prioritized Experience Replay - 优先采样 (α=0.7, β=0.5)
5. N-step Learning - 多步自助法 (n=5)
6. Distributional RL - 51-atom支持
7. Multi-step + Prioritized Replay整合

离散动作空间: Buy/Sell/Hold (3个动作)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

from config import Config


class NoisyLinear(nn.Module):
    """
    Noisy Network层
    对应论文 Section 3.4: parametric noise injection (initial σ=0.17)
    
    用参数化噪声替代ε-greedy探索策略
    """
    
    def __init__(self, in_features: int, out_features: int, sigma_init: float = Config.NOISY_SIGMA_INIT):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        # 权重参数
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        # 偏置参数
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """初始化参数"""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        """重置噪声"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """生成缩放噪声"""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)


class DuelingNetwork(nn.Module):
    """
    Dueling Network架构
    对应论文 Section 3.4: Dueling networks with value-advantage stream separation
    
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_actions: int = Config.DISCRETE_ACTIONS,
                 use_noisy: bool = Config.USE_NOISY_NET):
        super(DuelingNetwork, self).__init__()
        
        self.num_actions = num_actions
        self.use_noisy = use_noisy
        
        # 选择层类型
        LinearLayer = NoisyLinear if use_noisy else nn.Linear
        
        # 价值流 (Value Stream)
        self.value_fc = LinearLayer(input_dim, hidden_dim)
        self.value_out = LinearLayer(hidden_dim, 1)
        
        # 优势流 (Advantage Stream)
        self.advantage_fc = LinearLayer(input_dim, hidden_dim)
        self.advantage_out = LinearLayer(hidden_dim, num_actions)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 状态表示 [batch_size, input_dim]
            
        Returns:
            Q值 [batch_size, num_actions]
        """
        # 价值流
        value = F.elu(self.value_fc(x))
        value = self.value_out(value)  # [batch, 1]
        
        # 优势流
        advantage = F.elu(self.advantage_fc(x))
        advantage = self.advantage_out(advantage)  # [batch, num_actions]
        
        # Dueling公式: Q = V + (A - mean(A))
        # 论文原文: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
    
    def reset_noise(self):
        """重置Noisy Layers的噪声"""
        if self.use_noisy:
            self.value_fc.reset_noise()
            self.value_out.reset_noise()
            self.advantage_fc.reset_noise()
            self.advantage_out.reset_noise()


class DistributionalDuelingNetwork(nn.Module):
    """
    分布式Dueling Network
    对应论文 Section 3.4: 51-atom support for distributional RL
    
    输出价值分布而非单点估计
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_actions: int = Config.DISCRETE_ACTIONS,
                 n_atoms: int = Config.N_ATOMS,
                 v_min: float = Config.V_MIN,
                 v_max: float = Config.V_MAX,
                 use_noisy: bool = Config.USE_NOISY_NET):
        super(DistributionalDuelingNetwork, self).__init__()
        
        self.num_actions = num_actions
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.use_noisy = use_noisy
        
        # 计算支撑集
        self.register_buffer('support', torch.linspace(v_min, v_max, n_atoms))
        
        LinearLayer = NoisyLinear if use_noisy else nn.Linear
        
        # 价值流 (输出分布)
        self.value_fc = LinearLayer(input_dim, hidden_dim)
        self.value_out = LinearLayer(hidden_dim, n_atoms)
        
        # 优势流 (输出分布)
        self.advantage_fc = LinearLayer(input_dim, hidden_dim)
        self.advantage_out = LinearLayer(hidden_dim, num_actions * n_atoms)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 状态表示 [batch_size, input_dim]
            
        Returns:
            (Q值 [batch, num_actions], 分布 [batch, num_actions, n_atoms])
        """
        batch_size = x.size(0)
        
        # 价值流
        value = F.elu(self.value_fc(x))
        value_dist = self.value_out(value).view(batch_size, 1, self.n_atoms)
        
        # 优势流
        advantage = F.elu(self.advantage_fc(x))
        advantage_dist = self.advantage_out(advantage).view(batch_size, self.num_actions, self.n_atoms)
        
        # Dueling公式（分布版本）
        q_dist = value_dist + (advantage_dist - advantage_dist.mean(dim=1, keepdim=True))
        
        # Softmax得到概率分布
        q_dist = F.softmax(q_dist, dim=-1)
        
        # 计算期望Q值
        q_values = torch.sum(q_dist * self.support, dim=-1)
        
        return q_values, q_dist
    
    def reset_noise(self):
        """重置Noisy Layers的噪声"""
        if self.use_noisy:
            self.value_fc.reset_noise()
            self.value_out.reset_noise()
            self.advantage_fc.reset_noise()
            self.advantage_out.reset_noise()


class RainbowDQN(nn.Module):
    """
    Rainbow DQN完整模型
    对应论文 Section 3.4: Enhanced Rainbow DQN architecture
    
    整合所有改进的DQN网络
    """
    
    def __init__(self,
                 input_dim: int = Config.ENCODER_HIDDEN_DIM,
                 hidden_dim: int = 128,
                 num_actions: int = Config.DISCRETE_ACTIONS,
                 use_distributional: bool = Config.USE_DISTRIBUTIONAL,
                 use_noisy: bool = Config.USE_NOISY_NET):
        """
        初始化Rainbow DQN
        
        Args:
            input_dim: 输入维度（来自编码器的64维潜在表示）
            hidden_dim: 隐藏层维度
            num_actions: 动作数量
            use_distributional: 是否使用分布式RL
            use_noisy: 是否使用Noisy Networks
        """
        super(RainbowDQN, self).__init__()
        
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.use_distributional = use_distributional
        self.use_noisy = use_noisy
        
        # 选择网络类型
        if use_distributional:
            self.network = DistributionalDuelingNetwork(
                input_dim, hidden_dim, num_actions, use_noisy=use_noisy
            )
        else:
            self.network = DuelingNetwork(
                input_dim, hidden_dim, num_actions, use_noisy=use_noisy
            )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 状态表示 [batch_size, input_dim]
            
        Returns:
            Q值 [batch_size, num_actions]
        """
        if self.use_distributional:
            q_values, _ = self.network(state)
            return q_values
        else:
            return self.network(state)
    
    def get_distribution(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取Q值分布（仅用于分布式RL）
        
        Args:
            state: 状态表示
            
        Returns:
            (Q值, 分布)
        """
        if self.use_distributional:
            return self.network(state)
        else:
            q_values = self.network(state)
            return q_values, None
    
    def select_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        选择动作
        
        Args:
            state: 状态表示 [input_dim]
            epsilon: ε-greedy参数（如果使用Noisy Net则忽略）
            
        Returns:
            动作索引
        """
        if not self.use_noisy and np.random.random() < epsilon:
            # ε-greedy探索
            return np.random.randint(self.num_actions)
        
        with torch.no_grad():
            state = state.unsqueeze(0) if state.dim() == 1 else state
            q_values = self.forward(state)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def reset_noise(self):
        """重置噪声（用于Noisy Networks）"""
        if self.use_noisy:
            self.network.reset_noise()


class DoubleDQNLoss(nn.Module):
    """
    Double DQN损失函数
    对应论文 Section 3.4: Double DQN to decouple action selection and evaluation
    
    减少过估计偏差
    """
    
    def __init__(self, gamma: float = Config.GAMMA):
        super(DoubleDQNLoss, self).__init__()
        self.gamma = gamma
    
    def forward(self,
                q_values: torch.Tensor,
                actions: torch.Tensor,
                rewards: torch.Tensor,
                next_q_values: torch.Tensor,
                target_next_q_values: torch.Tensor,
                dones: torch.Tensor) -> torch.Tensor:
        """
        计算Double DQN损失
        
        Args:
            q_values: 当前Q值 [batch, num_actions]
            actions: 执行的动作 [batch]
            rewards: 奖励 [batch]
            next_q_values: 下一状态Q值(online network) [batch, num_actions]
            target_next_q_values: 下一状态Q值(target network) [batch, num_actions]
            dones: 终止标志 [batch]
            
        Returns:
            损失值
        """
        # 当前Q值
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: 用online network选择动作，用target network评估
        next_actions = next_q_values.argmax(dim=1)
        next_q = target_next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        
        # 目标Q值
        target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Huber损失（比MSE更鲁棒）
        loss = F.smooth_l1_loss(current_q, target_q.detach())
        
        return loss


def test_rainbow_dqn():
    """测试Rainbow DQN"""
    print("=" * 60)
    print("测试Rainbow DQN (Rainbow DQN Test)")
    print("=" * 60)
    
    # 创建模型
    model = RainbowDQN(
        input_dim=Config.ENCODER_HIDDEN_DIM,
        use_distributional=True,
        use_noisy=True
    )
    
    print(f"\n模型配置:")
    print(f"  输入维度: {Config.ENCODER_HIDDEN_DIM}")
    print(f"  动作数量: {Config.DISCRETE_ACTIONS}")
    print(f"  使用分布式RL: {Config.USE_DISTRIBUTIONAL}")
    print(f"  使用Noisy Net: {Config.USE_NOISY_NET}")
    
    # 测试前向传播
    batch_size = 32
    state = torch.randn(batch_size, Config.ENCODER_HIDDEN_DIM)
    
    q_values = model(state)
    print(f"\nQ值形状: {q_values.shape}")
    print(f"Q值示例: {q_values[0]}")
    
    # 测试动作选择
    single_state = torch.randn(Config.ENCODER_HIDDEN_DIM)
    action = model.select_action(single_state)
    print(f"\n选择的动作: {action}")
    
    # 测试分布式输出
    if Config.USE_DISTRIBUTIONAL:
        q_vals, q_dist = model.get_distribution(state)
        print(f"\nQ分布形状: {q_dist.shape}")
    
    # 测试噪声重置
    if Config.USE_NOISY_NET:
        model.reset_noise()
        print(f"\n✓ 噪声重置成功")
    
    # 参数统计
    print(f"\n模型参数:")
    print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


if __name__ == "__main__":
    test_rainbow_dqn()

