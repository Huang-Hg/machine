"""
经验回放缓冲区
对应论文 Section 3.4: Prioritized Experience Replay (α=0.7, β=0.5)

支持:
1. 标准经验回放
2. 优先经验回放 (Prioritized Experience Replay)
3. N-step Returns (n=5)
4. 混合动作空间存储
"""

import numpy as np
import torch
from typing import Tuple, Optional
from collections import deque
import random

from config import Config


class ReplayBuffer:
    """
    标准经验回放缓冲区
    """
    
    def __init__(self, 
                 capacity: int = Config.BUFFER_SIZE,
                 state_shape: tuple = (Config.SEQUENCE_LENGTH, Config.ENCODER_INPUT_DIM)):
        """
        初始化缓冲区
        
        Args:
            capacity: 缓冲区容量
            state_shape: 状态形状
        """
        self.capacity = capacity
        self.state_shape = state_shape
        
        # 存储
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.discrete_actions = np.zeros(capacity, dtype=np.int64)
        self.continuous_actions = np.zeros((capacity, Config.CONTINUOUS_ACTION_DIM), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.position = 0
        self.size = 0
    
    def add(self,
           state: np.ndarray,
           discrete_action: int,
           continuous_action: np.ndarray,
           reward: float,
           next_state: np.ndarray,
           done: bool):
        """
        添加经验
        
        Args:
            state: 状态
            discrete_action: 离散动作
            continuous_action: 连续动作
            reward: 奖励
            next_state: 下一状态
            done: 是否结束
        """
        self.states[self.position] = state
        self.discrete_actions[self.position] = discrete_action
        self.continuous_actions[self.position] = continuous_action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple:
        """
        采样批次
        
        Args:
            batch_size: 批次大小
            
        Returns:
            (states, discrete_actions, continuous_actions, rewards, next_states, dones)
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        states = torch.FloatTensor(self.states[indices])
        discrete_actions = torch.LongTensor(self.discrete_actions[indices])
        continuous_actions = torch.FloatTensor(self.continuous_actions[indices])
        rewards = torch.FloatTensor(self.rewards[indices])
        next_states = torch.FloatTensor(self.next_states[indices])
        dones = torch.FloatTensor(self.dones[indices])
        
        return states, discrete_actions, continuous_actions, rewards, next_states, dones
    
    def __len__(self):
        return self.size


class PrioritizedReplayBuffer:
    """
    优先经验回放缓冲区
    对应论文 Section 3.4: Prioritized Experience Replay (α=0.7, β=0.5)
    """
    
    def __init__(self,
                 capacity: int = Config.BUFFER_SIZE,
                 state_shape: tuple = (Config.SEQUENCE_LENGTH, Config.ENCODER_INPUT_DIM),
                 alpha: float = Config.PRIORITY_ALPHA,
                 beta_start: float = Config.PRIORITY_BETA_START,
                 beta_frames: int = Config.PRIORITY_BETA_FRAMES):
        """
        初始化优先回放缓冲区
        
        Args:
            capacity: 缓冲区容量
            state_shape: 状态形状
            alpha: 优先级指数 (论文: 0.7)
            beta_start: 重要性采样初始值 (论文: 0.5)
            beta_frames: β增长到1的帧数
        """
        self.capacity = capacity
        self.state_shape = state_shape
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        # 存储
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.discrete_actions = np.zeros(capacity, dtype=np.int64)
        self.continuous_actions = np.zeros((capacity, Config.CONTINUOUS_ACTION_DIM), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        # 优先级
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
        
        self.position = 0
        self.size = 0
    
    def add(self,
           state: np.ndarray,
           discrete_action: int,
           continuous_action: np.ndarray,
           reward: float,
           next_state: np.ndarray,
           done: bool):
        """添加经验（初始优先级为最大值）"""
        self.states[self.position] = state
        self.discrete_actions[self.position] = discrete_action
        self.continuous_actions[self.position] = continuous_action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        # 新经验使用最大优先级
        self.priorities[self.position] = self.max_priority
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple:
        """
        基于优先级采样
        
        Args:
            batch_size: 批次大小
            
        Returns:
            (states, discrete_actions, continuous_actions, rewards, next_states, 
             dones, indices, weights)
        """
        # 计算采样概率
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 采样索引
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # 计算重要性采样权重
        beta = self._get_beta()
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()  # 归一化
        
        # 获取数据
        states = torch.FloatTensor(self.states[indices])
        discrete_actions = torch.LongTensor(self.discrete_actions[indices])
        continuous_actions = torch.FloatTensor(self.continuous_actions[indices])
        rewards = torch.FloatTensor(self.rewards[indices])
        next_states = torch.FloatTensor(self.next_states[indices])
        dones = torch.FloatTensor(self.dones[indices])
        weights = torch.FloatTensor(weights)
        
        self.frame += 1
        
        return states, discrete_actions, continuous_actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        更新优先级
        
        Args:
            indices: 样本索引
            priorities: 新的优先级（通常是TD误差）
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
        
        self.max_priority = max(self.max_priority, priorities.max())
    
    def _get_beta(self) -> float:
        """计算当前β值（线性增长）"""
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
    
    def __len__(self):
        return self.size


class NStepReplayBuffer:
    """
    N-step经验回放缓冲区
    对应论文 Section 3.4: N-step Learning (n=5)
    
    累积多步奖励以获得更好的信用分配
    """
    
    def __init__(self,
                 capacity: int = Config.BUFFER_SIZE,
                 state_shape: tuple = (Config.SEQUENCE_LENGTH, Config.ENCODER_INPUT_DIM),
                 n_step: int = Config.N_STEP,
                 gamma: float = Config.GAMMA):
        """
        初始化N-step缓冲区
        
        Args:
            capacity: 缓冲区容量
            state_shape: 状态形状
            n_step: 步数 (论文: 5)
            gamma: 折扣因子
        """
        self.capacity = capacity
        self.state_shape = state_shape
        self.n_step = n_step
        self.gamma = gamma
        
        # 主缓冲区
        self.buffer = ReplayBuffer(capacity, state_shape)
        
        # N-step缓冲区（临时存储）
        self.n_step_buffer = deque(maxlen=n_step)
    
    def add(self,
           state: np.ndarray,
           discrete_action: int,
           continuous_action: np.ndarray,
           reward: float,
           next_state: np.ndarray,
           done: bool):
        """
        添加经验（计算N-step return）
        """
        # 添加到N-step缓冲区
        self.n_step_buffer.append((state, discrete_action, continuous_action, reward, next_state, done))
        
        # 如果缓冲区满了或episode结束，计算N-step return
        if len(self.n_step_buffer) == self.n_step or done:
            # 获取第一个经验
            state_0, discrete_action_0, continuous_action_0, _, _, _ = self.n_step_buffer[0]
            
            # 计算N-step累积奖励
            n_step_reward = 0
            for i, (_, _, _, r, _, d) in enumerate(self.n_step_buffer):
                n_step_reward += (self.gamma ** i) * r
                if d:
                    break
            
            # 获_n = self.n_step_buffer[-1]
            
            # 添加到主缓冲区
            self.buffer.add(
                state_0,
                discrete_action_0,
                continuous_action_0,
                n_step_reward,
                next_state_n,
                done_n
            )
            
            # 如果episode结束，清空N-step缓冲区
            if done:
                self.n_step_buffer.clear()
    
    def sample(self, batch_size: int) -> Tuple:
        """从主缓冲区采样"""
        return self.buffer.sample(batch_size)
    
    def __len__(self):
        return len(self.buffer)


class HybridReplayBuffer:
    """
    混合缓冲区（整合所有特性）
    支持: 优先回放 + N-step + 混合动作空间
    """
    
    def __init__(self,
                 capacity: int = Config.BUFFER_SIZE,
                 state_shape: tuple = (Config.SEQUENCE_LENGTH, Config.ENCODER_INPUT_DIM),
                 use_prioritized: bool = Config.USE_PRIORITIZED_REPLAY,
                 use_n_step: bool = Config.USE_N_STEP):
        """
        初始化混合缓冲区
        
        Args:
            capacity: 容量
            state_shape: 状态形状
            use_prioritized: 是否使用优先回放
            use_n_step: 是否使用N-step
        """
        self.use_prioritized = use_prioritized
        self.use_n_step = use_n_step
        
        # 选择基础缓冲区类型
        if use_prioritized:
            self.buffer = PrioritizedReplayBuffer(capacity, state_shape)
        else:
            self.buffer = ReplayBuffer(capacity, state_shape)
        
        # 如果使用N-step，包装一层
        if use_n_step:
            self.n_step_buffer = deque(maxlen=Config.N_STEP)
            self.gamma = Config.GAMMA
    
    def add(self, *args):
        """添加经验"""
        if self.use_n_step:
            self._add_with_nstep(*args)
        else:
            self.buffer.add(*args)
    
    def _add_with_nstep(self,
                       state: np.ndarray,
                       discrete_action: int,
                       continuous_action: np.ndarray,
                       reward: float,
                       next_state: np.ndarray,
                       done: bool):
        """使用N-step添加"""
        self.n_step_buffer.append((state, discrete_action, continuous_action, reward, next_state, done))
        
        if len(self.n_step_buffer) == Config.N_STEP or done:
            state_0, discrete_action_0, continuous_action_0, _, _, _ = self.n_step_buffer[0]
            
            n_step_reward = 0
            for i, (_, _, _, r, _, d) in enumerate(self.n_step_buffer):
                n_step_reward += (self.gamma ** i) * r
                if d:
                    break
            
            _, _, _, _, next_state_n, done_n = self.n_step_buffer[-1]
            
            self.buffer.add(
                state_0,
                discrete_action_0,
                continuous_action_0,
                n_step_reward,
                next_state_n,
                done_n
            )
            
            if done:
                self.n_step_buffer.clear()
    
    def sample(self, batch_size: int):
        """采样"""
        return self.buffer.sample(batch_size)
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """更新优先级（仅用于优先回放）"""
        if self.use_prioritized:
            self.buffer.update_priorities(indices, priorities)
    
    def __len__(self):
        return len(self.buffer)


def test_replay_buffer():
    """测试回放缓冲区"""
    print("=" * 60)
    print("测试经验回放缓冲区 (Replay Buffer Test)")
    print("=" * 60)
    
    # 测试标准缓冲区
    print("\n1. 测试标准缓冲区")
    buffer = ReplayBuffer(capacity=1000)
    
    # 添加经验
    for i in range(100):
        state = np.random.randn(Config.SEQUENCE_LENGTH, Config.ENCODER_INPUT_DIM)
        next_state = np.random.randn(Config.SEQUENCE_LENGTH, Config.ENCODER_INPUT_DIM)
        buffer.add(
            state=state,
            discrete_action=np.random.randint(0, 3),
            continuous_action=np.random.random(Config.CONTINUOUS_ACTION_DIM),
            reward=np.random.randn(),
            next_state=next_state,
            done=False
        )
    
    print(f"  缓冲区大小: {len(buffer)}")
    
    # 采样
    batch = buffer.sample(32)
    print(f"  采样批次: {len(batch)} 个张量")
    print(f"  状态形状: {batch[0].shape}")
    
    # 测试优先回放缓冲区
    print("\n2. 测试优先回放缓冲区")
    pri_buffer = PrioritizedReplayBuffer(capacity=1000)
    
    for i in range(100):
        state = np.random.randn(Config.SEQUENCE_LENGTH, Config.ENCODER_INPUT_DIM)
        next_state = np.random.randn(Config.SEQUENCE_LENGTH, Config.ENCODER_INPUT_DIM)
        pri_buffer.add(
            state=state,
            discrete_action=np.random.randint(0, 3),
            continuous_action=np.random.random(Config.CONTINUOUS_ACTION_DIM),
            reward=np.random.randn(),
            next_state=next_state,
            done=False
        )
    
    batch = pri_buffer.sample(32)
    print(f"  缓冲区大小: {len(pri_buffer)}")
    print(f"  采样批次: {len(batch)} 个元素 (包括indices和weights)")
    
    # 测试混合缓冲区
    print("\n3. 测试混合缓冲区")
    hybrid_buffer = HybridReplayBuffer(
        capacity=1000,
        use_prioritized=True,
        use_n_step=True
    )
    
    for i in range(100):
        state = np.random.randn(Config.SEQUENCE_LENGTH, Config.ENCODER_INPUT_DIM)
        next_state = np.random.randn(Config.SEQUENCE_LENGTH, Config.ENCODER_INPUT_DIM)
        hybrid_buffer.add(
            state=state,
            discrete_action=np.random.randint(0, 3),
            continuous_action=np.random.random(Config.CONTINUOUS_ACTION_DIM),
            reward=np.random.randn(),
            next_state=next_state,
            done=np.random.random() < 0.01
        )
    
    print(f"  缓冲区大小: {len(hybrid_buffer)}")
    print(f"  使用优先回放: {hybrid_buffer.use_prioritized}")
    print(f"  使用N-step: {hybrid_buffer.use_n_step}")
    
    print(f"\n✓ 所有缓冲区测试通过！")


if __name__ == "__main__":
    test_replay_buffer()

