"""
时空特征编码器
对应论文 Section 3.3: Temporal-Spatial Feature Enhancement

核心组件:
1. TCN (Temporal Convolutional Network) - 空间注意力加权
2. GRU (Gated Recurrent Unit) - 时间依赖提取
3. Attention Mechanism - 特征重要性动态调整

输出: 64维潜在表示 (64-dimensional latent representation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from config import Config


class TemporalConvNet(nn.Module):
    """
    时间卷积网络 (TCN)
    对应论文 Section 3.3: Dilated Causal Convolutions with dilation factors [1,2,3]
    
    特点:
    - 因果卷积 (Causal Convolution): 不使用未来信息
    - 膨胀卷积 (Dilated Convolution): 扩大感受野
    - 残差连接 (Residual Connection): 缓解梯度消失
    """
    
    def __init__(self, 
                 input_channels: int,
                 hidden_channels: list = Config.TCN_CHANNELS,
                 kernel_size: int = Config.TCN_KERNEL_SIZE,
                 dilation_rates: list = Config.TCN_DILATION_RATES):
        """
        初始化TCN
        
        Args:
            input_channels: 输入通道数（特征维度）
            hidden_channels: 隐藏层通道数列表
            kernel_size: 卷积核大小
            dilation_rates: 膨胀率列表 [1, 2, 3]
        """
        super(TemporalConvNet, self).__init__()
        
        self.layers = nn.ModuleList()
        num_levels = len(hidden_channels)
        
        for i in range(num_levels):
            dilation = dilation_rates[i] if i < len(dilation_rates) else 1
            in_channels = input_channels if i == 0 else hidden_channels[i-1]
            out_channels = hidden_channels[i]
            
            # 因果卷积: padding = (kernel_size - 1) * dilation
            padding = (kernel_size - 1) * dilation
            
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation
            )
            
            self.layers.append(nn.Sequential(
                conv,
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(0.1)
            ))
        
        # 1x1卷积用于维度匹配（残差连接）
        if input_channels != hidden_channels[-1]:
            self.downsample = nn.Conv1d(input_channels, hidden_channels[-1], 1)
        else:
            self.downsample = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch_size, channels, seq_len]
            
        Returns:
            [batch_size, hidden_channels[-1], seq_len]
        """
        res = x if self.downsample is None else self.downsample(x)
        
        for layer in self.layers:
            x = layer(x)
            # 裁剪到原始长度（因果卷积会增加长度）
            x = x[:, :, :res.size(2)]
        
        # 残差连接
        x = F.relu(x + res)
        return x


class SpatialAttention(nn.Module):
    """
    空间注意力机制
    对应论文 Section 3.3: Spatial attention weighting dynamically adjusts feature importance
    
    动态调整不同特征的重要性
    """
    
    def __init__(self, hidden_dim: int):
        super(SpatialAttention, self).__init__()
        
        self.attention_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算注意力权重
        
        Args:
            x: [batch_size, seq_len, hidden_dim]
            
        Returns:
            (加权后的特征, 注意力权重)
        """
        # 计算注意力分数
        attn_scores = self.attention_fc(x)  # [batch, seq_len, 1]
        attn_weights = F.softmax(attn_scores, dim=1)  # 归一化
        
        # 加权求和
        context_vector = torch.sum(attn_weights * x, dim=1)  # [batch, hidden_dim]
        
        return context_vector, attn_weights


class SpatioTemporalEncoder(nn.Module):
    """
    时空特征编码器
    对应论文 Section 3.3 & 3.4: Shared Spatial-Temporal Feature Encoder
    
    架构:
    1. TCN层 - 提取空间特征（跨特征相关性）
    2. GRU层 - 提取时间依赖
    3. 注意力层 - 动态加权
    
    输出: 64维潜在表示（论文固定维度）
    """
    
    def __init__(self, 
                 input_dim: int = Config.ENCODER_INPUT_DIM,
                 hidden_dim: int = Config.ENCODER_HIDDEN_DIM,
                 tcn_channels: list = Config.TCN_CHANNELS,
                 gru_layers: int = Config.GRU_NUM_LAYERS):
        """
        初始化时空编码器
        
        Args:
            input_dim: 输入特征维度（PCA降维后，默认21）
            hidden_dim: 隐藏层维度（输出维度64）
            tcn_channels: TCN通道数列表
            gru_layers: GRU层数
        """
        super(SpatioTemporalEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # ==================== TCN路径 (空间特征提取) ====================
        # 论文: dilated convolutions with dilation factors [1,2,3] detect cross-feature dependencies
        self.tcn = TemporalConvNet(
            input_channels=input_dim,
            hidden_channels=tcn_channels,
            kernel_size=Config.TCN_KERNEL_SIZE,
            dilation_rates=Config.TCN_DILATION_RATES
        )
        
        # TCN输出映射到GRU输入维度
        self.tcn_to_gru = nn.Linear(tcn_channels[-1], hidden_dim)
        
        # ==================== GRU路径 (时间依赖提取) ====================
        # 论文: GRU networks to extract multi-scale temporal dependencies
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=0.1 if gru_layers > 1 else 0
        )
        
        # ==================== 注意力机制 ====================
        # 论文: temporal attention mechanism
        self.spatial_attention = SpatialAttention(hidden_dim)
        
        # ==================== 特征融合 ====================
        # 融合TCN和GRU的输出
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, input_dim]
            
        Returns:
            (context_vector: [batch_size, hidden_dim], 
             intermediates: 中间结果字典)
        """
        batch_size, seq_len, input_dim = x.size()
        
        # ==================== TCN分支 ====================
        # 转置为 [batch, channels, seq_len] 格式
        x_tcn = x.permute(0, 2, 1)
        tcn_out = self.tcn(x_tcn)  # [batch, tcn_channels[-1], seq_len]
        
        # 转回 [batch, seq_len, channels] 并映射到hidden_dim
        tcn_out = tcn_out.permute(0, 2, 1)  # [batch, seq_len, tcn_channels[-1]]
        tcn_features = self.tcn_to_gru(tcn_out)  # [batch, seq_len, hidden_dim]
        
        # ==================== GRU分支 ====================
        gru_out, gru_hidden = self.gru(x)  # out: [batch, seq_len, hidden_dim]
        
        # ==================== 注意力加权 ====================
        # 对GRU输出应用注意力
        gru_context, attn_weights = self.spatial_attention(gru_out)
        
        # 对TCN输出也应用注意力
        tcn_context, _ = self.spatial_attention(tcn_features)
        
        # ==================== 特征融合 ====================
        # 论文: combines both instantaneous market states and historical context embeddings
        combined = torch.cat([gru_context, tcn_context], dim=1)  # [batch, hidden_dim*2]
        context_vector = self.fusion(combined)  # [batch, hidden_dim]
        
        # 保存中间结果用于可视化和分析
        intermediates = {
            'tcn_features': tcn_features,
            'gru_output': gru_out,
            'attention_weights': attn_weights,
            'context_vector': context_vector
        }
        
        return context_vector, intermediates
    
    def get_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取特征重要性（通过注意力权重）
        
        Args:
            x: [batch_size, seq_len, input_dim]
            
        Returns:
            attention_weights: [batch_size, seq_len, 1]
        """
        _, intermediates = self.forward(x)
        return intermediates['attention_weights']


class EncoderWithGradientStopping(nn.Module):
    """
    带梯度停止的编码器
    对应论文 Section 3.4: gradient stopping layers
    
    用于混合动作空间架构中，确保特征一致性的同时允许专门化优化
    """
    
    def __init__(self, 
                 input_dim: int = Config.ENCODER_INPUT_DIM,
                 hidden_dim: int = Config.ENCODER_HIDDEN_DIM):
        super(EncoderWithGradientStopping, self).__init__()
        
        self.encoder = SpatioTemporalEncoder(input_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, stop_gradient: bool = False) -> torch.Tensor:
        """
        前向传播（可选梯度停止）
        
        Args:
            x: [batch_size, seq_len, input_dim]
            stop_gradient: 是否停止梯度传播
            
        Returns:
            context_vector: [batch_size, hidden_dim]
        """
        context_vector, _ = self.encoder(x)
        
        if stop_gradient:
            # 论文: gradient stopping layers - discrete/continuous policy heads 
            # maintain separate MLPs yet share base network parameters
            context_vector = context_vector.detach()
        
        return context_vector


def test_encoder():
    """测试编码器"""
    print("=" * 60)
    print("测试时空编码器 (Spatio-Temporal Encoder Test)")
    print("=" * 60)
    
    # 创建模拟输入
    batch_size = 32
    seq_len = Config.SEQUENCE_LENGTH
    input_dim = Config.ENCODER_INPUT_DIM
    
    x = torch.randn(batch_size, seq_len, input_dim)
    print(f"\n输入形状: {x.shape}")
    
    # 创建编码器
    encoder = SpatioTemporalEncoder(input_dim=input_dim)
    
    # 前向传播
    context_vector, intermediates = encoder(x)
    
    print(f"\n输出形状:")
    print(f"  Context Vector: {context_vector.shape}")
    print(f"  TCN Features: {intermediates['tcn_features'].shape}")
    print(f"  GRU Output: {intermediates['gru_output'].shape}")
    print(f"  Attention Weights: {intermediates['attention_weights'].shape}")
    
    # 验证输出维度
    assert context_vector.shape == (batch_size, Config.ENCODER_HIDDEN_DIM), \
        f"Expected shape ({batch_size}, {Config.ENCODER_HIDDEN_DIM}), got {context_vector.shape}"
    
    print(f"\n✓ 编码器测试通过！")
    print(f"  参数总数: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"  可训练参数: {sum(p.numel() for p in encoder.parameters() if p.requires_grad):,}")


if __name__ == "__main__":
    test_encoder()

