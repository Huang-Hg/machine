"""
DFCR (DenseNet for CAPTCHA Recognition) PyTorch实现
基于论文: CAPTCHA recognition based on deep convolutional neural network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from typing import List, Tuple, Optional


# ============================================================================
# 第一部分：Dense Block 核心组件
# ============================================================================

class DenseLayer(nn.Module):
    """
    DenseNet的基本层单元
    结构：BN → ReLU → Conv(1×1) → BN → ReLU → Conv(3×3)
    """
    def __init__(self, in_channels: int, growth_rate: int):
        super(DenseLayer, self).__init__()
        self.growth_rate = growth_rate
        
        # Bottleneck层：1×1卷积降维
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            4 * growth_rate,  # 通常是growth_rate的4倍
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        
        # 3×3卷积
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(
            4 * growth_rate,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        
    def forward(self, x):
        # Bottleneck
        out = self.bn1(x)
        out = F.relu(out, inplace=True)
        out = self.conv1(out)
        
        # 3×3 Conv
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        
        # 跨层连接：拼接输入和输出
        out = torch.cat([x, out], dim=1)
        return out


class DenseBlock(nn.Module):
    """
    Dense Block：包含多个DenseLayer
    每层的输入是前面所有层的输出拼接
    """
    def __init__(self, num_layers: int, in_channels: int, growth_rate: int):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        
        # 创建多个DenseLayer
        layers = []
        for i in range(num_layers):
            layers.append(
                DenseLayer(
                    in_channels + i * growth_rate,
                    growth_rate
                )
            )
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        # 顺序通过每个DenseLayer
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionLayer(nn.Module):
    """
    Transition层：降采样
    结构：BN → Conv(1×1) → AvgPool(2×2)
    """
    def __init__(self, in_channels: int, compression: float = 0.5):
        super(TransitionLayer, self).__init__()
        self.compression = compression
        out_channels = int(in_channels * compression)
        
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        out = self.bn(x)
        out = F.relu(out, inplace=True)
        out = self.conv(out)
        out = self.pool(out)
        return out


# ============================================================================
# 第二部分：DFCR主模型
# ============================================================================

class DFCR(nn.Module):
    """
    DFCR完整模型
    支持三种数据集配置
    """
    def __init__(
        self,
        input_channels: int = 3,
        num_classes_per_char: int = 62,  # 每个字符的类别数
        num_chars: int = 5,               # 验证码字符数
        growth_rate: int = 32,
        dataset_type: int = 1             # 1, 2, 或 3
    ):
        super(DFCR, self).__init__()
        
        self.num_classes_per_char = num_classes_per_char
        self.num_chars = num_chars
        self.growth_rate = growth_rate
        self.dataset_type = dataset_type
        
        # ========== 初始卷积和池化 ==========
        # 7×7卷积，stride=2
        self.conv1 = nn.Conv2d(
            input_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        
        # 3×3最大池化，stride=2
        self.pool1 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )
        # 输出：56×56
        
        # 计算每个Dense Block后的通道数
        num_channels = 64
        
        # ========== Dense Block 1 (6层) ==========
        self.dense_block1 = DenseBlock(
            num_layers=6,
            in_channels=num_channels,
            growth_rate=growth_rate
        )
        num_channels += 6 * growth_rate
        
        # Transition 1
        self.transition1 = TransitionLayer(
            in_channels=num_channels,
            compression=0.5
        )
        num_channels = int(num_channels * 0.5)
        # 输出：28×28
        
        # ========== Dense Block 2 (6层) ==========
        # 论文中从12层减少到6层，降低内存消耗
        self.dense_block2 = DenseBlock(
            num_layers=6,
            in_channels=num_channels,
            growth_rate=growth_rate
        )
        num_channels += 6 * growth_rate
        
        # Transition 2
        self.transition2 = TransitionLayer(
            in_channels=num_channels,
            compression=0.5
        )
        num_channels = int(num_channels * 0.5)
        # 输出：14×14
        
        # ========== Dense Block 3 (24层) ==========
        self.dense_block3 = DenseBlock(
            num_layers=24,
            in_channels=num_channels,
            growth_rate=growth_rate
        )
        num_channels += 24 * growth_rate
        
        # Transition 3
        self.transition3 = TransitionLayer(
            in_channels=num_channels,
            compression=0.5
        )
        num_channels = int(num_channels * 0.5)
        # 输出：7×7
        
        # ========== Dense Block 4 (16层) ==========
        self.dense_block4 = DenseBlock(
            num_layers=16,
            in_channels=num_channels,
            growth_rate=growth_rate
        )
        num_channels += 16 * growth_rate
        
        # 最后的BN
        self.bn_final = nn.BatchNorm2d(num_channels)
        
        # ========== 全局平均池化 ==========
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # ========== 分类层（多任务） ==========
        if dataset_type == 1:
            # Dataset #1: 5字符，62类（10数字+26大写+26小写）
            self.classifiers = nn.ModuleList([
                nn.Linear(num_channels, num_classes_per_char)
                for _ in range(num_chars)
            ])
        elif dataset_type == 2:
            # Dataset #2: 4字符，36类（10数字+26大写）
            self.classifiers = nn.ModuleList([
                nn.Linear(num_channels, num_classes_per_char)
                for _ in range(num_chars)
            ])
        else:  # dataset_type == 3
            # Dataset #3: 中文字符，单分类器
            self.classifier = nn.Linear(num_channels, num_classes_per_char)
    
    def forward(self, x):
        # 初始卷积和池化
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.pool1(x)
        
        # Dense Block 1 + Transition 1
        x = self.dense_block1(x)
        x = self.transition1(x)
        
        # Dense Block 2 + Transition 2
        x = self.dense_block2(x)
        x = self.transition2(x)
        
        # Dense Block 3 + Transition 3
        x = self.dense_block3(x)
        x = self.transition3(x)
        
        # Dense Block 4
        x = self.dense_block4(x)
        
        # 最后的BN和ReLU
        x = self.bn_final(x)
        x = F.relu(x, inplace=True)
        
        # 全局平均池化
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        
        # 分类
        if self.dataset_type == 3:
            return self.classifier(x)
        else:
            outputs = [classifier(x) for classifier in self.classifiers]
            return outputs


# ============================================================================
# 第三部分：数据处理模块
# ============================================================================

class CaptchaDataset(Dataset):
    """
    验证码数据集
    支持图像文件夹格式
    """
    def __init__(
        self,
        image_folder: str,
        char_set: str = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
        num_chars: int = 5,
        transform: Optional[transforms.Compose] = None
    ):
        self.image_folder = image_folder
        self.char_set = char_set
        self.num_chars = num_chars
        self.char_to_idx = {char: idx for idx, char in enumerate(char_set)}
        self.idx_to_char = {idx: char for idx, char in enumerate(char_set)}
        
        # 获取所有图像文件
        self.image_files = []
        self.labels = []
        
        for filename in os.listdir(image_folder):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                # 文件名即为标签
                label = os.path.splitext(filename)[0]
                self.image_files.append(os.path.join(image_folder, filename))
                self.labels.append(label)
        
        # 数据转换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载图像
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # 处理标签
        label_text = self.labels[idx]
        labels = self._label_to_indices(label_text)
        
        return image, labels
    
    def _label_to_indices(self, label_text: str) -> List[int]:
        """将标签文本转换为索引列表"""
        indices = []
        for i, char in enumerate(label_text[:self.num_chars]):
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                indices.append(0)  # 未知字符用0填充
        
        # 如果标签不足，填充0
        while len(indices) < self.num_chars:
            indices.append(0)
        
        return indices
    
    def decode_predictions(self, predictions: torch.Tensor) -> str:
        """将模型预测转换为文本"""
        label_text = ''
        for pred in predictions:
            idx = pred.argmax().item()
            label_text += self.idx_to_char.get(idx, '?')
        return label_text
