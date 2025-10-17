"""
DFCR (DenseNet for CAPTCHA Recognition) PyTorch实现
基于论文: CAPTCHA recognition based on deep convolutional neural network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from time import time
import os
from typing import List, Tuple, Optional,Union


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
        num_chars: int = 4,               # 验证码字符数
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
            # Dataset #2: 4字符，62类（10数字+26大写+26小写）
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
# ============================================================================s

class CaptchaDataset(Dataset):
    """
    - 验证码数据集（文件夹格式）
    - 训练/验证/推理都调用这里的工具方法
    """
    def __init__(
        self,
        image_folder: str,
        char_set: str = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
        num_chars: int = 4,
        transform: Optional[transforms.Compose] = None
    ):
        self.image_folder = image_folder
        self.char_set = char_set
        self.num_chars = num_chars

        # 收集样本
        self.image_files: List[str] = []
        self.labels: List[str] = []
        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.webp')):
                label = os.path.splitext(filename)[0]
                self.image_files.append(os.path.join(image_folder, filename))
                self.labels.append(label)

        # 预处理
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


    # 编码/解码
    def encode_label(self, label_text: str) -> List[int]:
        """将标签文本编码为索引列表（长度不足会右侧填充0；未知字符映射到0）"""
        indices: List[int] = []
        for ch in label_text[:self.num_chars]:
            idx = self.char_set.find(ch)
            indices.append(0 if idx == -1 else idx)
        while len(indices) < self.num_chars:
            indices.append(0)
        return indices

    def decode_indices(self, indices: List[int]) -> str:
        """将索引列表解码为字符串（越界索引按0处理）"""
        out_chars = []
        for i in indices[:self.num_chars]:
            i_int = int(i)
            if 0 <= i_int < len(self.char_set):
                out_chars.append(self.char_set[i_int])
            else:
                out_chars.append(self.char_set[0])
        return ''.join(out_chars)

    @staticmethod
    def decode_logits_list(
        logits_list: Union[List[torch.Tensor], torch.Tensor],
        char_set: str
    ) -> List[str]:
        """
        将模型输出（按字符位的 logits 列表）解码为 batch 的字符串列表。
        期待 logits_list 为长度=num_chars 的列表，每个 tensor 形状 [B, C]。
        """
        def safe_idx_to_char(idx: int) -> str:
            return char_set[idx] if 0 <= idx < len(char_set) else char_set[0]

        if isinstance(logits_list, torch.Tensor):
            # 有些实现会直接返回 [B, C]，视作单字符
            preds = logits_list.argmax(dim=1).tolist()  # [B]
            return [''.join(safe_idx_to_char(p) for p in preds)]

        # 标准：list[ num_chars * (B, C) ]
        batch_size = logits_list[0].shape[0]
        num_chars = len(logits_list)
        # [num_chars, B]
        idx_stacked = torch.stack([lg.argmax(dim=1) for lg in logits_list], dim=0)  # long
        # 转为 [B, num_chars]
        idx_transposed = idx_stacked.permute(1, 0).tolist()
        decoded = []
        for b in range(batch_size):
            decoded.append(''.join(safe_idx_to_char(i) for i in idx_transposed[b][:num_chars]))
        return decoded


    # DataLoader 的 collate 函数
    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, List[int]]]):
        """
        处理多字符标签：
        - images: [B, C, H, W]
        - labels_transposed: list[num_chars]，其中每个元素形状为 [B]
        """
        images = torch.stack([item[0] for item in batch], dim=0)
        labels = [item[1] for item in batch]  # list[B][num_chars]
        num_chars = len(labels[0])
        labels_transposed = [
            torch.tensor([labels[i][j] for i in range(len(labels))], dtype=torch.long)
            for j in range(num_chars)
        ]
        return images, labels_transposed


    # Dataset接口
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label_text = self.labels[idx]
        label_indices = self.encode_label(label_text)
        return image, label_indices



# ============================================================================
# 第四部分：推理示例    
# ============================================================================

class DFCRTrainer:
    """DFCR训练器"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        save_dir: str = './checkpoints',
        num_chars: int = 4
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_chars = num_chars
        self.best_val_acc = 0.0
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
    
    def train_epoch(self, epoch: int) -> tuple:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            
            # 将labels转换为list of tensors
            if not isinstance(labels, list):
                labels = [labels[:, i].to(self.device) for i in range(labels.shape[1])]
            else:
                labels = [label.to(self.device) for label in labels]
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # 计算损失
            if isinstance(outputs, list):
                # 多输出情况
                loss = sum([self.criterion(output, label) 
                           for output, label in zip(outputs, labels)])
                
                # 计算准确率（所有字符都正确才算正确）
                batch_correct = 0
                for i in range(len(images)):
                    all_correct = all([
                        outputs[j][i].argmax() == labels[j][i]
                        for j in range(len(outputs))
                    ])
                    if all_correct:
                        batch_correct += 1
                correct += batch_correct
            else:
                # 单输出情况
                loss = self.criterion(outputs, labels[0])
                correct += (outputs.argmax(1) == labels[0]).sum().item()
            
            total += len(images)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                print(f'  Batch [{batch_idx+1}/{len(self.train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        print(f'Epoch {epoch} - 训练完成, 耗时: {epoch_time:.2f}s')
        print(f'  平均损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}')
        
        return avg_loss, accuracy
    
    def validate(self, epoch: int) -> tuple:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                
                if not isinstance(labels, list):
                    labels = [labels[:, i].to(self.device) for i in range(labels.shape[1])]
                else:
                    labels = [label.to(self.device) for label in labels]
                
                # 前向传播
                outputs = self.model(images)
                
                # 计算损失
                if isinstance(outputs, list):
                    loss = sum([self.criterion(output, label) 
                               for output, label in zip(outputs, labels)])
                    
                    # 计算准确率
                    batch_correct = 0
                    for i in range(len(images)):
                        all_correct = all([
                            outputs[j][i].argmax() == labels[j][i]
                            for j in range(len(outputs))
                        ])
                        if all_correct:
                            batch_correct += 1
                    correct += batch_correct
                else:
                    loss = self.criterion(outputs, labels[0])
                    correct += (outputs.argmax(1) == labels[0]).sum().item()
                
                total += len(images)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        print(f'Epoch {epoch} - 验证结果:')
        print(f'  平均损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}')
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs
        }
        
        # 保存最新模型
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f'保存检查点: {checkpoint_path}')
        
        # 如果是最佳模型，额外保存
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f'保存最佳模型: {best_path}')
    
    def train(self, num_epochs: int, scheduler=None):
        """完整训练流程"""
        print(f'\n开始训练，共 {num_epochs} 个epochs')
        print(f'训练集大小: {len(self.train_loader.dataset)}')
        print(f'验证集大小: {len(self.val_loader.dataset)}')
        print(f'设备: {self.device}\n')
        
        for epoch in range(1, num_epochs + 1):
            print(f'\n{"="*60}')
            print(f'Epoch {epoch}/{num_epochs}')
            print(f'{"="*60}')
            
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # 验证
            val_loss, val_acc = self.validate(epoch)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # 学习率调整
            if scheduler is not None:
                scheduler.step()
                print(f'  当前学习率: {scheduler.get_last_lr()[0]:.6f}')
            
            # 保存模型
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                print(f'新的最佳准确率: {val_acc:.4f}')
            
            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(epoch, val_acc, is_best)
        
        print(f'\n{"="*60}')
        print(f'训练完成！')
        print(f'最佳验证准确率: {self.best_val_acc:.4f}')
        print(f'{"="*60}\n')