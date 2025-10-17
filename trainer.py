"""
DFCR完整训练脚本示例
包含数据加载、训练、验证和保存模型
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
import time
from pathlib import Path

from DFCR import DFCR, CaptchaDataset


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
        save_dir: str = './checkpoints'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
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
                print(f'  🎉 新的最佳准确率: {val_acc:.4f}')
            
            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(epoch, val_acc, is_best)
        
        print(f'\n{"="*60}')
        print(f'训练完成！')
        print(f'最佳验证准确率: {self.best_val_acc:.4f}')
        print(f'{"="*60}\n')


def collate_fn(batch):
    """自定义collate函数，处理多标签情况"""
    images = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    
    # 转置标签，使其成为 [num_chars, batch_size]
    num_chars = len(labels[0])
    labels_transposed = [
        torch.tensor([labels[i][j] for i in range(len(labels))], dtype=torch.long)
        for j in range(num_chars)
    ]
    
    return images, labels_transposed


def setup_training(
    data_folder: str,
    char_set: str = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
    num_chars: int = 5,
    num_classes_per_char: int = 62,
    batch_size: int = 16,
    num_workers: int = 4,
    val_split: float = 0.2
):
    """设置训练环境"""
    
    # 创建数据集
    print("加载数据集...")
    full_dataset = CaptchaDataset(
        image_folder=data_folder,
        char_set=char_set,
        num_chars=num_chars
    )
    
    # 分割训练集和验证集
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"数据集总数: {len(full_dataset)}")
    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 创建模型
    print("\n创建DFCR模型...")
    model = DFCR(
        input_channels=3,
        num_classes_per_char=num_classes_per_char,
        num_chars=num_chars,
        growth_rate=32,
        dataset_type=1  # Dataset #1
    ).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 创建优化器
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        nesterov=True,
        weight_decay=1e-4
    )
    
    # 学习率调度器
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    return model, train_loader, val_loader, optimizer, scheduler, criterion, device


def main():
    """主函数"""
    # 配置参数
    config = {
        'data_folder': './data/captcha_images',  # 数据文件夹路径
        'char_set': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
        'num_chars': 5,
        'num_classes_per_char': 62,
        'batch_size': 16,
        'num_workers': 4,
        'val_split': 0.2,
        'num_epochs': 100,
        'save_dir': './checkpoints'
    }
    
    # 设置训练环境
    model, train_loader, val_loader, optimizer, scheduler, criterion, device = setup_training(
        data_folder=config['data_folder'],
        char_set=config['char_set'],
        num_chars=config['num_chars'],
        num_classes_per_char=config['num_classes_per_char'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        val_split=config['val_split']
    )
    
    # 创建训练器
    trainer = DFCRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_dir=config['save_dir']
    )
    
    # 开始训练
    trainer.train(
        num_epochs=config['num_epochs'],
        scheduler=scheduler
    )


def inference_example(model_path: str, image_path: str, char_set: str):
    """推理示例"""
    from torchvision import transforms
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DFCR(
        input_channels=3,
        num_classes_per_char=len(char_set),
        num_chars=5,
        growth_rate=32,
        dataset_type=1
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 准备图像
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # 解码结果
    char_to_idx = {char: idx for idx, char in enumerate(char_set)}
    idx_to_char = {idx: char for idx, char in enumerate(char_set)}
    
    predicted_text = ''
    if isinstance(outputs, list):
        for output in outputs:
            idx = output.argmax(1).item()
            predicted_text += idx_to_char[idx]
    else:
        idx = outputs.argmax(1).item()
        predicted_text = idx_to_char[idx]
    
    print(f"预测结果: {predicted_text}")
    return predicted_text


if __name__ == "__main__":
    print("""
    DFCR训练脚本
    
    使用方法:
    1. 准备数据: 将CAPTCHA图片放在文件夹中，文件名作为标签（例如: ABC12.jpg）
    2. 修改config中的data_folder路径
    3. 运行: python train_dfcr.py
    
    推理使用:
    predicted_text = inference_example(
        model_path='./checkpoints/best_model.pth',
        image_path='test_image.jpg',
        char_set='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    )
    """)
    
    # 取消注释以运行训练
    # main()