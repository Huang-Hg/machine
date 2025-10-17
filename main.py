import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from PIL import Image

from DFCR import DFCR, CaptchaDataset, DFCRTrainer


# =============================================================================
# 训练/验证环境
# =============================================================================

def setup_training(
    data_folder: str,
    char_set: str = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
    num_chars: int = 4,
    num_classes_per_char: int = 62,
    batch_size: int = 16,
    num_workers: int = 4,
    val_split: float = 0.2
):
    """设置训练/验证环境"""
    print("加载数据集...")
    full_dataset = CaptchaDataset(
        image_folder=data_folder,
        char_set=char_set,
        num_chars=num_chars
    )

    # 训练/验证划分
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"数据集总数: {len(full_dataset)}")
    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")

    # DataLoader（直接使用 CaptchaDataset.collate_fn）
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=CaptchaDataset.collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=CaptchaDataset.collate_fn,
        pin_memory=True
    )

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 模型
    print("\n创建DFCR模型...")
    model = DFCR(
        input_channels=3,
        num_classes_per_char=num_classes_per_char,
        num_chars=num_chars,
        growth_rate=32,
        dataset_type=1  # Dataset #1
    ).to(device)

    # 模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 优化器 & 调度器 & 损失
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        nesterov=True,
        weight_decay=1e-4
    )
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    return model, train_loader, val_loader, optimizer, scheduler, criterion, device, full_dataset


def main():
    """主函数"""
    config = {
        'data_folder': './data/captcha_images',
        'char_set': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
        'num_chars': 4,
        'num_classes_per_char': 62,
        'batch_size': 16,
        'num_workers': 4,
        'val_split': 0.2,
        'num_epochs': 100,
        'save_dir': './checkpoints'
    }

    (model, train_loader, val_loader,
     optimizer, scheduler, criterion, device, ds) = setup_training(
        data_folder=config['data_folder'],
        char_set=config['char_set'],
        num_chars=config['num_chars'],
        num_classes_per_char=config['num_classes_per_char'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        val_split=config['val_split']
    )

    trainer = DFCRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_dir=config['save_dir']
    )

    trainer.train(
        num_epochs=config['num_epochs'],
        scheduler=scheduler
    )

def inference_example(
    model_path: str,
    image_path: str,
    char_set: str,
    num_chars: int
) -> str:
    """
    推理示例：
    - 与训练保持一致的 char_set / num_chars
    - 直接复用 CaptchaDataset.decode_logits_list 进行解码
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 复用数据集的预处理 & 字符表
    dummy_ds = CaptchaDataset(
        image_folder=os.path.dirname(image_path) or '.',
        char_set=char_set,
        num_chars=num_chars
    )
    preprocess = dummy_ds.transform

    # 构建模型
    model = DFCR(
        input_channels=3,
        num_classes_per_char=len(char_set),
        num_chars=num_chars,
        growth_rate=32,
        dataset_type=1
    ).to(device)

    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    state = checkpoint.get('model_state_dict', checkpoint)     # 兼容不同保存格式
    model.load_state_dict(state, strict=True)
    model.eval()

    # 读图并推理
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)

    # 解码
    texts = CaptchaDataset.decode_logits_list(outputs, dummy_ds.idx_to_char)
    predicted_text = texts[0] if texts else ""
    print(f"预测结果: {predicted_text}")
    return predicted_text


if __name__ == "__main__":
    print("""
    DFCR训练脚本（重构版）
    
    使用方法:
    1. 准备数据: 将 CAPTCHA 图片放在文件夹中，文件名作为标签（例如: ABC12.jpg）
    2. 修改 config 中的 data_folder 路径
    3. 运行: python main.py
    
    推理使用:
    predicted_text = inference_example(
        model_path='./checkpoints/best_model.pth',
        image_path='test_image.jpg',
        char_set='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
        num_chars=4
    )
    """)
    
    main()