from main import initialize_model,setup_training,setup_training,setup_seed,predict_api
from DFCR import DFCRTrainer,DFCR,CaptchaDataset,Features_plt
import torch
import os
from PIL import Image


def fileimg_predict(
    image_folder: str,
    num_chars: int,
    char_set: str = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
    model_path: str = './checkpoints/best_model.pth'
):
    """
    批量预测指定文件夹下的验证码图片并计算整体准确率。
    简化版：内部加载模型、预处理、推理与评估，无需 predict_api。
    """
    model,device,preprocess = initialize_model(char_set, num_chars, model_path)
    model.eval()

    # 收集样本文件
    image_files = []
    labels = []
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.webp')):
            labels.append(os.path.splitext(filename)[0])  # 文件名即标签
            image_files.append(os.path.join(image_folder, filename))

    # 推理与结果收集
    correct = 0
    total = len(image_files)

    for img_path, true_label in zip(image_files, labels):

        image = Image.open(img_path).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)

        # 调用decode_predictions
        pred_text = CaptchaDataset.decode_predictions(outputs, num_chars, char_set)

        if pred_text == true_label:
            correct += 1

        print(f"{os.path.basename(img_path)} => 预测: {pred_text} | 正确: {true_label}")

    # 统计结果
    acc = correct / total if total > 0 else 0
    print(f"\n总验证集数: {total}, 正确数: {correct}, 准确率: {acc:.4f}")
    return acc

def fine_tuning(num_chars: int, char_set: str, data_folder: str, pretrained_model_path: str = './checkpoints/best_model.pth'):
    """
    使用新数据(./data_folder下图片)微调模型10轮，
    将微调后的模型保存到./checkpoints/tuned_model  
    
    Args:
        num_chars: 验证码字符数
        char_set: 字符集
        data_folder: 新数据文件夹路径
        pretrained_model_path: 预训练模型路径
    
    Returns:
        model: 微调后的模型
        device: 设备
    """
    # 加载预训练模型
    model,device,_ = initialize_model(char_set, num_chars, pretrained_model_path)
    print("预训练模型加载成功！")
    
    # 准备微调数据
    print(f"\n准备微调数据集：{data_folder}")
    (_, train_loader, val_loader,
     optimizer, scheduler, criterion, _, _) = setup_training(
        data_folder=data_folder,
        char_set=char_set,
        num_chars=num_chars,
        num_classes_per_char=len(char_set),
        batch_size=16,
        num_workers=4,
        val_split=0.2
    )
    
    # 创建微调训练器
    trainer = DFCRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_dir='./checkpoints/tuned_model',
        num_chars=num_chars
    )
    
    # 开始微调
    print("\n开始微调训练（10轮）...")
    trainer.train(
        num_epochs=10,
        scheduler=scheduler
    )
    
    print(f"\n微调完成！模型已保存到 ./checkpoints/tuned_model/")
    return model, device

def visualize_features_example():
    """特征可视化示例"""
    from DFCR import features_plt
    
    model_path = './checkpoints/best_model.pth'
    image_path = './data/captch/1.jpeg'
    char_set = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    num_chars = 4
    
    print("加载模型...")
    model, device, preprocess = initialize_model(char_set, num_chars, model_path)
    
    visualizer = Features_plt(model=model, device=device)
    visualizer.visualize(image_path=image_path, preprocess=preprocess)


if __name__ == "__main__":
    print("""
    DFCR训练与可视化脚本
    
    使用方法:
    1. 训练模型: 运行 main()
    2. 预测单张图片: 运行 predict_api()
    3. 特征可视化: 运行 visualize_features_example()
    """)
    
    # 设置随机数种子
    setup_seed(3741)
    visualize_features_example()

