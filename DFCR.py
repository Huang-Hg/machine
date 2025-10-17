"""
DFCR (DenseNet for CAPTCHA Recognition) 完整实现
基于论文: CAPTCHA recognition based on deep convolutional neural network
"""

import tensorflow as tf
from keras import layers, models
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import numpy as np
from PIL import Image
import os

# ============================================================================
# 第一部分：Dense Block 核心组件
# ============================================================================

class DenseLayer(layers.Layer):
    """
    DenseNet的基本层单元
    结构：BN → ReLU → Conv(1×1) → BN → ReLU → Conv(3×3)
    """
    def __init__(self, growth_rate, **kwargs):
        super(DenseLayer, self).__init__(**kwargs)
        self.growth_rate = growth_rate
        
        # Bottleneck层：1×1卷积降维
        self.bn1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(
            filters=4 * growth_rate,  # 通常是growth_rate的4倍
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=False
        )
        
        # 3×3卷积
        self.bn2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(
            filters=growth_rate,
            kernel_size=3,
            strides=1,
            padding='same',
            use_bias=False
        )
        
    def call(self, inputs):
        # Bottleneck
        x = self.bn1(inputs)
        x = layers.Activation('relu')(x)
        x = self.conv1(x)
        
        # 3×3 Conv
        x = self.bn2(x)
        x = layers.Activation('relu')(x)
        x = self.conv2(x)
        
        # 跨层连接：拼接输入和输出
        return layers.Concatenate()([inputs, x])


class DenseBlock(layers.Layer):
    """
    Dense Block：包含多个DenseLayer
    每层的输入是前面所有层的输出拼接
    """
    def __init__(self, num_layers, growth_rate, name=None, **kwargs):
        super(DenseBlock, self).__init__(name=name, **kwargs)
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        
        # 创建多个DenseLayer
        self.dense_layers = [
            DenseLayer(growth_rate, name=f'dense_layer_{i}')
            for i in range(num_layers)
        ]
    
    def call(self, inputs):
        x = inputs
        # 顺序通过每个DenseLayer
        for layer in self.dense_layers:
            x = layer(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_layers': self.num_layers,
            'growth_rate': self.growth_rate
        })
        return config


class TransitionLayer(layers.Layer):
    """
    Transition层：降采样
    结构：BN → Conv(1×1) → AvgPool(2×2)
    """
    def __init__(self, compression=0.5, name=None, **kwargs):
        super(TransitionLayer, self).__init__(name=name, **kwargs)
        self.compression = compression
        self.bn = layers.BatchNormalization()
        self.pool = layers.AveragePooling2D(pool_size=2, strides=2)
        
    def build(self, input_shape):
        # 动态计算输出通道数
        num_filters = int(input_shape[-1] * self.compression)
        self.conv = layers.Conv2D(
            filters=num_filters,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=False
        )
        
    def call(self, inputs):
        x = self.bn(inputs)
        x = layers.Activation('relu')(x)
        x = self.conv(x)
        x = self.pool(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({'compression': self.compression})
        return config


# ============================================================================
# 第二部分：DFCR主模型
# ============================================================================

class DFCR:
    """
    DFCR完整模型
    支持三种数据集配置
    """
    def __init__(self, 
                 input_shape=(224, 224, 3),
                 num_classes_per_char=62,  # 每个字符的类别数
                 num_chars=5,               # 验证码字符数
                 growth_rate=32,
                 dataset_type=1):           # 1, 2, 或 3
        
        self.input_shape = input_shape
        self.num_classes_per_char = num_classes_per_char
        self.num_chars = num_chars
        self.growth_rate = growth_rate
        self.dataset_type = dataset_type
        self.model = None
        
    def build_model(self):
        """构建完整的DFCR网络"""
        
        # 输入层
        input_img = layers.Input(shape=self.input_shape, name='input')
        
        # ========== 初始卷积和池化 ==========
        # 7×7卷积，stride=2
        x = layers.Conv2D(
            filters=64,
            kernel_size=7,
            strides=2,
            padding='same',
            use_bias=False,
            name='conv1'
        )(input_img)
        x = layers.BatchNormalization(name='bn1')(x)
        x = layers.Activation('relu', name='relu1')(x)
        
        # 3×3最大池化，stride=2
        x = layers.MaxPooling2D(
            pool_size=3,
            strides=2,
            padding='same',
            name='pool1'
        )(x)
        # 输出：56×56
        
        # ========== Dense Block 1 (6层) ==========
        x = DenseBlock(
            num_layers=6,
            growth_rate=self.growth_rate,
            name='dense_block_1'
        )(x)
        
        # Transition 1
        x = TransitionLayer(
            compression=0.5,
            name='transition_1'
        )(x)
        # 输出：28×28
        
        # ========== Dense Block 2 (6层) - 关键改进！==========
        # 论文中从12层减少到6层，降低内存消耗
        x = DenseBlock(
            num_layers=6,  # 原DenseNet-121是12层
            growth_rate=self.growth_rate,
            name='dense_block_2'
        )(x)
        
        # Transition 2
        x = TransitionLayer(
            compression=0.5,
            name='transition_2'
        )(x)
        # 输出：14×14
        
        # ========== Dense Block 3 (24层) ==========
        x = DenseBlock(
            num_layers=24,
            growth_rate=self.growth_rate,
            name='dense_block_3'
        )(x)
        
        # Transition 3
        x = TransitionLayer(
            compression=0.5,
            name='transition_3'
        )(x)
        # 输出：7×7
        
        # ========== Dense Block 4 (16层) ==========
        x = DenseBlock(
            num_layers=16,
            growth_rate=self.growth_rate,
            name='dense_block_4'
        )(x)
        
        # 最后的BN和ReLU
        x = layers.BatchNormalization(name='bn_final')(x)
        x = layers.Activation('relu', name='relu_final')(x)
        
        # ========== 全局平均池化 ==========
        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        # 输出特征向量
        
        # ========== 分类层（多任务） ==========
        outputs = []
        
        if self.dataset_type == 1:
            # Dataset #1: 5字符，62类（10数字+26大写+26小写）
            for i in range(self.num_chars):
                out = layers.Dense(
                    self.num_classes_per_char,
                    activation='softmax',
                    name=f'output_char_{i+1}'
                )(x)
                outputs.append(out)
                
        elif self.dataset_type == 2:
            # Dataset #2: 4字符，36类（10数字+26大写）
            for i in range(self.num_chars):
                out = layers.Dense(
                    self.num_classes_per_char,
                    activation='softmax',
                    name=f'output_char_{i+1}'
                )(x)
                outputs.append(out)
                
        else:  # dataset_type == 3
            # Dataset #3: 中文字符，单分类器
            outputs = layers.Dense(
                self.num_classes_per_char,
                activation='softmax',
                name='output'
            )(x)
        
        # 创建模型
        self.model = models.Model(
            inputs=input_img,
            outputs=outputs,
            name='DFCR'
        )
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """编译模型"""
        if self.model is None:
            raise ValueError("请先调用build_model()构建模型")
        
        # 优化器
        optimizer = SGD(
            learning_rate=learning_rate,
            momentum=0.9,
            nesterov=True
        )
        
        # 损失函数
        if isinstance(self.model.output, list):
            # 多输出：每个字符一个分类器
            losses = ['categorical_crossentropy'] * len(self.model.output)
            metrics = [['accuracy']] * len(self.model.output)
        else:
            # 单输出
            losses = 'categorical_crossentropy'
            metrics = ['accuracy']
        
        # 编译
        self.model.compile(
            optimizer=optimizer,
            loss=losses,
            metrics=metrics
        )
        
        print("✓ 模型编译完成")
        
    def summary(self):
        """打印模型结构"""
        if self.model is None:
            raise ValueError("请先调用build_model()构建模型")
        self.model.summary()


# ============================================================================
# 第三部分：数据处理模块
# ============================================================================

class CaptchaDataGenerator:
    """
    验证码数据生成器
    支持TFRecord格式和图像文件夹格式
    """
    def __init__(self, 
                 image_size=(224, 224),
                 num_chars=5,
                 char_set='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'):
        
        self.image_size = image_size
        self.num_chars = num_chars
        self.char_set = char_set
        self.char_to_idx = {char: idx for idx, char in enumerate(char_set)}
        self.idx_to_char = {idx: char for idx, char in enumerate(char_set)}
        
    def preprocess_image(self, image_path):
        """图像预处理"""
        # 读取图像
        img = Image.open(image_path).convert('RGB')
        
        # 调整大小
        img = img.resize(self.image_size)
        
        # 转换为数组并归一化
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0  # 归一化到[0,1]
        
        return img_array
    
    def label_to_onehot(self, label_text):
        """
        将标签文本转换为one-hot编码
        例如："ABC12" -> [one_hot(A), one_hot(B), one_hot(C), one_hot(1), one_hot(2)]
        """
        labels = []
        for char in label_text[:self.num_chars]:
            if char in self.char_to_idx:
                # 创建one-hot向量
                one_hot = np.zeros(len(self.char_set))
                one_hot[self.char_to_idx[char]] = 1
                labels.append(one_hot)
        
        # 如果标签不足，填充零向量
        while len(labels) < self.num_chars:
            labels.append(np.zeros(len(self.char_set)))
        
        return labels
    
    def onehot_to_label(self, predictions):
        """
        将模型预测的one-hot转换回文本
        predictions: list of arrays, 每个array是一个字符的概率分布
        """
        label_text = ''
        for pred in predictions:
            idx = np.argmax(pred)
            label_text += self.idx_to_char[idx]
        return label_text
    
    def create_dataset_from_folder(self, folder_path, batch_size=16):
        """
        从文件夹创建数据集
        文件夹结构：
        folder_path/
            ABC12.jpg
            XYZ89.png
            ...
        """
        image_paths = []
        labels = []
        
        # 遍历文件夹
        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                # 文件名即为标签
                label = os.path.splitext(filename)[0]
                image_paths.append(os.path.join(folder_path, filename))
                labels.append(label)
        
        # 创建数据生成器
        def generator():
            for img_path, label in zip(image_paths, labels):
                # 加载图像
                img = self.preprocess_image(img_path)
                
                # 转换标签
                label_onehot = self.label_to_onehot(label)
                
                yield img, tuple(label_onehot)
        
        # 创建tf.data.Dataset
        output_signature = (
            tf.TensorSpec(shape=(*self.image_size, 3), dtype=tf.float32),
            tuple([tf.TensorSpec(shape=(len(self.char_set),), dtype=tf.float32) 
                   for _ in range(self.num_chars)])
        )
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )
        
        # 数据增强和批处理
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


# ============================================================================
# 第四部分：训练管理器
# ============================================================================

class TrainingManager:
    """训练管理器：负责模型训练、验证和保存"""
    
    def __init__(self, model, save_dir='checkpoints'):
        self.model = model
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def get_callbacks(self):
        """创建训练回调"""
        callbacks = [
            # 学习率调度
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            
            # 早停
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            
            # 模型检查点
            ModelCheckpoint(
                filepath=os.path.join(self.save_dir, 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            
            # TensorBoard日志
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(self.save_dir, 'logs'),
                histogram_freq=1
            )
        ]
        
        return callbacks
    
    def train(self, 
              train_dataset, 
              val_dataset, 
              epochs=100,
              initial_epoch=0):
        """训练模型"""
        
        print("=" * 60)
        print("开始训练 DFCR 模型")
        print("=" * 60)
        
        # 训练
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=self.get_callbacks(),
            verbose=1
        )
        
        print("\n✓ 训练完成！")
        
        return history
    
    def evaluate(self, test_dataset):
        """评估模型"""
        print("\n评估模型性能...")
        results = self.model.evaluate(test_dataset, verbose=1)
        
        # 解析结果
        if isinstance(results, list):
            print("\n评估结果：")
            print(f"总损失: {results[0]:.4f}")
            for i, acc in enumerate(results[1:]):
                print(f"字符 {i+1} 准确率: {acc:.4f}")
        else:
            print(f"\n准确率: {results:.4f}")
        
        return results


# ============================================================================
# 第五部分：使用示例
# ============================================================================

def example_usage():
    """完整使用示例"""
    
    # 1. 构建模型（Dataset #1: 5字符，62类）
    print("步骤1: 构建DFCR模型...")
    dfcr = DFCR(
        input_shape=(224, 224, 3),
        num_classes_per_char=62,  # 10数字 + 26大写 + 26小写
        num_chars=5,
        growth_rate=32,
        dataset_type=1
    )
    
    model = dfcr.build_model()
    dfcr.compile_model(learning_rate=0.001)
    
    # 打印模型结构
    print("\n模型结构：")
    dfcr.summary()
    
    # 2. 准备数据
    print("\n步骤2: 准备数据...")
    data_gen = CaptchaDataGenerator(
        image_size=(224, 224),
        num_chars=5
    )
    
    # 创建数据集（假设数据在这些文件夹中）
    train_dataset = data_gen.create_dataset_from_folder(
        'data/train',
        batch_size=16
    )
    val_dataset = data_gen.create_dataset_from_folder(
        'data/val',
        batch_size=16
    )
    test_dataset = data_gen.create_dataset_from_folder(
        'data/test',
        batch_size=16
    )
    
    # 3. 训练模型
    print("\n步骤3: 训练模型...")
    trainer = TrainingManager(model, save_dir='checkpoints')
    history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=100
    )
    
    # 4. 评估模型
    print("\n步骤4: 评估模型...")
    trainer.evaluate(test_dataset)
    
    # 5. 预测示例
    print("\n步骤5: 预测示例...")
    # 加载一张测试图像
    test_img = data_gen.preprocess_image('data/test/ABC12.jpg')
    test_img = np.expand_dims(test_img, axis=0)  # 添加batch维度
    
    # 预测
    predictions = model.predict(test_img)
    predicted_text = data_gen.onehot_to_label(predictions)
    print(f"预测结果: {predicted_text}")


if __name__ == "__main__":
    # 运行示例
    example_usage()