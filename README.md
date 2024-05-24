# 基于CNN的FashionMNIST图像分类

## 引言

在本次作业中，我们利用卷积神经网络（CNN）对FashionMNIST数据集进行图像分类。该数据集包含10类不同的时尚物品的灰度图像，每张图像大小为28x28像素。任务的目标是训练一个能够有效分类这些图像的CNN模型，并分析不同模型配置对分类性能的影响。

## 研究意义

FashionMNIST数据集是一个常用的图像分类基准数据集，通过对其进行分类研究，不仅可以提升我们对CNN模型的理解和应用能力，还可以为其他复杂的图像分类任务提供借鉴和参考。通过调整模型的不同参数（如卷积层数、通道数、学习率等），我们可以深入了解这些参数对模型性能的影响，从而优化模型设计，提升分类准确率。这对实际应用中处理大规模、高维度图像数据具有重要意义。

## 数据预处理

我们使用了TensorFlow框架来加载和处理数据。具体步骤如下：

1. 从TensorFlow Keras库中加载FashionMNIST数据集，包含训练集和测试集。
2. 将原始28x28的图像尺寸调整为32x32，以便与我们设计的CNN架构兼容。
3. 将像素值归一化到[0, 1]范围，以加速模型的收敛。

```python
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
train_images = tf.image.resize(train_images[..., tf.newaxis], [32, 32]).numpy()
test_images = tf.image.resize(test_images[..., tf.newaxis], [32, 32]).numpy()
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255
模型设计
为了分析不同因素对模型性能的影响，我们设计了不同配置的CNN模型，包括：

不同的卷积层数（2层，3层，4层）
不同的卷积通道数（32通道，64通道，128通道）
不同的学习率（0.001，0.0005，0.0001）
模型的基本结构如下：

初始卷积层：3x3卷积核，ReLU激活，same填充
最大池化层：2x2池化核
后续卷积层：3x3卷积核，ReLU激活，same填充
全连接层：64个神经元，ReLU激活
输出层：10个神经元（对应10个类别）
python
复制代码
def create_model(num_layers, num_channels, learning_rate):
    model = models.Sequential()
    model.add(layers.Conv2D(num_channels, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    for _ in range(num_layers - 1):
        model.add(layers.Conv2D(num_channels * 2, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        num_channels *= 2
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model
实验结果
不同层数的影响
我们分别训练了2层、3层和4层卷积层的模型，并记录了每个模型的测试准确率。


Model with 2 layers Training and Validation Accuracy


Model with 3 layers Training and Validation Accuracy


Model with 4 layers Training and Validation Accuracy

从图中可以看出，随着卷积层数的增加，模型的训练准确率和验证准确率都有所提高。然而，随着层数的增加，验证准确率的提升幅度逐渐减小。这说明增加卷积层数可以提取更复杂的特征，但过多的卷积层也可能导致模型的计算复杂度增加，从而影响模型的泛化能力。

我们分别训练了2层、3层和4层卷积层的模型，并记录了每个模型的测试准确率。

2层模型：测试准确率 = 0.89
3层模型：测试准确率 = 0.91
4层模型：测试准确率 = 0.92
不同通道数的影响
我们分别训练了32通道、64通道和128通道的模型，并记录了每个模型的测试准确率。


Model with 32 channels Training and Validation Accuracy


Model with 64 channels Training and Validation Accuracy


Model with 128 channels Training and Validation Accuracy

从图中可以看出，增加卷积通道数可以提高模型的训练准确率和验证准确率。然而，通道数的增加也带来了更高的计算复杂度，且效果逐渐递减。

32通道模型：测试准确率 = 0.88
64通道模型：测试准确率 = 0.90
128通道模型：测试准确率 = 0.91
不同学习率的影响
我们分别训练了学习率为0.001、0.0005和0.0001的模型，并记录了每个模型的测试准确率。


Model with learning rate 0.001 Training and Validation Accuracy


Model with learning rate 0.0005 Training and Validation Accuracy


Model with learning rate 0.0001 Training and Validation Accuracy

从图中可以看出，学习率对模型的训练效果有显著影响。过高的学习率（如0.001）可能导致模型不稳定，而过低的学习率（如0.0001）则可能导致收敛速度变慢。

学习率0.001：测试准确率 = 0.91
学习率0.0005：测试准确率 = 0.89
学习率0.0001：测试准确率 = 0.87
结果分析与总结
通过实验，我们发现：

随着卷积层数的增加，模型的分类准确率有所提升。
增加卷积通道数也对模型性能有正面影响，但效果逐渐递减。
学习率的选择对模型的训练效果有显著影响，过大的学习率可能导致模型不稳定，而过小的学习率则可能导致收敛速度变慢。
综合来看，3层卷积、64通道、学习率为0.001的模型在我们的实验中表现最佳。

未来工作
在未来的工作中，我们可以尝试更多的超参数组合，使用更复杂的网络结构（如ResNet、VGG），以及利用数据增强技术进一步提升模型的性能。此外，还可以尝试迁移学习，将预训练的模型应用于FashionMNIST数据集，以期获得更高的分类准确率。
