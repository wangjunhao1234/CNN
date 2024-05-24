import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import os

# 创建结果文件夹
results_folder = 'results'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# 加载FashionMNIST数据集
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

# 增加输入图像的尺寸
train_images = tf.image.resize(train_images[..., tf.newaxis], [32, 32]).numpy()
test_images = tf.image.resize(test_images[..., tf.newaxis], [32, 32]).numpy()

# 预处理数据
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255


# 定义一个通用的函数来创建模型
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


# 参数列表
layer_configs = [2, 3, 4]
channel_configs = [32, 64, 128]
learning_rate_configs = [0.001, 0.0005, 0.0001]

# 训练和评估不同层数的模型
for layers_num in layer_configs:
    model = create_model(num_layers=layers_num, num_channels=32, learning_rate=0.001)
    history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'\nModel with {layers_num} layers - Test accuracy: {test_acc}')

    model.save(os.path.join(results_folder, f'model_layers_{layers_num}.h5'))
    with open(os.path.join(results_folder, f'model_layers_{layers_num}_results.txt'), 'w') as f:
        f.write(f'Model with {layers_num} layers\n')
        f.write(f'Test accuracy: {test_acc}\n')

    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Model with {layers_num} layers Training and Validation Accuracy')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(results_folder, f'model_layers_{layers_num}_accuracy.png'))
    plt.close()

# 训练和评估不同通道数的模型
for channels_num in channel_configs:
    model = create_model(num_layers=2, num_channels=channels_num, learning_rate=0.001)
    history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'\nModel with {channels_num} channels - Test accuracy: {test_acc}')

    model.save(os.path.join(results_folder, f'model_channels_{channels_num}.h5'))
    with open(os.path.join(results_folder, f'model_channels_{channels_num}_results.txt'), 'w') as f:
        f.write(f'Model with {channels_num} channels\n')
        f.write(f'Test accuracy: {test_acc}\n')

    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Model with {channels_num} channels Training and Validation Accuracy')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(results_folder, f'model_channels_{channels_num}_accuracy.png'))
    plt.close()

# 训练和评估不同学习率的模型
for lr in learning_rate_configs:
    model = create_model(num_layers=2, num_channels=32, learning_rate=lr)
    history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'\nModel with learning rate {lr} - Test accuracy: {test_acc}')

    model.save(os.path.join(results_folder, f'model_lr_{lr}.h5'))
    with open(os.path.join(results_folder, f'model_lr_{lr}_results.txt'), 'w') as f:
        f.write(f'Model with learning rate {lr}\n')
        f.write(f'Test accuracy: {test_acc}\n')

    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Model with learning rate {lr} Training and Validation Accuracy')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(results_folder, f'model_lr_{lr}_accuracy.png'))
    plt.close()
