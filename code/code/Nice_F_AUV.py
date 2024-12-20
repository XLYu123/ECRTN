import os
import pickle
import torch
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from keras import backend as K
from utils import whitening
from utils.Network_Operate import build_NICE1, build_NICE2, build_NICE_reverse

# 常量定义
BATCH_SIZE = 32
ORIGINAL_DIM = 3072
DATA_PATH =
OUTPUT_PATH =
EPOCHS = 100  # 修改训练次数为5000

def load_data_pickle(path, name, mode='rb'):
    """从 pickle 文件加载数据。"""
    with open(os.path.join(path, f"{name}.pkl"), mode) as f:
        return pickle.load(f)

def preprocess_data(x1, x2, batch_size, conditional=True):
    """准备 TensorFlow 数据集以供训练。"""
    dataset = tf.data.Dataset.from_tensor_slices((x1, x2))
    if conditional:
        dataset = dataset.shuffle(231)
    return dataset.batch(batch_size)

def fft_normalize(x):
    """执行 FFT 并归一化幅度。"""
    fft_result = np.abs(np.fft.fft(x)) * 2 / len(x)
    return fft_result[:len(x)]

def frequency_analysis(x, sampling_frequency=12800):
    """对输入数据进行频率分析。"""
    x -= np.mean(x)
    fft_result = fft_normalize(x)
    freq_axis = sampling_frequency * np.arange(len(fft_result) // 2) / len(fft_result)
    return freq_axis, fft_result

def build_and_summarize_model(model_func, input_dim):
    """构建并总结模型。"""
    x_in, x = model_func(input_dim)
    model = Model(x_in, x)
    model.summary()
    return model

def train_loss(encoder1, encoder2, X_train1, X_train2, optimizer):
    """计算并应用梯度进行训练。"""
    with tf.GradientTape() as tape:
        pred_y1 = encoder1(X_train1)
        pred_y2 = encoder2(X_train2)
        pred_y1 = tf.cast(pred_y1, tf.float64)
        loss = tf.reduce_mean(0.5 * K.sum((X_train1 - pred_y1) ** 2, axis=1))
        gradients = tape.gradient(loss, encoder1.trainable_variables)
        optimizer.apply_gradients(zip(gradients, encoder1.trainable_variables))
    return loss

def train_model(train_db, encoder1, encoder2, optimizer, epochs=100):
    """在数据集上训练模型，并记录损失。"""
    loss_values = []
    for epoch in range(epochs):
        for images_batch in train_db:
            images1, images2 = images_batch
            loss = train_loss(encoder1, encoder2, images1, images2, optimizer)
            loss_values.append(loss.numpy())
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy()}")
    return loss_values

def save_data_pickle(path, name, data):
    """将数据保存到 pickle 文件中。"""
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, f"{name}_test.pkl"), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print(f'成功保存到 {path}')

def save_model(model, model_name):
    """保存训练好的模型。"""
    model.save(os.path.join(OUTPUT_PATH, model_name))
    print(f'模型 {model_name} 已成功保存。')

def test_model(model, test_data):
    """使用测试数据评估模型性能。"""
    predictions = model.predict(test_data)
    return predictions

def plot_loss_curve(loss_values):
    """绘制损失曲线图。"""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label='train loss', color='blue')
    plt.title('loss curve')
    plt.xlabel('epochs')
    plt.ylabel('loss value')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_PATH, 'loss_curve.png'))
    plt.show()

# 加载和预处理数据
data = load_data_pickle(DATA_PATH, 'C5')
data = data.reshape(10, ORIGINAL_DIM)

_, x_freq = frequency_analysis(data)

# ZCA 处理
zca_time = whitening.ZCA(x=data)  # 创建 ZCA 对象
X_time_zca = zca_time.apply(data)  # 应用 ZCA
zca_freq = whitening.ZCA(x=x_freq)  # 创建频率 ZCA 对象
X_freq_zca = zca_freq.apply(x_freq)  # 应用频率 ZCA

# 准备数据集
train_db = preprocess_data(X_time_zca, X_freq_zca, BATCH_SIZE)

# 构建模型
encoder1 = build_and_summarize_model(build_NICE1, ORIGINAL_DIM)
encoder2 = build_and_summarize_model(build_NICE2, ORIGINAL_DIM)

# 优化器选择
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# 训练模型并记录损失
loss_values = train_model(train_db, encoder1, encoder2, optimizer, epochs=EPOCHS)

# 保存模型
# save_model(encoder1, 'encoder1_model.h5')
# save_model(encoder2, 'encoder2_model.h5')

# Inverse-Sample Generated Data
# Time series
x_in,x=build_NICE_reverse(ORIGINAL_DIM)
decoder1 = Model(x_in, x)
decoder1.summary()

# Frequency
x_in_f,x_f=build_NICE_reverse(ORIGINAL_DIM)
decoder2 = Model(x_in_f, x_f)
decoder2.summary()


# 生成样本
data_z =[]
for i in range (300):
    z_sample=np.array(np.random.randn(1,ORIGINAL_DIM))
    x_decoded = encoder1.predict(z_sample)
    data_z.extend(x_decoded)
data_z_trans = np.array(data_z)

# 反 ZCA 处理
predictions_inverse = zca_time.invert(data_z_trans)  # 使用 zca_time 对象的 invert 方法

# 打印预测结果
print("预测结果 (Encoder1):", predictions_inverse)
# print("预测结果 (Encoder2):", predictions2)
save_data_pickle(OUTPUT_PATH, 'C5', predictions_inverse)

# 绘制损失曲线图
plot_loss_curve(loss_values)


