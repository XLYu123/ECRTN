import os
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Conv1D, Dense, Flatten, Input, Reshape, BatchNormalization, Activation
from keras.models import Model
# from keras.optimizer.v2 import Adam
import pickle


# 设置随机种子以确保可重复性
def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'


set_seed(900)


# 数据加载函数
def LoadData_pickle_1(path, name, type='rb'):
    with open(path + name + '.pkl', type) as f:
        data, label = pickle.load(f)
    return data, label


def LoadData_pickle_T(path, name, type='rb'):
    with open(path + name + '.pkl', type) as f:
        data, label = pickle.load(f)
    return data, label


# 数据预处理函数
def TFData_preprocessing(x, y, batch_size, conditional=True):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if conditional:
        dataset = dataset.shuffle(10000).batch(batch_size)
    else:
        dataset = dataset.batch(batch_size)
    return dataset


def one_hot_MPT(y, depth):
    return tf.keras.utils.to_categorical(y, num_classes=depth)


# 定义用于迁移学习的CNN模型
class ECA(keras.Model):
    def __init__(self, img_shape):
        super().__init__()
        self.img_shape = img_shape
        self.eca_model = self._get_1dcnn()
        self.opt = keras.optimizers.Adam(1e-4)
        self.loss_bool = keras.losses.CategoricalCrossentropy(from_logits=True)

    def _get_1dcnn(self):
        in_image = Input(shape=self.img_shape)
        x = Reshape((3072, 1))(in_image)

        # 特征提取层（在迁移学习过程中将被冻结）
        enc = Conv1D(32, 5, 2, 'same')(x)
        enc = BatchNormalization()(enc)
        enc = Activation('relu')(enc)

        enc = Conv1D(16, 5, 2, 'same')(enc)
        enc = BatchNormalization()(enc)
        enc = Activation('relu')(enc)

        # 迁移学习：在冻结的层上添加更多层
        enc = Conv1D(8, 5, 2, 'same')(enc)
        enc = BatchNormalization()(enc)
        enc = Activation('relu')(enc)

        enc = Conv1D(16, 5, 2, 'same')(enc)
        enc = BatchNormalization()(enc)
        enc = Activation('relu')(enc)

        enc = Flatten()(enc)
        logits = Dense(7, activation='softmax')(enc)

        model = Model(inputs=in_image, outputs=logits)
        return model

    # 训练模型
    def train(self, train_db, test_db):
        self.eca_model.compile(optimizer=self.opt, loss=self.loss_bool, metrics=['accuracy'])
        self.eca_model.fit(train_db, epochs=200, validation_data=test_db)

    # 微调模型
    def fine_tune(self, train_db, test_db):
        # 解冻部分层，并在新数据上进行微调
        for layer in self.eca_model.layers[:4]:
            layer.trainable = False
        for layer in self.eca_model.layers[4:]:
            layer.trainable = True

        self.eca_model.compile(optimizer=Adam(1e-5), loss=self.loss_bool, metrics=['accuracy'])
        self.eca_model.fit(train_db, epochs=10, validation_data=test_db)

    # 评估模型
    def evaluate(self, test_db):
        results = self.eca_model.evaluate(test_db)
        print(f'测试集损失: {results[0]}, 测试集准确率: {results[1]}')

from sklearn.preprocessing import MinMaxScaler
# 归一化函数
def normalize_data(data):
    scaler = MinMaxScaler()
    # 将数据 reshape 为 2D 以适应 MinMaxScaler
    data_reshaped = data.reshape(-1, data.shape[-1])
    data_normalized = scaler.fit_transform(data_reshaped)
    # 重新 reshape 回原始形状
    data_normalized = data_normalized.reshape(data.shape)
    return data_normalized

# 主程序
if __name__ == '__main__':
     # 加载源域数据集
    faults_1 = ['C0','C1','C2','C3','C4','C5','C0_test','C1_test','C2_test', 'C3_test','C4_test','C5_test']
    x_target_list, y_target_list = [], []
    for fault in faults_1:
        x, y = LoadData_pickle_1(path=, name=fault)
        x_target_list.extend(x)
        y_target_list.extend(y)

    x_target = np.array(x_target_list)
    y_target = np.array(y_target_list).reshape(x_target.shape[0], )
    y_target_one_hot = one_hot_MPT(y_target, depth=7)  # 确保目标域数据标签也进行one-hot编码

    # 加载目标域数据集
    faults_2 = ['C0_test','C1_test','C2_test', 'C3_test','C4_test','C5_test']
    x_test_list, y_test_list = [], []
    for fault in faults_2:
        x, y = LoadData_pickle_T(path=, name=fault)
        x_test_list.extend(x)
        y_test_list.extend(y)

    x_test = np.array(x_test_list)
    y_test = np.array(y_test_list).reshape(x_test.shape[0], )
    y_test_one_hot = one_hot_MPT(y_test, depth=7)  # 确保测试数据集标签也进行one-hot编码

    # 生成训练和测试数据集
    train_db = TFData_preprocessing(x_target, y_target_one_hot, batch_size=32)
    test_db = TFData_preprocessing(x_test, y_test_one_hot, batch_size=32)

    # 初始化模型
    model = ECA(img_shape=(3072,))

    # 在源域和目标域数据上训练模型
    model.train(train_db, test_db)

    # 在目标域数据上微调模型
    model.fine_tune(train_db, test_db)

    # 使用测试数据评估模型


