from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=8000)  # 前8000个单词(每局评论至多包含8000个单词)
print('shape of train data is ', train_data.shape)
print('shape of train labels is ', train_labels.shape)
print('an example of train data is ', train_data[5])

# 处理数据集
import numpy as np


# 神经网络的输入必须是tensor而不是list，所以需要将数据集处理为25000*8000
def vectorize_sequences(sequences, dimension=8000):
    # 生成25000*8000的二维Numpy数组
    results = np.zeros((len(sequences), dimension))
    # one-hot编码
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# 设计网络结构
from keras import models
from keras import layers


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(8000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',  # 还可以通过optimizer = optimizers.RMSprop(lr=0.001)来为优化器指定参数
                  loss='binary_crossentropy',  # 等价于loss = losses.binary_crossentropy
                  metrics=['accuracy'])  # 等价于metrics = [metircs.binary_accuracy]
    return model


model = build_model()

# 划分验证集用于选择超参数epochs
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# 训练模型
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,  # 在全数据集上迭代4次
                    batch_size=512,  # 每个batch的大小为512
                    validation_data=(x_val, y_val))

history_dict = history.history
history_dict.keys()

# 绘制loss和accuracy
import matplotlib.pyplot as plt

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()

results = model.evaluate(x_test, y_test)
print(results)
