from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
import numpy as np

# 设定随机数种子
np.random.seed(7)

# 导入数据
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
# 分割输入x和输出Y
x = dataset[:, 0: 8]
Y = dataset[:, 8]

# 创建模
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
history = model.fit(x=x, y=Y, epochs=150, batch_size=10, validation_split=0.2)

# 评估模型
scores = model.evaluate(x=x, y=Y)
print('\n%s : %.2f%%' % (model.metrics_names[1], scores[1] * 100))

# # calculate predictions
# predictions = model.predict(X)
# # round predictions
# rounded = [round(x) for x in predictions]
# print(rounded)

print(history.history.keys())

# accuracy的历史
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# loss的历史
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
