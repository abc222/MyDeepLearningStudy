from sklearn import datasets
digits = datasets.load_digits()
x = digits.data                                              #输入数据
y = digits.target
# import numpy as np
# # 导入数据
# dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
# # 分割输入x和输出Y
# x = dataset[:, 0 : 8]
# Y = dataset[:, 8]
#输出数据
from sklearn import decomposition
pca = decomposition.PCA()
pca.fit(x)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(pca.explained_variance_, 'k', linewidth=2)
plt.xlabel('n_components', fontsize=16)
plt.ylabel('explained_variance_', fontsize=16)
plt.show()