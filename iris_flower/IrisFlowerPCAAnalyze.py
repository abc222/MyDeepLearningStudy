import matplotlib.pyplot as plt  # 加载matplotlib用于数据的可视化
from sklearn.decomposition import PCA  # 加载PCA算法包
from sklearn.datasets import load_iris

# Python sklearn库实现PCA
# 其中样本总数为150，鸢尾花的类别有三种，分别标记为0，1，2

data = load_iris()
y = data.target
x = data.data
pca = PCA(n_components=2)  # 加载PCA算法，设置降维后主成分数目为2
reduced_x = pca.fit_transform(x)  # 对样本进行降维

red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []

for i in range(len(reduced_x)):
    if y[i] == 0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])

    elif y[i] == 1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])

    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])

# 可视化
plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()

# https://blog.csdn.net/qq_38825002/article/details/81356377
