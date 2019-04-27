# # 参考：https://blog.csdn.net/CanLiu1992/article/details/83271695
#
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn import tree
# from sklearn.metrics import accuracy_score
#
# import ssl
#
# ssl._create_default_https_context = ssl._create_unverified_context
#
# iris = sns.load_dataset("iris")
# print(iris.head())
# print(iris.shape)
# print(iris.describe())
#
# # sns.set(style="ticks")
# # sns.pairplot(iris, hue="species", palette="bright")
# # plt.show()
#
# y = iris.species
# X = iris.drop('species', axis=1)
#
# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)
#
# from sklearn.tree import DecisionTreeClassifier
#
# # 决策树算法
# clf = tree.DecisionTreeClassifier()
# clf.fit(X_train, y_train)
# # DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
# #                        max_features=None, max_leaf_nodes=None,
# #                        min_impurity_decrease=0.0, min_impurity_split=None,
# #                        min_samples_leaf=1, min_samples_split=2,
# #                        min_weight_fraction_leaf=0.0, presort=False, random_state=None,
# #                        splitter='best')
# DecisionTreeClassifier()
# from sklearn.datasets import load_iris
#
# iris = load_iris()
# tree.export_graphviz(clf, out_file="iris.dot", feature_names=iris.feature_names, class_names=iris.target_names,
#                      filled=True, rounded=True, special_characters=True)
#
# y_pred = (clf.predict(X_test))
# print("Accuracy Score")
# print(accuracy_score(y_test, y_pred) * 100)


# 参考：https://blog.csdn.net/oxuzhenyi/article/details/76427704
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
iris_feature = iris.data
iris_target = iris.target

feature_train, feature_test, target_train, target_test = train_test_split(iris_feature, iris_target, test_size=0.33,
                                                                          random_state=56)
dt_model = DecisionTreeClassifier()
dt_model.fit(feature_train, target_train)
predict_results = dt_model.predict(feature_test)
scores = dt_model.score(feature_test, target_test)

print(accuracy_score(predict_results, target_test))
print(scores)
