from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np

# 导入数据
dataset = np.loadtxt('../pima-indians-diabetes.csv', delimiter=',')
# 分割输入x和输出Y
x = dataset[:, 0: 8]
Y = dataset[:, 8]

feature_train, feature_test, target_train, target_test = train_test_split(x, Y, test_size=0.2, random_state=56)

# 简单训练
rf_model = RandomForestClassifier(n_estimators=170)
rf_model.fit(feature_train, target_train)
predict_results = rf_model.predict(feature_test)

print(accuracy_score(predict_results, target_test))

# 调参1：n_estimators
# param_test1 = {'n_estimators': range(80, 201, 10)}
# gsearch1 = GridSearchCV(estimator=RandomForestClassifier(min_samples_split=100,
#                                                          min_samples_leaf=20, max_depth=8, max_features='sqrt',
#                                                          random_state=10),
#                         param_grid=param_test1, scoring='roc_auc', cv=5)
# gsearch1.fit(x, Y)
# print(gsearch1.best_params_, gsearch1.best_score_)

# 调参2：max_depth，min_samples_split
# param_test2 = {'max_depth': range(3, 14, 2), 'min_samples_split': range(50, 201, 20)}
# gsearch2 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=170,
#                                                          min_samples_leaf=20, max_features='sqrt', oob_score=True,
#                                                          random_state=10),
#                         param_grid=param_test2, scoring='roc_auc', iid=False, cv=5)
# gsearch2.fit(x, Y)
# print(gsearch2.best_params_, gsearch2.best_score_)

# # 调参3：min_samples_split，min_samples_leaf
# param_test3 = {'min_samples_split': range(80, 150, 20), 'min_samples_leaf': range(10, 60, 10)}
# gsearch3 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=170, max_depth=7,
#                                                          max_features='sqrt', oob_score=True, random_state=10),
#                         param_grid=param_test3, scoring='roc_auc', iid=False, cv=5)
# gsearch3.fit(x, Y)
# print(gsearch3.best_params_, gsearch3.best_score_)

# 调参4：max_features
# param_test4 = {'max_features': range(3, 8, 1)}
# gsearch4 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=170, max_depth=7, min_samples_split=80,
#                                                          min_samples_leaf=10, oob_score=True, random_state=10),
#                         param_grid=param_test4, scoring='roc_auc', iid=False, cv=5)
# gsearch4.fit(x, Y)
# print(gsearch4.best_params_, gsearch4.best_score_)
