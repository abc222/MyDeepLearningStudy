from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 导入数据
from sklearn import datasets
iris = datasets.load_iris()
iris_feature = iris.data
iris_target = iris.target

feature_train, feature_test, target_train, target_test = train_test_split(iris_feature, iris_target, test_size=0.33,
                                                                          random_state=56)

rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(feature_train, target_train)
predict_results = rf_model.predict(feature_test)

print(accuracy_score(predict_results, target_test))
