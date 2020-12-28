from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import preprocess as pp
import evaluate as evl

# TODO: add GridSearchCV to improve performance

# get data
data = pp.import_data()
x_train, x_test, y_train, y_test = pp.preprocess(data)

# random forest classifier
rfc = RandomForestClassifier(n_estimators=100)
model = rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)
#

# cross validation score
evl.cross_val_eval(rfc, x_train, y_train, 5)

# accuracy, recall, precision, f1
evl.evaluate_model(y_test, y_pred)

# confusion matrix
evl.confusion_matrix(y_test, y_pred, data, False)

# important features
feature_importance = model.feature_importances_
feature_importance = pd.Series(feature_importance, index=data.drop('quality', axis=1).columns)\
    .sort_values(ascending=True)

plt.figure()
plt.title("Feature importance: Random Forest")
plt.ylabel("Features")
plt.xlabel("Importance score")
feature_importance.plot(kind='barh', figsize=(15, 10))
plt.savefig("./graphs/feature_importance_rfc.jpg")
plt.show()
#
