from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import preprocess as pp


# get data
data = pp.import_data()
x_train, x_test, y_train, y_test = pp.preprocess(data)

# random forest classifier
rfc = RandomForestClassifier(n_estimators=100)
model = rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)
#

# evaluation
df_y_test = pd.DataFrame(y_test)
print("Test set class distribution: \n", df_y_test.value_counts())
# TODO: add cross validation score
print("Report:\n", metrics.classification_report(y_test, y_pred))
print('F1 Score:\n', metrics.f1_score(y_test, y_pred, average='micro'))
print('Precision Score:\n', metrics.precision_score(y_test, y_pred, average="micro"))
print("Accuracy:\n", metrics.accuracy_score(y_test, y_pred))
#

# confusion matrix
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", conf_matrix)

class_names = data['quality'].unique()
df_conf = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
sns.heatmap(df_conf, annot=True,  fmt='g')
plt.title("Confusion matrix: Random Forest")
plt.xlabel("False")
plt.ylabel("True")
plt.savefig("./graphs/confusion_matrix_rfc.jpg")
plt.show()
#

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
