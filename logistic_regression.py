from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import preprocess as pp
import evaluate as eval


# get data
data = pp.import_data()
x_train, x_test, y_train, y_test = pp.preprocess(data)

# Logistic regression model
lrm = LogisticRegression(multi_class="ovr", class_weight="balanced")
lrm.fit(x_train, y_train)
y_pred = lrm.predict(x_test)
#

# cross validation score
eval.cross_val_eval(lrm, x_train, y_train, 5)

# accuracy, recall, precision, f1
eval.evaluate_model(y_test, y_pred)

# confusion matrix
eval.confusion_matrix(y_test, y_pred, data, True, "logistic_regression")
