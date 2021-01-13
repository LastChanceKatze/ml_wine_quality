from sklearn.linear_model import LogisticRegression
from scipts import evaluate as evl, preprocess as pp

# get data
data = pp.import_data()
x_train, x_test, y_train, y_test = pp.preprocess(data)

# Logistic regression model
# tested for different solvers, no significant difference
lrm = LogisticRegression(multi_class="multinomial", solver="saga",
                         random_state=0, max_iter=1000)
lrm.fit(x_train, y_train)
y_pred = lrm.predict(x_test)
#

# cross validation score
evl.cross_val_eval(lrm, x_train, y_train, 5)

# accuracy, recall, precision, f1
evl.evaluate_model(y_test, y_pred)

# confusion matrix
evl.confusion_matrix(y_test, y_pred, data, False, "logistic_regression")
