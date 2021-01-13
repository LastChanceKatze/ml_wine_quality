
#import necessary modules
from sklearn.tree import DecisionTreeClassifier
from scipts.preprocess import *
from scipts.evaluate import *

data=import_data()

data_train, data_test, target_train, target_test=preprocess(data)

from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=51, learning_rate=1)
clf.fit(data_train, target_train)

pred=clf.predict(data_test)

# cross validation score
cross_val_eval(clf, data_train, target_train, 5)

# accuracy, recall, precision, f1
evaluate_model(target_test, pred)

# confusion matrix
confusion_matrix(target_test, pred, data, False, "adaboost")