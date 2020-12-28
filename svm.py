#import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm

from preprocess import *
from evaluate import *

#from sklearn.svm import SVC
#import preprocess
data=import_data()

data_train, data_test, target_train, target_test = preprocess(data)
clf = svm.SVC()

clf.fit(data_train, target_train)
pred=clf.predict(data_test)

# cross validation score
cross_val_eval(clf, data_train, target_train, 5)

# accuracy, recall, precision, f1
evaluate_model(target_test, pred)

# confusion matrix
confusion_matrix(target_test, pred, data, False, "svm")

