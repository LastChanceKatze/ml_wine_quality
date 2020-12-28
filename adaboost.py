
#import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import *
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostRegressor
#from descriptive_statistics import *
from sklearn.svm import SVC

data=import_data()

data_train, data_test, target_train, target_test=preprocess(data)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification

clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=51, learning_rate=1)
clf.fit(data_train, target_train)

pred=clf.predict(data_test)

print ("ADa boost accuracy score : ", accuracy_score(target_test, pred))