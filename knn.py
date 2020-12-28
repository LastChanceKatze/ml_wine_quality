from preprocess import *
import sklearn.metrics as metrics
#import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from preprocess import *
from evaluate import *

data=import_data()

data_train, data_test, target_train, target_test=preprocess(data)
#create object of the lassifier
clf = KNeighborsClassifier(n_neighbors=9, metric='manhattan')
#Train the algorithm
clf.fit(data_train, target_train)
# predict the response
pred = clf.predict(data_test)
# evaluate accuracy

# cross validation score
cross_val_eval(clf, data_train, target_train, 5)

# accuracy, recall, precision, f1
evaluate_model(target_test, pred)

# confusion matrix
confusion_matrix(target_test, pred, data, False, "KNN")


