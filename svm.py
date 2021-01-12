#import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import pandas as pd
from preprocess import *
from evaluate import *
from feature_selection import *

#best params {'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'}
def grid_search(is_grid_search=False):

    tuned_parameters = [{'kernel': ['rbf'],
                     'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}]
    data=import_data()

    data_train, data_test, target_train, target_test = preprocess(data)


    clf = svm.SVC()

    if is_grid_search:
        clf=GridSearchCV(clf, tuned_parameters)
    else:
        clf=svm.SVC(C=1000, gamma=0.001, kernel='rbf')

    # pca feature selection
    #data_train, data_test = fs.pca_selection(data_train, data_test, 9)
    #

    # recursive elimination feature selection
    data_train, data_test = fs.recursive_f_elimination(estimator=clf, x_train=data_train, y_train=target_train, x_test=data_test) 


    clf.fit(data_train, target_train)

    if is_grid_search:
        print(clf.best_params_)
        print(pd.concat([pd.DataFrame(clf.cv_results_["params"]),pd.DataFrame(clf.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1))



    pred=clf.predict(data_test)

    #cross validation score
    cross_val_eval(clf, data_train, target_train, 5)

    # accuracy, recall, precision, f1
    evaluate_model(target_test, pred)

    # confusion matrix
    confusion_matrix(target_test, pred, data, False, "svm")


grid_search(False)


