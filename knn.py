
import sklearn.metrics as metrics
#import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from preprocess import *
from evaluate import *
from sklearn.model_selection import GridSearchCV
def grid_search(is_grid_search=False):

    data=import_data()

    data_train, data_test, target_train, target_test=preprocess(data)
    #create object of the lassifier

    tuned_parameters = [{'n_neighbors': [3,5,11,19,23],
                        'metric': ['euclidean','manhattan'],
                        'weights': ['uniform', 'distance']}]

    #best params {'metric': 'manhattan', 'n_neighbors': 19, 'weights': 'distance'}

    clf=KNeighborsClassifier()

    if is_grid_search:
        clf=GridSearchCV(clf, tuned_parameters)
    else:
        clf=KNeighborsClassifier( metric = 'manhattan', n_neighbors = 19, weights ='distance')


    #Train the algorithm
    clf.fit(data_train, target_train)

    if is_grid_search:
        print(clf.best_params_)
        print(pd.concat([pd.DataFrame(clf.cv_results_["params"]),pd.DataFrame(clf.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1))
   
    # predict the response
    pred = clf.predict(data_test)

    # cross validation score
    cross_val_eval(clf, data_train, target_train, 5)

    # accuracy, recall, precision, f1
    evaluate_model(target_test, pred)

    # confusion matrix
    confusion_matrix(target_test, pred, data, False, "KNN")

grid_search(True)
