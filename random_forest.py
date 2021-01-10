from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import preprocess as pp
import evaluate as evl
import feature_selection as fs


def grid_search(classifier):
    param_dict = {
        "n_estimators": [50, 100, 150, 200, 250],
        "min_samples_leaf": range(1, 5),
        "criterion": ['gini', 'entropy'],
        "class_weight": ["balanced", None, "balanced_subsample"]
    }

    rfc_gs = GridSearchCV(estimator=classifier, param_grid=param_dict, scoring="accuracy",
                          cv=5, verbose=True, n_jobs=-1)
    rfc_gs.fit(x_train, y_train)

    print("Best estimator:\n", rfc_gs.best_params_)
    print("Best score: ", rfc_gs.best_score_)

    return rfc_gs.best_estimator_


def plot_feature_importance(fitted_model):
    """
    plot feature importance
    """
    feature_importance = fitted_model.feature_importances_
    feature_importance = pd.Series(feature_importance, index=data.drop('quality', axis=1).columns)\
        .sort_values(ascending=True)

    plt.figure()
    plt.title("Feature importance: Random Forest")
    plt.ylabel("Features")
    plt.xlabel("Importance score")
    feature_importance.plot(kind='barh', figsize=(15, 10))
    plt.savefig("./graphs/feature_importance_rfc.jpg")
    plt.show()


# get data
data = pp.import_data()
x_train, x_test, y_train, y_test = pp.preprocess(data, hnd_outliers=False, var_threshold=False)

# pca feature selection
# x_train, x_test = fs.pca_selection(x_train, x_test, 9)
#

# random forest classifier
rfc = RandomForestClassifier(random_state=20, n_estimators=150, criterion="entropy",
                             min_samples_leaf=1)
rfc = grid_search(RandomForestClassifier())

# recursive elimination feature selection
# x_train, x_test = fs.recursive_f_elimination(estimator=rfc, x_train=x_train, y_train=y_train, x_test=x_test)
#

# fit and predict
model = rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)
#

# cross validation score
evl.cross_val_eval(rfc, x_train, y_train, 5)

# accuracy, recall, precision, f1
evl.evaluate_model(y_test, y_pred)

# confusion matrix
evl.confusion_matrix(y_test, y_pred, data, False)
