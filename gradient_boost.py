from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import preprocess as pp
import evaluate as evl
import feature_selection as fs


def grid_search(classifier):
    param_dict = {
        "n_estimators": [50, 100, 150, 200, 250],
        "criterion": ["friedman_mse", "mse"],
        "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
        "max_depth": range(3, 6)
    }

    rfc_gs = GridSearchCV(estimator=classifier, param_grid=param_dict, scoring="accuracy",
                          cv=5, verbose=True,
                          n_jobs=-1)
    rfc_gs.fit(x_train, y_train)

    print("Best estimator:\n", rfc_gs.best_params_)
    print("Best score: ", rfc_gs.best_score_)

    return rfc_gs.best_estimator_


# get data
data = pp.import_data()
x_train, x_test, y_train, y_test = pp.preprocess(data, hnd_outliers=True, var_threshold=False)

# pca feature selection
# x_train, x_test = fs.pca_selection(x_train, x_test, 9)
#

# random forest classifier
rfc = GradientBoostingClassifier(random_state=20, n_estimators=200, max_depth=5,
                                 learning_rate=0.2, criterion="mse")

# recursive elimination feature selection
x_train, x_test = fs.recursive_f_elimination(estimator=rfc, x_train=x_train, y_train=y_train, x_test=x_test)
#

model = rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)
#

# cross validation score
evl.cross_val_eval(rfc, x_train, y_train, 5)

# accuracy, recall, precision, f1
evl.evaluate_model(y_test, y_pred)

# confusion matrix
evl.confusion_matrix(y_test, y_pred, data, True, "gradient_boost")
