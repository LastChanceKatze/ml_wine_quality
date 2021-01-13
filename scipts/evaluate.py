import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score


def evaluate_model(y_test, y_pred):
    # evaluation
    # df_y_test = pd.DataFrame(y_test)
    # print("Test set class distribution: \n", df_y_test.value_counts())
    print("Report:\n", metrics.classification_report(y_test, y_pred))
    print('F1 Score:\n', metrics.f1_score(y_test, y_pred, average='micro'))
    print('Precision Score:\n', metrics.precision_score(y_test, y_pred, average="micro"))
    print("Accuracy:\n", metrics.accuracy_score(y_test, y_pred))


def confusion_matrix(y_test, y_pred, data, plot=False, classifier_name=""):
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", conf_matrix)

    if plot:
        class_names = data['quality'].unique()
        df_conf = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
        sns.heatmap(df_conf, annot=True, fmt='g')
        plt.title("Confusion matrix: " + classifier_name)
        plt.xlabel("False")
        plt.ylabel("True")
        plt.savefig("../graphs/confusion_matrix_" + classifier_name + ".jpg")
        plt.show()


def cross_val_eval(estimator, x_train, y_train, cv):
    cv_eval = cross_val_score(estimator=estimator, X=x_train, y=y_train, cv=cv, scoring="accuracy")
    print("Cross validation - Accuracy:\n", cv_eval.mean())
    cv_eval = cross_val_score(estimator=estimator, X=x_train, y=y_train, cv=cv, scoring="f1_micro")
    print("Cross validation - F1:\n", cv_eval.mean())
