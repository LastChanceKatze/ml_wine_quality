from sklearn.feature_selection import VarianceThreshold, RFECV
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


def variance_threshold(x_train, x_test):
    selector = VarianceThreshold(threshold=0.05)
    x_train = selector.fit_transform(x_train)
    x_test = selector.transform(x_test)
    print("X train shape - Var Threshold", x_train.shape)
    return x_train, x_test


def recursive_f_elimination(estimator, x_train, y_train, x_test):
    rfecv = RFECV(estimator=estimator, step=1, cv=5, scoring='accuracy')
    rfecv.fit(x_train, y_train)
    x_train = rfecv.transform(x_train)
    x_test = rfecv.transform(x_test)
    print("No. of best features: ", rfecv.n_features_)
    print("Selected features: ", rfecv.support_)
    return x_train, x_test


def plot_pcs(variance):
    plt.plot(variance)
    plt.xlabel('Number of components')
    plt.ylabel("Variance")
    plt.show()


def pca_selection(x_train, x_test, num_features):
    pca = PCA(n_components=num_features, random_state=200, whiten=True)
    pca.fit(x_train)

    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

    var_ratio = pca.explained_variance_ratio_*100
    print("Variance ratio - PCA\n", var_ratio)
    print("Total variance - PCA: ", np.sum(var_ratio))

    return x_train, x_test



