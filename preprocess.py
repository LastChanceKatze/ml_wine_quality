import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import feature_selection as fs
desired_width = 500
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)


# import data
def import_data():
    data = pd.read_csv("./dataset/winequality-white.csv", delimiter=';')
    print(data.head(5))
    return data


# extract data
# independent variables
def extract_data(data):
    x = data.iloc[:, :-1].values
    # print("Data: \n" + str(x))
    # dependent variable
    y = data.iloc[:, -1].values
    # print("Output: \n" + str(y))
    return x, y


# check missing values
def check_missing_vals(data):
    # check for missing values
    missing_vals = data.isna().sum()
    print("Missing values: \n" + str(missing_vals))
    return missing_vals.sum() != 0


# create classes for wine quality
# 1-3: bad
# 4-7: good
# 8-10: excellent
def redefine_classes(data):
    bins = (0, 4, 6, 10)
    labels = ['bad', 'good', 'excellent']
    data['quality'] = pd.cut(data['quality'], bins=bins, labels=labels)
    return data


# encode labels
def encode_labels(data):
    encode_quality = LabelEncoder()
    data['quality'] = encode_quality.fit_transform(data['quality'])
    return data


# split dataset
def split_dataset(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test


# feature scaling
def scale_features(x_train, x_test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    return x_train, x_test


def replace_outliers_bound(feature):
    q1, q3 = np.percentile(feature, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    new_feature = feature
    new_feature = np.where(new_feature > upper_bound, upper_bound, new_feature)
    new_feature = np.where(new_feature < lower_bound, lower_bound, new_feature)
    return new_feature


def replace_outliers_log(feature):
    return np.log(feature)


def handle_outliers(x):
    all_indices = list(range(0, x.shape[1]))

    """
        fixed acidity, sugar,
        free sul. dioxide, total sul. dioxide
    """
    log_indices = [0, 3, 5, 6]
    for ind in log_indices:
        new_feature = replace_outliers_log(x[:, ind])
        x[:, ind] = new_feature

    bound_indices = np.setdiff1d(all_indices, log_indices)
    for ind in bound_indices:
        new_feature = replace_outliers_bound(x[:, ind])
        x[:, ind] = new_feature

    return x


def preprocess(data, hnd_outliers=False, var_threshold=False):
    # no missing values
    # check_missing_vals(data)

    data = redefine_classes(data)
    data = encode_labels(data)

    x, y = extract_data(data)
    x_train, x_test, y_train, y_test = split_dataset(x, y)

    # handle outliers
    if hnd_outliers:
        x_train = handle_outliers(x_train)
        x_test = handle_outliers(x_test)

    # remove features with variance below a threshold
    if var_threshold:
        x_train, x_test = fs.variance_threshold(x_train, x_test)

    x_train, x_test = scale_features(x_train, x_test)

    return x_train, x_test, y_train, y_test



