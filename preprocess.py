import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
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
    print("Data: \n" + str(x))
    # dependent variable
    y = data.iloc[:, -1].values
    print("Output: \n" + str(y))
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
    bins = (0, 3, 7, 10)
    labels = ['bad', 'good', 'excellent']
    data['quality'] = pd.cut(data['quality'], bins=bins, labels=labels)
    return data


# encode labels
def encode_labels(data):
    encode_quality = LabelEncoder()
    data['quality'] = encode_quality.fit_transform(data['quality'])
    print(data['quality'].value_counts())
    return data


# split dataset
def split_dataset(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test


# feature scaling
def scale_features(x_train, x_test):
    scaler = StandardScaler()
    scaler.fit_transform(x_train)
    scaler.fit_transform(x_test)

#
# data = import_data()
# data = redefine_classes(data)
# print(data.head(5))
# data = encode_labels(data)
