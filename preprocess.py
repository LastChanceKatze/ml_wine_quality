import pandas as pd


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


def check_missing_vals(data):
    # check for missing values
    missing_vals = data.isna().sum()
    print("Missing values: \n" + str(missing_vals))
    return missing_vals.sum() != 0

# feature scaling

# split dataset
