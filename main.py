import pandas as pd

data = pd.read_csv("./dataset/winequality-white.csv", delimiter=';')
print(data.head(5))
print(data.describe())
