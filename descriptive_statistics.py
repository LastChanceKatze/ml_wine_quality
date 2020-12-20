import preprocess as pp
import matplotlib.pyplot as plt
import seaborn as sns

data = pp.import_data()
print("Describe data: \n" + str(data.describe(include='all')))
print("----------------------------------------------------------")
print("Print median: \n" + str(data.median()))
print("----------------------------------------------------------")
print("Skewness: \n" + str(data.skew()))
print("----------------------------------------------------------")


def plot_density_graph(data):
    for column in data.columns:
        fig = plt.figure()
        data[column].plot(kind="density", figsize=(17, 17))

        plt.vlines(data[column].mean(), ymin=0, ymax=0.5, linewidth=0.5)

        plt.vlines(data[column].median(), ymin=0, ymax=0.5, linewidth=2.0, color="red")

        plt.savefig("density_" + str(column)+".jpg")


def plot_boxplot(data, y_coordinate):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 8))
    sns.boxplot(x='quality', y=y_coordinate, data=data)
    plt.show()


def plot_histogram(data_column):
    plt.figure(figsize=(14, 8))
    sns.distplot(data_column, kde=False)
    plt.show()


# plot_density_graph(data)
# plot_boxplot(data, y_coordinate="density")
plot_histogram(data['density'])
