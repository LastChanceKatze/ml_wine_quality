import preprocess as pp
import matplotlib.pyplot as plt
import seaborn as sns


# TODO: Remove, unused
def plot_density_graph(data):
    for column in data.columns:
        fig = plt.figure()
        data[column].plot(kind="density", figsize=(17, 17))

        plt.vlines(data[column].mean(), ymin=0, ymax=0.5, linewidth=0.5)

        plt.vlines(data[column].median(), ymin=0, ymax=0.5, linewidth=2.0, color="red")

        plt.savefig("density_" + str(column)+".jpg")


def plot_boxplot(data):
    data.drop('quality', axis=1).plot(kind='box', figsize=(10, 10), subplots=True, layout=(2, 6),
                                      sharex=False, sharey=False,
                                      title='Box Plot for each input variable')
    plt.subplots_adjust(wspace=0.5)
    plt.savefig('features_box_plot.jpg')
    plt.show()


# TODO: uses seaborn
def plot_barplots(data):
    num_rows = 3
    num_cols = 4
    fig, axs = plt.subplots(num_rows, num_cols)
    plt.subplots_adjust(hspace=0.5, wspace=.5, top=0.95)

    for i in range(0, num_rows):
        for j in range(0, num_cols):
            index = j * num_rows + i
            col_name = data.columns[index]
            sns.barplot(x='quality', y=col_name, data=data, ax=axs[i][j])

    plt.savefig("bar_plot.jpg")
    plt.show()


def plot_histogram(data, data_column):
    plt.figure(figsize=(14, 8))
    data[data_column].plot(kind='hist', figsize=(8, 8))
    # sns.histplot(data_column, kde=True)
    plt.savefig("./graphs/" + data_column + "_histogram.jpg")
    plt.show()


def plot_count(data, data_column):
    plt.figure(figsize=(14, 8))
    sns.countplot(x=data_column, data=data)
    plt.savefig("./graphs/" + data_column + "_countplot.jpg")
    plt.show()


# TODO: uses seaborn
def correlation_matrix(data):
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, ax=ax)
    plt.savefig("correlation_matrix.png")
    plt.show()

    corr = abs(corr['quality']).sort_values(ascending=False)
    return corr


def exploratory_analysis():
    data = pp.import_data()
    """
    outliers can be seen in:
    free sulfur dioxide, total sulfur dioxide, residual sugar, alcohol, sulphates
    """
    print("Describe data: \n" + str(data.describe(include='all')))
    print("----------------------------------------------------------")
    print("Print median: \n" + str(data.median()))
    print("----------------------------------------------------------")
    print("Skewness: \n" + str(data.skew()))
    print("----------------------------------------------------------")

    """
    quality histogram
    the dataset is unbalanced
    much more good wines than bad or excellent ones
    """
    # plot_histogram(data=data, data_column='quality')

    """
    box plots for each variable
    outliers can be seen
    as well as data range
    """
    # plot_boxplot(data)

    """
    bar plot for each variable by quality
    """
    # plot_barplots(data)

    # correlation matrix
    # corr = correlation_matrix(data)
    # print("Sorted correlation values: \n" + str(corr))

    """
    quality count plot with redefined classes
    """
    data = pp.redefine_classes(data)
    plot_count(data, 'quality')


exploratory_analysis()
