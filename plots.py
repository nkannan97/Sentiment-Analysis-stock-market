import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

STOCKS = ['COP', 'EMR', 'ENB', 'EOG', 'GE', 'HAL', 'OXY', 'PSX', 'VLO', 'XOM']


def Plot1():
    """
    This function plots the accuracy of each plot with text data alone and
    text data boosted by the indicator data.
    :return: a plot
    """

    fig = plt.figure(figsize=(7,11))

    for i, stock in enumerate(STOCKS):


        loadfile = 'results/Text-TextwIndicatorsAnalysis/'+stock+'.csv'
        data = pd.read_csv(loadfile)

        x_axis = np.array(data['Unnamed: 0'])
        text = np.array(data['text'])
        both = np.array(data['both'])

        plt.subplot(5,2,i+1)

        plt.plot(x_axis, text,'r-o', label = 'Text', linewidth = 0.5)
        plt.plot(x_axis, both,'b-o', label = 'Text+Indicators', linewidth = 0.5)
        plt.ylim((0,1))
        plt.legend(loc = 3)
        plt.title(stock)

    plt.show()


def plot2():
    """
    This function plots a heatmap that indicates accuracy between performace of a range
    of classifiers across the given stocks in the 2-month period.
    :return:
    """

    loadfile = 'results/AccuracyOnDifferentClassifiers/accuracyOnDifferentClassifiers.csv'
    data = pd.read_csv(loadfile)
    classifiers = np.array(data['Unnamed: 0'])
    data = data[STOCKS]

    fig, ax = plt.subplots(figsize = (6,5))
    sns.heatmap(data.transpose(), cmap='RdYlGn', ax=ax, vmin = 0, vmax=1, cbar_kws={'label': 'Accuracy'})
    ax.set_xticklabels(['Naive Bayes', 'Support Vector Machine','Random Forest'], va = 'center')

    plt.show()


def plot3() :
    """
    This function plots a graph that indicates how classification accuracy varies as
    we change the number of ngrams we take into consideration from the news articles
    :return:
    """


    fig = plt.figure(figsize=(10,10))

    for i, stock in enumerate(STOCKS):


        loadfile = 'results/NGramAnalysis/'+stock+'.csv'
        data = pd.read_csv(loadfile)

        x_axis = np.array(data['Unnamed: 0'])
        columns = np.array(data.columns)
        columns = columns[1:len(columns)]
        cols_plotting = np.array(['[1,1]', '[1,2]', '[1,3]', '[2,2]', '[2,3]', '[3,3]'])
        plt.subplot(2, 5, i+1)

        for i, ngram in enumerate(columns):
            y_vals = np.array(data[ngram])
            plt.plot(x_axis, y_vals, '-o' , label = cols_plotting[i])
            plt.ylim((0,1))

        plt.title(stock)
        plt.legend(loc=3)
        plt.xticks(rotation = 90)



    plt.show()










