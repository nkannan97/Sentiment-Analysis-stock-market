# This is the main script to run classification
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def loadData(stock, news_sources):
    """
    Loads all related data of a given stock
    :param stock: stock tickers
    :param news_sources: sources of news information

    :return: X - contains indicators and news data
    :return y - contains corresponding label for whether stock moved up or down in given timeframe
    """

    if isinstance(news_sources, list):

        data = pd.DataFrame(columns=['date', 'news'])

        for source in news_sources:
            # get the source path
            source_path = 'data/NewsData/' + source + '/' + source + 'Data_Clean/' + stock + '-' + source + '.txt'

            # read in the data
            stockDF = pd.read_csv(source_path)

            # store the data
            data = pd.concat([data, stockDF[['date', 'news']]])

        data = data.dropna(how='any')  # drop a row if any of the values in that row are nan
        data.columns = ['Date', 'News']

        # now, obtain the stock data for each stock
        source_path = 'data/StockData/StockData_withLabels_CSV/StockData_withLabels_CSV/' + stock + '.csv'
        stock_data = pd.read_csv(source_path)

        # obtain technical data
        source_path = 'data/StockData/StockData_withLabels_CSV/StockData_withLabels_CSV/' + stock + '-indicatorData.csv'
        indicator_data = pd.read_csv(source_path)

        # obtain all indicator column data
        indicator_cols = list(indicator_data.keys())
        indicator_cols = indicator_cols[1:len(indicator_cols)]

        # append to stock dat
        for i in indicator_cols:
            stock_data[i] = indicator_data[i]

        # now that we have merged the stock and indicator data together, we must match the data for those dates that
        # are given by our text data
        data = data.merge(stock_data, on='Date')

        X = data[['Volume', 'News', 'TSI', 'Momentum', 'Volatility', 'AroonUp', 'AroonDown', 'ATR', 'BollingerB',
                  'UltimateOscillator', 'SlowStochastics', 'MACDLine', 'MACDSignal', 'RSI']]
        y = data[['5 Days', '10 Days', '2 Weeks', '1 Month', '2 Months']]

        return X, y



def textWeightingScheme(text, type, ngram_range = (1,1), y=None):
    """
    Returns a matrix with a weight for each word
    :param text: plain text corpus
    :param type: if type == 'tfidf' uses tfidf vectorizer, if type == 'labelweighted' : uses label based weighting
    :return: matrix with weightage values
    """

    if type == 'tfidf':

        vectorizer = TfidfVectorizer(min_df=3, ngram_range=ngram_range)
        X = vectorizer.fit_transform(text)
        X = X.todense()
    elif type == 'count':
        vectorizer = CountVectorizer(min_df=3, ngram_range=ngram_range)
        X = vectorizer.fit_transform(text)
        X = X.todense()
    elif type == 'SGDWeighted':
        assert y
        clf = SGDClassifier(loss='hinge', penalty='l2', max_iter=5)
        vectorizer = CountVectorizer(min_df=3, ngram_range=ngram_range)
        X = vectorizer.fit_transform(text)
        X_temp = X.todense()
        clf.fit(X_temp, y)
        labels = vectorizer.get_feature_names()
        weights = clf.coef_[0]
        for i in range(X_temp.shape[0]):
            X_temp[i,:] =  np.array(X_temp[i,:]) * weights
        X = X_temp

    return X

def generateTrainingData(X, text_weight_type, data_type,  ngram_range = (1,1), y = None):
    X_vals = None
    if data_type == 'both':
        text_matrix = textWeightingScheme(X['News'], text_weight_type, ngram_range)
        indicator_data  = np.array(X[['Volume', 'TSI','Momentum','Volatility','AroonUp','AroonDown','ATR','BollingerB',
                                      'UltimateOscillator','SlowStochastics','MACDLine','MACDSignal','RSI']])
        X_vals = np.append(text_matrix, indicator_data, axis=1)
    elif data_type == 'indicators':
        indicator_data  = np.array(X[['Volume', 'TSI','Momentum','Volatility','AroonUp','AroonDown','ATR','BollingerB',
                                      'UltimateOscillator','SlowStochastics','MACDLine','MACDSignal','RSI']])
        X_vals = indicator_data
    elif data_type == 'text':
        text_matrix = textWeightingScheme(X['News'], text_weight_type, ngram_range)
        X_vals = text_matrix
    elif data_type == 'weightedtext':
        text_matrix = textWeightingScheme(X['News'], text_weight_type, ngram_range, y)
        X_vals = text_matrix
    elif data_type == 'weightedtext+indicator':
        text_matrix = textWeightingScheme(X['News'], text_weight_type, ngram_range, y)
        indicator_data  = np.array(X[['Volume', 'TSI','Momentum','Volatility','AroonUp','AroonDown','ATR','BollingerB',
                                      'UltimateOscillator','SlowStochastics','MACDLine','MACDSignal','RSI']])
        X_vals = np.append(text_matrix, indicator_data, axis=1)

    return X_vals



def analysis1(stock):
    """
    Computes what the accuracy of using text, indicators and both
    :param X: features
    :param Y: labels
    :return: mean accuracy from 10 fold cross validation
    """

    data_types = ['text', 'indicators', 'both']
    duration_period = ['5 Days', '10 Days', '2 Weeks', '1 Month', '2 Months']

    results = pd.DataFrame(index=duration_period, columns=data_types)

    #load the data
    X, Y = loadData(stock, ['Investopedia','SeekingAlpha'])

    for i, dtype in enumerate(data_types):
        for j, duration in enumerate(duration_period):

            print('Processing ', dtype, '-',duration)
            X_temp = generateTrainingData(X, 'tfidf', data_type=dtype)
            Y_temp = Y[duration]

            clf = RandomForestClassifier(n_estimators=100)
            scores = cross_val_score(clf, X_temp, Y_temp, cv = 10)
            print('Score = ',scores.mean())
            results[dtype].loc[duration] = scores.mean()

    return results


def analysis2(stock):
    """
    Computes the performance on different classifiers (Done on 2 Month)
    :param stock:
    :return: results for a given stock on multiple classifiers
    """

    X, Y = loadData(stock, ['Investopedia','SeekingAlpha'])
    duration = '2 Months'
    dtype = 'both'

    X_temp = generateTrainingData(X, 'tfidf', data_type=dtype)
    Y_temp = Y[duration]

    mean_accuracies = np.zeros(3)
    # naive bayes
    gnb = GaussianNB()
    scores = cross_val_score(gnb, X_temp, Y_temp, cv = 10)
    mean_accuracies[0] = scores.mean()

    # svm
    svm = SVC()
    scores = cross_val_score(svm, X_temp, Y_temp, cv = 10)
    mean_accuracies[1] = scores.mean()

    # Random forest
    rf = RandomForestClassifier(n_estimators=100)
    scores = cross_val_score(rf, X_temp, Y_temp, cv = 10)
    mean_accuracies[2] = scores.mean()

    return mean_accuracies

def Analysis3(stock):
    """
    Here, we contrast how the number of n-grams used varies in the accuracy of prediction
    :param stock:
    :return:
    """

    ngram_combinations = [(1,1), (1,2), (1,3), (2,2) , (2,3), (3,3)]
    duration_period = ['5 Days', '10 Days', '2 Weeks', '1 Month', '2 Months']
    X, Y = loadData(stock, ['Investopedia','SeekingAlpha'])

    results = pd.DataFrame(index=duration_period, columns=ngram_combinations)

    for combo in ngram_combinations:
        for duration in duration_period:

            print('Processing ', combo, '-',duration)
            X_temp = generateTrainingData(X, 'tfidf', data_type='both', ngram_range=combo)
            Y_temp = Y[duration]

            clf = RandomForestClassifier(n_estimators=100)
            scores = cross_val_score(clf, X_temp, Y_temp, cv = 10)
            print('Score = ',scores.mean())
            results[combo].loc[duration] = scores.mean()

    return results






if __name__ == '__main__':


    # Analysis 1
    STOCKS = ['COP', 'EMR', 'ENB', 'EOG', 'GE', 'HAL', 'OXY', 'PSX', 'SU', 'VLO', 'XOM']

    # for stock in STOCKS:
    #     results = analysis1(stock)
    #     filepath = 'results/Text-TextwIndicatorsAnalysis/'+stock+'.csv'
    #     results.to_csv(path_or_buf=filepath)
    #

    # Analysis 2

    # results = pd.DataFrame(index=['gnb','svm','rf'], columns=STOCKS)
    # for stock in STOCKS:
    #     print('Stock = ', stock)
    #     results[stock] = analysis2(stock)
    #
    # results.to_csv('results/AccuracyOnDifferentClassifiers/accuracyOnDifferentClassifiers.csv')
    #
    #
    #

    # Analysis 3
    # for stock in STOCKS:
    #     results = Analysis3(stock)
    #     filepath = 'results/NGramAnalysis/'+stock+'.csv'
    #     results.to_csv(filepath)


