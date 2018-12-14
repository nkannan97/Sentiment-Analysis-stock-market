
import os
import re
from datetime import datetime

import pandas as pd


def cleanData(filename, source, stopwords = []):
    """
    Cleans up news in text-files
    :param filename: file name
    :param source: news source (investopedia/seekingalpha)
    :param stopwords: list of stopwords to be removed from each sentence
    :return: date, cleaned data

    """

    # read in data
    data = open(filename).readlines()
    text = None
    date = None

    if source == 'seekingalpha':

        # extract date
        date = data[0][0:10]

        # extract text data
        text = data[1:len(data) - 3]

    elif source == 'investopedia':
        try:
            # extract date
            date = data[0].split('|')[1].split('—')[0].strip()
            date = date.strip('Updated ') # some articles get updated .. apparently
            date = datetime.strptime(date,'%B %d, %Y')

            if len(str(date.day)) == 1:
                date = str(date.year) + '-' + str(date.month) + '-' + '0' + str(date.day)
            else:
                date = str(date.year) + '-' + str(date.month) + '-' + str(date.day)

        except IndexError:
            Warning('Date unavailable for article')

        text = data[1:len(data)]

    # join text data together for condensed cleanup
    text = ''.join(text)

    # remove numbers and special characters
    text = re.sub('[^A-Za-z ]+', ' ', text)

    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    # remove text with stock tickers (usually 1-4 in length and in CAPS) and stopwords
    text = text.split(' ')

    for i in range(0, len(text)):
        if text[i].isupper() and (len(text[i]) >= 1 or len(text[i]) <= 4):
            text[i] = ''

        if text[i] != '' and text[i] in stopwords:
            text[i] = ''

    text = ' '.join(text)

    # re-remove excess spaces
    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    return date, text


if __name__ == '__main__':

    # read in stopwords
    stopwords_file = 'src/stopwords/english'
    stopwords = open(stopwords_file).readlines()
    stopwords = [x[0:(len(x)-1)] for x in stopwords]

    # list of stocks
    STOCKS = ['COP','EMR','ENB','EOG','GE','HAL','OXY','PSX','SU','VLO','XOM']

    # source sites
    sourceSite = ['Investopedia','SeekingAlpha']

    for source in sourceSite:

        for STOCK in STOCKS:

            source_folder = 'data/NewsData/'+source+'/'+source+'Data/'+STOCK+'-'+source+'-NewsData/'
            print('Extracting Data from', source_folder)

            # obtain the list of articles
            files = os.listdir(source_folder)

            # create dataframe to hold new data
            clean_news = pd.DataFrame(index = range(0, len(files)), columns = ['date','news'])

            for i, file in enumerate(files):
                filename = source_folder+file
                clean_news.loc[i] = cleanData(filename, source.lower(), stopwords)

            print(clean_news)

            write_file_name = 'data/NewsData/'+source+'/'+source+'Data_Clean/'+STOCK+'-'+source+'.txt'
            clean_news.to_csv(write_file_name, ',') # write clean data to csv file








