# Contains all indicator generating functions
import math
import statistics

import numpy as np
import pandas as pd


def simple_moving_average(price, n=14):
    """
    Returns moving average of a stock
    :param a: closing price for period
    :param n: lookback period
    :return: vector of moving average for each 'n' day period
    """
    mv_ave = np.zeros(len(price))
    for i in range(n, len(price)):
        mv_ave[i] = np.average(price[i - n:i])

    # fill in unfilled moving averages for first n values
    mv_ave[0:n] = mv_ave[n + 1]

    return mv_ave


def exponential_moving_average(price, n=10):
    """
    Returns exponential moving average for a given vector of prices
    :param price: price information for each period
    Refer: https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_averages
    :param n: lookback period
    :return: exponential moving average vector
    """

    # create the exponential moving average
    ema = np.zeros(len(price))

    # compute first EMA as the SMA for the first n-day period
    ema[n - 1] = np.average(price[0:n])

    # compute the multiplier
    multiplier = (2 / (n + 1))

    # compute all consecutive emas
    for i in range(n, len(price)):
        ema[i] = (price[i] - ema[i - 1]) * multiplier + ema[i - 1]

    return ema


def getVolatility(price, n=20):
    """
    Computes volatility (standard deviation) for a set of prices).
    Uses 20-day moving average. Volatility based on 2 standard deviations.
    Refer: https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:standard_deviation_volatility
    :param price: list containing prices
    :param n: lookback period
    :return: np.array with same size of price with volatility for each period
    """

    av_std = np.zeros(len(price))
    for i in range(n, len(price)):
        av_std[i] = statistics.stdev(price[i - n:i])
    av_std *= 2
    return av_std


def getMomentum(price, n=20):
    """
    Computes momentum of a stock based on lookback
    Refer: https://www.investopedia.com/articles/technical/081501.asp
    :param price: numpy array of prices
    :param n: lookback period
    :return: vector of momentum
    """

    momentum = np.zeros(len(price))
    for i in range(n, len(price)):
        momentum[i] = price[i] - price[i - n]

    return momentum


def getADX(price):
    pass


def getATR(close, high, low, n=14):
    """
    Compute the Average True Range for a given stock over the defined time period
    Refer: https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_true_range_atr
    :param close: closing prices
    :param high: highest value of stock on given day (np array for given time period)
    :param low: lowest value of stock on given day (np array for given time period)
    :param n: lookback period range
    :return: np array containing ATR values for each day
    """

    atr = np.zeros(len(high))
    tr = np.zeros(len(high))

    # Because there must be a beginning, the first TR value is simply the High minus the Low, and the first 14-day ATR
    # is the average of the daily TR values for the last 14 days.
    # After that, Wilder sought to smooth the data by incorporating the previous period's ATR value.

    tr[0] = high[0] - low[0]

    # for the next 13 values, we would compute the TR values for each individual
    for i in range(1, n):
        H_L = high[i] - low[i]  # current high less the current low
        H_PC = math.fabs(high[i] - close[i - 1])  # current high less the previous close (gap down)
        L_PC = math.fabs(low[i] - close[i - 1])  # current low less the previous close (gap up)

        tr[i] = np.max([H_L, H_PC, L_PC])

    # compute 14th day ATR as average of all the TR's so far (n=14, so index of 14th day is n-1 = 13 cuz 0 indexing)
    atr[n - 1] = np.average(tr[0:n])

    # for the subsequent values, we must calculate the ATR using the following:
    # Current ATR = [(Prior ATR x 13) + Current TR] / 14
    #   - Multiply the previous 14-day ATR by 13.
    #   - Add the most recent day's TR value.
    #   - Divide the total by 14

    for i in range(n, len(high)):
        H_L = high[i] - low[i]  # current high less the current low
        H_PC = math.fabs(high[i] - close[i - 1])  # current high less the previous close (gap down)
        L_PC = math.fabs(low[i] - close[i - 1])  # current low less the previous close (gap up)

        tr[i] = np.max([H_L, H_PC, L_PC])

        atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n

    return atr


def getRSI(close, n=14):
    """
    Computes the Relative Strength Index of a trend
    Refer: https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:relative_strength_index_rsi
    :param close: closing prices
    :param n: lookback period
    :return: a np array containing the RSI for all periods spanning closing price data
    """

    # compute a vector for change between daily closing prices
    change = np.zeros(len(close))
    for i in range(1, len(close)):
        change[i] = close[i - 1] - close[i]

    # compute a vector of gains and losses
    gain = np.zeros(len(close))
    loss = np.zeros(len(close))

    for i in range(1, len(close)):
        if change[i] >= 0:
            gain[i] = change[i]
        else:
            loss[i] = math.fabs(change[i])

    avg_gain = np.zeros(len(close))
    avg_loss = np.zeros(len(close))

    avg_gain[n] = np.average(gain[0:n])
    avg_loss[n] = np.average(loss[0:n])

    for i in range(n, len(close)):
        avg_gain[i] = (avg_gain[i - 1] * (n - 1) + gain[i]) / n
        avg_loss[i] = (avg_loss[i - 1] * (n - 1) + loss[i]) / n

    RS = np.zeros(len(close))
    for i in range(n, len(close)):
        RS[i] = avg_gain[i] / avg_loss[i]

    RSI = np.zeros(len(close))
    for i in range(n, len(close)):
        RSI[i] = 100 - (100 / (1 + RS[i]))

    return RSI


def getMACD(close, n=14):
    """
    Returns the MACD for a given time period of closing price data
    Refer: https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_average_convergence_divergence_macd
    :param close: closing price
    :param n: lookback period
    :return macd: np array containing MACD for each period
    :return signal: np array containing points for signal line for each period
    """

    EMA_12 = exponential_moving_average(close, 12)  # compute the 12 day moving average
    EMA_24 = exponential_moving_average(close, 24)  # compute the 24 day moving average
    MACD = EMA_12 - EMA_24  # MACD line values
    signal = exponential_moving_average(MACD, 9)  # compute the 9 day moving average of the MACD

    return MACD, signal


def getTrueStrengthIndex(close):
    """
    Computes the True Strength Index of a given stock.
    Uses 25 day ema for first smoothing
    uses 15 day ema for second smooth.
    Refer: https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:true_strength_index
    :param close: closing prices
    :return: np array containing True Strength Index for each period
    """

    # first compute daily momentum
    momentum = getMomentum(close, 1)

    # compute 25 day ema of the momentum
    EMA_25 = exponential_moving_average(momentum, 25)

    # compute 13 period ema of EMA_25
    EMA_25_13 = exponential_moving_average(EMA_25, 13)

    # take absolute value of the momentum
    absolute_momentum = np.abs(momentum)

    # compute 25 day ema of the absolute momentum
    EMA_25_abs = exponential_moving_average(absolute_momentum, 25)

    # compute the 13 day ema of EMA_25_abs
    EMA_25_13_abs = exponential_moving_average(EMA_25_abs, 13)

    # compute the TSI
    TSI = np.zeros(len(close))

    for i in range(0, len(close)):
        if EMA_25_13_abs[i] != 0:
            TSI[i] = 100 * (EMA_25_13[i] / EMA_25_13_abs[i])

    return TSI


def getFastStochastics(high, low, close, n=14):
    """
    Returns fast stochastics
    Refer: https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:stochastic_oscillator_fast_slow_and_full
    :param high: daily highs
    :param low: daily lows
    :param close: daily close
    :param n: lookback period
    :return: np array containing slow stochastics values
    """

    # create highest high and lowest low
    highest_high = np.zeros(len(close))
    lowest_low = np.zeros(len(close))

    for i in range(n - 1, len(close)):
        highest_high[i] = np.max(high[(i + 1 - n):(i + 1)])
        lowest_low[i] = np.min(low[(i + 1 - n):(i + 1)])

    # compute %K
    percentK = np.zeros(len(close))
    for i in range(len(close)):
        if highest_high[i] - lowest_low[i] != 0:
            percentK[i] = ((close[i] - lowest_low[i]) / (highest_high[i] - lowest_low[i])) * 100

    # compute 3 day moving average of %K
    percentD = np.zeros(len(close))
    for i in range(3, len(close)):
        percentD[i] = np.average(percentK[i - 3:i])

    return percentD


def getUltimateOscillator(close, high, low):
    """
    Ultimate oscillator indicator
    Refer: https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ultimate_oscillator
    :param close: closing prices
    :param high: daily highest
    :param low: daily lowest
    :return: np array containing ultimate oscillator indicator values
    """

    # compute buying pressure
    BP = np.zeros(len(close))
    for i in range(1, len(close)):
        BP[i] = close[i] - min(low[i], close[i - 1])

    # compute the True Range
    TR = np.zeros(len(close))
    for i in range(1, len(close)):
        TR[i] = max(high[i], close[i - 1]) - min(low[i], close[i - 1])

    # 7-day average
    average7 = np.zeros(len(close))
    for i in range(7, len(close)):
        average7[i] = np.sum(BP[i - 7:i]) / np.sum(TR[i - 7:i])

    # 14 day average
    average14 = np.zeros(len(close))
    for i in range(14, len(close)):
        average14[i] = np.sum(BP[i - 14:i]) / np.sum(TR[i - 14:i])

    # 28 day average
    average28 = np.zeros(len(close))
    for i in range(28, len(close)):
        average28[i] = np.sum(BP[i - 28:i]) / np.sum(TR[i - 28:i])

    UltOsc = np.zeros(len(close))
    for i in range(28, len(close)):
        UltOsc[i] = 100 * (4 * average7[i] + 2 * average14[i] + average28[i]) / (4 + 2 + 1)

    return UltOsc


def getBollingerPercentB(close, n=20):
    """
    Computes the %B for a given band.
    :param close: closing prices
    :param n: lookback period
    :return: %B values for each period
    """

    # first, compute the 20 day SMA
    SMA_20 = simple_moving_average(close, n)

    # compute the standard deviation of closing price and obtain upper and lower bands
    std_dev = np.zeros(len(close))
    for i in range(n, len(close)):
        std_dev[i] = statistics.stdev(close[i - n:i])

    upper_band = SMA_20 + 2 * std_dev
    lower_band = SMA_20 - 2 * std_dev

    percentB = np.zeros(len(close))
    for i in range(0, len(close)):
        if upper_band[i] - lower_band[i] != 0:
            percentB[i] = (close[i] - lower_band[i]) / (upper_band[i] - lower_band[i])

    return percentB


def getAroon(high, low, n=20):
    """
    Aroon indicator
    Refer: https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:aroon
    :param high: daily highs
    :param low: daily lows
    :param n: lookback period
    :return: Aroon up, Aroon down
    """

    # compute a vector of n day highs and lows
    highs = np.zeros(len(high))
    lows = np.zeros(len(low))
    aroon_up = np.zeros(len(high))
    aroon_down = np.zeros(len(low))

    for i in range(n, len(high)):
        # compute the new high and lows
        aroon_up[i] = (high[i - n:i].argmax()) / n * 100
        aroon_down[i] = (low[i - n:i].argmin()) / n * 100

    return aroon_up, aroon_down


if __name__ == '__main__':

    # list of stocks
    STOCKS = ['COP', 'CVX', 'EMR', 'ENB', 'EOG', 'GE', 'HAL', 'OXY', 'PSX', 'SU', 'VLO', 'XOM']

    source_folder = "data/StockData/StockData_withLabels_CSV/StockData_withLabels_CSV/"

    for STOCK in STOCKS:
        print('Obtaining data for ', STOCK)
        filename = source_folder + STOCK + '.csv'
        data = pd.read_csv(filename)

        close = np.array(data['Close'])
        high = np.array(data['High'])
        low = np.array(data['Low'])

        print('Getting TSI')
        # get true strength index
        TSI = getTrueStrengthIndex(close)

        print('Getting Momentum')
        # get momentum
        momentum = getMomentum(close)

        print('Getting Volatility')
        # get volatility
        volatility = getVolatility(close)

        print('Getting Aroon indicators')
        # get aroon indicator
        aroon_up, aroon_down = getAroon(high, low)

        print('Getting ATR')
        # get ATR
        ATR = getATR(close, high, low)

        print('Getting Bollinger %B')
        # get bollinger %B
        bollingerB = getBollingerPercentB(close)

        print('Getting Ultimate Oscillator')
        # get ultimate osciallator
        ultiOsc = getUltimateOscillator(close, high, low)

        print('Getting slow stochastics')
        # get slow stochastics
        slowStochastics = getFastStochastics(high, low, close)

        print('Getting MACD')
        # get MACD
        MACD, signal = getMACD(close)

        print('Getting RSI')
        # get RSI
        RSI = getRSI(close)

        indicatorData = pd.DataFrame(
            columns=['TSI', 'Momentum', 'Volatility', 'AroonUp', 'AroonDown', 'ATR', 'BollingerB',
                     'UltimateOscillator', 'SlowStochastics', 'MACDLine', 'MACDSignal', 'RSI'])

        indicatorData['TSI'] = TSI
        indicatorData['Momentum'] = momentum
        indicatorData['Volatility'] = volatility
        indicatorData['AroonUp'] = aroon_up
        indicatorData['AroonDown'] = aroon_down
        indicatorData['ATR'] = ATR
        indicatorData['BollingerB'] = bollingerB
        indicatorData['UltimateOscillator'] = ultiOsc
        indicatorData['SlowStochastics'] = slowStochastics
        indicatorData['MACDLine'] = MACD
        indicatorData['MACDSignal'] = signal
        indicatorData['RSI'] = RSI

        print('Writing to file')
        write_to_file_path = source_folder + STOCK + "-indicatorData.csv"
        indicatorData.to_csv(write_to_file_path)
