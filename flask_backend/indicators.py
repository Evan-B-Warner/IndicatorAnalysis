import yfinance as yf
import random


def rsi(close_prices, period=14):
    # determine gain and loss per day
    gains = [max(0, close_prices[i]/close_prices[i-1] - 1) for i in range(1, len(close_prices))]
    losses = [max(0, 1 - close_prices[i]/close_prices[i-1]) for i in range(1, len(close_prices))]

    # calculate the first value
    prev_gains = sum(gains[:period]) / period
    prev_losses = sum(losses[:period]) / period
    if prev_losses == 0:
        rsi_values = [100]
    else:
        rsi_values = [100 - (100 / (1 + prev_gains/prev_losses))]
    
    # iteratively compute remaining values
    for i in range(period, len(gains)):
        avg_gain = (prev_gains*(period-1) + gains[i]) / period
        avg_loss = (prev_losses*(period-1) + losses[i]) / period
        if avg_loss == 0:
            rsi_values.append(100)
        else:
            rsi_values.append(100 - (100 / (1 + avg_gain/avg_loss)))
        prev_gains, prev_losses = avg_gain, avg_loss
    
    return rsi_values


def ma(close_prices, period):
    # the moving average is the mean closing price over `period` days
    return [sum(close_prices[i-period:i])/period for i in range(period, len(close_prices)+1)]


def exponential_ma(close_prices, period):
    # the initial ema value is the simple ma over the first `period` days
    ema_values = [ma(close_prices[:period], period)[0]]

    # calculate smoothing factor
    smoothing = 2 / (period + 1)

    # iteratively compute remaining values
    for i in range(period, len(close_prices)):
        ema_values.append(close_prices[i]*smoothing + ema_values[-1]*(1 - smoothing))
    
    return ema_values


def macd(close_prices, fast_period=12, slow_period=26, signal_period=9):
    # compute fast and slow exponential moving averages
    period_diff = slow_period - fast_period
    fast_ema = exponential_ma(close_prices, fast_period)[period_diff:]
    slow_ema = exponential_ma(close_prices, slow_period)

    # compute MACD
    macd_values = [fast_ema[i] - slow_ema[i] for i in range(len(slow_ema))]

    # compute MACD signal
    macd_signal_values = exponential_ma(macd_values, signal_period)

    return macd_values, macd_signal_values


def stochastic_oscillator(close_prices, high_prices, low_prices, period=14):
    stoch_osc_values = []
    for i in range(period-1, len(close_prices)):
        lowest = min(low_prices[i-period+1:i+1])
        highest = max(high_prices[i-period+1:i+1])
        if highest - lowest == 0:
            stoch_osc_values.append(100)
        else:
            stoch_osc_values.append(((close_prices[i] - lowest) / (highest - lowest)) * 100)
    return stoch_osc_values


def true_range(close_prices, high_prices, low_prices):
    # true range is the largest of
    # 1. current high - current low
    # 2. abs(current high - prev close)
    # 3. abs(current low - prev close)
    tr_values = []
    for i in range(1, len(close_prices)):
        high_minus_low = high_prices[i] - low_prices[i]
        high_minus_close = abs(high_prices[i] - close_prices[i-1])
        low_minus_close = abs(low_prices[i] - close_prices[i-1])
        tr_values.append(max(high_minus_low, high_minus_close, low_minus_close))
    return tr_values


def average_true_range(close_prices, high_prices, low_prices, period=14):
    # determine the true range
    tr_values = true_range(close_prices, high_prices, low_prices)

    # compute the initial ATR value
    atr_values = [sum(tr_values[:period]) / period]

    # iteratively compute the remaining values
    for i in range(period, len(tr_values)):
        atr_values.append((atr_values[-1] * (period - 1) + tr_values[i]) / period)
    
    return atr_values


def directional_moves(high_prices, low_prices):
    # determine raw move value
    n = len(high_prices)
    dm_plus = [high_prices[i] - high_prices[i-1] for i in range(1, n)]
    dm_minus = [low_prices[i-1] - low_prices[i] for i in range(1, n)]

    # set lower value to 0
    for i in range(n-1):
        if dm_plus[i] >= dm_minus[i]:
            dm_minus[i] = 0
        else:
            dm_plus[i] = 0

    return dm_plus, dm_minus


def directional_indicator(dm_plus, dm_minus, atr_values, period=14):
    # determine exponential ma of directional moves
    exp_dm_plus = exponential_ma(dm_plus, period=period)
    exp_dm_minus = exponential_ma(dm_minus, period=period)
    
    # multiply by 100 and divide by ATR
    for i in range(len(exp_dm_plus)):
        if atr_values[i] == 0:
            exp_dm_plus[i] = 100
        else:
            exp_dm_plus[i] = exp_dm_plus[i] * 100 / atr_values[i]

    for i in range(len(exp_dm_minus)):
        if atr_values[i] == 0:
            exp_dm_minus[i] = 100
        else:
            exp_dm_minus[i] = exp_dm_minus[i] * 100 / atr_values[i]
    
    return exp_dm_plus, exp_dm_minus


def directional_index(plus_dm, minus_dm):
    di_values = []
    for i in range(len(plus_dm)):
        if plus_dm[i] + minus_dm[i] == 0:
            di_values.append(100)
        else:
            di_values.append(abs(plus_dm[i] - minus_dm[i]) / (plus_dm[i] + minus_dm[i]) * 100)
    return di_values


def average_directional_index(close_prices, high_prices, low_prices, period=14):
    # compute ATR and directional moves
    atr_values = average_true_range(close_prices, high_prices, low_prices, period=period)
    dm_plus, dm_minus = directional_moves(high_prices, low_prices)

    # compute plus and minus DI
    plus_dm, minus_dm = directional_indicator(dm_plus, dm_minus, atr_values, period)

    # compute directional index
    dx = directional_index(plus_dm, minus_dm)

    # compute ADX
    adx = exponential_ma(dx, period=period)
    adx = [adx[i] * 100 for i in range(len(adx))]

    return adx


def get_prices(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    dates = hist.index.tolist()
    close_prices = hist["Close"].tolist()
    return dates, close_prices


def get_stock_info(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    dates = hist.index.tolist()
    close_prices = hist["Close"].tolist()
    high_prices = hist["High"].tolist()
    low_prices = hist["Low"].tolist()
    return close_prices, high_prices, low_prices


def compute_all_indicators(ticker):
    # pull stock price info from yfinance
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")

    # extract necessary pricing data
    dates = hist.index.tolist()
    close_prices = hist["Close"].tolist()
    high_prices = hist["High"].tolist()
    low_prices = hist["Low"].tolist()
    
