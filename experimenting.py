import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

csu = yf.Ticker("CSU.TO")

def display_as_df(cols, colnames, head_only=True):
    display_df = pd.DataFrame()
    for i in range(len(cols)):
        display_df[colnames[i]] = cols[i]
    if head_only:
        print(display_df.head())
    else:
        print(display_df)

def make_plot(x, y, lines=None):
    plt.plot(x, y)
    for line in lines:
        plt.plot(line[0], line[1])
    plt.show()

def rsi(hist, period=14):
    # extract closing prices and gain per day
    close_prices = hist["Close"].tolist()
    gains = [max(0, close_prices[i]/close_prices[i-1] - 1) for i in range(1, len(close_prices))]
    losses = [max(0, 1 - close_prices[i]/close_prices[i-1]) for i in range(1, len(close_prices))]

    # calculate the first value
    prev_gains = sum(gains[:period]) / period
    prev_losses = sum(losses[:period]) / period
    rsi_values = [100 - (100 / (1 + prev_gains/prev_losses))]
    
    # recursively compute remaining values
    for i in range(period, len(gains)):
        avg_gain = (prev_gains*(period-1) + gains[i]) / period
        avg_loss = (prev_losses*(period-1) + losses[i]) / period
        rsi_values.append(100 - (100 / (1 + avg_gain/avg_loss)))
        prev_gains, prev_losses = avg_gain, avg_loss
    
    #display_as_df([hist.index.tolist()[14:], close_prices[14:], rsi_values], ["date", "close", "RSI"], head_only=False)
    #make_plot(hist.index.tolist(), hist["Close"].tolist(), [[hist.index.tolist()[14:], rsi_values]])
    
    return rsi_values


def 

    
        

# get all stock info
hist = csu.history(period="6mo")
rsi(hist)