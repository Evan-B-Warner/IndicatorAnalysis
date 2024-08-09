import csv
import numpy as np
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from indicators import compute_all_indicators


def write_to_csv(rows, save_path, header=None):
    """Writes each row in `rows` to the specified save path"""
    with open(save_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
        if header is not None:
            csv_writer.writerow(header)
        for row in rows:
            csv_writer.writerow(row)


def read_tickers(txts):
    """Reads tickers from multiple txt's, returns concatenated list of tickers"""
    all_tickers = []
    for txt_path in txts:
        with open(txt_path) as f:
            # extract all lines excluding header
            lines = f.readlines()[1:]
            all_tickers.extend([line.split()[0] for line in lines])
    return all_tickers


def generate_initial_train_data(tickers, save_path="data/train_data.csv"):
    """Generates raw training data csv containing data for tickers in `tickers`"""
    # compute indicator data for each ticker
    train_data = []
    for ticker in tqdm(tickers):
        train_data.extend(compute_all_indicators(ticker))
    
    # write data to csv
    header = [
        "rsi_5",
        "rsi_5-1",
        "rsi_5-5",
        "rsi_5-20",
        "rsi_14",
        "rsi_14-1",
        "rsi_14-5",
        "rsi_14-20",
        "rsi_30",
        "rsi_30-1",
        "rsi_30-5",
        "rsi_30-20",
        "ma_5",
        "ma_5-1",
        "ma_5-5",
        "ma_5-20",
        "ma_20",
        "ma_20-1",
        "ma_20-5",
        "ma_20-20",
        "ma_50",
        "ma_50-1",
        "ma_50-5",
        "ma_50-20",
        "ma_100",
        "ma_100-1",
        "ma_100-5",
        "ma_100-20",
        "ema_5",
        "ema_5-1",
        "ema_5-5",
        "ema_5-20",
        "ema_20",
        "ema_20-1",
        "ema_20-5",
        "ema_20-20",
        "ema_50",
        "ema_50-1",
        "ema_50-5",
        "ema_50-20",
        "ema_100",
        "ema_100-1",
        "ema_100-5",
        "ema_100-20",
        "macd",
        "macd-1",
        "macd-5",
        "macd-20",
        "macd_signal",
        "macd_signal-1",
        "macd_signal-5",
        "macd_signal-20",
        "so_5",
        "so_5-1",
        "so_5-5",
        "so_5-20",
        "so_14",
        "so_14-1",
        "so_14-5",
        "so_14-20",
        "so_30",
        "so_30-1",
        "so_30-5",
        "so_30-20",
        "atr_5",
        "atr_5-1",
        "atr_5-5",
        "atr_5-20",
        "atr_14",
        "atr_14-1",
        "atr_14-5",
        "atr_14-20",
        "atr_30",
        "atr_30-1",
        "atr_30-5",
        "atr_30-20",
        "adx_5",
        "adx_5-1",
        "adx_5-5",
        "adx_5-20",
        "adx_14",
        "adx_14-1",
        "adx_14-5",
        "adx_14-20",
        "adx_30",
        "adx_30-1",
        "adx_30-5",
        "adx_30-20",
        "close_price",
        "high_price",
        "low_price",
        "date",
        "total_revenue",
        "revenue_per_share",
        "1d",
        "5d",
        "20d"
    ]
    write_to_csv(train_data, save_path, header=header)


def normalize_train_data(X):
    """Normalizes each numeric column in X by subtracting mean and dividing by SD"""
    X = np.array(X).astype("float64")
    for col in range(len(X[0])):
        vals = X[:, col]
        max, min = np.max(vals), np.min(vals)
        X[:, col] = (vals - max) / (max - min)
    return X


def process_initial_train_data(save_path="data/train_data.csv"):
    """Normalizes raw training data, splits into train and validation data,
    and saves to separate csv's
    """
    # read csv and separate into X and y
    X, y = [], []
    with open(save_path) as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if "nan" in row:
                continue
            X_row = row[:-6]
            y.append(row[-6:-3])
            X_row.extend(row[-3:])
            X.append(X_row)
    X = np.array(X)

    # subtract each backstep from current indicator
    for i in range(0, 85, 4):
        for ii in range(i+1, i+4):
            X[:, ii] = X[:, i].astype("float64") - X[:, ii].astype("float64")
    # should probably use division for MA, EMA, MACD/MACD_signal
    # ADX calculation is wrong

    # normalize model inputs in X
    #X[:, :-3] = normalize_train_data(X[:, :-3])
    y = np.array(y).astype("float64")
    y = y*100

    # save to csv
    save_folder = "/".join(save_path.split("/")[:-1])
    X_header = [
        "rsi_5",
        "rsi_5-1",
        "rsi_5-5",
        "rsi_5-20",
        "rsi_14",
        "rsi_14-1",
        "rsi_14-5",
        "rsi_14-20",
        "rsi_30",
        "rsi_30-1",
        "rsi_30-5",
        "rsi_30-20",
        "ma_5",
        "ma_5-1",
        "ma_5-5",
        "ma_5-20",
        "ma_20",
        "ma_20-1",
        "ma_20-5",
        "ma_20-20",
        "ma_50",
        "ma_50-1",
        "ma_50-5",
        "ma_50-20",
        "ma_100",
        "ma_100-1",
        "ma_100-5",
        "ma_100-20",
        "ema_5",
        "ema_5-1",
        "ema_5-5",
        "ema_5-20",
        "ema_20",
        "ema_20-1",
        "ema_20-5",
        "ema_20-20",
        "ema_50",
        "ema_50-1",
        "ema_50-5",
        "ema_50-20",
        "ema_100",
        "ema_100-1",
        "ema_100-5",
        "ema_100-20",
        "macd",
        "macd-1",
        "macd-5",
        "macd-20",
        "macd_signal",
        "macd_signal-1",
        "macd_signal-5",
        "macd_signal-20",
        "so_5",
        "so_5-1",
        "so_5-5",
        "so_5-20",
        "so_14",
        "so_14-1",
        "so_14-5",
        "so_14-20",
        "so_30",
        "so_30-1",
        "so_30-5",
        "so_30-20",
        "atr_5",
        "atr_5-1",
        "atr_5-5",
        "atr_5-20",
        "atr_14",
        "atr_14-1",
        "atr_14-5",
        "atr_14-20",
        "atr_30",
        "atr_30-1",
        "atr_30-5",
        "atr_30-20",
        "adx_5",
        "adx_5-1",
        "adx_5-5",
        "adx_5-20",
        "adx_14",
        "adx_14-1",
        "adx_14-5",
        "adx_14-20",
        "adx_30",
        "adx_30-1",
        "adx_30-5",
        "adx_30-20",
        "date",
        "total_revenue",
        "revenue_per_share"
    ]
    write_to_csv(X, f"{save_folder}/X.csv", header=X_header)
    write_to_csv(y, f"{save_folder}/y.csv", header=["1d", "5d", "20d"])


def model_ready_data(x_path, y_path, split=0.2):
    # read X and y
    X, y = [], []
    with open(x_path) as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        for row in csv_reader:
            X.append(row)

    with open(y_path) as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        for row in csv_reader:
            y.append(row)
    
    # perform train test split
    y = np.array(y).astype("float64")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split, random_state=507)

    # separate auxiliary variables
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    auxiliary_train = X_train[:, -3:]
    auxiliary_val = X_val[:, -3:]
    X_train = X_train[:, :-3].astype("float64")
    X_val = X_val[:, :-3].astype("float64")

    return X_train, X_val, y_train, y_val, auxiliary_train, auxiliary_val


if __name__ == "__main__":
    # generate raw training data
    tickers = read_tickers(["data/NYSE.txt", "data/TSX.txt", "data/NASDAQ.txt"])
    generate_initial_train_data(tickers)

    # normalize and process data
    #process_initial_train_data("data/train_data.csv")

    # generate model ready data
    #X_train, X_val, y_train, y_val, auxiliary_train, auxiliary_val = model_ready_data("data/X.csv", "data/y.csv")