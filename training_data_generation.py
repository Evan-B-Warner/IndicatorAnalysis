import csv
import numpy as np
import random
from sklearn.model_selection import train_test_split

from indicators import compute_all_indicators


def write_to_csv(rows, save_path):
    """Writes each row in `rows` to the specified save path"""
    with open(save_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
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
    write_to_csv(train_data, save_path)


def normalize_train_data(X):
    """Normalizes each numeric column in X by subtracting mean and dividing by SD"""
    X = np.array(X).astype("float64")
    for col in range(len(X[0])):
        vals = X[:, col]
        mean, std = np.mean(vals), np.std(vals)
        X[:, col] = (vals - mean) / std
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

    # normalize model inputs in X
    X[:, :-3] = normalize_train_data(X[:, :-3])
    y = np.array(y).astype("float64")
    y = y*100

    # save to csv
    save_folder = "/".join(save_path.split("/")[:-1])
    write_to_csv(X, f"{save_folder}/X.csv")
    write_to_csv(y, f"{save_folder}/y.csv")


def model_ready_data(x_path, y_path, split=0.2):
    # read X and y
    X, y = [], []
    with open(x_path) as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            X.append(row)

    with open(y_path) as f:
        csv_reader = csv.reader(f)
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
    #tickers = read_tickers(["data/NYSE.txt", "data/TSX.txt", "data/NASDAQ.txt"])
    #generate_train_data(tickers)

    # normalize and process data
    process_initial_train_data("data/train_data.csv")

    # generate model ready data
    X_train, X_val, y_train, y_val, auxiliary_train, auxiliary_val = model_ready_data("data/X.csv", "data/y.csv")