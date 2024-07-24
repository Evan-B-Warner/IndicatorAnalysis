import csv
import numpy as np
import random
from tqdm import tqdm

import keras
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split

from indicators import compute_all_indicators

def read_tickers(txts):
    all_tickers = []
    for txt_path in txts:
        with open(txt_path) as f:
            # extract all lines excluding header
            lines = f.readlines()[1:]
            all_tickers.extend([line.split()[0] for line in lines])
    return all_tickers


def generate_train_data(tickers, save_path="data/train_data.csv"):
    selected_tickers = random.sample(tickers, 150)
    train_data = []
    for ticker in tqdm(selected_tickers):
        train_data.extend(compute_all_indicators(ticker))
    
    with open(save_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
        for row in train_data:
            csv_writer.writerow(row)


def load_train_data(save_path="data/train_data.csv"):
    X, y_1, y_7, y_30, dates = [], [], [], [], []
    with open(save_path) as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            X.append(row[:-4])
            y_1.append(row[-4])
            y_7.append(row[-3])
            y_30.append(row[-2])
            dates.append(row[-1])
    return np.array(X).astype("float64"), np.array(y_1).astype("float64"), np.array(y_7).astype("float64"), np.array(y_30).astype("float64"), dates


def build_model(dense_architecture, input_shape):
    model = Sequential()
    
    # dense layers
    for i in range(len(dense_architecture)):
        width = dense_architecture[i]
        if i == 0:
            model.add(Dense(width, activation="relu", input_shape=input_shape))
        else:
            model.add(Dense(width, activation="relu"))
    
    # add output layer
    model.add(Dense(1, activation="linear"))

    # compile the model
    model.compile(loss="MeanSquaredError",
                optimizer="Adam",
                metrics=["MeanAbsoluteError"])
    
    return model



if __name__ == "__main__":
    #tickers = read_tickers(["data/NYSE.txt", "data/TSX.txt", "data/NASDAQ.txt"])
    #generate_train_data(tickers)
    X, y_1, y_7, y_30, dates = load_train_data()
    y = y_30
    model = build_model([64, 32, 16], X[0].shape)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=32,
        epochs=200,
        verbose=1
    )

    import matplotlib.pyplot as plt
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
