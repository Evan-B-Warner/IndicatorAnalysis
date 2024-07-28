import json
import keras
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import numpy as np
from bayes_opt import BayesianOptimization

from training_data_generation import model_ready_data


def build_model(dense_architecture, input_shape, learning_rate):
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
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="MeanSquaredError",
                optimizer=optimizer,
                metrics=["MeanAbsoluteError"])
    
    return model


def test_hyperparameters(num_layers, min_width, num_epochs, batch_size, learning_rate):
    # build the model using the specified hyperparameters
    min_width, num_layers = int(min_width), int(num_layers)
    architecture = [min_width*(2**i) for i in range(num_layers)]
    model = build_model(architecture, input_shape, learning_rate)

    # fit the model
    early_stopper = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)
    train_result = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=int(batch_size),
        epochs=int(num_epochs),
        verbose=2,
        callbacks=[early_stopper]
    )

    # return lowest loss encountered in training
    return -min(train_result.history['val_loss'])


def compute_optimal_hyperparameters():

    pbounds = {
        "num_layers": (2, 6), 
        "min_width": (64, 512),
        "num_epochs": (50, 500),
        "batch_size": (16, 256),
        "learning_rate": (0.0005, 0.02)
    }

    optimizer = BayesianOptimization(
        f=test_hyperparameters,
        pbounds=pbounds,
        verbose=2,
        random_state=507,
    )

    optimizer.maximize(
        init_points=2,
        n_iter=10,
    )

    return optimizer.max["params"]


if __name__ == "__main__":
    # load training data
    X_train, X_val, y_train_orig, y_val_orig, auxiliary_train, auxiliary_val = model_ready_data("data/X.csv", "data/y.csv")

    from scipy.stats import pearsonr

    selected = []
    for col in range(len(X_train[0])):
        _, p = pearsonr(X_train[:, col], y_train_orig[:, 1])
        if p < 0.01:
            selected.append(col)
    X_train = X_train[:, selected]
    X_val = X_val[:, selected]
    input_shape = X_train[0].shape
    print(input_shape)

    final_architecture = [32, 16]
    final_model = build_model(final_architecture, input_shape, 0.001)
    early_stopper = keras.callbacks.EarlyStopping(monitor="loss", patience=20, restore_best_weights=True)
    final_model.fit(
        X_train,
        y_train_orig[:, 0],
        validation_data=(X_val, y_val_orig[:, 0]),
        batch_size=64,
        epochs=500,
        verbose=1,
        callbacks=[early_stopper]
    )
    
    exit()

    # determine and produce the best model for each target
    model_labels = ["1d", "5d", "20d"]
    for i in range(len(model_labels)):
        print(f"\n\nWorking on {model_labels[i]} model...\n\n")
        # select desired response (1d, 5d, 20d)
        y_train = y_train_orig[:, i]
        y_val = y_val_orig[:, i]

        # find optimal hyperparameters
        opt_hp = compute_optimal_hyperparameters()

        # train final model
        final_architecture = [int(opt_hp["min_width"])*(2**i) for i in range(int(opt_hp["num_layers"]))]
        final_model = build_model(final_architecture, input_shape, opt_hp["learning_rate"])
        early_stopper = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
        final_model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            batch_size=int(opt_hp["batch_size"]),
            epochs=int(opt_hp["num_epochs"]),
            verbose=1,
            callbacks=[early_stopper]
        )
        
        # save hyperparameters and final model
        with open(f"models/nn_{model_labels[i]}.json", "w") as f:
            json.dump(opt_hp, f, indent=4)
        final_model.save_weights(f"models/nn_{model_labels[i]}.h5")



    
