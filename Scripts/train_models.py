"""
Not completed.
"""
import numpy as np
import pandas as pd


def train_with_features(x1_train, x2_train, x1_valid, x2_valid, y_train, y_valid, param_dict, model):
    """
    Train a model with textual data (X1) and additional features (X2) and passes a dictionary with
    all relevant training parameters. Returns the trained model and a history dataframe with loss and
    accuracy at every epoch.
    """
    history = model.fit([np.array(x1_train),
                         np.array(x2_train).reshape((
                             len(x2_train), param_dict["max_len"], param_dict["num_features"]))],
                        y_train, validation_data=([np.array(x1_valid),
                                                   np.array(x2_valid).reshape((
                                                       len(x2_valid), param_dict["max_len"],
                                                       param_dict["num_features"]))],
                                                  y_valid), batch_size=param_dict["batch_size"],
                        epochs=param_dict["epochs"], verbose=1)
    hist = pd.DataFrame(history.history)
    return model, hist


def train(x_train, x_valid, y_train, y_valid, param_dict, model):
    """
    Train a model with textual data (X1) and passes a dictionary with
    all relevant training parameters. Returns the trained model and a history
    dataframe with loss and accuracy at every epoch.
    """
    history = model.fit(np.array(x_train), y_train,
                        validation_data=(np.array(x_valid), y_valid),
                        batch_size=param_dict["batch_size"], epochs=param_dict["epochs"], verbose=1)
    hist = pd.DataFrame(history.history)
    return model, hist


