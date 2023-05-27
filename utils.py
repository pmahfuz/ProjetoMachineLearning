import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import keras


def plot_multiple_forecasts(X, Y, Y_pred, scaler):
    n_steps = X.shape[1]
    ahead = Y.shape[1]
    plot_series(scaler.inverse_transform(X[0, :]))
    plt.plot(
        np.arange(n_steps, n_steps + ahead),
        scaler.inverse_transform(Y[0, :]),
        "bo-",
        label="Actual",
    )
    plt.plot(
        np.arange(n_steps, n_steps + ahead),
        scaler.inverse_transform(Y_pred[0, :]),
        "rx-",
        label="Forecast",
        markersize=10,
    )
    plt.legend(fontsize=14)


def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])


def plot_learning_curves(loss, val_loss):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)


def plot_series(
    series, y=None, y_pred=None, x_label="$t$", y_label="$x(t)$", legend=True, n_steps=5
):
    plt.plot(series, ".-")
    if y is not None:
        plt.plot(n_steps, y, "bo", label="Target")
    if y_pred is not None:
        plt.plot(n_steps, y_pred, "rx", markersize=10, label="Prediction")
    plt.grid(True)
    if x_label:
        plt.xlabel(x_label, fontsize=16)
    if y_label:
        plt.ylabel(y_label, fontsize=16, rotation=0)
    if legend and (y or y_pred):
        plt.legend(fontsize=14, loc="upper left")
