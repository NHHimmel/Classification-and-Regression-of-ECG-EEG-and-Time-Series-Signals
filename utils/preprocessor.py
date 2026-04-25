from __future__ import annotations
import numpy as np
from sklearn.preprocessing import LabelEncoder


def encode_label(y_train, y_val, y_test) -> tuple:
    enc = LabelEncoder()
    enc.fit(np.concatenate([y_train, y_val, y_test]))
    y_train = enc.transform(y_train)
    y_val = enc.transform(y_val)
    y_test = enc.transform(y_test)

    return y_train, y_val, y_test


class Target_Normalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, y):
        self.mean = y.mean()
        self.std = y.std() if y.std() > 0 else 1

    def transform(self, y):
        assert self.mean is not None and self.std is not None, "Call the fit method first!"
        return (y - self.mean) / self.std

    def inverse_transform(self, y):
        return y * self.std + self.mean


class Normalizer:
    def __init__(self, config):
        self.config = config
        self.means = []
        self.stds = []

    def fit(self, X):
        self.means = []
        self.stds = []
        for C in range(self.config['input_shape'][0]):
            self.means.append(X[:, C, :].mean())
            self.stds.append(X[:, C, :].std() if X[:, C, :].std() > 0 else 1)

    def transform(self, X):
        assert len(self.means)==X.shape[1] ,"Shape mismatch!"
        assert self.means != [] and self.stds != [], "Call fit method first!"
        X = X.astype('float32', copy=True)
        for C in range(self.config['input_shape'][0]):
            X[:, C, :] = (X[:, C, :] - self.means[C]) / self.stds[C]
        return X
