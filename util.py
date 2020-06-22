import importlib
import os
import re
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, log_loss, r2_score, mean_squared_error, f1_score

def evaluate(y, y_pred, measure=None):
    if measure == "auc":
        return roc_auc_score(y, y_pred)
    elif measure == "mse":
        return mean_squared_error(y.ravel(), y_pred.ravel())
    elif measure == "cross_entropy":
        return log_loss(y, y_pred)
    elif measure == "f1_micro":
        return f1_score(np.argmax(y, axis=1), np.argmax(y_pred, axis=1), average='micro')
    else:
        return r2_score(y.ravel(), y_pred.ravel())

def encode_onehot(data_y):
    return np.eye(np.max(data_y) + 1)[data_y.astype("int32")]

def load_data(file, target, to_onehot = False, loader=None):
    if loader is not None:
        lib = load_lib(loader)
        return lib.load()

    df = pd.read_csv(file)
    df_x = df.drop(df.columns[target], axis=1)
    df_y = df.iloc[:, target]

    data_x = df_x.to_numpy()
    data_y = df_y.to_numpy()

    if to_onehot:
        data_y = encode_onehot(data_y[:, 0])

    return data_x, data_y

def load_lib(model_name, path = None):
    module_name = model_name if path is None else re.sub(r"[\/\\]", ".", os.path.splitext(path)[0])
    module = importlib.import_module(module_name)
    model = getattr(module, re.search(r"[^\.]+$", model_name).group())
    return model()

def cumulative_kl(x, y, fraction=0.5):
    # Implementation based on Pérez-Cruz (2008) - Kullback-Leibler divergence estimation of continuous distributions
    def ecdf(x):
        x = np.sort(x)
        u, c = np.unique(x, return_counts=True)
        n = len(x)
        y = (np.cumsum(c) - 0.5) / n

        def interpolate_(x_):
            yinterp = np.interp(x_, u, y, left=0.0, right=1.0)
            return yinterp

        return interpolate_

    dx = np.diff(np.sort(np.unique(x)))
    dy = np.diff(np.sort(np.unique(y)))
    ex = np.min(dx)
    ey = np.min(dy)
    e = np.min([ex, ey]) * fraction
    n = len(x)
    m = len(y)
    P = ecdf(x)
    Q = ecdf(y)
    KL = (1.0 / n) * np.sum(np.log((P(x) - P(x - e)) / (Q(x) - Q(x - e))))
    return KL

def reg_ratio(action_probs, ratio):
    return torch.abs(torch.mean(action_probs[:, 1]) - ratio)

def reg_iid(action_probs, y):
    y_prob = torch.matmul(action_probs.transpose(0, 1), y)
    p_train = y_prob[1] / torch.sum(y_prob[1])
    p_test = y_prob[0] / torch.sum(y_prob[0])
    return torch.sum(p_test * torch.log(p_test / p_train))

def reg_iidr(state, action):
    data_x, data_y = state
    train_y = data_y[action == 1].ravel()
    test_y = data_y[action == 0].ravel()

    train_kl = cumulative_kl(train_y, data_y)
    test_kl = cumulative_kl(test_y, data_y)
    objective_kl = train_kl + test_kl
    return np.exp(-objective_kl)
