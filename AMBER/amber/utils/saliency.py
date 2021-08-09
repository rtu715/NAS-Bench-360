# -*- coding: UTF-8 -*-
from __future__ import print_function

import numpy as np


def approx_grad(model, X, epsilon=0.01):
    nr, nc = X.shape
    grad = np.zeros(nc)
    for i in range(nc):
        X_tmp = np.copy(X)
        X_tmp[:, i] += epsilon
        y_plus = model.predict(X_tmp)
        X_tmp[:, i] -= 2 * epsilon
        y_minus = model.predict(X_tmp)
        grad[i] = np.mean((y_plus - y_minus) / 2. / epsilon)
    return grad


def approx_grad_array(model, X, epsilon=0.01):
    nr, nc = X.shape
    grad_array = np.zeros((nr, nc))
    for i in range(nc):
        X_tmp = np.copy(X)
        X_tmp[:, i] += epsilon
        y_plus = model.predict(X_tmp).flatten()
        X_tmp[:, i] -= 2 * epsilon
        y_minus = model.predict(X_tmp).flatten()
        grad_array[:, i] = (y_plus - y_minus) / 2. / epsilon
    return grad_array


def approx_hessian(model, x, epsilon=0.01):
    p = x.shape[1]
    hess = np.zeros((p, p))
    grad = np.zeros(p)

    def grad_func(x, idx):
        x_ = np.copy(x)
        y0 = model.predict(x_)
        # y0 = simulator.get_ground_truth(x_)
        x_[:, idx] += epsilon
        y1 = model.predict(x_)
        # y1 = simulator.get_ground_truth(x_)
        x_[:, idx] -= epsilon * 2
        y2 = model.predict(x_)
        # y2 = simulator.get_ground_truth(x_)
        return ((y1 - y0) + (y0 - y2)) / (2 * epsilon)

    for i in range(p):
        g0 = grad_func(x, i)
        grad[i] = g0
        for j in range(p):
            x1 = np.copy(x)
            x1[:, j] += epsilon
            g1 = grad_func(x1, i)
            x1[:, j] -= epsilon * 2
            g2 = grad_func(x1, i)
            hess[i, j] = ((g1 - g0) + (g0 - g2)) / (2 * epsilon)
    return hess


def approx_hessian_array(model, data, epsilon=0.01):
    hess = np.zeros((data.shape[1], data.shape[1]))
    for i in range(data.shape[0]):
        hess += approx_hessian(model, data[i].reshape(1, data.shape[1]), epsilon)
    hess /= data.shape[0]
    return hess
