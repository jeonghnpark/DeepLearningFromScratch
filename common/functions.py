import numpy as np


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=int)  # np.array(x>0) => bool array => int array로 변환


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x >= 0] = 1  # x>=0 : bool array
    return grad


def softmax(x):
    if x.ndim == 2:
        