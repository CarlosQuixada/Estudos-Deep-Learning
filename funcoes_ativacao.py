import numpy as np


def stepFunction(soma):
    if soma >= 1:
        return 1
    return 0


def sigmoidFunction(soma):
    return 1 / (1 + np.exp(-soma))


def relu(soma):
    if soma >= 0:
        return soma
    return 0
