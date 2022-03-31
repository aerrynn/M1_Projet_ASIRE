import Const as c
import numpy as np


def constantLearningRate(x):
    return c.LEARNING_RATE


def timeBasedDecay(x):
    return (1. / (1. + c.DECAY * x))*c.LEARNING_RATE


def stepDecay(x):
    return c.LEARNING_RATE * np.power(c.DECAY, np.floor(x / 10))


def exponentialDecay(t):
    return c.LEARNING_RATE * np.exp(-c.DECAY*t)


###  Adagrad, Adadelta, RMSprop, Adam
