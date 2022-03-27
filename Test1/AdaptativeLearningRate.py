def constantLearningRate(x):
    return x

def timeBasedDecay(x):
    return (1. / (1. + c.DECAY * x ))

def stepDecay(x):
    return c.LEARNING_RATE * np.power(c.DECAY, np.floor(x / 10))

def exponentialDecay(t):
    return LEARNING_RATE * np.exp(-c.DECAY*t)


###  Adagrad, Adadelta, RMSprop, Adam