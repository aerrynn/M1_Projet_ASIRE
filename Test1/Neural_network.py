import numpy as np
import Const as g
import random


def sigmoid(x: float) -> float:
    '''
    sigmoid:  bounded, differentiable, real function that is defined for all real input values 
    and has a non-negative derivative at each point[1] and exactly one inflection point
    '''
    if x.any() < -10:
        print(x)
    return 1.0 / (1.0 + np.exp(-x))


# Sigmoide derivative
def d_sigmoid(x: float) -> float:
    '''
    derivative of the aforementioned sigmoid function
    '''
    return sigmoid(x) * (1.0 - sigmoid(x))


class Layer:
    '''
    Une seule couche de neurones.
    '''

    def __init__(self, size: int, input_size: int):
        '''
        Constructor:
            :param size: Size of the previous layer
            :param input_size: Size of the new layer
        '''
        self.size = size
        self.input_size = input_size
        self.weights = np.random.randn(size, input_size)    #
        self.biases = np.random.randn(size)                 # Neuron Bias

    def forward(self, data: np.ndarray) -> np.ndarray:
        '''
        forward: propagates the input data through the layer
            :param data: the input data
            :return activation: the values computed
        '''
        aggregation = self.aggregation(data)
        activation = self.activation(aggregation)
        return activation

    def aggregation(self, data: np.ndarray) -> float:
        '''
        TODO: To fill
        aggregation:
            :param data:
        '''
        return np.dot(self.weights, data) + self.biases

    def activation(self, x: float) -> float:
        '''
        TODO: To fill
        activation:
        '''
        return sigmoid(x)

    def activation_prime(self, x: float) -> float:
        '''
        activation_prime: computes the value of the derivative activation of the 
        current layer
        '''
        return d_sigmoid(x)

    # gradient update
    def update_weights(self, gradient, learning_rate: float):
        '''
        update_weights: partially learns from the weights gradient to improve the layer
            :param gradient: The newly computed weights
            :param learning rate: The learning ratio applied to the new weights
        '''
        self.weights -= learning_rate * gradient

    # Bias update
    def update_biases(self, gradient, learning_rate: float):
        '''
        update_biases: partially learns from the biases gradient to improve the layer
            :param gradient: The newly computed biases
            :param learning rate: The learning ratio applied to the new biases
        '''
        self.biases -= learning_rate * gradient


class NeuralNetwork:
    '''
    Actual set of neurons
    '''

    def __init__(self, input_dim: int, nb_layers: int = 1, size_layer: int = 8):
        '''
        Constructor
            :param input_dim: input dimension
            :param nb_layers: amount of layers of the network (1 = perceptron)
            :param size_layer: number of neurons for each layer
        '''
        self.input_dim = input_dim
        self.layers = []
        for _ in range(nb_layers):
            self.add_layer(size_layer)
        self.add_layer(2)

    def add_layer(self, size: int):
        '''
        add_layer: Add a single layer to the neural network
            :param size: amount of neurons in the new layer
        '''
        if len(self.layers) > 0:
            input_dim = self.layers[-1].size
        else:
            input_dim = self.input_dim

        self.layers.append(Layer(size, input_dim))

    def feedforward(self, input_data: np.ndarray):
        '''
        feedforward: Layer to layer propagation
            :input input_data: data detected by the sensors
            :return activation: computed data by the neural network
        '''
        activation = input_data
        for layer in self.layers:
            activation = layer.forward(activation)
        return activation

    def to_output(self, last_layer):
        '''
        to_output: Changes the neural network output range from ([0,1],[0,1]) to ([0,1],[-0.5,0.5])
            :@param last_layer: the vector calculated by the last layer of the neural network
        '''
        return(last_layer[0], last_layer[1]-0.5)

    def ff_to_output(self, observations: np.ndarray):
        '''
        ff_to_output: Compute the complete neural network results and turns it into a vector the agent can use
            :param observations: A vector containing the sensors results.
        '''
        return self.to_output(self.feedforward(observations))

    def train(self, X, Y, steps: int = 30, learning_rate: float = 0.3, batch_size: int = 10):
        '''
        train: Train the neural network with the retropropagation algorithm over a dataset
            :param X: list of inputs of the data
            :param Y: list of outputs of the data
            :param steps: Amount of times the network iterates over the data
            :param learning rate: Neural network learning rate
            :batch size: Mac amount of input/output tuples in each batch
        '''
        n = Y.size
        for i in range(steps):                                  # Repeats steps times
            # Iterates over each element of the data subdivided into smaller batches
            for batch_start in range(0, n, batch_size):
                X_batch, Y_batch = X[batch_start:batch_start +
                                     batch_size], Y[batch_start:batch_start + batch_size]
                self.train_batch(X_batch, Y_batch, learning_rate)

    def train_batch(self, X: np.ndarray, Y: np.ndarray, learning_rate: float):
        '''
        TODO: To fill / to comment
        train_batch:
            :param X:
            :param Y:
            :param learning_rate:
        '''
        # Initialising gradients
        weight_gradient = [np.zeros(layer.weights.shape)
                           for layer in self.layers]
        bias_gradient = [np.zeros(layer.biases.shape) for layer in self.layers]

        # Retropropagation
        for (x, y) in zip(X, Y):
            new_weight_gradient, new_bias_gradient = self.backprop(x, y)
            weight_gradient = [wg + nwg for wg,
                               nwg in zip(weight_gradient, new_weight_gradient)]
            bias_gradient = [bg + nbg for bg,
                             nbg in zip(bias_gradient, new_bias_gradient)]

        # Average of each gradient
        avg_weight_gradient = [wg / max(1, Y.size) for wg in weight_gradient]
        avg_bias_gradient = [bg / max(1, Y.size) for bg in bias_gradient]

        # Updating biaises and weights
        for layer, weight_gradient, bias_gradient in zip(self.layers,
                                                         avg_weight_gradient,
                                                         avg_bias_gradient):
            layer.update_weights(weight_gradient, learning_rate)
            layer.update_biases(bias_gradient, learning_rate)

    def backprop(self, x: np.ndarray, y: np.ndarray):
        '''
        TODO: To fill / To comment
        backprop:
            :param x:
            :param y:
            :return weight_gradients:
            :return bias_gradient:
        '''
        aggregations = []
        activation = x
        activations = [activation]
        for layer in self.layers:
            aggregation = layer.aggregation(activation)
            aggregations.append(aggregation)
            activation = layer.activation(aggregation)
            activations.append(activation)
        target = y
        delta = self.get_output_delta(activation, target)
        deltas = [delta]
        nb_layers = len(self.layers)
        for l in reversed(range(nb_layers - 1)):
            layer = self.layers[l]
            next_layer = self.layers[l + 1]
            activation_prime = layer.activation_prime(aggregations[l])
            delta = activation_prime * \
                np.dot(next_layer.weights.transpose(), delta)
            deltas.append(delta)
        deltas = list(reversed(deltas))
        weight_gradient = []
        bias_gradient = []
        for l in range(len(self.layers)):
            prev_activation = activations[l]
            weight_gradient.append(np.outer(deltas[l], prev_activation))
            bias_gradient.append(deltas[l])

        return weight_gradient, bias_gradient

    def get_output_delta(self, activation, target):
        '''
        get_output_delta: Computes the difference between the targeted value and the
        calculated value for each component of the output vector
            :param activation: the computed value of the last layer of the network
            :param target: The targeted output
            :param delta: The difference of the expected action and the wanted action
        '''
        activation = self.to_output(activation)
        delta = activation - target
        return delta

    def mutate(self):
        # TODO: do mutate function, using gaussians
        pass

# To test the actual code
if __name__ == '__main__':
    data_x = []
    data_y = []

    empty_inpt = [float(x % 2) for x in range(1, 17)]

    food_left = [float(x % 2) for x in range(1, 17)]
    food_left[2] = 0.8
    food_left[3] = g.FOOD_ID

    food_front = [float(x % 2) for x in range(1, 17)]
    food_front[4] = 0.7
    food_front[5] = g.FOOD_ID

    food_right = [float(x % 2) for x in range(1, 17)]
    food_right[6] = 0.8
    food_right[7] = g.FOOD_ID

    wall_front = [float(x % 2) for x in range(1, 17)]
    wall_front[4] = 0.8
    wall_front[5] = g.WALL_ID

    wall_left = [float(x % 2) for x in range(1, 17)]
    wall_left[2] = 0.8
    wall_left[3] = g.WALL_ID
    wall_left[4] = 0.7
    wall_left[5] = g.WALL_ID

    wall_right = [float(x % 2) for x in range(1, 17)]
    wall_right[4] = 0.7
    wall_right[5] = g.WALL_ID
    wall_right[6] = 0.8
    wall_right[7] = g.WALL_ID

    for _ in range(7):
        data_x.append(empty_inpt)
        data_y.append([1, 0])

    data_x.append(food_left)
    data_y.append([1, 0.5])

    data_x.append(food_front)
    data_y.append([1, 0])

    data_x.append(food_right)
    data_y.append([1, -.5])

    for _ in range(7):
        data_x.append(wall_front)
        data_y.append([1, -0.5])

    data_x.append(wall_left)
    data_y.append([1.0, -0.5])

    data_x.append(wall_right)
    data_y.append([1.0, 0.5])

    data_x = np.array(data_x)
    data_y = np.array(data_y)

    # print(data_x)
    # print(data_y)
    eval_x = []
    attendu_y = []

    eval_x.append(empty_inpt)
    attendu_y.append([1, 0])

    eval_x.append(food_left)
    attendu_y.append([1, 0.5])

    eval_x.append(food_right)
    attendu_y.append([1, -.5])

    eval_x.append(wall_front)
    attendu_y.append([1, -0.5])

    eval_x.append(wall_right)
    attendu_y.append([1.0, .5])

    eval_x = np.array(eval_x)

    nn = NeuralNetwork(16, 1, 16)
    # nn.train(np.array([wall_left]), [1,0.5])
    nn.train(data_x, data_y, 1000)
    for i, (each, att) in enumerate(zip(eval_x, attendu_y)):
        print(f"{i}, {each}\n{nn.ff_to_output(each)}, expected {att}")
    # pass
