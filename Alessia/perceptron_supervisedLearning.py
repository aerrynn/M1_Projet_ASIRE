
#                                   perceptron_supervisedLearning


################################################################################################################
# IMPORTS
################################################################################################################

import numpy as np
import copy
import math
import random


################################################################################################################
# PARAMETERS
################################################################################################################

firstForwardPropagation = False     # tests if a first ForwardPropagation has been done
firstBackPropagation = False        # tests if a first BackPropagation has been done


################################################################################################################
# PERCEPTRON
################################################################################################################


class Perceptron(): 
    
    def __init__(self, genome, tabExtSensors, genomeExpert, tabSensorsExpert, lR, aE, maxIt, verbose = False): 

        assert len(genomeExpert) == len(tabSensorsExpert)

        self.genomeExpert = np.array(genomeExpert)
        self.tabSensorsExpert = np.array(tabSensorsExpert)

        self.data_y = np.multiply(self.tabSensorsExpert, self.genomeExpert)    # data_y : labels' set
        self.data_x = np.array(copy.deepcopy(tabExtSensors))                   # data_x : descriptors' set
        self.data_w = np.array(copy.deepcopy(genome))                          # data_w : genome (weights)
        
        self.bestW = []
        self.nbMaxIt = maxIt
        self.learningRate = lR
        self.allowedError = aE

        self.verbose = verbose


    def train(self):
        convergence = False
        old_data_w = copy.deepcopy(self.data_w)
        nbIt = 0

        while not(convergence):

            for i in range(len(self.data_x)) :
                x = self.data_x[i]
                w = self.data_w[i]
                y = self.data_y[i]

                # If the difference between this robot's prediction and the expert label is too big,
                # then modify the robot's weights to get closer to the expert perception of the environment
                prediction = self.prediction(x, w)
                diff = np.abs(prediction - y)
                if diff > 0.001 :
                    old_data_w = self.data_w
                    direction = self.direction(prediction, y)
                    self.data_w[i] = self.data_w[i] + ( diff * self.learningRate * direction )

                    if self.verbose:
                        if direction == 1:
                            imgSigne = ">"
                        else:
                            imgSigne = "<"
                        print("\n[PERCEPTRON new gene setted] : sensor*old_w =", prediction, "label =", y, ", direction =", imgSigne, "\n\t---> sensor*new_w =", (x*self.data_w[i]))

            # stop condition loop
            if np.sqrt(np.sum ((self.data_w - old_data_w)**2) ) < self.learningRate or nbIt > self.nbMaxIt :
                convergence = True
            
            nbIt += 1

        return self.data_w


    def prediction(self, x, w): 
        """ rend la prediction su senseur x * son poids w
            x : une description
            w : une case du genome (weight)
        """
        return np.multiply(x, w)
     

    def direction(self, p, y):
        """ rend le signe correspondant à la direction vers la valeur y
            p : une prediction
            y : un label
        """
        if y > p: 
            return 1
        return -1



#---------------------------------------------------------------------------------------------------------------


class neuralNetwork():

    def __init__(self, nb_neuronsPerInputs, nb_hiddenLayers, nb_neuronsPerHidden, nb_neuronsPerOutputs):
        """
        :param nb_neuronsPerInputs : number of neurons in the input layer
        :param nb_hiddenLayers : number of hidden layers
        :param nb_neuronsPerHidden : number of neurons in each hidden layer
        :param nb_neuronsPerOutputs : number of neurons in the output layer
        """
        
        self.n_neuronsPerInputs = nb_neuronsPerInputs
        self.n_hiddenLayers = nb_hiddenLayers
        self.n_neuronsPerHidden = nb_neuronsPerHidden
        self.n_neuronsPerOutputs = nb_neuronsPerOutputs

        self.weights = {}
        self.neurons = {}
        self.errors = {}
        cptLayers = 0

        # Hidden layers' weights : for every neuron on a hidden layer, we stock the weight 
        # linking that neuron with all the neurons and the bias of the previous layer.
        # We apply the same procedure to all the hidden layers.
        # NB. The first hidden layer will be linked with the input layer.
        self.n_neuronsPreviousLayer = self.n_neuronsPerInputs
        for layer in range(self.n_hiddenLayers):
            cptLayers += 1
            s = "Layer" + str(cptLayers)
            self.weights[s] = []
            for neuron in range(self.n_neuronsPerHidden):
                self.weights[s].append([random.random() for i in range(self.n_neuronsPreviousLayer + 1)])
                self.n_neuronsPreviousLayer = self.n_neuronsPerHidden

        # Output layers' weights : for every neuron on the output layer, we stock the weight 
        # linking that neuron with all the neurons and the bias of the previous layer
        cptLayers += 1
        s = "Layer" + str(cptLayers)
        self.weights[s] = []
        for neuron in range(self.n_neuronsPerOutputs):
            if self.n_hiddenLayers > 0 :   
                self.weights[s].append([random.random() for i in range(self.n_neuronsPerHidden + 1)])
            else:
                self.weights[s].append([random.random() for i in range(self.n_neuronsPerInputs + 1)])


    #-------------------------------------------------------------
    
    def getWeights(self):
        """Returns the network's weights (dictionnary), organised in layers (key)
        """
        return self.weights


    def getNetworkInformation(self):
        """Returns :
            - general structure
            - neurons values (dict)
            - weights (dict)
            - error estimation (dict)
        """

        print(f"\nNeural network with {self.n_hiddenLayers} hidden layers,")
        print(f"composed by {self.n_neuronsPerInputs} input neurons + bias (bias value = 1), {self.n_neuronsPerHidden} hidden neurons + bias (bias value = 1) for each hidden layer, {self.n_neuronsPerOutputs} outputs neurons.\n")
        print("neurons (dict) :", self.neurons, "\n")
        print("weights (dict) :", self.weights, "\n")
        print("error (dict) :", self.errors, "\n")

        return self.neurons, self.weights, self.errors


    #-------------------------------------------------------------
    
    def neuronActivation(self, weightsToPreviousLayer, inputsPreviousLayer):
        """Returns the weighted sum of the inputs from the previous layer
        
        :param weightsToPreviousLayer : list of weights connecting this neuron with the previous layer
        :param inputsPreviousLayer : list of neuron values of the previous layer
        """
        activation = 0

        # activation = sum (neuron_i * weight_i)
        for i in range(len(weightsToPreviousLayer)-1):
            activation += inputsPreviousLayer[i] * weightsToPreviousLayer[i]
        
        # we add the bias weight to the sum (we consider that bias = 1, so weight * bias = weight)
        activation += weightsToPreviousLayer[-1]

        return activation


    #-------------------------------------------------------------

    def stableSigmoid(self, x):
        """Returns the résult of the function sigmoid(x) avoiding overflow numeric issue
        """
        if x >= 0:
            z = math.exp(-x)
            sig = 1.0 / (1.0 + z)
            return sig
        else:
            z = math.exp(x)
            sig = z / (1.0 + z)
            return sig


    #------------------------------------------------------------- 

    def transferNeuronActivation(self, activation):
        """Returns the résult of the function sigmoid(activation)
        """
        return self.stableSigmoid(activation)


    #------------------------------------------------------------- 

    def forwardPropagation(self, inputsNeuronsValues):
        """Returns the output layer, given the input layer

        :param inputsNeuronsValues : values of the first neuron's layer (inputs)
        """
        # for each layer in 'weights' (hidden + output):
        inputLayer = inputsNeuronsValues
        for layer in self.weights.keys(): 
            self.neurons[layer] = []        
            newInputLayer = []  

            # computing the value of the neurons in the current layer, from  the input values
            # of the previous layer. The current layer becomes the new input layer for the next iteration.
            # For each 'neuron':
            for weightsNeuron in self.weights[layer]:
                activation = self.neuronActivation(weightsNeuron, inputLayer) # sum(input_i * w_i) previous Layer
                self.neurons[layer].append(self.transferNeuronActivation(activation)) # sigmoid function
                newInputLayer.append(self.neurons[layer][-1])
            inputLayer = newInputLayer

        global firstForwardPropagation
        firstForwardPropagation = True

        # we return the last inpurLayer which corresponds to the output layer of the NN
        return inputLayer


    #------------------------------------------------------------- 
    #------------------------------------------------------------- 
    #------------------------------------------------------------- 

    def derivative(self, x):
        """Returns the résult of the derivative(x)
        """
        return x * (1.0 - x)


    #------------------------------------------------------------- 

    def computeErrorOutputLayer(self, outputNeuron, label):
        """Returns the error for each neuron in the outputLayer
        """
        return (outputNeuron - label) * self.derivative(outputNeuron)


    #------------------------------------------------------------- 

    def computeErrorHiddenLayer(self, hiddenNeuronsCurrentLayer, weightsToTheNextLayer, errorsNextLayer):
        """Returns the error for each neuron in the hiddenLayer
        """
        errorsHiddenLayer = []

        # error of one neuron in the hidden layer = (error_outputNeuron_j * weight_kj linking the outputNeuron_j to the hiddenNeuron_k) * derivative(output of the current hiddenNeuron_k)
        for h in range(len(hiddenNeuronsCurrentLayer)):
            hiddenError = 0
            for e in range(len(errorsNextLayer)):
                hiddenError += errorsNextLayer[e] * weightsToTheNextLayer[e][h]
            errorsHiddenLayer.append(hiddenError * self.derivative(hiddenNeuronsCurrentLayer[h]))

        return errorsHiddenLayer


    #------------------------------------------------------------- 
    
    def backPropagation(self, labels):
        """Returns the error estimation (dictionnary) for each neuron in the network,
        obteined by backpropaging the output's error e = (output - expected label) * derivative function (output)
        from the outputLayer back to the inputLayer.

        :param labels : list of each expected value for the output neuron in the output layer
        """

        assert firstForwardPropagation == True

        # Loop from the outputLayer to the inputLayer, backtracking
        layersLabels = list(self.weights.keys())
        for i in range(len(self.weights), 0, -1):
            self.errors[layersLabels[i-1]] = []

            # Getting error estimation for each neuron in the outputLayer
            if i == len(self.weights):
                for neuron in range(len(self.neurons[layersLabels[i-1]])):
                    self.errors[layersLabels[i-1]].append(self.computeErrorOutputLayer(self.neurons[layersLabels[i-1]][neuron], labels[neuron]))
            
            # Getting error estimation for each neuron in the hiddenLayer
            else:
                self.errors[layersLabels[i-1]] = self.computeErrorHiddenLayer(self.neurons[layersLabels[i-1]], self.weights[layersLabels[i]], self.errors[layersLabels[i]])

        global firstBackPropagation
        firstBackPropagation = True

        # we return the dictionnary of errors
        return self.errors


    #------------------------------------------------------------- 
    #------------------------------------------------------------- 
    #------------------------------------------------------------- 

    def updateWeights(self, inputLayer, learningRate = 1.0):
        """Returns the updated weights of the network

        :param inputLayer : dataset of values as entry of the network
        :param learningRate : pourcentage that controls how much modify the weight to correct for the error. learningRate = 0.1 ---> 10% of correction is applied
        """
        assert firstBackPropagation == True

        inputPreviousLayer = inputLayer

        for layer in range(len(self.weights)): # { 0, 1, ... , nbLayers from hiddenLayer to outputLayer}

            # The layers differents from the inputLayer, have inputs = neuron values from the previous layer
            if layer != 0:
                s = 'Layer' + str(layer)
                inputPreviousLayer = self.neurons[s]

            s = 'Layer' + str(layer+1)
            for neuron in range(len(self.neurons[s])): # { 0, 1, ... , nbNeuronsInCurrentLayer but bias}
                listWeightsCurrentNeuron = self.weights[s][neuron]
                for w in range(len(listWeightsCurrentNeuron)-1):  # { 0, 1, ... , n-1}, weight associated to bias is excluded
                    listWeightsCurrentNeuron[w] = listWeightsCurrentNeuron[w] - (learningRate * self.errors[s][neuron] * inputPreviousLayer[w])
                listWeightsCurrentNeuron[w+1] = listWeightsCurrentNeuron[w+1] - (learningRate * self.errors[s][neuron]) # particular computing for bias' weight


    #------------------------------------------------------------- 

    def train(self, trainingDataset, labelsDataset, nb_epoch, learningRate = 1.0):
        """Returns the trained network (updated optimised weights). Procedure:
        For each input row from the training set 'inputLayer':
            - to propagate inputs from the inputLayer to the outputLayer and get the output
            - to backpropagate errors from the outputLayer to the inputLayer and get the errors dictionnary
            - to update the network weights

        :param trainingDataset : dataset of values as entry of the network
        :param labelsDataset : dataset of expected values as outputs of the network
        :param nb_epoch : number of iterations 
        :param learningRate : pourcentage that controls how much modify the weight to correct for the error. learningRate = 0.1 ---> 10% of correction is applied 
        """

        assert len(trainingDataset) == len(labelsDataset)
        datasetsSize = len(trainingDataset)

        print("\nTraining process :")
        for epoch in range(nb_epoch):
            costFunction = 0 # negative gradient
            for nbRow in range(datasetsSize):
                inputLayer = trainingDataset[nbRow]
                outputs = self.forwardPropagation(inputLayer)
                expectedOutput = labelsDataset[nbRow]
                
                costFunction += sum([(expectedOutput[i] - outputs[i])**2 for i in range(len(outputs))])

                self.backPropagation(expectedOutput)
                self.updateWeights(inputLayer, learningRate)

            print(f"Epoch : {epoch}, \tLearning rate : {learningRate}, \tcostFunction : {costFunction}")

        return self.weights


    #------------------------------------------------------------- 

    def predict(self, inputLayer):
        outputs = self.forwardPropagation(inputLayer)
        return outputs
        #return self.interpretRobotAction(outputs)


   #------------------------------------------------------------- 

    # def interpretRobotAction(self, networkOutputs):
    #     if len(networkOutputs) == 2 :
    #         return networkOutputs[0], networkOutputs[1]
    # ici il faut faire en sorte que l action puisse avoir une valeur 1 et -1
    # verifier le range des valeurs output







####################################################
# TESTS
####################################################

# network = neuralNetwork(3, 1, 1, 3)
# network.getWeights()
# network.forwardPropagation([1, 1, 1])
# network.backPropagation([0.5, 0.5, 0.5])
# network.updateWeights([1, 1, 1])

#------------------------------------------------------------- 

# dataset = [[2.7810836,2.550537003],
# 	[1.465489372,2.362125076],
# 	[3.396561688,4.400293529],
# 	[1.38807019,1.850220317],
# 	[3.06407232,3.005305973],
# 	[7.627531214,2.759262235],
# 	[5.332441248,2.088626775],
# 	[6.922596716,1.77106367],
# 	[8.675418651,-0.242068655],
# 	[7.673756466,3.508563011]]

# expected = [[1,0],
# 	[0,0],
# 	[0,1],
# 	[0,0],
# 	[1,1],
# 	[0,1],
# 	[0,1],
# 	[1,0],
# 	[1,1],
# 	[0,1]]


# network = neuralNetwork(2, 2, 1, 2)
# network.train(dataset, expected, 20, 0.5)

# network.getNetworkInformation()

#------------------------------------------------------------- 
