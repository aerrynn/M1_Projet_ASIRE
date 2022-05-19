
#                                  PART 2 : DIFFUSION, FORAGING TASK
#                                            NEURAL NETWORK


################################################################################################################
# IMPORTS
################################################################################################################

import math
import random
import copy




################################################################################################################
# PARAMETERS
################################################################################################################

firstForwardPropagation = False     # tests if a first ForwardPropagation has been done
firstBackPropagation = False        # tests if a first BackPropagation has been done

strRobotId = None
debug = None


################################################################################################################
# NEURAL NETWORK, FORWARD PROPAGATION and BACKPROPAGATION
################################################################################################################


class neuralNetwork():

    def __init__(self, nb_neuronsPerInputs, nb_hiddenLayers, nb_neuronsPerHidden, nb_neuronsPerOutputs, debugNN=False, strRId=None):
        """
        :param nb_neuronsPerInputs : number of neurons in the input layer
        :param nb_hiddenLayers : number of hidden layers
        :param nb_neuronsPerHidden : number of neurons in each hidden layer
        :param nb_neuronsPerOutputs : number of neurons in the output layer
        """
        
        global strRobotId, debug
        strRobotId = strRId
        debug = debugNN

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


    #-------------------------------------------------------------

    def getWeightsList(self):

        weightsList = []
        for layer in self.weights.keys():
            for neuron in self.weights[layer]:
                for w in neuron:
                    weightsList.append(w)

        return weightsList
                    

    #-------------------------------------------------------------

    def setWeightsFromList(self, tabWeights):
        weightsDict = {}
        tabW = copy.deepcopy(tabWeights)

        cptLayers = 0
        for layer in range(self.n_hiddenLayers):
            cptLayers += 1
            s = "Layer" + str(cptLayers)
            weightsDict[s] = []
            for neuron in range(self.n_neuronsPerHidden):
                weightsDict[s].append([tabW.pop(0) for i in range(self.n_neuronsPreviousLayer + 1)])
                self.n_neuronsPreviousLayer = self.n_neuronsPerHidden

        # Output layers' weights : for every neuron on the output layer, we stock the weight 
        # linking that neuron with all the neurons and the bias of the previous layer
        cptLayers += 1
        s = "Layer" + str(cptLayers)
        weightsDict[s] = []
        for neuron in range(self.n_neuronsPerOutputs):
            if self.n_hiddenLayers > 0 :   
                weightsDict[s].append([tabW.pop(0) for i in range(self.n_neuronsPerHidden + 1)])
            else:
                weightsDict[s].append([tabW.pop(0) for i in range(self.n_neuronsPerInputs + 1)])

        self.weights = weightsDict


    #-------------------------------------------------------------

    def getNetworkInformation(self):
        """Returns :
            - general structure
            - neurons values (dict)
            - weights (dict)
            - error estimation (dict)
        """
        return self.neurons, self.weights, self.errors


    #-------------------------------------------------------------

    def printNetworkInformation(self):
        print(f"\n{strRobotId}\tNEURAL NETWORK INFORMATION")
        print(f"\nNeural network composed by:")
        print(f"\t{self.n_hiddenLayers} hidden layers,")
        print(f"\t{self.n_neuronsPerInputs} input neurons + bias (bias value = 1),")
        print(f"\t{self.n_neuronsPerHidden} hidden neurons + bias (bias value = 1) for each hidden layer,")
        print(f"\t{self.n_neuronsPerOutputs} outputs neurons.")
        print("\n***************************************************************")


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

        # we return the last inputLayer, which corresponds to the output layer of the NN
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

        if debug:
            print(f"\n{strRobotId}\tTRAINING PROCESS DETAILS :")
            print(f"\t\tLearning rate : {learningRate}")
            print(f"\t\tNumber epochs : {nb_epoch}")
            print(f"\nTRAINING DATASET and LABELS recived from expert :\t(len = {len(trainingDataset)})")
            for row in range(len(trainingDataset)):
                print("(" + ' '.join(str(elem) for elem in trainingDataset[row]) + ") :", labelsDataset[row])


        for epoch in range(nb_epoch+1):
            costFunction = 0 # negative gradient
            for nbRow in range(datasetsSize):
                inputLayer = trainingDataset[nbRow]
                outputs = self.forwardPropagation(inputLayer)
                expectedOutput = labelsDataset[nbRow]
                
                costFunction += sum([(expectedOutput[i] - outputs[i])**2 for i in range(len(outputs))])

                self.backPropagation(expectedOutput)
                self.updateWeights(inputLayer, learningRate)

            if debug:
                # Select the following line to show ALL the epochs
                print(f"\tEpoch : {epoch} \tcostFunction : {costFunction}")
                #-----------------------------------------------------
                # Select the following lines to show ONLY the first and the last epoch
                # if epoch == 1 or epoch == nb_epoch:
                #     print(f"Epoch : {epoch}, \tLearning rate : {learningRate}, \tcostFunction : {costFunction}")
        
        if debug:
            print("\n---------------------------------------------------------------")

        return self.weights


    #------------------------------------------------------------- 

    def predict(self, inputLayer):
        outputs = self.forwardPropagation(inputLayer)
        return outputs


    #------------------------------------------------------------- 



