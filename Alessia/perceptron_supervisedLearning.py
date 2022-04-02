
#                                   perceptron_supervisedLearning


################################################################################################################
# IMPORTS
################################################################################################################

from re import X
import numpy as np
import copy


################################################################################################################
# PARAMETERS
################################################################################################################




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
