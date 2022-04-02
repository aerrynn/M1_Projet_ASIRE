
#                                   perceptron_supervisedLearning


################################################################################################################
# IMPORTS
################################################################################################################

import numpy as np
import copy


################################################################################################################
# PARAMETERS
################################################################################################################




################################################################################################################
# PERCEPTRON
################################################################################################################


class Perceptron(): 
    
    def __init__(self, genome, tabSensors, genomeExpert, tabSensorsExpert, learning_rate, allowed_error, nbMaxIt, verbose = False): 
        
        assert len(genomeExpert) == len(tabSensorsExpert)
        self.data_y = tabSensorsExpert * genomeExpert       # data_y : labels' set

        self.learning_rate = learning_rate
        self.allowed_error = allowed_error
        self.data_x = copy.deepcopy(tabSensors)             # data_x : descriptors' set
        self.data_w = copy.deepcopy(genome)                 # data_w : genome (weights)
        self.bestW = []
        self.nbMaxIt = nbMaxIt


    def train(self):
        convergence = False
        old_data_w = copy.deepcopy(self.data_w)
        nbIt = 0

        while not(convergence):

            for i in range(len(self.data_x)) :
                x = self.data_x[i]
                w = self.data_w[i]
                y = self.data_y[i]

                if np.abs(self.predict(x, w), y) > self.allowed_error :
                    old_data_w = self.data_w
                    self.data_w[i] = self.data_w[i] + self.learning_rate * y

            # stop condition loop
            if np.sqrt(np.sum((self.data_w - old_data_w)**2)) < self.learning_rate or nbIt > self.nbMaxIt :                                                \
                convergence = True
            
            nbIt += 1

        return self.data_w



    def score(self, x, w): 
        """ rend le score de prédiction sur x (valeur réelle) 
            x : une description
            w : une case du genome (weight)
        """
        return np.dot(x, w)
     

    def predict(self, x, w):
        """ rend la prediction sur x (soit -1 ou soit +1) 
            x : une description
            w : une case du genome (weight)
        """
        if self.score(x, w) < 0: 
            return -1
        return 1



#---------------------------------------------------------------------------------------------------------------
