
#                                                robotsBehaviors


################################################################################################################
# IMPORTS
################################################################################################################

import numpy as np


################################################################################################################
# PARAMETERS
################################################################################################################




################################################################################################################


################################################################################################################


def avoidEverything(self):
    t = 1
    r = 0

    if self.get_distance_at(1) < 1 or self.get_distance_at(2) < 1 :
        r = 0.5
    elif self.get_distance_at(3) < 1 :
        r = -0.5

    return testControllersValidRange(t, r)


#--------------------------------------------------------------------------------------------------------------

def compute_neuralNetwork(self):
    """
    Linear combination of n "distance" sensory inputs and genome vector (weights)

    """
    t = 0
    r = 0

    assert len(self.genome) == len(self.tabSensors)

    halfSizeGenome = int(np.ceil(len(self.genome)/2))

    t = 1 + np.dot(self.genome[0:halfSizeGenome], self.tabSensors[0:halfSizeGenome])
    r = np.random.choice([-1, 1]) * 0.5 + np.dot(self.genome[halfSizeGenome:len(self.genome)], self.tabSensors[halfSizeGenome:len(self.tabSensors)])
    
    return testControllersValidRange(t, r)
    

#--------------------------------------------------------------------------------------------------------------

def testControllersValidRange(t, r):
    # Limits the values of transition and rotation from -1 to +1
    t = max(-1, min(t, 1))
    r = max(-1, min(r, 1))
    return t, r


