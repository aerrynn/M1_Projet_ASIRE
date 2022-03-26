
#                                                robotsBehaviors


################################################################################################################
# IMPORTS
################################################################################################################

import numpy as np


################################################################################################################
# PARAMETERS
################################################################################################################




################################################################################################################
# BRAITENBERG BEHAVIORS
################################################################################################################


def braitenberg_avoiderRobotsWalls(self):
    t = self.sensors["sensor_front"]["distance_to_wall"] + self.sensors["sensor_front"]["distance_to_robot"]
    r = ( 1 - self.sensors["sensor_front"]["distance_to_robot"] - self.sensors["sensor_front_left"]["distance_to_robot"] + self.sensors["sensor_front_right"]["distance_to_robot"] ) + ( self.sensors["sensor_front"]["distance_to_wall"] - 1 - self.sensors["sensor_front_left"]["distance_to_wall"] + self.sensors["sensor_front_right"]["distance_to_wall"] )

    return testControllersValidRange(t, r)
    
#--------------------------------------------------------------------------------------------------------------
    
def braitenberg_loveWalls(self):
    t = 1
    r = self.sensors["sensor_front_left"]["distance_to_wall"] - self.sensors["sensor_front_right"]["distance_to_wall"]

    return testControllersValidRange(t, r)

#--------------------------------------------------------------------------------------------------------------

def braitenberg_hateWalls(self):
    t = self.sensors["sensor_front"]["distance_to_wall"]
    r = 1 - self.sensors["sensor_front"]["distance_to_wall"] - self.sensors["sensor_front_left"]["distance_to_wall"] + self.sensors["sensor_front_right"]["distance_to_wall"]

    return testControllersValidRange(t, r)
    
#--------------------------------------------------------------------------------------------------------------

def braitenberg_loveBot(self):
    t = 1
    r = self.sensors["sensor_front_left"]["distance_to_robot"] - self.sensors["sensor_front_right"]["distance_to_robot"]

    return testControllersValidRange(t, r)
    
#--------------------------------------------------------------------------------------------------------------  
    
def braitenberg_hateBot(self):
    t = self.sensors["sensor_front"]["distance_to_robot"]
    r = 1 - self.sensors["sensor_front"]["distance_to_robot"] - self.sensors["sensor_front_left"]["distance_to_robot"] + self.sensors["sensor_front_right"]["distance_to_robot"]

    return testControllersValidRange(t, r)



################################################################################################################
# SUBSOMPTION BEHAVIORS
################################################################################################################

def avoid_all(self): # used in roborobo examples, exploration behavior
    t = 1
    r = 0

    if self.get_distance_at(1) < 1 or self.get_distance_at(2) < 1 :
        r = 0.5
    elif self.get_distance_at(3) < 1 :
        r = -0.5

    return testControllersValidRange(t, r)

#--------------------------------------------------------------------------------------------------------------  

def avoidRobotsWalls_getObjects(self):
    t = 1
    r = 0

    if self.sensors["sensor_left"]["distance_to_object"] < 1                \
        or self.sensors["sensor_front_left"]["distance_to_object"] < 1      \
        or self.sensors["sensor_front"]["distance_to_object"] < 1 :
        r = -0.5
    elif self.sensors["sensor_front_right"]["distance_to_object"] < 1       \
        or self.sensors["sensor_right"]["distance_to_object"] < 1 :
        r = 0.5
    else:
        return avoid_all(self)

    return testControllersValidRange(t, r)

#--------------------------------------------------------------------------------------------------------------  

def avoidRobotsWalls_getObjects_strongVersion(self):
    t = 0.5
    r = 0

    if self.sensors["sensor_left"]["distance_to_object"] < 1                \
        or self.sensors["sensor_front_left"]["distance_to_object"] < 1      \
        or self.sensors["sensor_front"]["distance_to_object"] < 1 :
        t = -1
        r = -1
    elif self.sensors["sensor_front_right"]["distance_to_object"] < 1       \
        or self.sensors["sensor_right"]["distance_to_object"] < 1 :
        t = -1
        r = 1
    else:
        return avoid_all(self)

    return testControllersValidRange(t, r)




################################################################################################################
# NEURAL NETWORK BEHAVIORS
################################################################################################################


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
    



################################################################################################################
# TOOLS
################################################################################################################


def testControllersValidRange(t, r):
    # Limits the values of transition and rotation from -1 to +1
    t = max(-1, min(t, 1))
    r = max(-1, min(r, 1))
    return t, r


