
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


def braitenberg_avoiderRobotsWalls(sensors):
    t = sensors["sensor_front"]["distance_to_wall"] + sensors["sensor_front"]["distance_to_robot"]
    r = ( 1 - sensors["sensor_front"]["distance_to_robot"] - sensors["sensor_front_left"]["distance_to_robot"] + sensors["sensor_front_right"]["distance_to_robot"] ) + ( sensors["sensor_front"]["distance_to_wall"] - 1 - sensors["sensor_front_left"]["distance_to_wall"] + sensors["sensor_front_right"]["distance_to_wall"] )

    return testControllersValidRange(t, r)
    
#--------------------------------------------------------------------------------------------------------------
    
def braitenberg_loveWalls(sensors):
    t = 1
    r = sensors["sensor_front_left"]["distance_to_wall"] - sensors["sensor_front_right"]["distance_to_wall"]

    return testControllersValidRange(t, r)

#--------------------------------------------------------------------------------------------------------------

def braitenberg_hateWalls(sensors):
    t = sensors["sensor_front"]["distance_to_wall"]
    r = 1 - sensors["sensor_front"]["distance_to_wall"] - sensors["sensor_front_left"]["distance_to_wall"] + sensors["sensor_front_right"]["distance_to_wall"]

    return testControllersValidRange(t, r)
    
#--------------------------------------------------------------------------------------------------------------

def braitenberg_loveBot(sensors):
    t = 1
    r = sensors["sensor_front_left"]["distance_to_robot"] - sensors["sensor_front_right"]["distance_to_robot"]

    return testControllersValidRange(t, r)
    
#--------------------------------------------------------------------------------------------------------------  
    
def braitenberg_hateBot(sensors):
    t = sensors["sensor_front"]["distance_to_robot"]
    r = 1 - sensors["sensor_front"]["distance_to_robot"] - sensors["sensor_front_left"]["distance_to_robot"] + sensors["sensor_front_right"]["distance_to_robot"]

    return testControllersValidRange(t, r)



################################################################################################################
# SUBSOMPTION BEHAVIORS
################################################################################################################

def avoid_all(sensors): # used in roborobo examples, exploration behavior
    t = 1
    r = 0

    if sensors["sensor_front_left"]["distance_to_robot"] < 1            \
        or sensors["sensor_front_left"]["distance_to_object"] < 1       \
        or sensors["sensor_front_left"]["distance_to_wall"] < 1         \
        or sensors["sensor_front"]["distance_to_robot"] < 1             \
        or sensors["sensor_front"]["distance_to_object"] < 1            \
        or sensors["sensor_front"]["distance_to_wall"] < 1:
        r = 0.5
    elif sensors["sensor_front_right"]["distance_to_robot"] < 1         \
        or sensors["sensor_front_right"]["distance_to_object"] < 1      \
        or sensors["sensor_front_right"]["distance_to_wall"] < 1:
        r = -0.5

    return testControllersValidRange(t, r)


#1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0): [1, 0]
#--------------------------------------------------------------------------------------------------------------  

def avoidRobotsWalls_getObjects(sensors):
    t = 1
    r = 0

    if sensors["sensor_left"]["distance_to_object"] < 1                \
        or sensors["sensor_front_left"]["distance_to_object"] < 1      \
        or sensors["sensor_front"]["distance_to_object"] < 1 :
        r = -0.5
    elif sensors["sensor_front_right"]["distance_to_object"] < 1       \
        or sensors["sensor_right"]["distance_to_object"] < 1 :
        r = 0.5
    else:
        return avoid_all(sensors)

    return testControllersValidRange(t, r)

#--------------------------------------------------------------------------------------------------------------  

def avoidRobotsWalls_getObjects_strongVersion(sensors):
    t = 0.5
    r = 0

    if sensors["sensor_left"]["distance_to_object"] < 1                \
        or sensors["sensor_front_left"]["distance_to_object"] < 1      \
        or sensors["sensor_front"]["distance_to_object"] < 1 :
        t = -1
        r = -1
    elif sensors["sensor_front_right"]["distance_to_object"] < 1       \
        or sensors["sensor_right"]["distance_to_object"] < 1 :
        t = -1
        r = 1
    else:
        return avoid_all(sensors)

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

    assert len(self.genome) == len(self.tabExtSensorsFloat)

    halfSizeGenome = int(np.ceil(len(self.genome)/2))

    t = 1 + np.dot(self.genome[0:halfSizeGenome], self.tabExtSensorsFloat[0:halfSizeGenome])
    r = np.random.choice([-1, 1]) * 0.5 + np.dot(self.genome[halfSizeGenome:len(self.genome)], self.tabExtSensorsFloat[halfSizeGenome:len(self.tabExtSensorsFloat)])
    
    return testControllersValidRange(t, r)
    



################################################################################################################
# TOOLS
################################################################################################################

from itertools import combinations_with_replacement

def testControllersValidRange(t, r):
    # Limits the values of transition and rotation from -1 to +1
    t = max(-1, min(t, 1))
    r = max(-1, min(r, 1))
    return t, r


def buildExpertListBehaviors(minSensorValue, maxSensorValue, nbSignificativesValues, tailleSensors, maxSizeDictMyBehaviors, definedExpertBehavior):

    assert maxSensorValue > minSensorValue
    if nbSignificativesValues == 1:
        unit = abs(maxSensorValue - minSensorValue)
    else:
        unit = abs(maxSensorValue - minSensorValue) / (nbSignificativesValues - 1)
    units = [unit * i for i in range(nbSignificativesValues)]

    dictBehaviors = {}

    fictitiousSensors = [i for i in combinations_with_replacement(units, tailleSensors)]
    
    # dictBehaviors[tuple([maxSensorValue] * tailleSensors)] = [1, 0] # MODIFIER AVEC DEFAULT BEHAVIOR first behavior (sensors range)
    # sensors = inputLayerTo24Sensors(tuple([minSensorValue] * tailleSensors))
    # output_t, output_r = definedExpertBehavior(sensors)
    # dictBehaviors[tuple([minSensorValue] * tailleSensors)] = [output_t, output_r] # last behavior (sensors range)

    # print("---------------------------------------------------------------")
    # print(dictBehaviors)

    # for newRow in range(maxSizeDictMyBehaviors-2):
    #     fictitiousSensors = []
    #     for i in range(tailleSensors): # taille = nb sensors dans une liste de inputs
    #         fS = np.random.choice(units)
    #         fictitiousSensors.append(fS)
    
    #     sensors = inputLayerTo24Sensors(fictitiousSensors)
    #     output_t, output_r = definedExpertBehavior(sensors)
    #     dictBehaviors[tuple(fictitiousSensors)] = [output_t, output_r]


    fictitiousSensors = [
        [1,1,0.5, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        [1,1,1, 1,1,0.5, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        [1,1,1, 1,1,1, 1,1,0.5, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        [1,1,1, 1,1,1, 1,1,1, 1,1,0.5, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        [1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,0.5, 1,1,1, 1,1,1, 1,1,1],
        [1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,0.5, 1,1,1, 1,1,1],
        [1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,0.5, 1,1,1],
        [1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,0.5],

        [1,1,0.5, 1,1,0.5, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        [1,1,0.5, 1,1,1, 1,1,0.5, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        [1,1,0.5, 1,1,1, 1,1,1, 1,1,0.5, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        [1,1,0.5, 1,1,1, 1,1,1, 1,1,1, 1,1,0.5, 1,1,1, 1,1,1, 1,1,1],
        [1,1,0.5, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,0.5, 1,1,1, 1,1,1],
        [1,1,0.5, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,0.5, 1,1,1],
        [1,1,0.5, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,0.5],

        [1,1,1, 1,1,0.5, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        [1,1,1, 1,1,0.5, 1,1,0.5, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        [1,1,1, 1,1,0.5, 1,1,1, 1,1,0.5, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        [1,1,1, 1,1,0.5, 1,1,1, 1,1,1, 1,1,0.5, 1,1,1, 1,1,1, 1,1,1],

        [1,1,1, 1,1,1, 1,1,0.5, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        [1,1,1, 1,1,1, 1,1,0.5, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        [1,1,1, 1,1,1, 1,1,0.5, 1,1,0.5, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        [1,1,1, 1,1,1, 1,1,0.5, 1,1,1, 1,1,0.5, 1,1,1, 1,1,1, 1,1,1],

        [1,1,1, 1,1,1, 1,1,1, 1,1,0.5, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        [1,1,1, 1,1,1, 1,1,1, 1,1,0.5, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        [1,1,1, 1,1,1, 1,1,1, 1,1,0.5, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        [1,1,1, 1,1,1, 1,1,1, 1,1,0.5, 1,1,0.5, 1,1,1, 1,1,1, 1,1,1],
        #------------------------
        [1,1,0.5, 1,1,0.5, 1,1,0.5, 1,1,0.5, 1,1,0.5, 1,1,1, 1,1,1, 1,1,1],
        [1,1,0.5, 1,1,1, 1,1,0.5, 1,1,0.5, 1,1,0.5, 1,1,1, 1,1,1, 1,1,1],
        [1,1,0.5, 1,1,0.5, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        [1,1,0.5, 1,1,0.5, 1,1,0.5, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        [1,1,1, 1,1,1, 1,1,0.5, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        [1,1,1, 1,1,1, 1,1,1, 1,1,0.5, 1,1,0.5, 1,1,1, 1,1,1, 1,1,1],

        [1,1,0, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,0],
        [1,1,0, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,0, 1,1,1],
        [1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,0, 1,1,0, 1,1,1, 1,1,1],
        [1,1,0, 1,1,0, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        [1,1,1, 1,1,0, 1,1,0, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        [1,1,1, 1,1,1, 1,1,0, 1,1,0, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        [1,1,1, 1,1,1, 1,1,1, 1,1,0, 1,1,0, 1,1,1, 1,1,1, 1,1,1],
        [1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,0, 1,1,0, 1,1,1, 1,1,1],
        [1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,0, 1,1,0, 1,1,1],
        [1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,0, 1,1,0],

        [1,1,0, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,0, 1,1,0],
        [1,1,0, 1,1,0, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,0, 1,1,0],
        [1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,0, 1,1,0, 1,1,0],
        [1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,0, 1,1,0, 1,1,0, 1,1,0],

        [1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
        
    ]



    # fictitiousSensors = [
    #     #----------------------------------------------------------------
    #     # getObjects
    #     #----------------------------------------------------------------
    #     #----- cond1
    #     [1,0.5,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
    #     #----- or cond2
    #     [1,1,1, 1,0.5,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
    #     [1,0.5,1, 1,0.5,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
    #     #----- or cond3
    #     [1,0.5,1, 1,1,1, 1,0.5,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
    #     [1,1,1, 1,0.5,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
    #     [1,0.5,1, 1,0.5,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],

    #     [1,0.5,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
    #     [1,1,1, 1,0.5,1, 1,0.5,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
    #     [1,0.5,1, 1,0.5,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],

    #     [1,0.5,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
    #     [1,1,1, 1,0.5,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
    #     [1,0.5,1, 1,0.5,1, 1,0.5,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
    #     #----------------------------------------------------------------
    #     [1,1,1, 1,1,1, 1,1,1, 1,0,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],  # getObjects
    #     [1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,0,1, 1,1,1, 1,1,1, 1,1,1],  # getObjects
    #     [1,1,1, 1,1,1, 1,1,1, 1,0,1, 1,0,1, 1,1,1, 1,1,1, 1,1,1],  # getObjects
    #     #----------------------------------------------------------------
    #     #----------------------------------------------------------------
    #     # avoid all
    #     #----------------------------------------------------------------
    #     [1,1,1, 0.5,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
    #     [1,1,1, 1,1,0.5, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
    #     [1,1,1, 0.5,1,0.5, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],

    #     [1,1,1, 0.5,1,1, 0.5,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
    #     [1,1,1, 1,1,0.5, 1,1,0.5, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
    #     [1,1,1, 0.5,1,0.5, 0.5,1,0.5, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
    #     #------
    #     [1,1,1, 1,1,1, 1,1,1, 0.5,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
    #     [1,1,1, 1,1,1, 1,1,1, 1,1,0.5, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
    #     [1,1,1, 1,1,1, 1,1,1, 0.5,1,0.5, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
    #     #----------------------------------------------------------------
    #     #----------------------------------------------------------------
    #     # default
    #     #----------------------------------------------------------------
    #     [1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1]
    # ]


    for f in fictitiousSensors:
        sensors = inputLayerTo24Sensors(f)
        output = list(definedExpertBehavior(sensors))
        dictBehaviors[tuple(f)] = output

    return dictBehaviors


def buildDefaultListBehaviors(nbExtendedSensors, default_behavior):
    """Returns the default behavior for the apprentices: 
    if you don't feel anything around you (sesnor = 1), then go straight (t = 1) and don't turn (r = 0)
    """
    dictBehaviors = {}
    dictBehaviors[tuple([1] * nbExtendedSensors)] = default_behavior
    return dictBehaviors


import random

def inputLayerTo24Sensors(inputLayer):
    sensors = {}
    armsLabels = ["sensor_left", "sensor_front_left", "sensor_front", "sensor_front_right", "sensor_right", "sensor_back_right", "sensor_back", "sensor_back_left"]
    distLabels = ["distance_to_robot", "distance_to_object", "distance_to_wall"]

    assert len(armsLabels) * len(distLabels) == len(inputLayer)

    for i in range(len(armsLabels)):
        indice = i*len(distLabels)
        sensors[armsLabels[i]] = {             \
        distLabels[0] : inputLayer[indice],    \
        distLabels[1] : inputLayer[indice+1],  \
        distLabels[2] : inputLayer[indice+2]}

    return sensors


def addBehavior(self, sensoryInputs, newBehavior, maxSizeDictMyBehaviors, epsilon):
    if len(self.dictMyBehaviors) < maxSizeDictMyBehaviors:
        self.dictMyBehaviors[tuple(sensoryInputs)] = newBehavior
    else:
        # We replace the most similar old existing behavior with the newone
        _, nearestBehavior = findNearestBehavior(self, sensoryInputs, epsilon)
        del self.dictMyBehaviors[nearestBehavior]
        self.dictMyBehaviors[tuple(sensoryInputs)] = newBehavior


def behaviorsDistances(behavior1, behavior2):
    """Returns the list of distances between les éléménts in behavior1 (expert) and behavior2
    """
    d = []
    for i in range(len(behavior1)):
        if behavior1 == 1: # not setted behavior particule, because no obstacles (default)
            d.append(0)
        else:
            #d.append((behavior1[i] - behavior2[i])**2)
            d.append(abs(behavior1[i] - behavior2[i]))
    return d


def findNearestBehavior(self, sensoryInputs, epsilon):
    minDistance = np.inf
    nearestBehaviors = []
    nearestDistances = []
    # for b in self.dictMyBehaviors.keys():
    #     d_tmp = sum(behaviorsDistances(b, sensoryInputs))
    #     if d_tmp < d:
    #         nearestBehavior = b
    listBehaviors = list(self.dictMyBehaviors.keys())
    for b in range(len(listBehaviors)):
        distances = behaviorsDistances(listBehaviors[b], sensoryInputs) # tableau
        cptDistance = 0
        for i in range(len(distances)):
            if distances[i] >= epsilon : # we don't consider little differences
                cptDistance += distances[i]

        if cptDistance < minDistance:
            minDistance = cptDistance
            nearestBehaviors = [listBehaviors[b]]
            nearestDistances = [distances]
        if cptDistance == minDistance:
            nearestBehaviors.append(listBehaviors[b])
            nearestDistances.append((distances, cptDistance))

    i = random.choice(range(len(nearestBehaviors)))
    return nearestDistances[i], nearestBehaviors[i]


# def areBehaviorsDistants(behavior1, behavior2, epsilon):
#     distances = behaviorsDistances(behavior1, behavior2)
#     for d in distances:
#         if d > epsilon:
#             return distances, True
#     return distances, False


def getOwnAction(self, sensoryInputs, epsilon = 1.0):
    tabDistances, nearestB = findNearestBehavior(self, sensoryInputs, epsilon)
    return tabDistances, nearestB, self.dictMyBehaviors[nearestB]

    
    #     f, verdict = areBehaviorsDistants(sensoryInputs, st, epsilon)

    #     if verdict == False: # we have found an applicable behavior
    #         nearestStates.append(st)

    # if len(nearestStates) > 0: 
    #     print("Actions possibles:", nearestStates)       
    #     state = random.choice(nearestStates)   
    #     print("Action choisie:", state) 
    #     return f, state, self.dictMyBehaviors[state]
    # return f, None, None




# # test
# d = {}
# #fictitiousSensors = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
# #fictitiousSensors = [1,1,0,1,1,0,1,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1] # walls avoider
# fictitiousSensors = [1,1,1, 1,1,1, 1,1,1, 1,0,1, 1,0,1, 1,1,1, 1,1,1, 1,1,1] # getObject
# #fictitiousSensors = [0] * 24
# sensors = inputLayerTo24Sensors(fictitiousSensors)
# output_t, output_r = avoidRobotsWalls_getObjects(sensors)
# d[tuple(fictitiousSensors)] = [output_t, output_r]
# print("comportement", d)
