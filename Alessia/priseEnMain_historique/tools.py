
#                                              tools


################################################################################################################
# IMPORTS
################################################################################################################

import numpy as np
import random

from discretizeBehaviors import discretizeBehavior, inputLayerTo24Sensors




################################################################################################################
# PARAMETERS
################################################################################################################

debug = False





################################################################################################################
# INITIALIZATION BEHAVIORS
################################################################################################################


def buildDefaultListBehaviors(tailleSensors, defaultBehavior):
    """Returns the default behavior for the apprentices: 
    if you don't feel anything around you (sesnor = 1), then go straight (t = 1) and don't turn (r = 0)
    """
    dictBehaviors = {}
    dictBehaviors[tuple([1] * tailleSensors)] = defaultBehavior

    return dictBehaviors


#---------------------------------------------------------------------------------------------------------------

def buildExpertListBehaviors(minSensorValue, maxSensorValue, nbSignificativesValues, tailleSensors, significatifsArms, valuesPerArm, definedExpertBehavior, maxSizeDictMyBehaviors=None):

    assert maxSensorValue > minSensorValue
    if nbSignificativesValues == 1:
        unit = abs(maxSensorValue - minSensorValue)
    else:
        unit = abs(maxSensorValue - minSensorValue) / (nbSignificativesValues - 1)
    units = [round(unit * i, 1) for i in range(nbSignificativesValues)]

    dictBehaviors = discretizeBehavior(units, tailleSensors, significatifsArms, valuesPerArm, definedExpertBehavior, maxSizeDictMyBehaviors=None)

    return dictBehaviors


#---------------------------------------------------------------------------------------------------------------

def getExpertFixedBehavior(definedExpertBehavior):

    dictBehaviors = {}

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

        [1,1,1, 1,1,1, 0.5, 0.5, 0.5, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],

        [1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],      
    ]

    for f in fictitiousSensors:
        sensors = inputLayerTo24Sensors(f)
        output = list(definedExpertBehavior(sensors))
        dictBehaviors[tuple(f)] = output

    if debug:    
        for key in list(dictBehaviors.keys()):
            print(f"{key} : {dictBehaviors[key]}")

    return dictBehaviors




################################################################################################################
# BEHAVIORS' LIST MANAGEMENT
################################################################################################################


def addBehavior(self, sensoryInputs, newBehavior, maxSizeDictMyBehaviors, epsilon):
    if len(self.dictMyBehaviors) < maxSizeDictMyBehaviors:
        self.dictMyBehaviors[tuple(sensoryInputs)] = newBehavior
    else:
        # We replace the most similar old existing behavior with the newone
        _, nearestBehavior = findNearestBehavior(self, sensoryInputs, epsilon)
        del self.dictMyBehaviors[nearestBehavior]
        self.dictMyBehaviors[tuple(sensoryInputs)] = newBehavior


#---------------------------------------------------------------------------------------------------------------

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


#---------------------------------------------------------------------------------------------------------------

def findNearestBehavior(self, sensoryInputs, epsilon):
    minDistance = np.inf
    nearestBehaviors = []
    nearestDistances = []

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

    if len(nearestBehaviors) > 0:
        i = random.choice(range(len(nearestBehaviors)))
        return nearestDistances[i], nearestBehaviors[i]
    return None, None


#---------------------------------------------------------------------------------------------------------------

# def areBehaviorsDistants(behavior1, behavior2, epsilon):
#     distances = behaviorsDistances(behavior1, behavior2)
#     for d in distances:
#         if d > epsilon:
#             return distances, True
#     return distances, False


#---------------------------------------------------------------------------------------------------------------

def getOwnAction(self, sensoryInputs, epsilon = 1.0):
    tabDistances, nearestB = findNearestBehavior(self, sensoryInputs, epsilon)
    return tabDistances, nearestB, self.dictMyBehaviors[nearestB]