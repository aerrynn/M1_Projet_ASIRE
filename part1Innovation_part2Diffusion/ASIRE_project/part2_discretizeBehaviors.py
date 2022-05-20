
#                                            discretizeBehaviors


################################################################################################################
# IMPORTS
################################################################################################################

from itertools import product



################################################################################################################
# PARAMETERS
################################################################################################################

debug = False





################################################################################################################
# DISCRATIZATION METHODS FOR BEHAVIORS
################################################################################################################


def discretizeBehavior(units, tailleSensors, significatifsArms, valuesPerArm, definedExpertBehavior, maxSizeDictMyBehaviors=None):
    """
    Returns a dictionnary of all possible (tuple of sensors, action)
    """
    
    dictBehaviors = {}
    fictitiousSensors = []

    for p in product(units, repeat=valuesPerArm*len(significatifsArms)): # valuesPerArm = distance_to_robot / objects / walls
        
        fSLine = [1] * tailleSensors
        for i in range(len(significatifsArms)):
            indice = valuesPerArm*significatifsArms[i]
            fSLine[indice] = p[i]
            fSLine[indice+1] = p[i+1]
            fSLine[indice+2] = p[i+2]
        fictitiousSensors.append(fSLine)
    
    for f in fictitiousSensors: # it contains info about n arms only
        sensors = inputLayerTo24Sensors(f, arms=significatifsArms) # we write the concerned arms
        output = list(definedExpertBehavior(sensors))
        dictBehaviors[tuple(f)] = output

    if debug:    
        for key in list(dictBehaviors.keys()):
            print(f"{key} : {dictBehaviors[key]}")

    return dictBehaviors


#---------------------------------------------------------------------------------------------------------------

def inputLayerTo24Sensors(inputLayer, arms=None):
    """
    Converts a list containing sensors in a sensors dictionnary
    """

    sensors = {}
    armsLabels = ["sensor_left", "sensor_front_left", "sensor_front", "sensor_front_right", "sensor_right", "sensor_back_right", "sensor_back", "sensor_back_left"]
    distLabels = ["distance_to_robot", "distance_to_object", "distance_to_wall"]

    for i in range(len(armsLabels)):
        indice = i * len(distLabels)
        sensors[armsLabels[i]] = {                  \
        distLabels[0] : inputLayer[indice],         \
        distLabels[1] : inputLayer[indice+1],       \
        distLabels[2] : inputLayer[indice+2]}

    return sensors

