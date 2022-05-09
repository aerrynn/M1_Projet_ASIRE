
#                                              tools


################################################################################################################
# IMPORTS
################################################################################################################

import numpy as np
import random
from math import sqrt

from part2_discretizeBehaviors import discretizeBehavior, inputLayerTo24Sensors



################################################################################################################
# PARAMETERS
################################################################################################################

isFirstIteration = True



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
        f = np.float_(f)
        sensors = inputLayerTo24Sensors(f)
        output = list(definedExpertBehavior(sensors))
        dictBehaviors[tuple(f)] = output

    return dictBehaviors




################################################################################################################
# BEHAVIORS' LIST MANAGEMENT
################################################################################################################


def addBehavior(self, newSensoryInputs, newAction, maxSizeDictMyBehaviors, distanceEpsilon):
    if tuple(newSensoryInputs) not in self.dictMyBehaviors:
        if len(self.dictMyBehaviors) < maxSizeDictMyBehaviors:
            self.dictMyBehaviors[tuple(newSensoryInputs)] = newAction
        else:
            # we look for the most similar older existing behavior in the database
            nearestSensors, _ = findNearestBehavior(self, newSensoryInputs, distanceEpsilon)
            # we choose to keep the older or the newer behavior
            usefulBehavior = chooseUsefulBehavior(self.dictMyBehaviors, newSensoryInputs, nearestSensors)
            
            if usefulBehavior != None and usefulBehavior != nearestSensors:
                del self.dictMyBehaviors[nearestSensors]
                self.dictMyBehaviors[usefulBehavior] = newAction


#---------------------------------------------------------------------------------------------------------------

def chooseUsefulBehavior(dictMyBehaviors, newSensoryInputs, nearestSensors, debug=False):
    clusters = buildClusters(dictMyBehaviors)

    for c in range(len(clusters)):
        if nearestSensors in clusters[c]:
            selectedCluster = c
            break

    if debug:
        l = clusters[selectedCluster]
        print(l[0], "\n")
        for c in range(len(l)):
            if c != len(l)-1:
                print(l)
            else:
                print("\n", l)



    frontierHead = None
    frontierEnd = None
    if selectedCluster != 0:
        frontierHead = clusters[selectedCluster-1][-1]
    if selectedCluster != len(clusters)-1:
        frontierEnd = clusters[selectedCluster+1][0]

    # print("\nfrontierHead", frontierHead)
    # print("\nselected cluster", clusters[selectedCluster])
    # print("\nfrontierEnd", frontierEnd)


    # replace the current first frontier by our newSensoryInputs?
    if nearestSensors == clusters[selectedCluster][0]:
        if frontierHead != None:
            d1 = behaviorsEuclideanDistance(newSensoryInputs, frontierHead)
            d2 = behaviorsEuclideanDistance(nearestSensors, frontierHead)
            if d1 < d2:
                return newSensoryInputs
            return nearestSensors

    # replace the current last frontier by our newSensoryInputs?
    elif nearestSensors == clusters[selectedCluster][-1]:
        if frontierEnd != None:
            d1 = behaviorsEuclideanDistance(newSensoryInputs, frontierEnd)
            d2 = behaviorsEuclideanDistance(nearestSensors, frontierEnd)
            if d1 < d2:
                return newSensoryInputs
            return nearestSensors

    # objectif : better distribution of behaviors. Our newSensoryInputs can improve it?
    else:
        for row in range(len(clusters[selectedCluster])):
            if nearestSensors == clusters[selectedCluster][row]:
                
                d_left_i = behaviorsEuclideanDistance(clusters[selectedCluster][row-1], newSensoryInputs)
                d_right_i = behaviorsEuclideanDistance(newSensoryInputs, clusters[selectedCluster][row+1])

                d_left_n = behaviorsEuclideanDistance(clusters[selectedCluster][row-1], nearestSensors)
                d_right_n = behaviorsEuclideanDistance(nearestSensors, clusters[selectedCluster][row+1])
            
                d1 = max(d_left_i, d_right_i)
                d2 = max(d_left_n, d_right_n)
                
                if d1 < d2:
                    return newSensoryInputs
                return nearestSensors


#---------------------------------------------------------------------------------------------------------------

def buildClusters(dictMyBehaviors, debug=False):
    sensors = list(dictMyBehaviors.keys())
    sensors.sort()

    clusters = {}
    cpt = -1
    previousAction = None
    for s in sensors:
        a = dictMyBehaviors[s]
        if a == previousAction:
            clusters[cpt].append(s)
        else:
            cpt += 1
            clusters[cpt] = [s]
        previousAction = a
    
    if debug:
        print("\nCLUSTERS DETAILS :")
        for c in clusters:
            print("\nCluster", c)
            for row in clusters[c]:
                print(row, ":", dictMyBehaviors[row])

    return clusters


#---------------------------------------------------------------------------------------------------------------

def behaviorsDistance(behavior1, behavior2):
    """Returns the list of distances between les éléménts in behavior1 (expert) and behavior2
    """
    d = []
    for i in range(len(behavior1)):
        if behavior1 == 1: # not setted behavior particule, because no obstacles (default)
            d.append(0)
        else:
            d.append(abs(behavior1[i] - behavior2[i]))
    return d


#---------------------------------------------------------------------------------------------------------------

def behaviorsEuclideanDistance(behavior1, behavior2):
    """Returns the euclidean distance between les éléménts in behavior1 (expert) and behavior2
    """
    d = 0
    for i in range(len(behavior1)):
        d += (behavior1[i] - behavior2[i])**2
    return sqrt(d)

#---------------------------------------------------------------------------------------------------------------

def findNearestBehavior(self, sensoryInputs, distanceEpsilon):
    minDistance = np.inf
    nearestSensors = []
    listBehaviors = list(self.dictMyBehaviors.keys())

    for b in range(len(listBehaviors)):

        #---------------------------------------------------------------------------
        # variant distance 1
        # distances = behaviorsDistance(listBehaviors[b], sensoryInputs) # tableau des distances, senseur par senseur
        # cptDistance = 0
        # for i in range(len(distances)):
        #     if distances[i] >= distanceEpsilon : # we don't consider little differences
        #         cptDistance += distances[i]
        #---------------------------------------------------------------------------
        # variant distance 2 (euclidean distance)
        cptDistance = behaviorsEuclideanDistance(listBehaviors[b], sensoryInputs)
        #---------------------------------------------------------------------------

        if cptDistance < minDistance:
            minDistance = cptDistance
            nearestSensors = [listBehaviors[b]]
        if cptDistance == minDistance:
            nearestSensors.append(listBehaviors[b])

    # we choose one element from the list of sensors with same distance minimale
    if len(nearestSensors) > 0:
        i = random.choice(range(len(nearestSensors)))
        return nearestSensors[i], minDistance
    return None, None


#---------------------------------------------------------------------------------------------------------------

def getOwnAction(self, sensoryInputs, distanceEpsilon=1.0):
    nearestSensors, distanceBehaviors = findNearestBehavior(self, sensoryInputs, distanceEpsilon)
    action = self.dictMyBehaviors[nearestSensors]
    return nearestSensors, action, distanceBehaviors

     
#---------------------------------------------------------------------------------------------------------------

def buildFileConfig(nbRobots, nbFoodObjects):

    s = ""
    s +=   \
    "###############################################################################################\n"\
    "# MAIN SIMULATION PARAMETERS\n"\
    "###############################################################################################\n"\
    "\n"\
    f"gInitialNumberOfRobots = {nbRobots}             # number of robots\n"\
    "gNbOfLandmarks = 0\n"\
    f"gNbOfPhysicalObjects = {nbFoodObjects}              # number of objects (see also PHYSICAL OBJECTS section)\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "gSensorRange = 24\n"\
    "gSynchronization = true                 # not implemented\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "gDisplayMode = 0\n"\
    "gBatchMode = false\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "gRandomSeed = -1                        # random seed\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "gVerbose = false                        # show execution details\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "gMaxIt = 80400 #-1\n"\
    "gEvaluationTime =  400 \n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "\n"\
    "###############################################################################################\n"\
    "#\n"\
    "###############################################################################################\n"\
    "ConfigurationLoaderObjectName = DummyConfigurationLoader\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "gRobotMaskImageFilename = data/minirobot-mask.bmp\n"\
    "gRobotSpecsImageFilename = data/minirobot-specs-8sensors.bmp\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "# environment: 400, 600, 1000, 1400x800, 4000\n"\
    "gForegroundImageFilename = data/env_600_foreground.bmp\n"\
    "gEnvironmentImageFilename = data/env_600_environment.bmp\n"\
    "gBackgroundImageFilename = data/env_600_background.bmp\n"\
    "gFootprintImageFilename = data/env_600_footprints.bmp\n"\
    "gScreenWidth = 600\n"\
    "gScreenHeight = 600\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "# default environment\n"\
    "#gForegroundImageFilename =	data/default_foreground.bmp\n"\
    "#gEnvironmentImageFilename = data/default_environment.bmp\n"\
    "#gBackgroundImageFilename = data/default_background.bmp\n"\
    "#gFootprintImageFilename = data/default_footprints.bmp\n"\
    "#gScreenWidth = 1350\n"\
    "#gScreenHeight = 800\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "# gLogFilename = logs/log.txt # if commented, create a time-stamped file.\n"\
    "# gLogCommentText = (under-development)\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "gSnapshots = false # take snapshots\n"\
    "gSnapshotsFrequency = 10 # every N generations\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "\n"\
    "###############################################################################################\n"\
    "#\n"\
    "###############################################################################################\n"\
    "\n"\
    "gPauseMode = false\n"\
    "gDisplaySensors = 1   # 0: no, 1: only-contact, 2: all + contacts are red, 3: all (same color)\n"\
    "gDisplayTail = true\n"\
    "gRobotDisplayFocus = false\n"\
    "gDisplayGroundCaption = false\n"\
    "gNiceRendering = true\n"\
    "SlowMotionMode = false\n"\
    "gUserCommandMode = false\n"\
    "gRobotLEDdisplay = true\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "gFastDisplayModeSpeed = 60\n"\
    "gFramesPerSecond = 60\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "gLocationFinderMaxNbOfTrials = 1000 # 100?\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "gRobotIndexFocus = 0\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "gfootprintImage_restoreOriginal = false\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "gNumberOfRobotGroups = 1 # unused\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "gPhysicalObjectIndexStartOffset = 1\n"\
    "gRobotIndexStartOffset = 1048576  # 0x100000\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "\n"\
    "###############################################################################################\n"\
    "# MONITORING\n"\
    "###############################################################################################\n"\
    "\n"\
    "gVideoRecording = false                 # significantly slow down simulation\n"\
    "gTrajectoryMonitor = false              # significantly slow down simulation\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "gInspectorMode = false\n"\
    "gInspectorAgent = false\n"\
    "gMonitorRobot = false\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "\n"\
    "###############################################################################################\n"\
    "# INSPECTOR CURSOR ('god mode')\n"\
    "###############################################################################################\n"\
    "\n"\
    "gInspectorCursorHorizontalSpeed = 1\n"\
    "gInspectorCursorVerticalSpeed = 1\n"\
    "gInspectorAgentXStart = 1\n"\
    "gInspectorAgentYStart = 1\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "\n"\
    "###############################################################################################\n"\
    "# ROBOT DYNAMICS AND STRUCTURE\n"\
    "###############################################################################################\n"\
    "\n"\
    "gMaxTranslationalSpeed = 2              # value btw 0+ and robot width in pixels\n"\
    "gMaxTranslationalDeltaValue = 2         # value btw 0+ and gMaxRotationalSpeed\n"\
    "gMaxRotationalSpeed = 30\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "gInspectorCursorMaxSpeedOnXaxis = 5\n"\
    "gInspectorCursorMaxSpeedOnYaxis = 10\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "gLocomotionMode = 0\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "\n"\
    "###############################################################################################\n"\
    "# SIMULATION PARAMETERS\n"\
    "###############################################################################################\n"\
    "\n"\
    "gMonitorPositions = false               # slow down if true.\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "\n"\
    "###############################################################################################\n"\
    "# LANDMARKS\n"\
    "###############################################################################################\n"\
    "\n"\
    "VisibleLandmarks = true\n"\
    "gLandmarkRadius = 10.0\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "\n"\
    "###############################################################################################\n"\
    "# PHYSICAL OBJECTS\n"\
    "###############################################################################################\n"\
    "\n"\
    "# List of the objects that we will use in the simulation\n"

    for i in range(nbFoodObjects):
        s += f"physicalObject[{i}].pytype = xxx          # Tell that we want to use an object with the id 'xxx'\n"

    s += \
    "#----------------------------------------------------------------------------------------------\n"\
    "gPhysicalObjectsVisible = true\n"\
    "gPhysicalObjectsRedraw = false\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "gMovableObjects = true                  # Set 'true' if you override a MovableObject\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "gPhysicalObjectDefaultType = 1\n"\
    "gPhysicalObjectDefaultRelocate = false\n"\
    "gPhysicalObjectDefaultOverwrite = false\n"\
    "gPhysicalObjectDefaultRadius = 6\n"\
    "gPhysicalObjectDefaultFootprintRadius = 10\n"\
    "gPhysicalObjectDefaultDisplayColorRed = 192\n"\
    "gPhysicalObjectDefaultDisplayColorGreen = 255\n"\
    "gPhysicalObjectDefaultDisplayColorBlue = 128\n"\
    "gPhysicalObjectDefaultSolid_w = 16\n"\
    "gPhysicalObjectDefaultSolid_h = 16\n"\
    "gPhysicalObjectDefaultSoft_w = 22\n"\
    "gPhysicalObjectDefaultSoft_h = 22\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "gPhysicalObjectDefaultRegrowTimeMax = 100\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "gEnergyItemDefaultMode = 0\n"\
    "gEnergyItemDefaultInit = 100\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "\n"\
    "###############################################################################################\n"\
    "# \n"\
    "###############################################################################################\n"\
    "\n"\
    "# landmarks. Check gNbOfLandmarks for max value.\n"\
    "landmark[0].x = 200\n"\
    "landmark[0].y = 100\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "# Robots. Check gInitialNumberOfRobots for max value.\n"\
    "#robot[0].x = 100\n"\
    "#robot[0].y = 100\n"\
    "#robot[0].orientation = 90			# 0...359, clockwise -- default is 0.\n"\
    "#robot[0].groupId=0						# default is 0 anyway\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "#robot[256].x = 50\n"\
    "#robot[256].y = 50\n"\
    "#robot[256].orientation = 90			# 0...359, clockwise -- default is 0.\n"\
    "#robot[256].groupId=0						# default is 0 anyway\n"\
    "#----------------------------------------------------------------------------------------------\n"\
    "\n"\
    "###############################################################################################\n"\
    "# LOGS\n"\
    "###############################################################################################\n"\
    "\n"\
    "# gLogFilename = logs\n"\
    "# gLogDirectoryname = ./logs\n"\
    "# gLogCommentText = \n"\
    "\n"\
    "# gPhysicalObjectsInitAreaX = 10\n"\
    "# gPhysicalObjectsInitAreaY = 10\n"\
    "# gPhysicalObjectsInitAreaWidth = gAreaWidth-10\n"\
    "# gPhysicalObjectsInitAreaHeight = gAreaHeight-10\n"\
    "# gAgentsInitAreaX = 0\n"\
    "# gAgentsInitAreaY = 0\n"\
    "# gAgentsInitAreaWidth = gAreaWidth-10\n"\
    "# gAgentsInitAreaHeight = AreaHeight-10\n"\
    "# gFootprintImage_restoreOriginal = true\n"\
    "# gSensoryInputs_distanceToContact = true\n"\
    "# gSensoryInputs_physicalObjectType = true\n"\
    "# gSensoryInputs_isOtherAgent = true\n"\
    "# gSensoryInputs_otherAgentGroup = true\n"\
    "# gSensoryInputs_otherAgentOrientation = true\n"\
    "# gSensoryInputs_isWall = true\n"\
    "# gSensoryInputs_groundSensors = true\n"\
    "# gSensoryInputs_landmarkTrackerMode = 0\n"\
    "# gSensoryInputs_distanceToLandmark = true\n"\
    "# gSensoryInputs_orientationToLandmark = true\n"\
    "# gSensoryInputs_energyLevel = true\n"\
    "# gEnergyLevel = false\n"\
    "# gEnergyRefill = false\n"\
    "# gEnergyMax = 100\n"\
    "# gEnergyInit = 100\n"\
    "# gReentrantMapping_motorOutputs = false\n"\
    "# gReentrantMapping_virtualOutputs = false\n"\
    "# gVirtualOutputs = 0\n"\
    "# gScreenDisplayHeight = gScreenDisplayHeight\n"\
    "# gScreenDisplayWidth = gScreenDisplayWidth\n"\
    "# gTailLength = 16\n"\
    "# gLocationFinderExitOnFail = true\n"\
    "# gOutputImageFormat = PNG\n"\
    "# gCustomSnapshot_niceRendering = '1'\n"\
    "# gCustomSnapshot_showLandmarks = '1'\n"\
    "# gCustomSnapshot_showObjects = '1'\n"\
    "# gCustomSnapshot_showRobots = '1'\n"\
    "# gCustomSnapshot_showSensorRays = '0'\n"\
    "# gRobotDisplayImageFilename = '' "


    with open("config/config.properties", mode='w', encoding='utf-8') as f:
        f.write(s)
