
#                                                HIT-EE algorithm VERSION 1


################################################################################################################
# IMPORTS
################################################################################################################

import numpy as np



################################################################################################################
# PARAMETERS
################################################################################################################

# These parameters are initialized by hit_ee, do not set them

selfC = None

tabFitness = None

mutationRate = None
transferRate = None
maturationDelay = None

isFirstIteration = True
verbose = None



################################################################################################################
# HIT-EE : Horizontal Information Transfer for Embodied Evolution, VERSION 1
################################################################################################################


def hit_ee_v1(self, tabF, mR, tR, mD, setVerbose = False):
    
    # Initialization
    global tabFitness, verbose, selfC, mutationRate, transferRate, maturationDelay, isFirstIteration

    if isFirstIteration :
        mutationRate = mR
        transferRate = tR
        maturationDelay = mD
        verbose = setVerbose
        isFirstIteration = False

    tabFitness = tabF
    selfC = self

    
    # hit_ee algorithm
    newGenome = False
    if selfC.age >= maturationDelay:     # The robot is ready to learn or teach
        
        # Teaching knowledge to every robot in the neighborhood
        broadcast(selfC.genome, transferRate, tabFitness[selfC.id])

        # Learning knowledge from received packets
        for m in selfC.messages:
            if m[3] >= tabFitness[selfC.id]:     # m[3] = 4eme part of the message = fitnessRS
                newGenome = transferGenome(m)
                newGenome = True

            if newGenome:
                newGenome = False
                selfC.age = 0
                
    selfC.age += 1


def broadcast(genome, transferRate, fitness):
    for i in range (selfC.nb_sensors):
        robotDestId = selfC.get_robot_id_at(i)
        if robotDestId == -1:
            continue
        nbElemToReplace = int(len(genome) * transferRate)
        elemToReplace = np.random.choice(range(0, len(selfC.genome)), nbElemToReplace, False)
        selfC.rob.controllers[robotDestId].messages += [(selfC.id, selfC.genome, elemToReplace, fitness)]

        if verbose :
            print("[SENT MSG] I'm the robot n." + str(selfC.id) + " and I've sent a msg to robot n." + str(robotDestId))  


def transferGenome(message):        # RS : Robot Source du message
    robotSourceId, genomeRS, elemToReplaceRS, fitnessRS = message
    if fitnessRS >= tabFitness[selfC.id]:
        oldGenome = selfC.genome
        for index in elemToReplaceRS:
            selfC.genome[index] = genomeRS[index] * (1-mutationRate)
    
        if verbose :
            print("\n[RECEIVED MSG] I'm the robot n." + str(selfC.id) + " and I've received a good msg from robot n." + str(robotSourceId))  
            print("\tI've changed my genome :\n\tfrom oldGenome =", oldGenome, ", \n\tto newGenome =", selfC.genome, ", \n\tlearned by genomeRS =", genomeRS)
            print("\tbecause fitness robot n.", robotSourceId," (" , fitnessRS , ") is >= than our fitness, (", tabFitness[selfC.id], ")\n")