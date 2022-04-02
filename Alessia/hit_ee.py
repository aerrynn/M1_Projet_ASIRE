
#                                                HIT-EE algorithms


################################################################################################################
# IMPORTS
################################################################################################################

import numpy as np
import perceptron_supervisedLearning as perceptronSL


################################################################################################################
# PARAMETERS
################################################################################################################

# Following parameters are initialized by hit_ee, do not set them

# v1
selfC = None

mutationRate = None
transferRate = None
maturationDelay = None

isFirstIteration = True
verbose = None

# v3, to set the perceptron
learningRate = None
allowedError = None
nbMaxIt = None



################################################################################################################
# HIT-EE : Horizontal Information Transfer for Embodied Evolution, VERSION 1
################################################################################################################


class hit_ee_v1():
    
    def __init__(self, selfRobot, mR, tR, mD, setVerbose = False):
        
        # Initialization
        global fitness, verbose, selfC, mutationRate, transferRate, maturationDelay, isFirstIteration

        if isFirstIteration :
            mutationRate = mR
            transferRate = tR
            maturationDelay = mD
            verbose = setVerbose
            isFirstIteration = False

        selfC = selfRobot

        
        # hit_ee algorithm
        newGenome = False
        if selfC.age >= maturationDelay:     # The robot is ready to learn or teach
            
            # Teaching knowledge to every robot in the neighborhood
            self.broadcast()

            # Learning knowledge from received packets
            for m in selfC.messages:
                if m[3] >= selfC.fitness:     # m[3] = 4eme part of the message = fitnessRS
                    newGenome = self.transferGenome(m)
                    newGenome = True

                if newGenome:
                    newGenome = False
                    selfC.age = 0
                    
        selfC.age += 1


    def broadcast(self):
        for i in range (selfC.nb_sensors):
            robotDestId = selfC.get_robot_id_at(i)
            if robotDestId == -1:
                continue
            nbElemToReplace = int(len(selfC.genome) * transferRate)
            elemToReplace = np.random.choice(range(0, len(selfC.genome)), nbElemToReplace, False)
            selfC.rob.controllers[robotDestId].messages += [(selfC.id, selfC.genome, elemToReplace, selfC.fitness)]

            if verbose :
                print("[SENT MSG] I'm the robot n." + str(selfC.id) + " and I've sent a msg to robot n." + str(robotDestId))  


    def transferGenome(self, message):        # RS : Robot Source du message
        robotSourceId, genomeRS, elemToReplaceRS, fitnessRS = message
        if fitnessRS >= selfC.fitness:
            oldGenome = selfC.genome
            for index in elemToReplaceRS:
                selfC.genome[index] = genomeRS[index] * (1-mutationRate)
        
            if verbose :
                print("\n[RECEIVED MSG] I'm the robot n." + str(selfC.id) + " and I've received a good msg from robot n." + str(robotSourceId))  
                print("\tI've changed my genome :\n\tfrom oldGenome =", oldGenome, ", \n\tto newGenome =", selfC.genome, ", \n\tlearned by genomeRS =", genomeRS)
                print("\tbecause fitness robot n.", robotSourceId," (" , fitnessRS , ") is >= than our fitness, (", selfC.fitness, ")\n")


#---------------------------------------------------------------------------------------------------------------

class hit_ee_v3():
    
    def __init__(self, selfRobot, mD, lR, aE, maxIt, setVerbose = False):
        
        # Initialization
        global selfC, maturationDelay, learningRate, allowedError, nbMaxIt, verbose, isFirstIteration

        if isFirstIteration :
            maturationDelay = mD

            learningRate = lR
            allowedError = aE
            nbMaxIt = maxIt

            verbose = setVerbose
            isFirstIteration = False

        selfC = selfRobot

        
        # hit_ee algorithm
        newGenome = False
        if selfC.age >= maturationDelay:     # The robot is ready to learn or teach
            
            # Teaching knowledge to every robot in the neighborhood
            self.broadcast()

            # Learning knowledge from received packets
            for m in selfC.messages:
                if m[3] >= selfC.fitness:     # m[3] = 4eme part of the message = fitnessRS
                    newGenome = self.transferGenome(m)
                    newGenome = True

                if newGenome:
                    newGenome = False
                    selfC.age = 0
                    
        selfC.age += 1


    def broadcast(self):
        for i in range (selfC.nb_sensors):
            robotDestId = selfC.get_robot_id_at(i)
            if robotDestId == -1:
                continue
            selfC.rob.controllers[robotDestId].messages += [(selfC.id, selfC.genome, selfC.tabExtSensorsFloat, selfC.fitness)]

            if verbose :
                print("[SENT MSG] I'm the robot n." + str(selfC.id) + " and I've sent a msg to robot n." + str(robotDestId))  


    def transferGenome(self, message):        # RS : Robot Source du message
        idRS, genomeRS, tabExtSensorsFloatRS, fitnessRS = message
        if fitnessRS >= selfC.fitness:
            oldGenome = selfC.genome

            p = perceptronSL.Perceptron(selfC.genome, selfC.tabExtSensorsFloat, genomeRS, tabExtSensorsFloatRS, learningRate, allowedError, nbMaxIt, verbose = False)
            data_w = p.train()

            if len(data_w) == len(oldGenome):
                selfC.genome = data_w
        
            if verbose :
                print("\n[RECEIVED MSG] I'm the robot n." + str(selfC.id) + " and I've received a good msg from robot n." + str(idRS))  
                print("\tI've changed my genome :\n\tfrom oldGenome =", oldGenome, ", \n\tto newGenome =", selfC.genome, ", \n\tlearned by genomeRS =", genomeRS)
                print("\tbecause fitness robot n.", idRS," (" , fitnessRS , ") is >= than our fitness, (", selfC.fitness, ")\n")