
#                                                HIT-EE algorithms


################################################################################################################
# IMPORTS
################################################################################################################

import numpy as np
from tools import addBehavior


################################################################################################################
# PARAMETERS
################################################################################################################

# Following parameters are initialized by hit_ee, do not set them

# v1
selfC = None

mutationRate = None
transferRate = None
maturationDelay = None
maxSizeDictMyBehaviors = None

isFirstIteration = True
verbose = None

# v3, to set the perceptron
learningRate = None
allowedError = None
nbMaxIt = None

#NN
nb_epoch = 20
learningRate = 0.5


################################################################################################################
# HIT-EE : Horizontal Information Transfer for Embodied Evolution, VERSION 2 DIFFUSION
################################################################################################################

class hit_ee_versionDiffusion():
    
    def __init__(self, selfRobot, tR, mD, maxSizeBehaviors, epsilon, setVerbose = False):
        
        # Initialization
        global fitness, verbose, selfC, transferRate, maturationDelay, maxSizeDictMyBehaviors, isFirstIteration

        if isFirstIteration :
            transferRate = tR
            maturationDelay = mD
            maxSizeDictMyBehaviors = maxSizeBehaviors
            verbose = setVerbose
            isFirstIteration = False

        selfC = selfRobot

        
        for i in range (selfC.nb_sensors):
            robotDestId = selfC.get_robot_id_at(i)
            if robotDestId == -1:
                continue
            else:
                # hit_ee algorithm
                if selfC.age >= maturationDelay:     # The robot is ready to learn or teach
                    
                    # Teaching knowledge to every robot in the neighborhood
                    if selfC.id == 0 or selfC.id == 1:
                        self.broadcast(robotDestId)

                    # Learning knowledge from received packets
                    if selfC.id != 0 and selfC.id != 1:
                        for m in selfC.messages:
                            if m[3] >= selfC.fitness:     # m[3] = 4eme part of the message = fitnessRS
                                self.transferGenome(m, epsilon)
                                selfC.age = 0    
                selfC.age += 1


    def broadcast(self, robotDestId):

        nbElemToSend = int(len(selfC.dictMyBehaviors) * transferRate)
        print("nbElemToSend", nbElemToSend)

        #--------------------------
        tabChoice = []
        for key in selfC.dictMyBehaviors:
            tabChoice.append(list(key))
        #print(">>>>>>>>>>>>>>>tabChoice", tabChoice)
        #--------------------------

        h_choice = np.random.choice([i for i in range(len(tabChoice))], nbElemToSend, False)
        sensoryInputsToSend = [tuple(tabChoice[h]) for h in h_choice]
        #print("sensoryInputsToSend", sensoryInputsToSend)
        outputsToSend = [selfC.dictMyBehaviors[input] for input in sensoryInputsToSend]
        
        selfC.rob.controllers[robotDestId].messages += [(selfC.id, sensoryInputsToSend, outputsToSend, selfC.fitness)]

        if verbose :
            print("[SENT MSG] I'm the robot n." + str(selfC.id) + " and I've sent a msg to robot n." + str(robotDestId))  


    def transferGenome(self, message, epsilon):        # RS : Robot Source du message
        robotSourceId, inputsRS, outputsRS, fitnessRS = message
        if fitnessRS >= selfC.fitness:

            # this robot accepts and adds the list of behaviors sent by the RS in its behaviors collection
            selfC.dictMyBehaviors = {}
            for i in range(len(inputsRS)):
                addBehavior(selfC, inputsRS[i], outputsRS[i], maxSizeDictMyBehaviors, epsilon)

            dataset = list(selfC.dictMyBehaviors.keys())
            labels = [selfC.dictMyBehaviors[input] for input in dataset]
            
            if verbose:
                print("\nDATASET :", dataset)
                print("\nLABELS :", labels)
                print("Je effectue le train sur mon NN...")
            selfC.myNetwork.train(dataset, labels, nb_epoch, learningRate)

            if verbose :
                print("\n[RECEIVED MSG] I'm the robot n." + str(selfC.id) + " and I've received a good msg from robot n." + str(robotSourceId))  
                print("\tbecause fitness robot n.", robotSourceId, " (" , fitnessRS , ") is >= than my fitness, (", selfC.fitness, ")\n")
