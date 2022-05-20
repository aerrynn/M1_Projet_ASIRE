
#                                    HIT-EE algorithm (PART 2 VERSION DIFFUSION)


################################################################################################################
# IMPORTS
################################################################################################################

import numpy as np
from allParts_tools import addBehavior


################################################################################################################
# PARAMETERS
################################################################################################################

# Following parameters are initialized by hit_ee, do not set them

transferRate = None
maturationDelay = None
distanceEpsilon = None
maxSizeDictMyBehaviors = None
learningOnlyFromExperts = None
slidingWindowTime = None

debug = None
isFirstIteration = True



################################################################################################################
# HIT-EE : Horizontal Information Transfer for Embodied Evolution, VERSION DIFFUSION
################################################################################################################


class hit_ee_versionDiffusion():
    
    def __init__(self, selfRobot, nbExpertsRobots, tR, mD, distEpsilon, maxSizeBehaviors, lOnlyFromExperts, setDebug=False):
        
        # Initialization
        global transferRate, maturationDelay, distanceEpsilon, maxSizeDictMyBehaviors, learningOnlyFromExperts, isFirstIteration, debug

        if isFirstIteration :
            transferRate = tR
            maturationDelay = mD
            distanceEpsilon = distEpsilon
            maxSizeDictMyBehaviors = maxSizeBehaviors
            learningOnlyFromExperts = lOnlyFromExperts
            debug = setDebug
            isFirstIteration = False

        self.selfC = selfRobot
        self.nbExpertsRobots = nbExpertsRobots
        self.newMessage = False


    #----------------------------------------------

    def hit_ee(self):
        """ 
        HIT-EE : Horizontal Information Transfer for Embodied Evolution, VERSION DIFFUSION
        """
        self.newMessage = False

        for i in range (self.selfC.nb_sensors):
            robotDestId = self.selfC.get_robot_id_at(i)
            if robotDestId == -1:
                continue
            else:
                if self.selfC.age >= maturationDelay:     # This robot is ready to learn or teach
                    
                    if self.selfC.id < self.nbExpertsRobots:
                        # Teaching knowledge to every robot in the neighborhood
                        self.broadcast(robotDestId)

                    else:
                        if not learningOnlyFromExperts:
                            # Teaching knowledge to every robot in the neighborhood
                            self.broadcast(robotDestId)

                        # Learning knowledge from received packets
                        for m in self.selfC.messages:
                            self.transferGenome(m, distanceEpsilon)
                            self.selfC.age = 0 
                    
                self.selfC.age += 1


    #----------------------------------------------

    def broadcast(self, robotDestId):
        """
        Method to send the parameters of the current robot to robots in the neighborhood
        """

        nbElemToSend = int(len(self.selfC.dictMyBehaviors) * transferRate)

        tabChoice = []
        for key in self.selfC.dictMyBehaviors:
            tabChoice.append(list(key))

        toSendChoice = np.random.choice([i for i in range(len(tabChoice))], nbElemToSend, False)
        sensoryInputsToSend = [tuple(tabChoice[h]) for h in toSendChoice]
        outputsToSend = [self.selfC.dictMyBehaviors[input] for input in sensoryInputsToSend]
        
        self.selfC.rob.controllers[robotDestId].messages += [(self.selfC.id, sensoryInputsToSend, outputsToSend, self.selfC.fitness)]

        if debug :
            print(f"\n[Robot n.{self.selfC.id}] :\tHIT_EE DIFFUSION VERSION DETAILS : ***BROADCAST***")
            print(f"\t\tThis robot is sending a message to robot n.{robotDestId}.")
            print(f"\t\tNumber of behaviors sent : {nbElemToSend}")
            print(f"\t\tBehaviors rows are : {toSendChoice}")
            for row in range(len(sensoryInputsToSend)):
                print("(" + ' '.join(str(elem) for elem in sensoryInputsToSend[row]) + ") :", outputsToSend[row])
            print("\n---------------------------------------------------------------")



    #----------------------------------------------

    def transferGenome(self, message, distanceEpsilon):   # RS : Robot Source du message
        """
        Method to manage the reception of messages sent by robots in the neighborhood

        : param message: 4-uplet (robotSourceId, inputsRS, outputsRS, fitnessRS)
        : param distanceEpsilon : required maximal distance between behaviors to be considered similars
        """
        
        robotSourceId, inputsRS, outputsRS, fitnessRS = message

        if debug :
            print(f"\n[Robot n.{self.selfC.id}] :\tHIT_EE DIFFUSION VERSION DETAILS : ***TRANSFER***")
            print(f"\t\tThis robot is receiving a message from robot n.{robotSourceId}...")


        if fitnessRS >= self.selfC.fitness:
            # this robot accepts and adds the list of behaviors sent by the RS in its behaviors collection
            for i in range(len(inputsRS)):
                addBehavior(self.selfC, inputsRS[i], outputsRS[i], maxSizeDictMyBehaviors, distanceEpsilon)

            # this robot has modified its behaviors database
            self.newMessage = True

            if debug :
                print(f"\t\tMessage has been accepted because fitness from robot source ({fitnessRS}) >= my fitness ({self.selfC.fitness})")
                print(f"\t\tNumber of behaviors received : {len(inputsRS)}")
                for row in range(len(inputsRS)):
                    print("(" + ' '.join(str(elem) for elem in inputsRS[row]) + ") :", outputsRS[row])
                    print("\n---------------------------------------------------------------")
            
        else:
            if debug :
                print(f"\t\tMessage has been rejected because fitness from robot source ({fitnessRS}) < my fitness ({self.selfC.fitness})")


    #----------------------------------------------

    def newMessageReceived(self):
        """
        Returns 'true' if this robot has received a new message at the last hit_ee algorithm call
        """
        return self.newMessage


    #----------------------------------------------