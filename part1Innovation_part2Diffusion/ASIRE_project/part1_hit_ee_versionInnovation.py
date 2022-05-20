
#                                    HIT-EE algorithm (PART 1 VERSION INNOVATION)


################################################################################################################
# IMPORTS
################################################################################################################

import numpy as np



################################################################################################################
# PARAMETERS
################################################################################################################

# Following parameters are initialized by hit_ee, do not set them

transferRate = None
maturationDelay = None
learningOnlyFromExperts = None
slidingWindowTime = None

debug = None
isFirstIteration = True



################################################################################################################
# HIT-EE : Horizontal Information Transfer for Embodied Evolution, VERSION INNOVATION
################################################################################################################


class hit_ee_versionInnovation():
    
    def __init__(self, selfRobot, nbExpertsRobots, tR, mD, lOnlyFromExperts, setDebug=False):
        
        # Initialization
        global transferRate, maturationDelay, learningOnlyFromExperts, isFirstIteration, debug

        if isFirstIteration :
            transferRate = tR
            maturationDelay = mD
            learningOnlyFromExperts = lOnlyFromExperts
            debug = setDebug
            isFirstIteration = False

        self.selfC = selfRobot
        self.nbExpertsRobots = nbExpertsRobots
        self.newMessage = False


    #----------------------------------------------

    def hit_ee(self):
        """ 
        HIT-EE : Horizontal Information Transfer for Embodied Evolution, VERSION INNOVATION
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
                            self.transferGenome(m)
                            self.selfC.age = 0 
                    
                self.selfC.age += 1


    #----------------------------------------------

    def broadcast(self, robotDestId):
        """
        Method to send the parameters of the current robot to robots in the neighborhood
        """

        nbElemToSend = int(len(self.selfC.myGenomeList) * transferRate)
        toSendChoice = np.random.choice([i for i in range(len(self.selfC.myGenomeList))], nbElemToSend, False)
        weightsToSend = [self.selfC.myGenomeList[h] for h in toSendChoice]
              
        self.selfC.rob.controllers[robotDestId].messages += [(self.selfC.id, toSendChoice, weightsToSend, self.selfC.fitness)]

        if debug :
            print(f"\n[Robot n.{self.selfC.id}] :\tHIT_EE INNOVATION VERSION DETAILS : ***BROADCAST***")
            print(f"\t\tThis robot is sending a message to robot n.{robotDestId}.")
            print(f"\t\tNumber of genes sent : {nbElemToSend}")
            print(f"\t\tGenome sent is : {weightsToSend}")
            print("\n---------------------------------------------------------------")



    #----------------------------------------------

    def transferGenome(self, message):   # RS : Robot Source du message
        """
        Method to manage the reception of messages sent by robots in the neighborhood

        : param message: 4-uplet (robotSourceId, positionsRS, weightsRS, fitnessRS)
        """
        robotSourceId, positionsRS, weightsRS, fitnessRS = message


        if debug :
            print(f"\n[Robot n.{self.selfC.id}] :\tHIT_EE INNOVATION VERSION DETAILS : ***TRANSFER***")
            print(f"\t\tThis robot is receiving a message from robot n.{robotSourceId}...")


        if fitnessRS >= self.selfC.fitness:

            if debug :
                print(f"\t\tMessage has been accepted because fitness from robot source ({fitnessRS}) >= my fitness ({self.selfC.fitness})")
                print(f"\t\tNumber of genes received : {len(positionsRS)}")


            # this robot accepts and adds the list of weights sent by the RS in its genome
            for i in range(len(positionsRS)):
                oldGene = self.selfC.myGenomeList[positionsRS[i]]
                self.selfC.myGenomeList[positionsRS[i]] = weightsRS[i] 

                if debug :     
                    print(f"\t\tOld gene {oldGene} \t-----> New gene {self.selfC.myGenomeList[positionsRS[i]]} at position : {positionsRS[i]}")

            
            if debug :
                print("\n---------------------------------------------------------------")


            # this robot has modified its genome
            self.newMessage = True
            
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