
#                                           TRIAL 2 : FORAGING TASK


################################################################################################################
# IMPORTS
################################################################################################################

from pyroborobo import Pyroborobo
from pyroborobo import Controller
from pyroborobo import CircleObject

from hit_ee import hit_ee_v1
from extendedSensors import get24ExtendedSensors
import robotsBehaviors

import numpy as np



################################################################################################################
# PARAMETERS
################################################################################################################

fileConfig = "config/trial1.properties" 
nbRobots = 30                               # get this value in the trial1.properties file

nbSteps = 2000
cptStepsG = 0                               # global, used to know the passed number of steps

currentAgent = None                         # global, used to know wich agent hits one object
tabSumFood = [0] * nbRobots                 # global, used to store the fitness function

mutationRate = 0                            # global, used by HIT-EE algorithm
transferRate = 0.5  # 0.9                   # global, used by HIT-EE algorithm
maturationDelay = 400                       # global, used by HIT-EE algorithm

verbose = False                              # set true if you want to see execution details on terminal
isFirstIteration = [True] * nbRobots


################################################################################################################
# OBJECTS DEFINITION
################################################################################################################


class Food_Object(CircleObject):

    def __init__(self, id, data):           # put "data" even if it is not used
        CircleObject.__init__(self, id)
        self.str = "[Food_Object " + str(id) + "] : "
        self.cptSteps = 0
        # self.rob = Pyroborobo.get()       # get pyroborobo singleton ?

    def reset(self):
        pass

    def step(self):
        self.cptSteps += 1

        global cptStepsG
        cptStepsG = self.cptSteps           # value used in RobotsController too

        if verbose : 
            print(self.str + "I'm object n." + str(self.id) + ", cptSteps = " + str(self.cptSteps) )


    def is_touched(self, id):
        self.hide()
        self.unregister()

        global tabSumFood
        tabSumFood[currentAgent] += 1       # foraging task, fitness array

        if verbose :
            print(self.str + "is_touched by robot n." + str(currentAgent) )


    def inspect(self, prefix=""):
        return "\n" + self.str + f"[INSPECT] I'm the object #{self.id}\n\n"





################################################################################################################
# ROBOTS DEFINITION
################################################################################################################


class RobotsController(Controller):
    
    def __init__(self, world_model):
        Controller.__init__(self, world_model)

        self.rob = Pyroborobo.get()
        self.age = 0

        self.genome = []
        self.expertGenome = []
        self.sensors, self.tabSensors = get24ExtendedSensors(self)
        self.nbExtendedSensors = len(self.tabSensors)

        self.messages = []
        

    def reset(self):
        pass


    def step(self):

        global currentAgent                 # sets value of currentAgent, mandatory declaration of global var
        currentAgent = self.id              # used to tell which robot hits the object
        
        # Set sensors vectors
        self.sensors, self.tabSensors = get24ExtendedSensors(self)


        global isFirstIteration
        if isFirstIteration[currentAgent] :   # parameters initialization    
            self.genome = [self.randNotZero()/2 for _ in range(self.nbExtendedSensors)]
            self.expertGenome = [0] * self.nbExtendedSensors
            isFirstIteration[currentAgent] = False
      

        if verbose :
            print("\nRobot n." + str(self.id) + " au passage actuellement")
            print("\tgenome =", self.genome)
            print("\tsensors :", self.sensors)
            if cptStepsG % nbRobots == 0 :
                print("\ttabSumFood :", tabSumFood)  # fitness values


        # Robots' behaviours exchange (communication)
        hit_ee_v1(self, tabSumFood, mutationRate, transferRate, maturationDelay, verbose)       # uses global mutationRate, transferRate, maturationDelay

        # Expert behaviour : le robot n.0 et n.1 play the role of the experts
        if self.id == 0 or self.id == 1:
            t, r = self.expertBehaviour()
            self.set_translation(t)
            self.set_rotation(r)
            return

        # Swarm behaviour
        t, r = self.swarmBehaviour()
        self.set_translation(t)
        self.set_rotation(r)


    def randNotZero(self):
        aleaFloat = 0
        while aleaFloat == 0:
            aleaFloat = np.random.random()
        return aleaFloat


    def expertBehaviour(self):      
        if verbose :
            print ("Hello I'm " + str(self.id) + " and I'm super cool 'cause I'm expert")    

        return robotsBehaviors.avoidRobotsWalls_getObjects(self)


    def swarmBehaviour(self):
        return robotsBehaviors.compute_neuralNetwork(self)


    def inspect(self, prefix=""):
        return f"\n[INSPECT] I'm the robot #{self.id}\n\n"




################################################################################################################
# MAIN
################################################################################################################

rob = Pyroborobo.create(fileConfig,                                     \
                        controller_class = RobotsController,            \
                        object_class_dict = {'xxx': Food_Object})      

rob.start()                                 # activation of pyroborobo simulator 
rob.update(nbSteps)                         # activation of nbUpdates steps
rob.close()                                 # free memory
