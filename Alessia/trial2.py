
#                                           TRIAL 2 : FORAGING TASK


################################################################################################################
# IMPORTS
################################################################################################################

from pyroborobo import Pyroborobo
from pyroborobo import Controller
from pyroborobo import CircleObject

from hit_ee import hit_ee_v1, hit_ee_v3
from extendedSensors import extractExtSensors_float, get24ExtendedSensors
import robotsBehaviors

import numpy as np



################################################################################################################
# PARAMETERS
################################################################################################################

fileConfig = "config/trial1.properties" 
nbRobots = 30                               # get this value in the trial1.properties file

nbSteps = 500
cptStepsG = 0                               # global, used to know the passed number of steps

tabSumFood = [0] * nbRobots                 # global, used to store the fitness function

mutationRate = 0                            # global, used by HIT-EE algorithm
transferRate = 0.5  # 0.9                   # global, used by HIT-EE algorithm
maturationDelay = 400                       # global, used by HIT-EE algorithm

verbose = False                             # set true if you want to see execution details on terminal
isFirstIteration = [True] * nbRobots


################################################################################################################
# OBJECTS DEFINITION
################################################################################################################


class Food_Object(CircleObject):

    def __init__(self, id, data):           # put "data" even if it is not used
        CircleObject.__init__(self, id)
        self.cptSteps = 0


    def reset(self):
        pass


    def step(self):
        self.cptSteps += 1

        global cptStepsG
        cptStepsG = self.cptSteps           # value used in RobotsController too
        

    def is_touched(self, id):
        self.hide()
        self.unregister()

        global tabSumFood
        tabSumFood[id] += 1       # foraging task, fitness array

        if verbose :
            print("[GNAM GNAM] Object n.", self.id, "is_touched by robot n.", id)


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

        self.sensors = get24ExtendedSensors(self)
        self.tabExtSensorsFloat = extractExtSensors_float(self.sensors)
        self.nbExtendedSensors = len(self.tabExtSensorsFloat)

        self.fitness = tabSumFood[self.id]

        self.messages = []
        

    def reset(self):
        pass


    def step(self):

        # Set sensors vectors
        self.sensors = get24ExtendedSensors(self)
        self.tabExtSensorsFloat = extractExtSensors_float(self.sensors)


        global isFirstIteration
        if isFirstIteration[self.id] :   # parameters initialization    
            self.genome = [self.randNotZero()/10 for _ in range(self.nbExtendedSensors)]
            self.expertGenome = [0] * self.nbExtendedSensors
            isFirstIteration[self.id] = False


        if verbose :
            print("\nRobot n." + str(self.id) + " au passage actuellement")
            print("\ngenome =", self.genome)
            print("\nsensors :", self.sensors)
            print("\n---------------------------------------------------------------")

            if cptStepsG % nbRobots == 0 :
                print("\ntabSumFood :", tabSumFood)  # fitness values
                print("\n---------------------------------------------------------------")


        # Robots' behaviours exchange (communication)
        #hit_ee_v1(self, mutationRate, transferRate, maturationDelay, verbose)       # uses global mutationRate, transferRate, maturationDelay
        hit_ee_v3(self, maturationDelay, 0.5, 0.1, 10, verbose)


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
