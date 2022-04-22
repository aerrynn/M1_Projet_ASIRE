
#                                           TRIAL 5 : FORAGING TASK


################################################################################################################
# IMPORTS
################################################################################################################

from pyroborobo import Pyroborobo
from pyroborobo import Controller
from pyroborobo import CircleObject

from hit_ee_versionDiffusion import hit_ee_versionDiffusion
from extendedSensors import get24ExtendedSensors, extractExtSensors_float
import robotsBehaviors
import analyses
import perceptron_supervisedLearning

import numpy as np



################################################################################################################
# PARAMETERS
################################################################################################################

fileConfig = "config/trial5.properties" 
nbRobots = 20                               # get this value in the trial1.properties file

nbSteps = 2000
cptStepsG = 0                               # global, used to know the passed number of steps

tabSumFood = [0] * nbRobots                 # global, used to store the fitness function

transferRate = 1  # 0.9                   # global, used by HIT-EE algorithm
maturationDelay = 100                       # global, used by HIT-EE algorithm

verbose = False                              # set true if you want to see execution details on terminal
plot = False                                # set true if you want to plot results
isFirstIteration = [True] * nbRobots


# Neural network swarm
nb_hiddenLayers = 1
nb_neuronsPerHidden = 1
nb_neuronsPerOutputs = 2

definedExpertBehavior = None
defaultBehavior = [-1, 0]                    # default behavior : t=1, r=0
maxSizeDictMyBehaviors = 100

learningRate = 1
epsilon = 0.2 # NB check : epsilon = unit value. small epsilon = more accuracy required to match


################################################################################################################
# OBJECTS DEFINITION
################################################################################################################


class Food_Object(CircleObject):

    def __init__(self, id, data):           # put "data" even if it is not used
        CircleObject.__init__(self, id)
        self.str = "[Food_Object " + str(id) + "] : "
        self.regrow_time = 200
        self.cur_regrow = 0
        self.triggered = False
        self.cptSteps = 0


    def reset(self):
        self.show()
        self.register()
        self.triggered = False
        self.cur_regrow = 0


    def step(self):
        self.cptSteps += 1
        global cptStepsG
        cptStepsG = self.cptSteps

        if self.triggered :
            self.cur_regrow -= 1
            if self.cur_regrow <= 0:
                self.show()
                self.register()
                self.triggered = False


    def is_walked(self, id):
        self.triggered = True
        self.cur_regrow = self.regrow_time
        self.hide()
        self.unregister()

        global tabSumFood
        tabSumFood[id] += 1       # foraging task, fitness global array

        if verbose :
            print("[GNAM GNAM] Object n.", self.id, "is_walked by robot n.", id)


    def inspect(self, prefix=""):
        return "\n" + self.str + f"[INSPECT] I'm the object #{self.id}\n\n"





################################################################################################################
# ROBOTS DEFINITION
################################################################################################################


class RobotsController(Controller):
    
    def __init__(self, world_model):
        Controller.__init__(self, world_model)
        self.rob = Pyroborobo.get()

        # 24 sensory inputs for one single robot
        self.sensors = get24ExtendedSensors(self)
        self.tabExtSensorsFloat = extractExtSensors_float(self.sensors)
        self.nbExtendedSensors = len(self.tabExtSensorsFloat)

        # swarm performance estimation
        self.fitness = tabSumFood[self.id]
        
        # broadcast and transfer parameters
        self.age = 0
        self.messages = []

        # project part 2 : 'diffusion'. Each robots holds a limited memory of behaviors, 
        self.dictMyBehaviors = {}

        # Neural Network parameters
        self.myNetwork = None



    def reset(self):
        pass


    def step(self):
        
        # Set sensors vectors
        self.sensors = get24ExtendedSensors(self)
        self.tabExtSensorsFloat = extractExtSensors_float(self.sensors)


        # Robots' behaviors initialisation
        global isFirstIteration
        if isFirstIteration[self.id] :   # parameters initialization

            if self.id == 0 or self.id == 1:
                posInit = (400 * self.id + 20, 400 * self.id + 20)
                rob.controllers[self.id].set_position(posInit[0], posInit[1])
                rob.controllers[self.id].set_absolute_orientation(90 * self.id)
                self.dictMyBehaviors = robotsBehaviors.buildExpertListBehaviors(0, 1, 3, self.nbExtendedSensors, maxSizeDictMyBehaviors, robotsBehaviors.avoidRobotsWalls_getObjects)
            else:
                self.dictMyBehaviors = robotsBehaviors.buildDefaultListBehaviors(self.nbExtendedSensors, defaultBehavior)
                
                self.myNetwork = perceptron_supervisedLearning.neuralNetwork(   \
                    self.nbExtendedSensors,                                     \
                    nb_hiddenLayers,                                            \
                    nb_neuronsPerHidden,                                        \
                    nb_neuronsPerOutputs)
            
            isFirstIteration[self.id] = False


        # if verbose :
        #     print("\nRobot n." + str(self.id) + " au passage actuellement")
        #     print("\nsensors :", self.sensors)
        #     print("\n---------------------------------------------------------------")

        #     if cptStepsG % nbRobots == 0 :
        #         print("\ntabSumFood :", tabSumFood)  # fitness values
        #         print("\n---------------------------------------------------------------")


        if plot:
            if cptStepsG % nbRobots == 0 or cptStepsG == nbSteps:
                analyses.plotAverageFitness(tabSumFood, cptStepsG, nbSteps, funcObj = "maximisation")


        # Robots' behaviors exchange (communication)
        hit_ee_versionDiffusion(self, transferRate, maturationDelay, maxSizeDictMyBehaviors, epsilon, verbose)


        # Expert behavior : le robot n.0 et n.1 play the role of the experts
        if self.id == 0 or self.id == 1:
            t, r = self.expertBehavior(epsilon)
            self.set_translation(t)
            self.set_rotation(r)
            return

        # Swarm behaviour
        t, r = self.swarmBehavior()
        self.set_translation(t)
        self.set_rotation(r)


    def expertBehavior(self, epsilon): 
        sensoryInputs = self.tabExtSensorsFloat
        
        distanceBehaviors, s, action = robotsBehaviors.getOwnAction(self, sensoryInputs, epsilon)

        # If no available action corresponds to the current sensors situation, apply the default behavior
        if action == None :
            #print("Aucune action trouvée... on crée une nouvelle par default")
            #action = robotsBehaviors.addBehavior(self, sensoryInputs, defaultBehavior, maxSizeDictMyBehaviors) # default action
            #action = robotsBehaviors.getOwnAction(self, sensoryInputs, epsilon) # default action
            action = defaultBehavior

        if verbose and self.id == 0:
            print("\n[Robot " + str(self.id) + "] Expert choice : t=", action[0] , ", r=" , action[1], "car :", sensoryInputs, "et", s)
            print("Distances between behaviors:", distanceBehaviors[0])
            print("ScoreDistances:", distanceBehaviors[1])
        return action[0], action[1]


    def swarmBehavior(self):
        inputLayer = self.tabExtSensorsFloat
        action = self.myNetwork.predict(inputLayer)

        if verbose :
            print("[Robot " + str(self.id) + "] Apprentice choice : t=", action[0] , ", r=" , action[1])
        
        return action[0], action[1]


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
