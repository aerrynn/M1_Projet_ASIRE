
#                                  PART 2 : DIFFUSION, FORAGING TASK
#                                         K NEAREST NEIGHBORGS


################################################################################################################
# IMPORTS
################################################################################################################

from cgitb import reset
from pyroborobo import Pyroborobo
from pyroborobo import Controller
from pyroborobo import CircleObject

from part2_hit_ee_versionDiffusion import hit_ee_versionDiffusion
from allParts_extendedSensors import get24ExtendedSensors, extractExtSensors_float

import allParts_tools
import allParts_robotsBehaviors
import allParts_analyses
from part2_kNearestNeighbors import knn




################################################################################################################
# CONFIGURATION FILE PARAMETERS : 
# enable 'buildFileConfig' if you change one or more parameters in this section 
# from the last execution!
# If you don't want to set a parameter, set its value to 'None'
################################################################################################################

nbExpertsRobots = 2
nbNotExpertsRobots = 3
nbFoodObjects = 150

nbRobots = nbNotExpertsRobots + nbExpertsRobots
#buildFileConfig(nbRobots, nbFoodObjects)           # modify the configuration file
fileConfig = "config/config.properties"





################################################################################################################
# PARAMETERS
################################################################################################################

nbSteps = 1000
cptStepsG = 0                               # counter used to know the passed number of steps
tabSumFood = [0] * nbRobots                 # list used to store the robots' fitness function
isFirstIteration = [True] * nbRobots        # booleen used to initialize parameters once


# HIT-EE algorithm parameters
transferRate = 0.8                          # percentage of behaviors the expert sends in the broadcast phase
maturationDelay = 0                         # number of steps each robot waits before starting teaching or learning


# Storage behaviors mode parameters (used in HIT-EE algorithm)
maxSizeDictMyBehaviors = 100 #None          # maximal size allowed for storing behaviors. None=unlimited
slidingWindowTime = None                    # number of steps to reinitialize the behaviors'DB. None=unlimited time
resetBehaviorsDB = False
resetBehaviorsDBTime = 2000                 # behaviors DB will be reinitialized (conteining only the default behavior) at each resetBehaviorsDBTime


defaultBehavior = [1, 0.5]                  # translation and rotation

distanceEpsilon = 0.25                      # required maximal distance between behaviors to be considered similars
                                            # small epsilon = more accuracy required to match


# k nearest neighborgs
k = 5




################################################################################################################
# DEBUG and PLOT PARAMETERS
################################################################################################################

# set 'True' if you want to see execution details on terminal
selectedRobots = [2,3]              

debug_part2_diffusion = False
debug_objects = False
debug_extendedSensors = False
debug_hitDiffusion = False
debug_supervisedLearning = False         # only 'selectedRobots' details are shown
debug_knn = False
debug_knn_accuracy = False


# set 'True' if you want to plot results           
plot = False




################################################################################################################
# OBJECTS DEFINITION
################################################################################################################


class Food_Object(CircleObject):

    def __init__(self, id, data):           # put "data" even if it is not used
        CircleObject.__init__(self, id)
        self.regrow_time = 200
        self.cur_regrow = 0
        self.triggered = False
        self.cptSteps = 0


    #----------------------------------------------

    def reset(self):
        self.show()
        self.register()
        self.triggered = False
        self.cur_regrow = 0


    #----------------------------------------------

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


    #----------------------------------------------

    def is_walked(self, id):
        self.triggered = True
        self.cur_regrow = self.regrow_time
        self.hide()
        self.unregister()

        global tabSumFood
        tabSumFood[id] += 1       # foraging task, fitness global array

        if debug_objects :
            print(f"[GNAM GNAM] Object n.{self.id} is_walked by robot n.{id}")


    #----------------------------------------------

    def inspect(self, prefix=""):
        return f"\n[INSPECT OBJECT] I'm the object n.{self.id}\n\n"






################################################################################################################
# ROBOTS DEFINITION
################################################################################################################


class RobotsController(Controller):
    
    def __init__(self, world_model):
        Controller.__init__(self, world_model)
        self.rob = Pyroborobo.get()
        self.str = f"[Robot n.{self.id}] : "

        # 24 sensory inputs for one single robot
        self.sensors = get24ExtendedSensors(self)
        self.tabExtSensorsFloat = extractExtSensors_float(self.sensors)
        self.nbExtendedSensors = len(self.tabExtSensorsFloat)
        self.halfSize = int(self.nbExtendedSensors/2)

        # swarm performance estimation
        self.fitness = tabSumFood[self.id]
        
        # broadcast and transfer parameters
        self.myHitee = None
        self.age = 0
        self.messages = []

        # project part 2 : 'diffusion'. Each robots holds a limited memory of behaviors, 
        self.dictMyBehaviors = {}

        # KNN parameters
        self.myKnnClassifier = None

    #----------------------------------------------

    def reset(self):
        pass


    #----------------------------------------------

    def step(self):
        
        # Set sensors vectors
        self.sensors = get24ExtendedSensors(self)
        self.tabExtSensorsFloat = extractExtSensors_float(self.sensors)

        if debug_extendedSensors:
            print(f"\n{self.str} Current sensors are :")
            print("\tARM\t\t\tDIST_TO_ROBOT\tDIST_TO_OBJECT\tDIST_TO_WALL")
            for arm in self.sensors:
                if len(arm)>12:
                    print(f"\t{arm}\t", end='')
                else:
                    print(f"\t{arm}\t\t", end='')
                print(f"{round(self.sensors[arm]['distance_to_robot'],5)}\t\t{round(self.sensors[arm]['distance_to_object'],5)}\t\t{round(self.sensors[arm]['distance_to_wall'],5)}")
            print("\n---------------------------------------------------------------")



        # Robots' behaviors initialisation
        global isFirstIteration
        if isFirstIteration[self.id] :   # parameters initialization

            self.myHitee = hit_ee_versionDiffusion(self, 
                            nbExpertsRobots, 
                            transferRate, 
                            maturationDelay,
                            distanceEpsilon, 
                            maxSizeDictMyBehaviors, 
                            debug_hitDiffusion)

            if self.id < nbExpertsRobots: # if this robot is an expert
                # posInit = (400 * self.id + 20, 400 * self.id + 20)
                # rob.controllers[self.id].set_position(posInit[0], posInit[1])
                # rob.controllers[self.id].set_absolute_orientation(90 * self.id)
                #self.dictMyBehaviors = buildExpertListBehaviors(0, 1, 4, self.nbExtendedSensors, [0], 3, robotsBehaviors.avoidRobotsWalls_getObjects, maxSizeDictMyBehaviors=maxSizeDictMyBehaviors)
                self.dictMyBehaviors = allParts_tools.getExpertFixedBehavior(allParts_robotsBehaviors.avoidRobotsWalls_getObjects)
            else:
                self.dictMyBehaviors = allParts_tools.buildDefaultListBehaviors(self.nbExtendedSensors, defaultBehavior)
                

            self.myKnnClassifier = knn(debugKNN=debug_knn, debugAcc=debug_knn_accuracy, strRId=self.str)

            isFirstIteration[self.id] = False


        if debug_part2_diffusion and cptStepsG % nbRobots == 0 and self.id == 0:
            print("\n***************************************************************")
            print("Fitness values for each robot (tabSumFood):", tabSumFood)  # fitness values
            print("***************************************************************\n")

        if plot:
            if cptStepsG % nbRobots == 0 or cptStepsG == nbSteps:
                allParts_analyses.plotAverageFitness(tabSumFood, cptStepsG, nbSteps, funcObj="maximisation")


        # Robots' behaviors exchange (communication)
        self.myHitee.hit_ee()


        # Expert behavior
        if self.id < nbExpertsRobots:
            t, r = self.expertBehavior(distanceEpsilon)
            self.set_translation(t)
            self.set_rotation(r)
            return

        # Swarm behaviour
        t, r = self.swarmBehavior()
        self.set_translation(t)
        self.set_rotation(r)


    #----------------------------------------------

    def expertBehavior(self, distanceEpsilon): 
        sensoryInputs = self.tabExtSensorsFloat
        
        nearestSensors, action, distanceBehaviors = allParts_tools.getOwnAction(self, sensoryInputs, distanceEpsilon)

        # If no available action corresponds to the current sensors situation, apply the default behavior
        if action == None :
            action = defaultBehavior

        if debug_part2_diffusion:
            print(f"\n{self.str} \tExpert choice : t={action[0]}, r={action[1]}")
            print("\n\t\tbecause current sensors are :")
            print(f"\t\t{sensoryInputs[:self.halfSize]}")
            print(f"\t\t{sensoryInputs[self.halfSize:]}")
            print("\n\t\tand nearest database's sensors are :")
            print(f"\t\t{tuple(nearestSensors)[:self.halfSize]}")
            print(f"\t\t{tuple(nearestSensors)[self.halfSize:]} : {self.dictMyBehaviors[tuple(nearestSensors)]}")
            print(f"\n\t\tBehaviors' distance between them is {distanceBehaviors}")
            print("\n---------------------------------------------------------------")

        return action[0], action[1]


    #----------------------------------------------

    def swarmBehavior(self):
        inputLayer = self.tabExtSensorsFloat
        action = self.myKnnClassifier.predict(inputLayer, self.dictMyBehaviors, k)

        if debug_part2_diffusion :
            print(f"\n{self.str} \tApprentice choice : t={action[0]}, r={action[1]}")
            print(f"\t\tobtained with k nearest neighbors prediction")
            print("\n---------------------------------------------------------------")
        
        return action[0], action[1]


    #----------------------------------------------

    def inspect(self, prefix=""):
        return f"\n[INSPECT ROBOT] I'm the robot n.{self.id}\n\n"




################################################################################################################
# MAIN
################################################################################################################

rob = Pyroborobo.create(fileConfig,                                     \
                        controller_class = RobotsController,            \
                        object_class_dict = {'xxx': Food_Object})      

rob.start()                                 # activation of pyroborobo simulator 
rob.update(nbSteps)                         # activation of nbUpdates steps
rob.close()                                 # free memory
