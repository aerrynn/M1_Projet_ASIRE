
#                                  PART 1 : INNOVATION, FORAGING TASK


################################################################################################################
# IMPORTS
################################################################################################################


from pyroborobo import Pyroborobo
from pyroborobo import Controller
from pyroborobo import CircleObject

from part1_hit_ee_versionInnovation import hit_ee_versionInnovation
from allParts_extendedSensors import get24ExtendedSensors, extractExtSensors_float

import allParts_tools
import allParts_robotsBehaviors
import part2_supervisedLearning
import allParts_analysis





################################################################################################################
# CONFIGURATION FILE PARAMETERS : 
# enable 'buildFileConfig' if you change one or more parameters in this section 
# from the last execution!
################################################################################################################

nbExpertsRobots = 10
nbNotExpertsRobots = 90
nbFoodObjects = 100

nbRobots = nbNotExpertsRobots + nbExpertsRobots

allParts_tools.buildFileConfig(nbRobots, nbFoodObjects) # modify the configuration file
fileConfig = "config/config.properties"







################################################################################################################
# PARAMETERS
################################################################################################################

nbSteps = 20000
cptStepsG = 0                               # counter used to know the passed number of steps, starting at 0
tabSumFood = [0] * nbRobots                 # list used to store the robots' fitness function
isFirstIteration = [True] * nbRobots        # booleen used to initialize parameters once


# HIT-EE algorithm parameters
transferRate = 0.4                          # percentage of genes the expert sends in the broadcast phase
maturationDelay = 0                         # number of steps each robot waits before starting teaching or learning
learningOnlyFromExperts=True                # 'True'= only experts robots can broadcast. 'False'= all robots can broadcast 


swarmLearningMode = "innovation"

# Neural Network parameters, used only to set the expert's weights
nb_hiddenLayers = 1
nb_neuronsPerHidden = 16
nb_neuronsPerOutputs = 2

defaultBehavior = [1,0.5]                   # translation and rotation. NB: don't leave spaces between t and r

learningRate = 0.4                          # pourcentage that controls how much to modify the NN weights to correct the error.
                                            # learningRate = 0.1 ---> 10% of correction is applied

distanceEpsilon = 0.25                      # required maximal distance between behaviors to be considered similars
                                            # small epsilon = more accuracy required to match

nbEpoch = 1000                              # number of iterations for backpropagation training


# Not used in innovation part1
maxSizeDictMyBehaviors = False
k = False





################################################################################################################
# DEBUG and PLOT PARAMETERS
################################################################################################################

# set 'True' if you want to see execution details on terminal
#selectedRobots = [i for i in range(nbExpertsRobots, nbRobots)]                          
selectedRobots = [0,1]

debug_part1_innovation = False
debug_objects = False
debug_extendedSensors = False
debug_hitInnovation = False
debug_supervisedLearning = False            # only 'selectedRobots' details are shown


# set 'True' if you want to plot results           
plot = True
folderAnalysis = "allParts_results"

evaluationTime = 100                        # number of steps (period) inwhich evaluate performances. None=unlimited time
slidingWindowTime = 100                     # slidingWindowTime : when the curent robot trains the behaviors dataset
resetEvaluation = True
resetEvaluationTime = 2000                  # behaviors DB will be reinitialized (conteining only the default behavior) at each resetBehaviorsDBTime


performances = []
periods = []
expertSensorsPath = []
expertActions = []
strDetails = None
selectedComparisonMoments = [1000, 2500, nbSteps]   # to write the distance between expert et best not expert behaviors




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
        self.arenaHeight, self.arenaWidth = self.rob.arena_size
        self.posInit = self.absolute_position

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

        # Neural Network parameters
        self.myNetwork = None
        self.myGenomeList = None



    #----------------------------------------------

    def reset(self):
        pass


    #----------------------------------------------

    def step(self):
        

        # GETTING 24 SENSORS FROM THE ENVIRONNEMENT #####################################
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




        # INITIALISATION ROBOT'S PARAMETERS #############################################
        # Robots' behaviors initialisation
        global isFirstIteration
        if isFirstIteration[self.id] :   # parameters initialization

            self.myHitee = hit_ee_versionInnovation(self, 
                            nbExpertsRobots, 
                            transferRate, 
                            maturationDelay,
                            learningOnlyFromExperts, 
                            debug_hitInnovation)

            setDebug = False
            if self.id in selectedRobots and debug_supervisedLearning:
                setDebug = True

            self.myNetwork = part2_supervisedLearning.neuralNetwork(
                            self.nbExtendedSensors,
                            nb_hiddenLayers,
                            nb_neuronsPerHidden,
                            nb_neuronsPerOutputs,
                            debugNN=setDebug,
                            strRId=self.str)     


            if self.id < nbExpertsRobots: # if this robot is an expert
                self.posInit = (int(self.arenaHeight/2) + (self.id*25), int(self.arenaWidth/2) + (self.id*25))
            
                dictMyBehaviors = allParts_tools.getExpertFixedBehavior(allParts_robotsBehaviors.avoidRobotsWalls_getObjects)
                dataset = list(dictMyBehaviors.keys())
                labels = [dictMyBehaviors[input] for input in dataset]
                self.myNetwork.train(dataset, labels, nbEpoch, learningRate)
            else:
                self.posInit = self.absolute_position


            self.myGenomeList = self.myNetwork.getWeightsList()


            if self.id in selectedRobots and debug_supervisedLearning:
                self.myNetwork.printNetworkInformation()

            isFirstIteration[self.id] = False



        # INITIAL CONDITIONS RESET #######################################################
        # Reset of the evaluation settings for each robot
        if resetEvaluation and (cptStepsG == 1 or cptStepsG % resetEvaluationTime == 0):
            rob.controllers[self.id].set_position(self.posInit[0], self.posInit[1])
            tabSumFood[self.id] = 0
            self.age = 0

        if debug_part1_innovation and cptStepsG % nbRobots == 0 and self.id == 0:
            print("\n***************************************************************")
            print("Fitness values for each robot (tabSumFood):", tabSumFood)  # fitness values
            print("***************************************************************\n")




        # PLOT SECTION ##################################################################
        # Writing statistical data about the current evaluation
        if plot and self.id == 0: # we observe parameters when robot n.0 is passing by

            if cptStepsG in selectedComparisonMoments:
                # looking for the best not expert individual in the swarm
                swarmFitnesses = tabSumFood[nbExpertsRobots:]
                bestNotExpertRobotId = swarmFitnesses.index(max(swarmFitnesses)) + nbExpertsRobots
                bestNotExpertRobot = self.rob.controllers[bestNotExpertRobotId]

                allParts_analysis.writeExpertVsNotExpertDistances(bestNotExpertRobot,
                                                                swarmLearningMode,
                                                                cptStepsG,
                                                                k,
                                                                expertSensorsPath,
                                                                expertActions)


            if cptStepsG == 1 or cptStepsG % evaluationTime == 0 :               
                global strDetails, performances, periods
                
                if cptStepsG == 1: # first Step
                    periods.append(0)

                    strDetails = allParts_analysis.writeStrDetails(nbRobots,
                            nbExpertsRobots,
                            nbNotExpertsRobots,
                            nbFoodObjects,
                            maxSizeDictMyBehaviors,
                            transferRate,
                            maturationDelay,
                            learningOnlyFromExperts,
                            swarmLearningMode,
                            nb_hiddenLayers,
                            nb_neuronsPerHidden,
                            nb_neuronsPerOutputs,
                            defaultBehavior,
                            learningRate,
                            distanceEpsilon,
                            nbEpoch,
                            k,
                            nbSteps,
                            evaluationTime,
                            resetEvaluation,
                            resetEvaluationTime)
                else:
                    periods.append(cptStepsG)

                performances.append(list(tabSumFood))
                print("cptStepsG :", cptStepsG)  # monitoring passed time at terminal


            if cptStepsG == nbSteps:
                allParts_analysis.writePerformanceData(performances)




        # ROBOTS' COMMUNICATION #########################################################
        # Robots' behaviors exchange information management
        self.myHitee.hit_ee()




        # NEXT ACTION CHOICE ############################################################
        # Expert behavior
        if self.id < nbExpertsRobots:
            t, r = self.expertBehavior()
            self.set_translation(t)
            self.set_rotation(r)
            return

        # Swarm behaviour
        t, r = self.swarmBehavior()
        self.set_translation(t)
        self.set_rotation(r)


    #----------------------------------------------

    def expertBehavior(self): 
        inputLayer = self.tabExtSensorsFloat
        
        action = self.myNetwork.predict(inputLayer)

        if debug_part1_innovation:
            print(f"\n{self.str} \tExpert choice : t={action[0]}, r={action[1]}")
            print("\n\t\tbecause current sensors are :")
            print(f"\t\t{inputLayer[:self.halfSize]}")
            print(f"\t\t{inputLayer[self.halfSize:]}")
            print("\n---------------------------------------------------------------")

        if plot and self.id == 0:
            global expertSensorsPath, expertActions
            expertSensorsPath.append(inputLayer)
            expertActions.append(action)

        return action[0], action[1]


    #----------------------------------------------

    def swarmBehavior(self):
        
        # 'slidingWindowTime==None' means that this robot trains the behaviors dataset each time it meets an expert
        # 'cptStepsG%slidingWindowTime==0' means that this robot trains the behaviors dataset at each 'slidingWindowTime' time
        if (slidingWindowTime == None and self.myHitee.newMessageReceived())     \
            or (slidingWindowTime != None and slidingWindowTime != 0 and cptStepsG%slidingWindowTime==0):
      
            self.myNetwork.setWeightsFromList(self.myGenomeList)


        inputLayer = self.tabExtSensorsFloat
        action = self.myNetwork.predict(inputLayer)

        if debug_part1_innovation :
            print(f"\n{self.str}\tApprentice choice : t={action[0]}, r={action[1]}")
            print(f"\t\tobtained with supervised learning prediction 'forwardPropagation' method")
            print("\n\t\tbecause current sensors are :")
            print(f"\t\t{inputLayer[:self.halfSize]}")
            print(f"\t\t{inputLayer[self.halfSize:]}")
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
