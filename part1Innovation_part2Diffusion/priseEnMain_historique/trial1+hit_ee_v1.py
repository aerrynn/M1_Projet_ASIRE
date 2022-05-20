
#                                           TRIAL 1 : FORAGING TASK


################################################################################################################
# IMPORTS
################################################################################################################

from pyroborobo import Pyroborobo
from pyroborobo import Controller
from pyroborobo import CircleObject

import numpy as np



################################################################################################################
# PARAMETERS
################################################################################################################

fileConfig = "config/trial1.properties" 
nbRobots = 30                               # get this value in the trial1.properties file

nbSteps = 300
cptStepsG = 0                               # global, used to know the passed number of steps

tabSumFood = [0] * nbRobots                 # global, used to store the fitness function

mutationRate = 0                            # global, used by HIT-EE algorithm
transferRate = 0.5  # 0.9                   # global, used by HIT-EE algorithm
maturationDelay = 400                       # global, used by HIT-EE algorithm

verbose = False                             # set true if you want to see execution details on terminal




################################################################################################################
# OBJECTS DEFINITION
################################################################################################################


class Food_Object(CircleObject):

    def __init__(self, id, data):           # put "data" even if it is not used
        CircleObject.__init__(self, id)
        self.str = "[Food_Object " + str(id) + "] : "
        self.cptSteps = 0

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

        self.genome = [self.randNotZero()/2 for _ in range(self.nb_sensors)]
        self.expertGenome = [0] * self.nb_sensors                   # initialisation and definitive vector
        self.sensors = [0] * self.nb_sensors
        self.halfSizeGenome = int(np.ceil(len(self.genome)/2))
        self.halfSizeSensors = int(np.ceil(len(self.sensors)/2))

        self.messages = []
        

    def reset(self):
        pass


    def step(self):

        # Set sensors vector
        for i in range(self.nb_sensors):
            self.sensors[i] = self.get_distance_at(i)
            
        if verbose :
            print("\nRobot n." + str(self.id) + " au passage actuellement")
            print("\tgenome =", self.genome)
            print("\tSensors :", self.sensors)
            if cptStepsG % nbRobots == 0 :
                print("\ttabSumFood :", tabSumFood)  # fitness values


        # Robots' behaviours exchange (communication)
        self.hit_ee()       # uses global mutationRate, transferRate, maturationDelay

        # Expert behaviour : le robot n.0 plays the role of the expert
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
        t = 1
        r = 0

        if self.get_distance_at(1) < 1 or self.get_distance_at(2) < 1 :
            r = 0.5
        elif self.get_distance_at(3) < 1 :
            r = -0.5

        if verbose :
            print ("Hello I'm " + str(self.id) + " and I'm super cool 'cause I'm expert")    

        return self.testControllersValidRange(t, r)


    def swarmBehaviour(self):
        t = 0
        r = 0

        # Neural network 1 : linear combination of sensory imputs and genome vector (weights)
        if len(self.genome) == len(self.sensors):    
            t = 1 + np.dot(self.genome[0:self.halfSizeGenome], self.sensors[0:self.halfSizeSensors])
            r = np.random.choice([-1, 1]) * 0.5 + np.dot(self.genome[self.halfSizeGenome:len(self.genome)], self.sensors[self.halfSizeSensors:len(self.sensors)])
        
        return self.testControllersValidRange(t, r)


    def testControllersValidRange(self, t, r):
        # Limits the values of transition and rotation from -1 to +1
        t = max(-1, min(t, 1))
        r = max(-1, min(r, 1))
        return t, r


    def hit_ee(self):
        newGenome = False
        if self.age >= maturationDelay:     # The robot is ready to learn or teach
            
            # Teaching knowledge to every robot in the neighborhood
            self.broadcast(self.genome, transferRate, tabSumFood[self.id])

            # Learning knowledge from received packets
            for m in self.messages:
                if m[3] >= tabSumFood[self.id]:     # m[3] = fitnessRS
                    newGenome = self.transferGenome(m)
                    newGenome = True

                if newGenome:
                    newGenome = False
                    self.age = 0
                    #self.fitness = 0         # line code in the HIT-EE algorithm, not used in this trial
                    
        self.age += 1


    def broadcast(self, genome, transferRate, fitness):
        for i in range (self.nb_sensors):
            robotDestId = self.get_robot_id_at(i)
            if robotDestId == -1:
                continue
            nbElemToReplace = int(len(genome) * transferRate)
            elemToReplace = np.random.choice(range(0, len(self.genome)), nbElemToReplace, False)
            self.rob.controllers[robotDestId].messages += [(self.id, genome, elemToReplace, fitness)]

            if verbose :
                print("[SENT MSG] I'm the robot n." + str(self.id) + " and I've sent a msg to robot n." + str(robotDestId))  


    def transferGenome(self, message):
        robotSourceId, genomeRS, elemToReplaceRS, fitnessRS = message
        if fitnessRS >= tabSumFood[self.id]:
            oldGenome = self.genome
            for index in elemToReplaceRS:
                self.genome[index] = genomeRS[index] * (1-mutationRate)
        
            if verbose :
                print("\n[RECEIVED MSG] I'm the robot n." + str(self.id) + " and I've received a good msg from robot n." + str(robotSourceId))  
                print("\tI've changed my genome :\n\tfrom oldGenome =", oldGenome, ", \n\tto newGenome =", self.genome, ", \n\tlearned by genomeRS =", genomeRS)
                print("\tbecause fitness robot n.", robotSourceId," (" , fitnessRS , ") is > than our fitness, (", tabSumFood[self.id], ")\n")


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
