
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
nbRobots = 20                               # get this value in the trial1.properties file

nbSteps = 2000
cptSteps = 0                                # global, used to know the passed number of steps

currentAgent = None                         # global, used to know wich agent hits one object
tabSumFood = [0] * nbRobots                 # global, used to store the fitness function

mutationRate = 0                            # global, used by HIT-EE algorithm
transferRate = 0.5  # 0.9                   # global, used by HIT-EE algorithm
maturationDelay = 100                       # global, used by HIT-EE algorithm

verbose = True                              # set true if you want to see execution details on terminal




################################################################################################################
# OBJECT DEFINITION
################################################################################################################


class Food_Object(CircleObject):

    def __init__(self, id, data):           # put "data" even if it is not used
        CircleObject.__init__(self, id)
        self.str = "[Food_Object " + str(id) + "] : "
        # self.rob = Pyroborobo.get()       # get pyroborobo singleton ?

    def reset(self):
        pass

    def step(self):
        global cptSteps
        cptSteps += 1

        if verbose : 
            print(self.str + "I'm object n." + str(self.id) + ", cptSteps = " + str(cptSteps) )

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
# ROBOTS BEHAVIOURS
################################################################################################################


class RobotsController(Controller):
    
    def __init__(self, world_model):
        Controller.__init__(self, world_model)

        self.age = 0

        self.genome = self.genome = [self.randNotZero() for _ in range(self.nb_sensors)]
        self.sensors = [0] * self.nb_sensors
        self.halfSizeGenome = int(np.ceil(len(self.genome)/2))
        self.halfSizeSensors = int(np.ceil(len(self.sensors)/2))

        self.message = []
        

    def reset(self):
        pass


    def step(self):

        global currentAgent                 # sets value of currentAgent, mandatory declaration of global var
        currentAgent = self.id              # used to tell which robot hits the object
        
        if verbose :
            print("Robot n." + str(self.id) + " au passage actuellement")  
            if cptSteps % nbRobots == 0 :
                print("tabSumFood : ", tabSumFood)  # fitness values
       
        # Set sensors vector
        for i in range(self.nb_sensors):
            self.sensors[i] = self.get_distance_at(i)
        print("sensors", self.sensors)

        # Robots' behaviours exchange (communication)
        self.hit_ee(mutationRate, transferRate, maturationDelay)

        # Expert behaviour : le robot n.0 plays the role of the expert
        if self.id == 0 :
            t, r = self.expertBehaviour()
            self.set_translation(t)
            self.set_rotation(r)
            return

        # Swarm behaviour
        t, r = self.swarmBehaviour()
        self.set_translation(t)
        self.set_rotation(r)


    def randNotZero():
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
        # Neural network 1 : linear combination of sensory imputs and genome weights
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
        if self.age >= maturationDelay:     # The robot is ready to learn or teach
            
            # Teaching knowledge to 
            self.broadcast(self.genome, transferRate, tabSumFood[self.id])

            # Learning knowledge from received packets
            for p in self.incomingPackets:
                if p.fitness >= self.fitness:
                    newGenome = self.transferGenome(p)




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
