
#                                               ROBOROBO TUTORIAL 3
#                                       see the prova3.properties file too


################################################################################################################
# IMPORTS
################################################################################################################

from pyroborobo import Pyroborobo
from pyroborobo import Controller
from pyroborobo import CircleObject



################################################################################################################
# PARAMETERS
################################################################################################################

fileConfig = "config/trial1.properties" 
nbSteps = 200



################################################################################################################
# OVERRIDE C++ OBJECTS (reset and step methods are mandatory to implement)
################################################################################################################


# Rappel : set "gMovableObjects = true" in your configuration file to use MovableObjects

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
        print(self.str + "I'm object n." + str(self.id) + ", cptSteps = " + str(self.cptSteps) )

    def is_pushed(self, id, speed):
        print(self.str + "is_pushed by robot n." + str(currentAgent) ) # reads value of currentAgent, don't need declaration of global var

    def is_touched(self, id):
        print(self.str + "is_touched by robot n." + str(currentAgent) )

    def is_walked(self, id):
        print(self.str + "is_walked by robot n." + str(currentAgent) )

    def inspect(self, prefix=""):
        return "\n" + self.str + f"[INSPECT] I'm the object #{self.id}\n\n"





################################################################################################################
# ROBOTS BEHAVIOURS
################################################################################################################

currentAgent = None

class RobotsController(Controller):
    
    def __init__(self, world_model):
        Controller.__init__(self, world_model)

    def reset(self):
        pass

    def step(self):

        if self.id == 0 :
            self.expertbehaviour()

        # Simple default definition of translation and rotation
        self.set_translation(1)
        self.set_rotation(0)

        #-------------------------------------------------------------------------------------------------------

        # Robot manipulation, world_model méthods (PyWorldModel C++ class)
        
        if self.get_distance_at(1) < 1 or self.get_distance_at(2) < 1 :
            self.set_rotation(0.5)
        elif self.get_distance_at(3) < 1 :
            self.set_rotation(-0.5)


    def inspect(self, prefix=""):
        return f"\n[INSPECT] I'm the robot #{self.id}\n\n"


    def expertbehaviour(self):
        global currentAgent     # sets value of currentAgent, mandatory declaration of global var
        currentAgent = self.id
        print ("Hello I'm " + str(self.id) + " and I'm super cool")



################################################################################################################
# MAIN
################################################################################################################

rob = Pyroborobo.create(fileConfig,                                     \
                        controller_class = RobotsController,            \
                        object_class_dict = {'xxx': Food_Object})      

rob.start()                                 # activation of pyroborobo simulator 
rob.update(nbSteps)                         # activation of nbUpdates steps
rob.close()                                 # free memory
