
#                                               ROBOROBO TUTORIAL 3
#                                       see the prova3.properties file too


################################################################################################################
# IMPORTS
################################################################################################################

from pyroborobo import Pyroborobo
from pyroborobo import Controller
from pyroborobo import CircleObject, MovableObject



################################################################################################################
# PARAMETERS
################################################################################################################

fileConfig = "config/prova3.properties" 
nbSteps = 200



################################################################################################################
# OVERRIDE C++ OBJECTS (reset and step methods are mandatory to implement)
################################################################################################################


class Kale_A_Object(CircleObject):

    def __init__(self, id, data):           # put "data" even if it is not used
        CircleObject.__init__(self, id)
        self.str = "[Kale_A_Object] : "
        self.cptSteps = 0
        # self.rob = Pyroborobo.get()       # get pyroborobo singleton ?

    def reset(self):
        pass

    def step(self):
        self.cptSteps += 1
        print(self.str + "I'm object n." + str(self.id) + ", cptSteps = " + str(self.cptSteps) )

    def is_pushed(self, id, speed):
        print(self.str + "is_pushed")

    def is_touched(self, id):
        print(self.str + "is_touched")

    def is_walked(self, id):
        print(self.str + "is_walked")

    def inspect(self, prefix=""):
        return "\n" + self.str + f"[INSPECT] I'm the object #{self.id}\n\n"


#---------------------------------------------------------------------------------------------------------------

# Rappel : set "gMovableObjects = true" in your configuration file to use MovableObjects

class Kale_C_Object(MovableObject):

    def __init__(self, id, data):           # put "data" even if it is not used
        MovableObject.__init__(self, id)
        self.str = "[Kale_C_Object] : "
        self.cptSteps = 0
        # self.rob = Pyroborobo.get()       # get pyroborobo singleton ?

    def reset(self):
        pass

    def step(self):
        self.cptSteps += 1
        print(self.str + "I'm object n." + str(self.id) + ", cptSteps = " + str(self.cptSteps) )

    def is_pushed(self, id, speed):
        print(self.str + "is_pushed")

    def is_touched(self, id):
        print(self.str + "is_touched")

    def is_walked(self, id):
        print(self.str + "is_walked")

    def inspect(self, prefix=""):
        return "\n" + self.str + f"[INSPECT] I'm the object #{self.id}\n\n"





################################################################################################################
# OVERRIDE C++ CONTROLLER
################################################################################################################

class PythonController(Controller):
    
    def __init__(self, world_model):
        Controller.__init__(self, world_model)

    def reset(self):
        pass

    def step(self):

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



################################################################################################################
# MAIN
################################################################################################################

rob = Pyroborobo.create(fileConfig,                                     \
                        controller_class = PythonController,            \
                        object_class_dict = {'xxx': Kale_A_Object,      \
                                            'yyy': Kale_C_Object})      

rob.start()                                 # activation of pyroborobo simulator 
rob.update(nbSteps)                         # activation of nbUpdates steps
rob.close()                                 # free memory
