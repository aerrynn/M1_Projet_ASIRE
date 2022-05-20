
#                                               ROBOROBO TUTORIAL 2


################################################################################################################
# IMPORTS
################################################################################################################

from pyroborobo import Pyroborobo
from pyroborobo import Controller
from pyroborobo import SquareObject, CircleObject, MovableObject       # in OVERRIDE C++ OBJECTS


################################################################################################################
# PARAMETERS
################################################################################################################

fileConfig = "config/prova2.properties" 
nbSteps = 200




###############################################################################################################
# C++ classes (Python constructors)
################################################################################################################

# Il existe plusieurs C++ classes : Controllers, WorldModels, WorldObserver, PhysicalObjects, AgentObservers

# Controller        : allows to code the agent behaviour.
#                     From the Controller, we can access the world and the robots world models.

# WorldModels       : objet where the world model is stored.

# Les observers are useful for monitoring, logging, computing fitness updates, ...
# Methods : reset and step. These classes are run before the actual agent behavioural update
#   - WorldObserver : used for accessing state of the world (with all agents)
#   - AgentObservers : used for accessing state of agent. 


# To override the C++ classes by Python classes, create a class that encodes the new behaviour and 
# provide it to the Pyroborobo.create method. The new class will inherit from the corresponding C++ class.

# Templates : roborobo4/docs/api/_______.rst




################################################################################################################
# OVERRIDE C++ OBJECTS (reset and step methods are mandatory to implement)
################################################################################################################

# We can extend C++ classes CircleObject, SqureObject and MovableObject
# The constructor receives a data dictionary containing the configuration file's information
# An object must implement the reset and step function
# To use movable objects, do not forget to set "gMovableObjects = true" in your properties file

# In the configuration file, for each object, we tell that we want to use the objects that we defined.
# Set the number of objects : gNbOfPhysicalObjects = 3
# Tell that we want to use an object with the id 'xxx' : physicalObject[0].pytype = xxx
# (NB. If we write physicalObjects[0].pytype = xxx, we have all the object of the same type (???) )
# Define other features aboout the object 'xxx' : physicalObjects[0].sendMessageTo = 0

# In the main, pass as parameter to Pyroborobo a dictionary mapping the pytype key to our object classes
# object_class_dict = {'xxx' : Kale_A_Object, 'yyy' : Kale_B_Object, 'zzz' : Kale_C_Object}


# NB. We can create default objects that don't need to declare their pytype. Procedure :
#   - in the configuration file, set existing gPhysicalObjectDefaultType to -1
#     gPhysicalObjectDefaultType = -1
#   - in the main, add the default object to the object_class_dict
#     object_class_dict = {'_default': Kale_D_Object}



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

class Kale_B_Object(SquareObject):

    def __init__(self, id, data):           # put "data" even if it is not used
        SquareObject.__init__(self, id)
        self.str = "[Kale_B_Object] : "
        self.cptSteps = 0
        self.rob = Pyroborobo.get()         # get pyroborobo singleton ?

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

rob = Pyroborobo.create(fileConfig, controller_class = PythonController, object_class_dict = {'xxx': Kale_A_Object, 'yyy': Kale_B_Object, 'zzz': Kale_C_Object})

rob.start()                                 # activation of pyroborobo simulator 
rob.update(nbSteps)                         # activation of nbUpdates steps
rob.close()                                 # free memory


