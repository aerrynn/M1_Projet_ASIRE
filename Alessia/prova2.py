
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
nbSteps = 50




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
# Tell that we want to use an object with the id 'xxx' : physicalObjects[0].pytype = xxx 
# Define other features aboout the object 'xxx' : physicalObjects[0].sendMessageTo = 0

# In the main, pass as parameter to Pyroborobo a dictionary mapping the pytype key to our object classes
# object_class_dict = {'xxx' : Kale_A_Object, 'yyy' : Kale_B_Object, 'zzz' : Kale_C_Object}


# NB. We can create a default object that doesn't need to declare their pytype. Procedure :
#   - in the configuration file, set existing gPhysicalObjectDefaultType to 1
#     gPhysicalObjectDefaultType = -1
#   - in the main, add the default object to the object_class_dict
#     object_class_dict = {'_default': Kale_D_Object}



class Kale_A_Object(CircleObject):

    def __init__(self, id):
        CircleObject.__init__(self, id)
        self.str = "[Kale_A_Object] : "

    def reset(self):
        print(self.str + "initialized")
        #super().reset()

    def step(self):
        print(self.str + "step")
        #super().step()

    # def is_pushed(self, id, speed):
    #     print(self.str + "is_pushed")
    #     #super().is_pushed(id_, speed)

    # def is_touched(self, id):
    #     print(self.str + "is_touched")
    #     #super().is_touched(id_)

    # def is_walked(self, id):
    #     print(self.str + "is_walked")
    #     #return super().is_walked(id_)

    def inspect(self, prefix=""):
        return f"[INFO] I'm the object #{self.id}\n"


#---------------------------------------------------------------------------------------------------------------

""" class Kale_B_Object(SquareObject):

    def __init__(self):
        SquareObject.__init__(self)
        str = "[Kale_B_Object] : "
        print(str + "initialized")

    def step(self):
        print(str + "step")
        #super().step()

    def is_pushed(self, id_, speed):
        print(str + "is_pushed")
        #super().is_pushed(id_, speed)

    def is_touched(self, id_):
        print(str + "is_touched")
        #super().is_touched(id_)

    def is_walked(self, id_):
        print(str + "is_walked")
        #return super().is_walked(id_)

    def inspect(self, prefix=""):
        return f"[INFO] I'm the object #{self.id}\n" """


#---------------------------------------------------------------------------------------------------------------

""" class Kale_C_Object(MovableObject):

    def __init__(self):
        MovableObject.__init__(self)
        str = "[Kale_C_Object] : "
        print(str + "step")

    def step(self):
        print(str + "step")
        #super().step()

    def is_pushed(self, id_, speed):
        print(str + "is_pushed")
        #super().is_pushed(id_, speed)

    def is_touched(self, id_):
        print(str + "is_touched")
        #super().is_touched(id_)

    def is_walked(self, id_):
        print(str + "is_walked")
        #return super().is_walked(id_)

    def inspect(self, prefix=""):
        return f"[INFO] I'm the object #{self.id}\n" """


#---------------------------------------------------------------------------------------------------------------

""" class Kale_D_Object(CircleObject):

    def __init__(self):
        CircleObject.__init__(self)
        str = "[Kale_D_Object] : "
        print(str + "initialized")

    def step(self):
        print(str + "step")
        #super().step()

    def is_pushed(self, id_, speed):
        print(str + "is_pushed")
        #super().is_pushed(id_, speed)

    def is_touched(self, id_):
        print(str + "is_touched")
        #super().is_touched(id_)

    def is_walked(self, id_):
        print(str + "is_walked")
        #return super().is_walked(id_)

    def inspect(self, prefix=""):
        return f"[INFO] I'm the object #{self.id}\n" """





################################################################################################################
# OVERRIDE C++ CONTROLLER
################################################################################################################

class PythonController(Controller):
    
    def __init__(self, world_model):
        Controller.__init__(self, world_model)
        #self.rob = Pyroborobo.get()
        print("I'm a Python controller\n")
        self.cptSteps = 0

    def reset(self):
        print("I'm initialized\n")

    def step(self):
        self.cptSteps += 1
        print("I'm robot n." + str(self.id) + ", cptSteps = " + str(self.cptSteps) )

        #-------------------------------------------------------------------------------------------------------

        # Simple default definition of translation and rotation

        self.set_translation(1)
        self.set_rotation(0)

        #-------------------------------------------------------------------------------------------------------

        # Robot manipulation, world_model méthods (PyWorldModel C++ class)
        
        if self.get_distance_at(1) < 1 or self.get_distance_at(2) < 1 :
            self.set_rotation(0.5)
        elif self.get_distance_at(3) < 1 :
            self.set_rotation(-0.5)

        #-------------------------------------------------------------------------------------------------------


    def inspect(self, prefix=""):
        return f"[INFO] I'm the robot #{self.id}\n"




################################################################################################################
# MAIN
################################################################################################################

rob = Pyroborobo.create(fileConfig, controller_class = PythonController, object_class_dict = {'xxx': Kale_A_Object})
#                        'yyy': Kale_B_Object,              \
#                        'zzz': Kale_C_Object,              \
#                        '_default': Kale_D_Object})        \

rob.start()                                 # activation of pyroborobo simulator 
rob.update(nbSteps)                         # activation of nbUpdates steps
rob.close()                                 # free memory


