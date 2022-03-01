
#                                               ROBOROBO TUTORIAL 1


################################################################################################################
# IMPORTS
################################################################################################################

from pyroborobo import Pyroborobo           # in MAIN : Creation d'un pyroborobo simulator (singleton object)
from pyroborobo import Controller           # in MAIN and OVERRIDE C++ CONTROLLER : override of C++ Controller to create a Python controller


################################################################################################################
# PARAMETERS
################################################################################################################

fileConfig = "config/prova1.properties"     # configuration file
nbSteps = 50                                # number of steps




################################################################################################################
# OVERRIDE C++ CONTROLLER
################################################################################################################

# Il existe plusieurs C++ classes : Controllers, WorldModels, WorldObserver, PhysicalObjects, AgentObservers
# To override the C++ classes by Python classes, create a class that encodes the new behaviour and 
# provide it to the Pyroborobo.create method. The new class will inherit from the corresponding C++ class.

# Templates : roborobo4/docs/api/_______.rst

class PythonController(Controller):         # override of C++ class Controller to create a new Python controller
    
    def __init__(self, world_model):        # world_model is a PyWorldModel, this class allows to access et manipulate the robor behaviour
        Controller.__init__(self, world_model) # link Python - C++ : call this super constructor BEFORE any other operation !
        print("I'm a Python controller\n")  # gInitialNumberOfRobots (config) affichages, un par robot
        self.cptSteps = 0

    def reset(self):                        # initialisation of the PythonController
        print("I'm initialized\n")

    def step(self):                         # step méthod is called at each time step for every robot :
        self.cptSteps += 1                       # ce compteur s'incrémente de nbSteps fois 
        print("I'm robot n." + str(self.id) + ", cptSteps = " + str(self.cptSteps) ) # nbSteps * gInitialNumberOfRobots (config) affichages

        #-------------------------------------------------------------------------------------------------------

        # Simple default definition of translation and rotation

        self.set_translation(1)             # go straight
        self.set_rotation(0)                # don't turn

        #-------------------------------------------------------------------------------------------------------

        # Robot manipulation, world_model méthods (PyWorldModel C++ class)
        # get_distance_at returns the distance to the object seen or "None" if any
        
        if self.get_distance_at(1) < 1 or self.get_distance_at(2) < 1 : # if we see something on our left or in front of us
            self.set_rotation(0.5)          # turn right
        elif self.get_distance_at(3) < 1 :  # if we see something on our right
            self.set_rotation(-0.5)         # turn left





################################################################################################################
# MAIN
################################################################################################################

# Creation d'un pyroborobo simulator (singleton object). Parameters :
#   - configuration file
#   - overrided C++ classes by Python classes
#       - controller_class                  <------ we tell roborobo to use the Python controller

rob = Pyroborobo.create(fileConfig, controller_class = PythonController)

#---------------------------------------------------------------------------------------------------------------

rob.start()                                 # activation of pyroborobo simulator 
rob.update(nbSteps)                         # activation of nbUpdates steps
rob.close()                                 # free memory


