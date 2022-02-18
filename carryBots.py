from hitAgents import Agent
import numpy as np

######################################## CONSTS ########################################
MAX_CAPACITY = 1

CARRYING_BONUS = 1000
STORING_BONUS = 2000

''' Interactible_related '''
STORAGE_BOON_DURATION = 100
REGROWTH_TIME = 10
MAX_RESSOURCE_LEVEL = 1
########################################################################################



########################################################################################
class CarrierAgent(Agent):
    '''
    CarrierAgent
        TODO: Fill Description 
        What ?
        Why ?
        How ?
    '''
    def __init__(self, wm) -> None:
        super().__init__(wm)
        ## This Agent can carry ressources
        self.max_capacity = MAX_CAPACITY
        self.current_capacity = 0
        # keeps in memory if he stored some ressource
        self.has_stored = 0

    def fitness(self, sensors_data):
        '''
        @Overwrite
        fitness : 
            :param sensors_data:
        '''
        if self.current_capacity > 0 :
            score += CARRYING_BONUS
        if self.has_stored > 0 :
            self.has_stored -= 1
            score += STORING_BONUS
        return score



class ExpertAgent(CarrierAgent):
    '''
    ExpertAgent
        TODO: Fill Description 
        What ?
        Why ?
        How ?
    '''
    def __init__(self, wm) -> None:
        super().__init__(wm)
        #TODO: How to set the expert neural network ?


########################################################################################

class StorageNode(CircleObject):
    '''
    StorageNode :
        TODO: Fill Description 
        What ?
        Why ?
        How ?
    '''

    def __init__(self, id_, data):
        CircleObject.__init__(self, id_)
        self.rob = Pyroborobo.get()

    def is_walked(self, rob_id):
        # If the agent can store the ressource, increase it
        if rob_id.current_ressource > 0:
            rob_id.has_stored += rob_id.current_ressource * STORAGE_BOON_DURATION
            print(f"{rob_id} deposited {rob_id.current_ressource} ressources")
            rob_id.current_ressource = 0

    def inspect(self, prefix=""):
        return f"Storage_Node_{self.id}\n \
        Size={self.ressource_level}"

########################################################################################

class ResourceNode(CircleObject):
    '''
    RessourceNode :
        TODO: Fill Description 
        What ?
        Why ?
        How ?
    '''
    def __init__(self, id_, data):
        CircleObject.__init__(self, id_)

        # After depleting, the node takes regrowth_time steps to regrow
        self.regrowth_time = REGROWTH_TIME
        # While regrow > 0, the node is depleted, and not interactible
        self.cur_regrow = 0
        # Max amount of ressources
        self.max_ressource_level = MAX_RESSOURCE_LEVEL
        self.ressource_level = 10		        # Current amount of ressources
        # Boolean (faster to check than cur_regrow > 0)
        self.regrowing = False
        self.rob = Pyroborobo.get()

    def reset(self):
        self.show()
        self.register()
        self.regrowing = False
        self.cur_regrow = 0

    def step(self):
        if self.regrowing:
            self.cur_regrow -= 1		        # Reduce the left regrowth time
            if self.cur_regrow <= 0: 	        # The node has finished regrowing
                self.show()				        # Displaying the node
                self.register()
                # Once regrown, set the node to max ressource
                self.ressource_level = self.max_ressource_level
                self.regrowing = False          # Regrowth ended

    def is_walked(self, rob_id):
        # If the agent can store the ressource, increase it
        if rob_id.max_ressource_level - rob_id.current_ressource > 0:
            rob_id.current_ressource += 1
            self.ressource_level -= 1
            if self.ressource_level <= 0:       # Start the regrowth cycle
                self.regrowing = True
                self.cur_regrow = self.regrowth_time
                self.hide()
                self.unregister()

    def inspect(self, prefix=""):
        return f"Ressource_Node_{self.id}\n \
        Size={self.ressource_level}"

########################################################################################
########################################  Main  ########################################
########################################################################################

def main():
    rob: Pyroborobo = Pyroborobo.create(
        "config/pywander_pyobj.properties",
        controller_class=Agent,
    )

    rob.start()
    rob.update(NB_ITER)
    rob.close()

if __name__ == "__main__":
    main()
