from pyroborobo import CircleObject, Pyroborobo, WorldObserver
from hitAgents import Agent
import numpy as np
import random

######################################## CONSTS ########################################
MAX_CAPACITY = 1

''' Interactible_related '''
REGROWTH_TIME = 150
MAX_RESSOURCE_LEVEL = 1

NB_ITER = 10000
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
        # This Agent can carry ressources
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
        global best_picker
        s = self.current_capacity
        self.current_capacity = 0
        return s

    def step(self):
        super().step()

    def reset(self):
        pass

    def inspect(self, prefix=""):
        return f'{self.id} : {self.current_capacity}'

    def pick_up(self):
        # print(f'{self.id}: I picked up some sugar !')
        self.current_capacity += 1


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
        # TODO: How to set the expert neural network ?


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
        if self.rob.controllers[rob_id].current_ressource > 0:
            print(
                f"{rob_id} deposited {self.rob.controllers[rob_id].current_ressource} ressources")
            self.rob.controllers[rob_id].current_ressource = 0
        return super.is_walked(self, rob_id)

    def inspect(self, prefix=""):
        return f"[INFO] I'm the object #{self.id}"

########################################################################################


class BushNode(CircleObject):
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
                self.regrowing = False          # Regrowth ended

    def is_walked(self, rob_id):
        # If the agent can store the ressource, increase it
        self.rob.controllers[rob_id].pick_up()
        self.regrowing = True
        self.cur_regrow = self.regrowth_time
        self.hide()
        self.unregister()

    def inspect(self, prefix):
        return f"Ressource_Node_{self.id}\n"

# World gen
#


class MyWorldObserver(WorldObserver):
    def __init__(self, world):
        super().__init__(world)
        self.rob = Pyroborobo.get()

    def init_pre(self):
        super().init_pre()

    def init_post(self):
        super().init_post()
        for i in range(50):
            bush = BushNode(i, {})
            self.rob.add_object(bush)

        x, y = 5, 5
        delta_X, delta_Y = 20, 20 # To organize the robots initialisation
        # for robot in self.rob.controllers:
        #     robot.set_absolute_orientation(random.randint(-180,180))
        #     robot.set_position(x, y)
        #     x = x + delta_X
        #     if x > 780:
        #         x = 20
        #         y += delta_Y

        print("End of worldobserver::init_post")

    def step_pre(self):
        super().step_pre()

    def step_post(self):
        super().step_post()

########################################################################################
########################################  Main  ########################################
########################################################################################


def main():
    rob = Pyroborobo.create(
        "config/test.properties",
        controller_class=CarrierAgent,
        world_observer_class=MyWorldObserver,
        object_class_dict={
            '_default': BushNode,
            'bush': BushNode
        }
    )

    rob.start()
    rob.update(NB_ITER)
    Pyroborobo.close()


if __name__ == "__main__":
    main()
