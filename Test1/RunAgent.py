from pyroborobo import CircleObject, Pyroborobo, WorldObserver
from hitAgents import Agent
import numpy as np
import random

######################################## CONSTS ########################################
MAX_CAPACITY = 1

''' Interactible_related '''

NB_ITER = 10000
########################################################################################

########################################################################################
class RunAgent(Agent):
    '''
    CarrierAgent
        TODO: Fill Description 
        What ?
        Why ?
        How ?
    '''

    def __init__(self, wm) -> None:
        super().__init__(wm)
        self.ite = 0
        self.fitness = 0
        self.best_time = 100
        self.t = 0

    def fitness(self, sensors_data):
        '''
        @Overwrite
        fitness : 
            :param sensors_data:
        '''
        self.fitness = (TP_POS_y - self.absolute_position[1])**2 + \
            (TP_POS_X - self.absolute_position[0])**2
        return self.fitness + 100 - self.best_time

    def step(self):
        super().step()
        self.t += 1

    def reset(self):
        self.best_time = self.t
        self.t = 0
        # self.set_position(self.absolute_position[0], 5)

    def inspect(self, prefix=""):
        return f'{self.id} : {self.fitness}'



class ExpertAgent(RunAgent):
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

class TP(CircleObject):
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
        self.rob.controllers[rob_id].reset()
        return super.is_walked(self, rob_id)

    def inspect(self, prefix=""):
        return f"[INFO] I'm the object #{self.id}"

########################################################################################

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
        tp = self.rob.add_object(TP(1,{}))
        # tp.set_position(TP_POS_X, TP_POS_y)
        x, y = 5, 5
        delta_X, delta_Y = 20, 0 # To organize the robots initialisation
        for robot in self.rob.controllers:
            robot.set_absolute_orientation(random.randint(-180,180))
            robot.set_position(x, y)
            x = x + delta_X
            if x > 780:
                x = 20
                y += delta_Y

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
        "config/RunAgent.properties",
        controller_class=RunAgent,
        world_observer_class=MyWorldObserver,
        object_class_dict={
            '_default': TP,
            'tp': TP
        }
    )

    rob.start()
    rob.update(NB_ITER)
    Pyroborobo.close()


if __name__ == "__main__":
    main()
