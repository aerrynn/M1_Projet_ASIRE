from pyroborobo import CircleObject, Pyroborobo, WorldObserver
from ExtendedAgent import Agent
from MemoryAgent import MemoryAgent
import numpy as np
import random
import Const as c

########################################################################################


class BushNode(CircleObject):
    '''
    RessourceNode : Simple interactible object that can be picked up by agents
    '''

    def __init__(self, id_, data):
        '''
        Const
        '''
        CircleObject.__init__(self, id_)

        # After depleting, the node takes regrowth_time steps to regrow
        self.regrowth_time = c.REGROWTH_TIME
        # While regrow > 0, the node is depleted, and not interactible
        self.cur_regrow = 0
        # Boolean (faster to check than cur_regrow > 0)
        self.regrowing = False
        self.rob = Pyroborobo.get()

    def reset(self):
        '''
        @Overwrite
        '''
        self.show()
        self.register()
        self.regrowing = False
        self.cur_regrow = 0

    def step(self):
        '''
        @Overwrite
        '''
        if self.regrowing:
            self.cur_regrow -= 1		        # Reduce the left regrowth time
            if self.cur_regrow <= 0: 	        # The node has finished regrowing
                self.show()				        # Displaying the node
                self.register()
                self.regrowing = False          # Regrowth ended

    def is_walked(self, rob_id):
        '''
        @Overwrite
        '''
        # If the agent can store the ressource, increase it
        self.rob.controllers[rob_id].pick_up()
        self.regrowing = True
        self.cur_regrow = self.regrowth_time
        self.hide()
        self.unregister()

    def inspect(self, prefix):
        return f"Ressource_Node_{self.id}\n"


########################################################################################


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

        # To organize the robots initialisation
        # x, y = 5, 5
        # delta_X, delta_Y = 20, 20
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
        "config/ForagingTask.properties",
        controller_class=MemoryAgent,
        world_observer_class=MyWorldObserver,
        object_class_dict={
            '_default': BushNode,
            'bush': BushNode
        }
    )

    rob.start()
    rob.update(c.NB_ITER)
    rob.close()


if __name__ == "__main__":
    main()
