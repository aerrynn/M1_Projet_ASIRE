from pyroborobo import CircleObject, Pyroborobo, WorldObserver
from Hit_Agents import Agent
import numpy as np
import random
import Const as c

########################################################################################


class CarrierAgent(Agent):
    '''
    CarrierAgent
        Learning agent for the foraging task
    '''

    def __init__(self, wm) -> None:
        '''
        Const
        '''
        super().__init__(wm)
        # This Agent can carry ressources
        self.max_capacity = c.MAX_CAPACITY
        self.current_capacity = 0
        # keeps in memory if he stored some ressource
        self.has_stored = 0

    def fitness(self, sensors_data):
        '''
        @Overwrite
        fitness : 
            :param sensors_data:
        '''
        if self.id <= 5:
            self.current_capacity += 1/c.EVALUATION_TIME
        global best_picker
        s = self.current_capacity
        self.current_capacity = 0
        return s

    def step(self):
        super().step()

    def inspect(self, prefix=""):
        return f'{self.id} : {self.current_capacity}'

    def pick_up(self):
        if c.VERBOSE:
            print(f'{self.id}: I picked up some sugar !')
        self.current_capacity += 1

    def apply_policy(self, observation):  # HACK TEMPORARY FIX
        '''
            :return translation, rotation:
        '''
        if self.id > 5:
            return super().apply_policy(observation)
        # We only look at the 3 frontal sensors :
        # Go straight for the object
        if observation[5] == c.FOOD_ID:
            return 1, 0
        if observation[7] == c.FOOD_ID:
            return 1, 0.5
        if observation[3] == c.FOOD_ID:
            return 1, -0.5
        if observation[4] < .9:
            if observation[2] < observation[6]:                 # Avoid to the left
                return 1, 0.5
            return 1, - 0.5                                     # Avoid to the right
        if observation[0] < 1 and observation[1] != c.FOOD_ID:
            return 1, 0.5
        if observation[8] < 1 and observation[9] != c.FOOD_ID:
            return 1, -0.5
        return 1, 0


########################################################################################


class ExpertAgent(CarrierAgent):
    '''
    ExpertAgent
        Agent with a fixed movement pattern
    '''

    def __init__(self, wm) -> None:
        '''
        Const
        '''
        super().__init__(wm)

    def fitness(self, sensors_data):
        '''
        @Overwrite
        fitness : 
            :param sensors_data:
        '''
        global best_picker
        s = self.current_capacity = 1/c.EVALUATION_TIME
        self.current_capacity = 0
        return s

    def apply_policy(self, observation):
        '''
        apply_policy: define how the agent should behave given a stimulation
        of its sensors
            :return translation, rotation:
        '''
        # We only look at the 3 frontal sensors :
        # print(f"l:{observation[2]};{observation[3]}, f:{observation[4]};{observation[5]}, r{observation[6]},{observation[7]}")
        if observation[5] == c.FOOD_ID:                     # Go straight for the object
            return 1, 0
        if observation[7] == c.FOOD_ID:
            return 1, 0.5
        if observation[3] == c.FOOD_ID:
            return 1, -0.5
        if observation[4] < 0.75:
            if observation[2] < observation[6]:             # Avoid to the left
                return 1, 0.5
            return 1, - 0.5                                 # Avoid to the right
        if observation[0] < 1 and observation[1] != c.FOOD_ID:
            return 1, 0.5
        if observation[8] < 1 and observation[9] != c.FOOD_ID:
            return 1, -0.5
        return 1, 0


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
        "config/Foraging_Task.properties",
        controller_class=CarrierAgent,
        # controller_class=ExpertAgent,
        world_observer_class=MyWorldObserver,
        object_class_dict={
            '_default': BushNode,
            'bush': BushNode
        }
    )

    rob.start()
    rob.update(c.NB_ITER)
    Pyroborobo.close()


if __name__ == "__main__":
    main()
