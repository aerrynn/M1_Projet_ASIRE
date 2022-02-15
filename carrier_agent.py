from hit_ee import Agent
import numpy as np

################################ CONSTS ################################
MAX_CAPACITY = 1
########################################################################


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