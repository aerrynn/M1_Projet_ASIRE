from ExtendedAgent import Agent
import numpy as np
import Const as c

class MemoryAgent(Agent):
    def __init__(self, wm) -> None:
        '''
        Const
        '''
        super().__init__(wm)
        if self.id <= c.NB_LEARNER:
            self.type = 0           # 0 -> learner
            self.act_function = self.memory_based_policy
        else : 
            self.type = 1           # 1 -> teacher
        self.memory = [None for _ in range(c.MEMORY_SIZE)]
        self.chunck_index = 0
        self.last_action = None
        self.last_observation = None

    def step(self):
        obs = self.sense()
        if self.type == 1 :
            mvm = self.expertPolicy(obs)
        else :
            mvm = self.memory_based_policy(obs)
        self.act(mvm)

        self.broadcast(self.last_observation, self.last_action, 0)
        self.last_observation = obs
        self.last_action = mvm
        self.learn_from_msg()

    def learn_from_msg(self):
        if self.type == 1:
            return
        # print(f"{self.id} received {len(self.messages)} messages !")
        for message in self.messages:
            _, obs, mvmt, _ = message
            self.memory[self.chunck_index] = (discretise(obs), mvmt)
            self.chunck_index += 1
            if self.chunck_index == c.MEMORY_SIZE:
                self.chunck_index = 0
        self.messages = []

    def memory_based_policy(self, obs):
        dobs = discretise(obs)
        for each in self.memory :
            if each == None:
                continue
            # print(each)
            if np.all(each[0] == dobs):
                return each[1]
        return super().apply_policy(obs)

    def expertPolicy(self, observation:np.ndarray) -> tuple:  
        '''
        @Overwrite
        '''
        # We only look at the 3 frontal sensors :
        # Go straight for the object
        # print('expert', observation)

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

    def sense(self):
        s, _ = super().sense()
        return s

    def broadcast(self, obs, mvm, score):
        if self.type == 0 :
            return
        super().broadcast(obs, mvm, score)


def discretise(obs):
    # print('discretise', obs)
    return np.array([(x//.1)/10 + .1 for x in obs])

