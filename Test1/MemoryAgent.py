from ExtendedAgent import Agent
import numpy as np
import Const as c


class MemoryAgent(Agent):
    def __init__(self, wm) -> None:
        '''
        Const
        '''
        super().__init__(wm)
        self.total_data_size = 0
        if self.id < c.NB_LEARNER:
            self.type = 0           # 0 -> learner
            self.act_function = super().apply_policy
        else:
            self.type = 1           # 1 -> teacher
            self.act_function = self.expertPolicy
        self.memory = []
        self.last_action = None
        self.last_observation = None

    def step(self):
        self.age += 1
        obs = self.sense()
        if self.type == 1:
            mvm = self.expertPolicy(obs)
        else:
            mvm = super().apply_policy(obs)
        self.act(mvm)

        self.broadcast(self.last_observation, self.last_action, 0)
        self.last_observation = obs
        self.last_action = mvm
        self.learn_from_msg()

    def learn_from_msg(self):
        if self.type == 1:
            return
        if self.age != c.LEARNING_GAP:
            return
        # print(f"{self.id} received {len(self.messages)} messages !")
        for message in self.messages:
            _, obs, mvmt, _ = message
            self.memory.append((obs, mvmt))
        self.theta.train(np.array([x[0] for x in self.memory]), np.array(
                [x[1] for x in self.memory]))
        n = len(self.messages)
        self.total_data_size += n
        if n != 0:
            print(f"{self.id} learned from {n} messages ({self.total_data_size} total)")
        self.messages[:]

    def expertPolicy(self, observation: np.ndarray) -> tuple:
        '''
        @Overwrite
        '''
        # We only look at the 3 frontal sensors :
        # Go straight for the object
        # print('expert', observation)


        # Check each frontal sensor for food :
        food_spots = []
        food_spotted = False
        for sensor_id in range(0,5):
            if observation[(sensor_id*4)+1]:
                food_spots.append(observation[sensor_id*4])
                food_spotted = True
            else :
                food_spots.append(2)
        print(food_spots)
        if food_spotted :
            direction = np.argmin(food_spots)
            if direction == 2:
                return 1, 0
            if direction < 2:
                return 1, 0.5
            return 1, -0.5
        # Check each frontal sensor for obstacles
        obstacles = []
        for sensor_id in range(0,5):
            if observation[(sensor_id*4)+2] + observation[(sensor_id*4)+3] > 0:
                obstacles.append(observation[sensor_id*4])
            else :
                obstacles.append(3/(np.abs(2-sensor_id)+1))
        print(obstacles)
        direction = np.argmax(obstacles)
        return 1, (direction-2) * 0.5

    def sense(self):
        s, _ = super().sense()
        return s

    def broadcast(self, obs, mvm, score):
        if self.type == 0:
            return
        super().broadcast(obs, mvm, score)


def discretise(obs):
    # print('discretise', obs)
    return np.array([(x//.1)/10 + .1 for x in obs])
