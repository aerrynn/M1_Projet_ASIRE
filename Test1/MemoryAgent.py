from ExtendedAgent import Agent
import numpy as np
import Const as c

data = dict()


def save_data(filename):
    with open(filename, 'w+') as f:
        for each in data.keys():
            f.write(str(data[each])+'\n')
    print('Done')


class MemoryAgent(Agent):
    def __init__(self, wm) -> None:
        '''
        Const
        '''
        super().__init__(wm)
        self.sliding_window = [0 for _ in range(c.LEARNING_GAP)]
        self.total_data_size = 0                        # Tracker for debugging purpose
        if self.id < c.NB_LEARNER:
            self.type = 0                               # 0 -> learner
            self.act_function = super().apply_policy    # Neural Network based_movement
        else:
            self.type = 1                               # 1 -> teacher
            self.act_function = self.expertPolicy

    def fitness(self, sensors_data):
        '''
        At each point of time, we track if the agent has picked up something
        if he has, we increase by one point its fitness for the step
        '''
        value = self.current_capacity
        self.current_capacity = 0
        return value

    def step(self):
        self.age += 1
        obs, fitness = self.sense()
        if self.type == 1:
            mvm = self.expertPolicy(obs)
            self.act(mvm)
            self.broadcast(obs, mvm, 0)
        else:
            mvm = super().apply_policy(obs)
            self.act(mvm)
            self.learn_from_msg()
        # Computing the fitness based off a sliding window of the picked up circles
        if self.age == c.LEARNING_GAP:
            self.age = 0
            self.current_capacity = 0
        self.sliding_window[self.age] = fitness

        # For dataCollection purposes
        if c.DATA_SAVE:
            try:
                data[self.id].append(np.mean(self.sliding_window))
            except KeyError:
                data[self.id] = [np.mean(self.sliding_window)]

    def learn_from_msg(self):
        if self.type == 1:                              # Experts Don't learn
            return
        if self.age != c.LEARNING_GAP or len(self.messages) == 0:
            return
        for message in self.messages:
            data_x = []
            data_y = []
            _, obs, mvmt, _ = message
            data_x.append(obs)
            data_y.append(mvmt)
        self.theta.train(np.array(data_x), np.array(data_y))
        n = len(self.messages)
        self.total_data_size += n
        if c.VERBOSE and n != 0:
            print(
                f"{self.id} learned from {n} messages ({self.total_data_size} total)")
        self.messages[:] = []

    def expertPolicy(self, observation: np.ndarray) -> tuple:
        '''
        @Overwrite
        '''

        # Check each frontal sensor for food :
        food_spots = []
        food_spotted = False
        for sensor_id in range(0, 5):
            if observation[(sensor_id*2)+1] == 1:
                food_spots.append(observation[sensor_id*2])
                food_spotted = True
            else:
                food_spots.append(2)
        if food_spotted:
            direction = np.argmin(food_spots)   # Go toward the closest food source
            return c.EXPERT_SPEED, (direction -2) * 0.5
        # Check each frontal sensor for obstacles
        if (observation[(2*2)]) == 1:  # There's nothing in front
            # if there's something on the left
            if (observation[(1*2)]) < 1:
                return c.EXPERT_SPEED, 0.5
            # if there's something on the right
            if (observation[(3*2)]) < 1:
                return c.EXPERT_SPEED, -0.5
            return c.EXPERT_SPEED, 0
        # If there's something in front
        if (observation[(0*2)]) < 1:  # and on the left
            return c.EXPERT_SPEED, 1  # Turn straight right
        return c.EXPERT_SPEED, -1  # else : turn straight left

    def broadcast(self, obs, mvm, score):
        if self.type == 0:
            return
        super().broadcast(obs, mvm, score)


def discretise(obs):
    # print('discretise', obs)
    return np.array([(x//.1)/10 + .1 for x in obs])
