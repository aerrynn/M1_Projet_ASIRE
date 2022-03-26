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

    def fitness(self, sensors_data):
        value = self.current_capacity
        self.current_capacity = 0
        return value

    def step(self):
        self.age += 1
        obs, fitness = self.sense()
        if self.type == 1:
            mvm = self.expertPolicy(obs)
        else:
            mvm = super().apply_policy(obs)
        self.act(mvm)

        self.broadcast(self.last_observation, self.last_action, 0)
        self.last_observation = obs
        self.last_action = mvm
        self.learn_from_msg()
        if self.age == c.LEARNING_GAP :
            self.age = 0
            self.current_capacity = 0
        self.sliding_window[self.age] = fitness
        try :
            data[self.id].append(np.mean(self.sliding_window))
        except KeyError:
            data[self.id] = [np.mean(self.sliding_window)]

    def learn_from_msg(self):
        if self.type == 1:
            return
        # print(f"{self.age}/{c.LEARNING_GAP}")
        if self.age != c.LEARNING_GAP:
            return
        if len(self.messages) == 0:
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
            print(
                f"{self.id} learned from {n} messages ({self.total_data_size} total)")
        self.messages[:]=[]
        self.memory[:] = []

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
        for sensor_id in range(0, 5):
            if observation[(sensor_id*4)+1]:
                food_spots.append(observation[sensor_id*4])
                food_spotted = True
            else:
                food_spots.append(2)
        if food_spotted:
            direction = np.argmin(food_spots)
            if direction == 2:
                return 1, 0
            if direction < 2:
                return 1, -0.5
            return 1, 0.5
        # Check each frontal sensor for obstacles
        if (observation[(2*4)+2] + observation[(2*4)+3]) == 0:  # There's nothing in front
            # if there's something on the left
            if (observation[(1*4)+2] + observation[(1*4)+3]) > 0:
                return 1, -0.5
            # if there's something on the right
            if (observation[(1*4)+2] + observation[(1*4)+3]) > 0:
                return 1, 0.5
            return 1, 0
        if (observation[(0*4)+2] + observation[(0*4)+3]) > 0:
            return 1, 1
        return 1, -1

    def broadcast(self, obs, mvm, score):
        if self.type == 0:
            return
        super().broadcast(obs, mvm, score)


def discretise(obs):
    # print('discretise', obs)
    return np.array([(x//.1)/10 + .1 for x in obs])
