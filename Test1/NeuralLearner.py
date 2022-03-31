from ExtendedAgent import Agent
from AdaptativeLearningRate import exponentialDecay
import numpy as np
import Const as c
import DataHandler


class NeuralLearner(Agent):
    def __init__(self, wm) -> None:
        '''
        Const
        '''
        super().__init__(wm)
        self.sliding_window = [0 for _ in range(c.EVALUATION_TIME)]
        self.total_data_size = 0                        # Tracker for debugging purpose
        if self.id < c.NB_LEARNER:
            self.type = 0                               # 0 -> learner
            self.set_color(255, 0, 0)
        else:
            self.type = 1                               # 1 -> teacher
            self.set_color(0, 0, 255)
            self.last_obs = []
            self.last_mvm = None

    def fitness(self, sensors_data=None):
        '''
        Return the amount of circles picked up during the evaluation
        interval
        '''
        return np.mean(self.sliding_window)

    def step(self):
        '''
        @Overwrite
        '''
        self.age += 1
        obs, fitness = self.sense()
        if self.type == 1:
            mvm = self.expertPolicy(obs)
            self.act(mvm)
            self.broadcast(obs, mvm, 100)
            self.last_mvm = mvm
            self.last_obs = obs
        else:
            mvm = super().apply_policy(obs)
            self.act(mvm)
            if c.PROPAGATION:
                self.broadcast(obs, mvm, self.fitness())
            self.learn_from_msg()
        # Computing the fitness based off a sliding window of the picked up circles
        if self.age == c.LEARNING_GAP:
            self.age = 0
            self.current_capacity = 0
        # For dataCollection purposes
        if self.id == 0:
            DataHandler.evaluation_iteration += 1
            DataHandler.iteration += 1
            print(f"\r{DataHandler.iteration}/{c.NB_ITER}", end='')
            if DataHandler.evaluation_iteration == c.EVALUATION_TIME:
                DataHandler.evaluation_iteration = 0
        self.sliding_window[DataHandler.evaluation_iteration] = 0
        self.save_data()

    def learn_from_msg(self):
        '''
        @Overwrite
        '''
        if self.type == 1:                              # Experts Don't learn
            return
        if self.age != c.LEARNING_GAP or len(self.messages) == 0:
            return
        for message in self.messages:
            data_x = []
            data_y = []
            _, obs, mvmt, score = message
            if score < self.fitness():
                continue
            data_x.append(obs)
            data_y.append(mvmt)
        self.theta.train_batch(np.array(data_x), np.array(
            data_y), c.DECAY_FUNCTION(DataHandler.iteration))
        n = len(self.messages)
        if c.VERBOSE and n != 0:
            print(
                f"{self.id} learned from {n} messages ({self.total_data_size} total)")

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
            # Go toward the closest food source
            direction = np.argmin(food_spots)
            return c.EXPERT_SPEED, (direction - 2) * 0.5
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
        '''
        @Overwrite
        '''
        if self.type == 1:
            score = 100  # We hardcode the increment
        if self.type == 1 and self.last_obs != []:
            super().broadcast(self.last_obs, self.last_mvm, score)
            return
        super().broadcast(obs, mvm, score)
