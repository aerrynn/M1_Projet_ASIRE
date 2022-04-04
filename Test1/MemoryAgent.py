from ExtendedAgent import Agent
import numpy as np
import Const as c
from AdaptativeLearningRate import exponentialDecay
import DataHandler

teach_data = [[] for _ in range(c.NB_ITER)]
iteration = 0

class MemoryAgent(Agent):
    def __init__(self, wm) -> None:
        '''
        Const
        '''
        super().__init__(wm)
        self.memory = {}
        self.total_data_size = 0                        # Tracker for debugging purpose

    def fitness(self, sensors_data=None):
        '''
        At each point of time, we track if the agent has picked up something
        if he has, we increase by one point its fitness for the step
        '''
        return np.sum(self.sliding_window)

    def step(self):
        '''
        @Overwrite
        '''
        global iteration
        self.sliding_window[DataHandler.evaluation_iteration] = 0
        self.age += 1
        obs, fitness = self.sense()
        self.theta.backprop(
            np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]), np.array([1, 0]))
        if self.type == 1:
            mvm = self.expertPolicy(obs)
            self.act(mvm)
            self.broadcast(obs, mvm, 0)
            self.last_mvm = mvm
            self.last_obs = obs
            DataHandler.add_teacher_data(self.id, self.fitness())
        else:
            mvm = self.learnerPolicy(obs)
            self.act(mvm)
            self.learn_from_msg()
            DataHandler.add_student_data(self.id, self.fitness())
        # Computing the fitness based off a sliding window of the picked up circles
        if self.age == c.LEARNING_GAP:
            self.age = 0
            self.current_capacity = 0
            # For dataCollection purposes
        if self.id == 28:
            iteration += 1
            # print(DataHandler.evaluation_iteration, self.sliding_window[DataHandler.evaluation_iteration])
            DataHandler.evaluation_iteration += 1
            DataHandler.iteration += 1
            print(f"\r{DataHandler.iteration}/{c.NB_ITER}", end='')
            if DataHandler.evaluation_iteration == c.EVALUATION_TIME:
                DataHandler.evaluation_iteration = 0
        self.save_data()

    def learn_from_msg(self):
        '''
        @Overwrite
        '''
        if self.type == 1:                              # Experts Don't learn
            return
        for message in self.messages:
            _, obs, mvmt, _ = message
            obs_d = discretise(obs)
            r_obs = discretise([obs[8], obs[9], obs[6], obs[7],
                               obs[4], obs[5], obs[2], obs[3], obs[0], obs[1]])
            r_mvmt = (mvmt[0], -mvmt[1])
            try:
                self.memory[obs_d]
            except KeyError:
                self.memory[obs_d] = mvmt
            try:
                self.memory[r_obs]
            except KeyError:
                self.memory[r_obs] = r_mvmt
        n = len(self.messages)
        self.total_data_size += n
        if c.VERBOSE and n != 0:
            print(
                f"{self.id} learned from {n} messages ({self.total_data_size} total, {len(self.memory.keys())} differents)")
        self.messages[:] = []

    def learnerPolicy(self, observation: np.ndarray) -> tuple:
        '''
        learnerPolicy: The learner agent will check in his memory if he learnt
        how to act given an observation, if so he will act this way, otherwise
        he'll follow a simple neural pattern.
            :param observation: The result of the sensor detection 
            during the current step
            :return mvm: a tuple containing both translation and rotation
            that the agent will execute during this step
        '''

        obs = discretise(observation)
        try:
            mvm = self.memory[obs]
            if c.LEARNT_BEHAVIOUR_PROPAGATION:
                self.broadcast(observation, mvm, 0)
        except KeyError:
            mvm = super().apply_policy(observation)
        return mvm

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

    def inspect(self, prefix=''):
        return str(self.id) + "  " + str(self.type) +'\n' + str(self.memory) 

    def broadcast(self, obs, mvm, score):
        '''
        @Overwrite
        '''
        if self.type == 1 and self.last_obs != []:
            super().broadcast(self.last_obs, self.last_mvm, score)
            return
        super().broadcast(obs, mvm, score)


def discretise(obs):
    '''
    discretise : discretise an input
        :param obs: the input observations of the agent
        :return : a string to hash for behaviour adaptation
    '''
    return ','.join([str((x//1)) for x in obs[:10]])
