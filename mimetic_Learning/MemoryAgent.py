from ExtendedAgent import Agent
import numpy as np
import Const as c
from AdaptativeLearningRate import exponentialDecay
import DataHandler
import rtree

iteration = 0




class Cluster:
    def __init__(self, observation, action, weight):
        self.obs = observation
        self.act = action
        self.weight = weight

    def dist_to(self, obj):
        return dist(self.obs, obj)

    def __str__(self):
        return str(self.obs) + '=>' + str(self.act) + "[" + str(self.weight) + "]"


def Clusterise(lst):
    mem = []
    while lst != []:
        curr = lst.pop(0)
        lst.sort(key=lambda x: curr.dist_to(x))
        # print(curr)
        while True:
            if lst == []:
                mem.append(curr)
                break
            each = lst[0]
            # print(f"\t{each}")
            if each.act == curr.act:
                curr = centroid(each, curr)
                lst.pop(0)
            else:
                mem.append(curr)
                break
    return mem


def dist(obj1, obj2):
    s = 0
    for i, j in zip(obj1, obj2.obs):
        s = max(np.abs(i-j), s)
    return s


def centroid(obj1, obj2):
    rtot = obj1.weight + obj2.weight
    newObs = obj1.obs * obj1.weight + obj2.obs * obj2.weight
    return Cluster(newObs, obj1.act, rtot)


class MemoryAgent(Agent):
    def __init__(self, wm) -> None:
        '''
        Const
        '''
        super().__init__(wm)
        self.learnt_behaviour = 0
        if c.DISCRETISE_RATIO == -1:
            self.memory = []
        else:
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
            self.broadcast(obs, mvm, 1)
            self.last_mvm = mvm
            self.last_obs = obs
            DataHandler.add_teacher_data(self.id, self.fitness())
        else:
            mvm, certainty = self.learnerPolicy(obs)
            if c.LEARNT_BEHAVIOUR_PROPAGATION:
                self.broadcast(obs, mvm, certainty)
            self.act(mvm)
            self.learn_from_msg()
            DataHandler.add_student_data(self.id, self.fitness())
            try :
                DataHandler.total_learnt[str(self.id)].append(self.learnt_behaviour)
            except KeyError:
                DataHandler.total_learnt[str(self.id)] = [self.learnt_behaviour]
        # Computing the fitness based off a sliding window of the picked up circles
        if self.age == c.LEARNING_GAP:
            self.age = 0
            self.current_capacity = 0
            # For dataCollection purposes
        if self.id == 28:
            print(len(self.memory))
            iteration += 1
            # print(DataHandler.evaluation_iteration, self.sliding_window[DataHandler.evaluation_iteration])
            DataHandler.evaluation_iteration += 1
            DataHandler.iteration += 1
            # print(f"\r{DataHandler.iteration}/{c.NB_ITER}", end='')
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
            _, obs, mvmt, score = message
            if mvmt == (0.5, 0.05) : continue
            if c.DISCRETISE_RATIO == -1:
                self.learn_message(obs, mvmt)
                continue
            obs_d = discretise(obs)
            try:
                self.memory[obs_d]
            except KeyError:
                self.memory[obs_d] = mvmt
                self.learnt_behaviour += 1
        n = len(self.messages)
        self.total_data_size += n
        if c.VERBOSE and n != 0:
            print(
                f"{self.id} learned from {n} messages ({self.total_data_size} total, {len(self.memory.keys())} differents)")
        self.messages[:] = []

    def learn_message(self, obs, mvmt):
        r = self.get_closest_obs(obs)
        if r == None :
            self.memory.append(Cluster(obs, mvmt, 1))
            return
        if r.act == mvmt :
            self.memory[0].obs = (self.memory[0].obs * r.weight + obs)/r.weight+1
            self.memory[0].weight += 1
        else :
            self.memory.append(Cluster(obs, mvmt, 1))

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
        certainty = 0
        if c.DISCRETISE_RATIO == -1:
            r = self.get_closest_obs(observation)
            if r != None : 
                certainty = self.compute_frontal_certainty(observation)
                return r.act, certainty 
            else :
                return (0.5,0.05), 0
        try:
            obs = discretise(observation)
            mvm = self.memory[obs]
            if c.LEARNT_BEHAVIOUR_PROPAGATION:
                self.broadcast(observation, mvm, 0)
        except KeyError:
            mvm = (0.5,0.05)
        return mvm, certainty

    def compute_frontal_certainty(self, observation):
        min_ = 1
        # BUG HORROR !!!
        for i, j in zip(observation[:10], self.memory[0].obs[:10]):
            s = np.abs(i-j)
            if s < min_:
                s = min_
        return s


    def get_closest_obs(self, observation):
        if self.memory == []:
            return None
        self.memory.sort(key=lambda x: dist(observation, x))
        return self.memory[0]

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
        return str(self.id) + "  " + str(self.type) + '\n' + str(''.join([str (x) for x in self.memory]))

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
    if c.DISCRETISE_RATIO == -1:
        return obs
    return ','.join([str((x//(1/(c.DISCRETISE_RATIO-1)))*(1/(c.DISCRETISE_RATIO-1))) for x in obs[:10]])
