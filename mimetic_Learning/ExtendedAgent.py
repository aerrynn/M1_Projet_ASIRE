from pyroborobo import Pyroborobo, Controller, AgentObserver
import numpy as np
from Neural_network import NeuralNetwork
import Const as c
import DataHandler

class Agent(Controller):
    '''
    Extended agent from controller
    '''

    def __init__(self, wm) -> None:
        '''
        Const
        '''
        Controller.__init__(self, wm)
        self.sliding_window = [0 for _ in range(c.EVALUATION_TIME)]
        self.theta = NeuralNetwork(2*self.nb_sensors, 2, c.NB_HIDDENS)
        self.messages = []  # used to store broadcasts
        self.current_capacity = 0
        self.rob = Pyroborobo.get()
        self.age = 0
        if self.id < c.NB_LEARNER:
            self.type = 0                               # 0 -> learner
            self.set_color(255, 0, 0)
        else:
            self.type = 1                               # 1 -> teacher
            self.set_color(0, 0, 255)
            self.last_obs = []
            self.last_mvm = None

    def step(self) -> None:
        global iteration
        self.age += 1
        if self.id == 0:
            iteration += 1
            if iteration == c.EVALUATION_TIME:
                iteration = 0
        self.sliding_window[iteration] = 0
        obs, score = self.sense()
        mvm = self.apply_policy(obs)
        self.act(mvm)
        self.broadcast(obs, mvm, score)
        self.learn_from_msg()

    def reset(self):
        pass

    def sense(self) -> tuple:
        '''
        sense : get the data from the sensors of the agent
            :return data: the data retrieved, for each sensor creates 3 inputs:
                - distance of a detected obstacle
                - type of the object
            :return fitness: the agent's score
        '''
        data = self.get_all_distances()
        data_plus = []
        for i, v in enumerate(data):
            # Adds the pos of the closest obstacle
            data_plus.append(v)
            # Adds a boolean, whether the object is a target, or an obstacle
            data_plus.append(self.get_object_at(i) != -1)
            # data_plus.append(self.get_wall_at(i) == 1 or (self.get_robot_id_at(i) != -1))
        data_plus = np.array(data_plus)
        fitness = self.fitness(data)
        return data_plus, fitness

    def fitness(self, sensors_data: np.ndarray) -> float:
        '''
        fitness : Function to be overwritten
            :param sensors_data:
        '''
        return self.current_capacity

    def act(self, action_vector: np.ndarray) -> None:
        '''
        act : compute the agent movement for this step
            :param action_vector: a deterministic vector
        '''
        self.set_translation(action_vector[0])
        self.set_rotation(action_vector[1])

    def broadcast(self, obs: np.ndarray, mvm: np.ndarray, score: float) -> None:
        '''
        broadcast sends a message containing theta, idx and score to nearby agents
            :param obs: A vector of tuples containing the inputs 
            :param mvm: outputs of the agent given its comportement
            :param score: The fitness to send
        '''
        for i in range(self.nb_sensors):
            rob_id = self.get_robot_id_at(i)
            if rob_id == -1:
                continue
            self.rob.controllers[rob_id].messages.append(
                (self.id, obs, mvm, score))

    def apply_policy(self, observations: np.ndarray) -> None:
        '''
        policy_function : Deterministic policy π_θ to compute the action vector 
            :param observation: the result vector of the observation
            :param theta: the policy
            :return a: an action vector
        '''
        out = self.theta.ff_to_output(observations)
        return np.clip(out, -1, 1)

    def save_data(self):
        '''
        save_data: saves the current fitness of the agent in a dict to be able
        to trace its evolution 
        '''
        if c.DATA_SAVE:
            if self.type == 1:
                DataHandler.add_teacher_data(self.id, self.fitness())
            else :
                DataHandler.add_student_data(self.id, self.fitness())

    def learn_from_msg(self):
        '''
        learn_from_msg: used to teach the agent how to act through messages received
        through broadcast
        '''
        self.messages[:] = []

    def pick_up(self) -> None:
        self.sliding_window[DataHandler.evaluation_iteration] = 1

########################################################################################
