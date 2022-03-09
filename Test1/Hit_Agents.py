from pyroborobo import Pyroborobo, Controller, AgentObserver
import numpy as np
from Neural_network import NeuralNetwork
import Const as c


class Agent(Controller):
    '''
    Simple agent attributes and functions for it to be able to learn from the HIT-EE method
    '''

    def __init__(self, wm) -> None:
        Controller.__init__(self, wm)
        self.theta = NeuralNetwork(2*self.nb_sensors, 2, c.NB_HIDDENS)
        self.res = [0 for _ in range(c.EVALUATION_TIME)]
        self.observation_data = np.array([None for _ in range(c.MEMORY_RANGE)])
        self.movements_data = np.array([None for _ in range(c.MEMORY_RANGE)])
        # Stores the last received message, empties when read
        self.message = []
        self.rob = Pyroborobo.get()
        self.time = 0

        # We don't want to learn more than once in a step from a single user
        self.teachers = set()

    def reset(self):
        pass

    def step(self):
        self.hit_algorithm()

    def sense(self):
        '''
        sense : get the data from the sensors of the agent
            :return data: the data retrieved, for each sensorn creates 3 inputs:
                - distance of a detected obstacle
                - type of the object
            :return fitness: the agent's score
        '''
        data = self.get_all_distances()
        data_plus = []
        for i, v in enumerate(data):
            # Adds the pos of the closest obstacle
            data_plus.append(v)
            id_ = 0
            if self.get_object_at(i) != -1:
                id_ = c.FOOD_ID
            elif self.get_wall_at(i):
                id_ = c.WALL_ID
            elif self.get_robot_id_at(i) != -1:
                id_ = c.ROBOT_ID
            # Adds the type_ID of the closest obstacle
            data_plus.append(id_)
        data_plus = np.array(data_plus)
        fitness = self.fitness(data)
        return data_plus, fitness

    def fitness(self, sensors_data):
        '''
        fitness : Function to be overwritten
            :param sensors_data:
        '''
        return 0

    def act(self, action_vector):
        '''
        act : compute the agent movement for this step
            :param action_vector: a deterministic vector
        '''
        self.set_translation(action_vector[0])
        self.set_rotation(action_vector[1])

    def broadcast(self, obs, mvm, score):
        '''
        broadcast sends a message containing theta, idx and score to nearby agents
            :param obs: A vector of tuples containing the inputs 
            :param mvm: outputs of the agent given its comportement
            :param score: The fitness to send
        '''
        if self.theta == 0:
            return
        for i in range(self.nb_sensors):
            rob_id = self.get_robot_id_at(i)
            self.rob.controllers[rob_id].message.append(
                (self.id, obs, mvm, score))

    def apply_policy(self, observations):
        '''
        policy_function : Deterministic policy π_θ to compute the action vector 
            :param observation: the result vector of the observation
            :param theta: the policy
            :return a: an action vector
        '''
        # print((observations.shape))
        out = self.theta.ff_to_output(observations)
        return np.clip(out, -1, 1)

    def transfer_function(self, G, message):
        '''
        transfer_function : If the sender of the message has a higher fitness score
        replace the theta[idx] of the receiver by the theta[idx] of the sender
            :param theta: The receiver agent policy
            :param G: The fitness score of the receiver
            :param message: the message to learn from
                s_ID: the ID of the teacher (for debugging purposes)
                s_O: the observations to learn from
                s_M: the movements to learn
                s_G: the sender's fitness score
            :return theta: The new policy
        '''
        s_ID, s_O, s_M, s_G = message
        if s_ID in self.teachers:
            return
        self.teachers.add(s_ID)
        if G <= s_G:                                                # If the sender has a lower fitness
            return                                                  # score don't do anything
        self.theta.train(s_O, s_M, c.LEARNING_STEPS, c.LEARNING_RATE)

    def gaussian_mutation(self):
        '''
        gaussian_mutation : Randomly mutates the agent, following a gaussian distribution
        '''
        self.theta.mutate()

    def hit_algorithm(self):
        '''
        hit_algorithm : applies the hit learning algorithm to the current agent
            evaluation_time: amount of steps evaluated (T in the paper)
            param policy_function: policy followed by the agent (π in the paper)
            self.theta = agent initialisation of policy
            o = observation vector
            r = reward scalar
            self.res = reward buffer of size T 
            self.movement_data = The memory of the previous movements
            a = action vector
            G = personal evaluation (sum of r on the whole evaluation time)
        '''
        o, r = self.sense()
        self.res[self.time % c.EVALUATION_TIME] = r
        a = self.apply_policy(o)
        # Save the current movement for other agents to learn from
        self.observation_data[self.time % c.MEMORY_RANGE] = o
        self.movements_data[self.time % c.MEMORY_RANGE] = a
        self.act(a)
        # While agent time <= evaluation time, he is maturing
        if self.time > c.EVALUATION_TIME:
            G = np.sum(self.res)
            self.broadcast(self.observation_data, self.movements_data, G)
            for m in self.message:                              # The agent received at least a message
                # Learning from the message
                self.transfer_function(G, m)
                # Enable / Disable Mutation of the agent
                # self.theta = gaussian_mutation(
                #     self.theta)
                # After any change in its NN, the agent resets its evaluation
                self.time = 0
            if self.time == 0:
                print(
                    f"{self.id} has learned from {' '.join([str(x) for x in self.teachers])}!")
                self.message.clear()
                self.teachers.clear()
        self.time += 1


########################################################################################
