from pyroborobo import Pyroborobo, Controller, AgentObserver
import numpy as np

'''
TODO: Increase convergence rate, but how ?!
'''

######################################## CONSTS ########################################
NB_HIDDENS = 10
EVALUATION_TIME = 100
ALPHA = 0.5
policy_size = 3*8 + 1 + NB_HIDDENS  # To set up
zero_to_m = list(range(policy_size))
########################################################################################

# Tracking tool
best_picker = 0



class Agent(Controller):
    '''
    Simple agent attributes and functions for it to be able to learn from the HIT-EE method
    '''

    def __init__(self, wm) -> None:
        Controller.__init__(self, wm)
        self.nb_hiddens = NB_HIDDENS
        self.theta = [np.random.normal(0, 1, (3*self.nb_sensors + 1, self.nb_hiddens)),
                      np.random.normal(0, 1, (self.nb_hiddens, 2))]
        self.res = [0 for _ in range(EVALUATION_TIME)]
        # Stores the last received message, empties when read
        self.message = []
        self.rob = Pyroborobo.get()
        self.time = 0

    def reset(self):
        pass

    def step(self):
        self.hit_algorithm()

    def sense(self):
        '''
        sense : get the data from the sensors of the agent
            :return data: the data retrieved, for each sensorn creates 3 inputs:
                - distance of a detected obstacle
                - distance of a detected robot
                - distance of a detected wall
            :return fitness: the agent's score
        '''
        data = self.get_all_distances()
        data_plus = []
        for i, v in enumerate(data):
            data_plus.append(v)                     # Add the pos of the closest obstacle
            if self.get_robot_id_at(i) != -1:       # Add the pos of the closest robot
                data_plus.append(v)
            else : data_plus.append(1)
            if self.get_wall_at(i):                 # Add the pos of the closest wall
                data_plus.append(v)
            else:
                data_plus.append(1)
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

    def broadcast(self, idx, score):
        '''
        broadcast sends a message containing theta, idx and score to nearby agents
            :param idx: The indexes of theta to teach
            :param score: The fitness to send
        '''
        if self.theta == 0:
            return
        for i in range(self.nb_sensors):
            rob_id = self.get_robot_id_at(i)
            self.rob.controllers[rob_id].message += [(self.theta, idx, score)]

    def hit_algorithm(self):
        '''
        hit_algorithm : applies the hit learning algorithm to the current agent
            alpha: transfert rate [0,1]
            evaluation_time: amount of steps evaluated (T in the paper)
            param policy_function: policy followed by the agent (π in the paper)
            self.theta = agent initialisation of policy
            policy_size = |self.theta|
            o = observation vector
            r = reward scalar
            self.res = reward buffer of size T 
            a = action vector
            G = personal evaluation (sum of r on the whole evaluation time)
        '''
        global best_picker
        o, r = self.sense()
        self.res[self.time % EVALUATION_TIME] = r
        a = policy_function(o, self.theta)
        self.act(a)
        # While agent time <= evaluation time, he is maturing
        if self.time > EVALUATION_TIME:
            G = np.sum(self.res)
            if G > best_picker :
                best_picker = G
                print(self.id, G)
            # random_pick = int(ALPHA * np.random.randint(0, policy_size))
            random_pick = int(ALPHA * policy_size)
            idx = np.random.choice(zero_to_m, random_pick, False)
            # self.broadcast(self.theta[idx], idx, G)           ## paper version
            self.broadcast(idx, G)
            for m in self.message:                              # The agent received at least a message
                self.theta = transfer_function(                 # Learning from the message
                    self.theta, G, m)
                # Enable / Disable Mutation of the agent
                # self.theta = gaussian_mutation(
                #     self.theta)                                 
                # After a mutation, the agent reset its evaluation
                self.time = 0
                self.message = []
        self.time += 1


########################################################################################

def policy_function(observations, theta):
    '''
    policy_function : Deterministic policy π_θ to compute the action vector 
        :param observation: the result vector of the observation
        :param theta: the policy
        :return a: an action vector
    '''
    # HACK: Change ? Or at least understand ... , this is a simple copy of wander_evolution
    # print((observations.shape))
    out = np.concatenate([[1], observations])
    for elem in theta[:-1]:
        out = np.tanh(out @ elem)
    out = out @ theta[-1]  # linear output for last layer
    return np.clip(out, -1, 1)


def transfer_function(theta, G, message):
    '''
    transfer_function : If the sender of the message has a higher fitness score
    replace the theta[idx] of the receiver by the theta[idx] of the sender
        :param theta: The receiver agent policy
        :param G: The fitness score of the receiver
        :param message: the message to learn from
            s_theta: the sender's policy
            s_idx: the id of the policy to learn from
            s_G: the sender's fitness score
        :return theta: The new policy
    '''
    # print(message)
    s_theta, s_idx, s_G = message
    # print(s_idx)
    if G <= s_G:                                                # If the sender has a lower fitness
        return theta                                            # score don't do anything
    # print(f"EVOLUTION! {s_G}")
    m = len(theta[0])
    for i in s_idx:
        i1 = i//m
        i2 = i%m
        theta[i1][i2] = s_theta[i1][i2]
    return theta


def gaussian_mutation(theta):
    '''
    gaussian_mutation : Randomly mutates the agent, following a gaussian distribution
        :param theta: the agent policy
        :return theta: the mutated policy
    TODO: Check that the += doesn't creates bugs albeit it reduces memory usage
    '''
    for layer in range(len(theta)):
        for i in range(len(theta[layer])):
            theta[layer][i] += np.random.normal()
    return theta
