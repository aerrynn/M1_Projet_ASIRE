from pyroborobo import Pyroborobo, Controller, AgentObserver
import numpy as np

'''
    Questions :
    - Un agent peut il recevoir un message durant sa phase de maturation ? 
    - Confirmer la manière dont marchent les signaux sur Roborobo
    - Devrait-on stocker plus d'un message ?
    - Confirmer / Réfuter l'implémentation de la fonction de stratégie
    - Quel était cet objet de roborobo qui permettait à tous les robots de connaitre sa position
'''

######################################## CONSTS ########################################
NB_HIDDENS = 10
EVALUATION_TIME = 1000
ALPHA = 0.5
policy_size = 10  # To set up
zero_to_m = list(range(policy_size))
########################################################################################


class Agent(Controller):
    '''
    Simple agent attributes and functions for it to be able to learn from the HIT-EE method
    '''

    def __init__(self, wm) -> None:
        Controller.__init__(self, wm)
        self.nb_hiddens = NB_HIDDENS
        self.theta = [np.random.normal(0, 1, (self.nb_sensors + 1, self.nb_hiddens)),
                      np.random.normal(0, 1, (self.nb_hiddens, 2))]
        self.res = [0 for _ in range(EVALUATION_TIME)]
        # Stores the last received message, empties when read
        self.message = None

    def reset(self):
        pass

    def step(self):
        self.hit_algorithm()

    def sense(self):
        '''
        sense : get the data from the sensors of the agent
            :return data: the data retrieved
            :return fitness: the agent's score
        '''
        data = self.get_all_distances()
        fitness = self.fitness(data)  # TODO: Create a function to compute this...
        return data, fitness

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
        # TODO: Confirm the way signals work
        '''
        broadcast sends a message containing theta, idx and score to nearby agents
            :param idx: The indexes of theta to teach
            :param score: The fitness to send
        '''
        for i in range(self.nb_sensors):
            rob = self.get_robot_id_at(i)
            rob.message = (self.theta, idx, score)

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
        o, r = self.sense()
        self.res[self.time % EVALUATION_TIME] = r
        a = policy_function(o, self.theta)
        self.act(a)
        # While agent time <= evaluation time, he is maturing
        if self.time > EVALUATION_TIME:
            G = np.sum(self.res)
            random_pick = int(ALPHA * np.random.randint(0, policy_size))
            idx = np.random.choice(zero_to_m, random_pick, False)
            # self.broadcast(self.theta[idx], idx, G)           ## paper version
            self.broadcast(self.theta, idx, G)
            if self.message:                                    # The agent received a message
                self.theta = transfer_function(                 # Learning from the message
                    self.theta, idx, G, self.message)
                self.theta = gaussian_mutation(
                    self.theta)                                 # Mutation of the agent
                # After a mutation, the agent reset its evaluation
                t = 0
                self.message = None
        t += 1


########################################################################################

def policy_function(observations, theta):
    '''
    policy_function : Deterministic policy π_θ to compute the action vector 
        :param observation: the result vector of the observation
        :param theta: the policy
        :return a: an action vector
    '''
    # HACK: Change, this is a simple copy of wander_evolution
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
    s_G, s_idx, s_theta = message
    if G <= s_G:                                                # If the sender has a lower fitness
        return                                                  # score don't do anything
    for i in s_idx:
        theta[i] = s_theta[i]
    return theta


def gaussian_mutation(theta):
    '''
    gaussian_mutation : Randomly mutates the agent, following a gaussian distribution
        :param theta: the agent policy
        :return theta: the mutated policy
    TODO: Check that the += doesn't creates bugs
    '''
    for i in range(len(theta)):
        theta[i] += np.random.normal()
    return theta
