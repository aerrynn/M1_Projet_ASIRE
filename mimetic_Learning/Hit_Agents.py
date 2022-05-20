from pyroborobo import Pyroborobo, Controller, AgentObserver
import numpy as np
from ExtendedAgent import Agent
from Neural_network import NeuralNetwork
import Const as c


class HitAgent(Agent):
    '''
    Simple agent attributes and functions for it to be able to learn from the HIT-EE method
    '''

    def __init__(self, wm) -> None:
        '''
        Const
        '''
        super.__init__(self, wm)
        self.theta = NeuralNetwork(2*self.nb_sensors, 2, c.NB_HIDDENS)
        self.res = [0 for _ in range(c.EVALUATION_TIME)]
        self.observation_data = np.array([None for _ in range(c.MEMORY_RANGE)])
        self.movements_data = np.array([None for _ in range(c.MEMORY_RANGE)])
        # Stores the last received message, empties when read

        self.time = 0

        # We don't want to learn more than once in a step from a single user
        self.teachers = set()

    def step(self) -> None:
        self.hit_algorithm()

    def broadcast(self, obs: np.ndarray, mvm: np.ndarray, score: float) -> None:
        '''
        @override
        '''
        if self.theta == 0:
            return
        super().broadcast(obs, mvm, score)

    def transfer_function(self, G: float, message: tuple) -> bool:
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
            return False
        self.teachers.add(s_ID)
        if G <= s_G:                                                # If the sender has a lower fitness
            return False                                            # score don't do anything
        self.theta.train(s_O, s_M, c.LEARNING_STEPS, c.LEARNING_RATE)
        return True

    def gaussian_mutation(self) -> None:
        '''
        gaussian_mutation : Randomly mutates the agent, following a gaussian distribution
        '''
        self.theta.mutate()

    def hit_algorithm(self) -> None:
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
                if (self.transfer_function(G, m)):
                    # After any change in its NN, the agent resets its evaluation
                    self.time = 0
            if self.time == 0:
                if c.VERBOSE:
                    print(
                        f"{self.id} has learned from {' '.join([str(x) for x in self.teachers])}!"
                    )
                self.gaussian_mutation()
                self.message.clear()
                self.teachers.clear()
        self.time += 1


########################################################################################
