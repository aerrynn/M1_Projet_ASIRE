from AdaptativeLearningRate import *


########################################################################################
#                                    Global                                            #
########################################################################################

# VERBOSE = True
VERBOSE = False
DATA_SAVE = True
# DATA_SAVE = False
SAVE_FILE = 'SavedData/data_memory'
OVERWRITE_FILE = False
########################################################################################
#                                   HitAgents                                          #
########################################################################################

NB_HIDDENS = 10
EVALUATION_TIME = 600
MEMORY_RANGE = 20
LEARNING_STEPS = 30
LEARNING_RATE = 0.8
MUTATION_RATE = 0.
########################################################################################
#                                 NeuralLearner                                        #
########################################################################################

PROPAGATION = False
DECAY_FUNCTION = constantLearningRate
DECAY_RATIO = 0.01
########################################################################################
#                                 Foraging Task                                        #
########################################################################################

# Carrier_Agents
MAX_CAPACITY = 1

# BushNode
NB_ITEMS = 100
REGROWTH_TIME = 10
CHANGE_POSITION = True
MAX_RESSOURCE_LEVEL = 1
NB_ITER = 100000
########################################################################################
#                                 MemoryAgents                                         #
########################################################################################

# between 0 and 1
EXPERT_SPEED = 1
LEARNING_GAP = 60
MEMORY_SIZE = 100
NB_LEARNER = 24
LEARNT_BEHAVIOUR_PROPAGATION = False
########################################################################################
