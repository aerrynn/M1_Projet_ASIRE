from AdaptativeLearningRate import *


########################################################################################
#                                    Global                                            #
########################################################################################

VERBOSE = False                                                                 # Bool
DATA_SAVE = True                                                                # Bool
SAVE_FILE = 'NewData/Disc'
OVERWRITE_FILE = False
########################################################################################
#                                ExtendedAgents                                        #
########################################################################################

NB_HIDDENS = 10
EVALUATION_TIME = 600
MEMORY_RANGE = 20
LEARNING_STEPS = 30
LEARNING_RATE = 0.8
MUTATION_RATE = 0.
########################################################################################
#                                 Foraging Task                                        #
########################################################################################

LEARNING_ALGORITHM = "adhoc"                 # "adhoc" or "neural"

NB_ITEMS = 100
REGROWTH_TIME = 10
CHANGE_POSITION = True
NB_ITER = 200000
########################################################################################
#                                 NeuralLearner                                        #
########################################################################################

PROPAGATION = False
DECAY_FUNCTION = constantLearningRate
DECAY_RATIO = 0.01
########################################################################################
#                                 MemoryAgents                                         #
########################################################################################

# between 0 and 1
EXPERT_SPEED = 1
LEARNING_GAP = 60
MEMORY_SIZE = 100
NB_LEARNER = 90
DISCRETISE_RATIO = 2                                   # if -1 : closest learnt
LEARNT_BEHAVIOUR_PROPAGATION = False
########################################################################################
