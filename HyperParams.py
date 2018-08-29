import numpy as np


###############################################################################
####################### ENVIRONMENT RELATED CONSTANTS #########################
STATE_DIM       = 4
ACTION_DIM      = 2

NUM_ITERATIONS  = 10
DT              = 0.01

MAX_ACTION      = np.array([0.7 ,  0.7])
MIN_ACTION      = np.array([-.1   , -0.7])

REWARD_SCALE    = 25
REWARD_MAX      = 10*DT*NUM_ITERATIONS*MAX_ACTION[0]
GAMMA           = 0.99
###############################################################################
###############################################################################


###############################################################################
############################# TRAINING CONSTANTS ##############################
BUFFER_SIZE         = 10**6
BATCH_SIZE          = 32
ACTOR_N_NEURONS_L0  = 300
ACTOR_N_NEURONS_L1  = 400
ACTOR_TAU           = 1e-3
ACTOR_LR            = 1e-4
CRITIC_N_NEURONS_L0 = 300
CRITIC_N_NEURONS_L1 = 400
CRITIC_TAU          = 1e-3
CRITIC_LR           = 1e-3
NUM_EPOCS           = 15
###############################################################################
###############################################################################


###############################################################################
###################### REINFORCEMENT LEARNIING CONSTANTS ######################
NUM_EPS         = 10**6
MAX_EPS_LEN     = 50

EVAL_INTERVAL   = 50
NUM_EVALS       = 15
EVAL_EPS_LEN    = 2*MAX_EPS_LEN


OU_MEAN         = [0.3, 0]
OU_THETA        = [0.6, 0.4]
OU_SIGMA        = [0.3, 0.4]
###############################################################################
###############################################################################
