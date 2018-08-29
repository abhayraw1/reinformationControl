import numpy as np
from Utils import *
from DDPG import DDPG
import HyperParams as HP
from FormationEnv import FormationEnvironment
import time
from PointEnvironment.Pose import Pose
from Utils import runExperiment, genTargetShape
import tensorflow as tf
from Critic import CriticNetwork
from keras import backend as K

tmodel = DDPG(None, train=False)
print tmodel.critic
tmodel.actor.model.load_weights('results/E_5650.h5')
visualOptions = { 'tailLength' : 4, # tail of agent's trajectory
                  'speedup' : 100, # realTime/simTime
                  'bounds': [-10,10,-10,10]# bounds of the environment [xmin, xmax, ymin, ymax]
                  }
config = tf.ConfigProto()
sess = tf.Session(config=config)
K.set_session(sess)
criticparams = {'sess': sess,
          'state_size':HP.STATE_DIM,
          'action_size':HP.ACTION_DIM,
          'tau':HP.CRITIC_TAU,
          'lr':HP.CRITIC_LR,
          'target': tmodel.critic}
targetshape = genTargetShape(genRandomPoints())
pts = [(0,2), (-1,0)]
ts = genTargetShape(pts)
print ts
env = FormationEnvironment(ts, num_iterations=20, visualize=True, visualOptions=visualOptions)
common_critic = CriticNetwork(**criticparams)
env.initEgdeModels(common_critic, tmodel)
env.startVisualiser()
env.reset({i:Pose(*pts[i]) for i in range(2)})
print runExperiment(env, 1, HP.EVAL_EPS_LEN*5, tmodel, eval=True, initPose={i:Pose(*pts[i]) for i in range(2)}, targetshape=ts)
