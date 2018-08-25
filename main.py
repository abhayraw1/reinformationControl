import numpy as np
from Utils import *
from DDPG import DDPG
import HyperParams as HP
from FormationEnv import FormationEnvironment
import time
from PointEnvironment.Pose import Pose

def runExperiment():
  pass

targetmodel = DDPG(train=False)
print "TARGET MODEL: ",targetmodel

visualOptions = { 'tailLength' : 4,
                  'speedup' : 20,
                  'bounds': [-10,10,-10,10]
                  }

targetshape = genTargetShape()
a = FormationEnvironment(targetshape, num_iterations=50, \
                         visualize=True, visualOptions=visualOptions)
a.initEgdeModels(targetmodel)
a.startVisualiser()
# time.sleep(3)
a.reset({k: Pose(k,k) for k in a.agents})

