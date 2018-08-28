import numpy as np
from Utils import *
from DDPG import DDPG
import HyperParams as HP
from FormationEnv import FormationEnvironment
import time
from PointEnvironment.Pose import Pose
from Utils import runExperiment, genTargetShape

tmodel = DDPG(train=False)
tmodel.actor.model.load_weights('results/E_44450.h5')
visualOptions = { 'tailLength' : 4, # tail of agent's trajectory
                  'speedup' : 100, # realTime/simTime
                  'bounds': [-10,10,-10,10]# bounds of the environment [xmin, xmax, ymin, ymax]
                  }
targetshape = genTargetShape(genRandomPoints())
pts = [(0,2), (-1,0), (1,0)]
ts = genTargetShape(pts)
print ts
env = FormationEnvironment(ts, num_iterations=50, visualize=True, visualOptions=visualOptions)
for i in env.agents.values():
  for e in i.edge_agents:
    e.model = tmodel
  i.initEdgeAgentModel(tmodel)
env.startVisualiser()
env.reset({i:Pose(*pts[i]) for i in range(3)})
runExperiment(env, 1, 10000, tmodel, eval=True, initPose={i:Pose(*pts[i]) for i in range(3)})
