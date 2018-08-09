import numpy as np
from PointEnvironment.Pose import Pose
from PointEnvironment.Agent import Agent

class RLAgent(Agent):
  def __init__(self, edges, partial_env, train, pose=Pose(),  \
              defaultPose=False, collisionRadius=0.15, lenTrajectory=100):
    self.type     = "RLAgent"
    assert type(edges) = dict
    self.edges    = edges
    self.env      = partial_env
    self.n_nhbrs  = len(self.egdes)
    super(RLAgent, self).__init__(agentpair)

  def initEdgeControllers(self):
    self.edgeControllers = {}
    for edge in self.edges.keys():
      self.edgeControllers[edge] = DDPG("""Pass args here!!""")

  # training is done for only two agents at a time to ensure that averaging
  # velocities don't have any negative imact on training...
  def act(self):
    action = np.zeros(2, 'f').reshape((1,2))
    for k in self.edgeControllers.keys():
      obs = self.edges[k].state().reshape((1, 3))
      action += controller.act(obs, train=self.train)
    action = action/len(self.edges)
    for k in self.edgeControllers.keys():
      next_state = partial_env.step(action)
      reward, done = partial_env.getReward()
      controller.replaybuffer.add(obs, action, reward, next_state, done)
