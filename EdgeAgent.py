import numpy as np
from PointEnvironment.Utils import R
from DDPG import DDPG
from PointEnvironment.Agent import Agent

class RLEdgeAgent(object):
  def __init__(self, i, j, target):
    assert isinstance(i, Agent)
    assert isinstance(j, Agent)
    self.i = i
    self.j = j
    self.target = target
    self.update()

  def setModel(self, targetmodel):
    self.model = DDPG(targetmodel, train=True)

  @property
  def state(self):
    dx, dy, phi = self.j.pose - self.i.pose
    d = np.linalg.norm([dx, dy])
    t = (np.matrix([dx, dy])/d)*R(self.i.pose.theta)
    assert np.abs(np.linalg.norm(t) - 1) < 1e-7 , (np.linalg.norm(t), t)
    _state = [d-self.target['d']] + (t-self.target['t']).tolist()[0] + [phi]
    return _state

  def getReward(self):
    scale = np.array([5, 2, 2, 1])
    _reward = -np.array(self.prevState)*(np.array(self.state) - self.prevState)
    print "REWD: ", _reward
    return _reward*scale

  def update(self):
    self.prevState = self.state

  def getAction(self):
    return self.model.act(self.state)

  @property
  def edge_maintained(self):
    scale = np.array([5, 2, 2, 1])
    n = np.linalg.norm(self.state)
    print self.state, sum(self.state)
    criteria = np.abs(self.state[:-1])<0.05
    # print "Edge Maintained for ({},{}) : {} {}".format(self.i.id, self.j.id, criteria.all(), criteria)
    return criteria.all()
