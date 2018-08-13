import numpy as np
import HyperParams as HP
from numpy.linalg import norm
from PointEnvironment.Utils import rreplace, wrap_angle, R

class Edge(object):
  def __init__(self, d, theta, r):
    self.TYPE   = "EDGE"
    self.d      = d
    self.t_ij   = np.matrix([np.cos(theta), np.sin(theta)])*R(r)
    self.theta  = np.arctan2(*self.t_ij.tolist()[0][::-1])

  def __repr__(self):
    info = self.TYPE + "\tr: {}, theta: {}, t: {}".format(self.d, self.theta, self.t_ij)
    return info

  def __sub__(self, other):
    ## Reward function not good :(
    error = self.d*self.t_ij - other.d*other.t_ij
    n = -norm(error)
    return n
    # return n*np.cos(np.arctan2(*error.tolist()[0][::-1])/2)*HP.REWARD_SCALE


class AgentEdge(Edge):
  def __init__(self, agent_pair):
    self.i, self.j = agent_pair
    self.ij = (self.i.id, self.j.id)
    super(AgentEdge, self).__init__(0, 0, 0)
    self.update()

  def update(self):
    dx, dy, self.r_ij = self.j.pose - self.i.pose
    self.d      = np.linalg.norm([dx, dy])
    self.r      = self.i.pose.theta
    self.t_ij   = np.matrix([dx, dy])*R(self.r)/self.d
    self.theta  = np.arctan2(dy, dx)

  def __repr__(self):
    info = super(AgentEdge, self).__repr__()
    return info +", r_ij: {}".format(self.r_ij)

  def state(self, targetshape):
    target = targetshape.edges[self.ij]
    targetstate = np.array([target.d, target.theta/np.pi, 0])
    state = np.array([self.d, self.theta/np.pi, self.r_ij/np.pi])
    return state - targetstate
