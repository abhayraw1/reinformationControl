import numpy as np
from PointEnvironment.Agent import Agent

class MobileAgent(Agent):
  def __init__(self, id, leader, ltn_ratio=2, **kwargs):
    self.leader       = leader
    self.ltn_ratio    = ltn_ratio
    if kwargs == None: kwargs = {}
    super(MobileAgent, self).__init__(id, **kwargs)

  def initEdgeAgents(self, edge_agents):
    self.edge_agents  = edge_agents
    self.n_neighbours = len(edge_agents)

  def initEdgeAgentModel(self, targetmodel):
    for i in self.edge_agents:
      i.setModel(targetmodel)

  def getFinalAction(self, explore=False):
    if self.id == self.leader:
      print "I am leader. I should be following a trajectory."
      return np.array([0., 0.])
    else:
      action = np.array([0, 0], 'f')
      for e in self.edge_agents:
        x = int(self.leader == e.j.id)
        action += (x*(self.ltn_ratio - 1) + 1)*e.getAction()
      #exploration handled in DDPG as of now
      # if explore:
      #   raise NotImplementedError, "Implement exploration noise"
      #   pass
      return action/ n_neighbours
