import numpy as np

from PointEnvironment.Agent import Agent
from PointEnvironment.Utils import R

from Edge import Edge, AgentEdge

class Shape(object):
  def __init__(self, edges):
    self.edges = edges
    self.TYPE  = "SHAPE OBJECT"

  def __str__(self):
    info = self.TYPE
    for i, j in self.edges.items():
        info += "\n{}: {}".format(i, j)
    return info

  def __getitem__(self, index):
    return self.edges[index]

  def __sub__(self, other):
    cost = {}
    for i,j in self.edges.keys():
      cost[i,j] = (self.edges[i,j]-other.edges[i,j])
    return cost

class ShapeByAgents(Shape):
  def __init__(self, agent_pairs):
    self.agent_pairs = agent_pairs
    self.TYPE = "SHAPE OBJECT BY AGENTS"
    edges = {}
    for i, j in self.agent_pairs:
      edges[i.id, j.id] = AgentEdge((i,j))
    super(ShapeByAgents, self).__init__(edges)
    self.update()

  def update(self):
    for i,j in self.agent_pairs:
      self.edges[i.id, j.id].update()

class ShapeByGeometry(Shape):
  def __init__(self, geometry):
    self.type = "SHAPE OBJECT BY GEOMETRY"
    c = geometry['coordinates']
    r = geometry['orientations']
    edges = {}
    for i in xrange(len(c)):
      for j in xrange(len(c)):
        if i == j:
          continue
        d     = np.linalg.norm(c[j] - c[i])
        t_ij  = np.matrix((c[j]-c[i]))
        theta = np.arctan2(*t_ij.tolist()[0][::-1])
        edges[i, j] = Edge(d, theta, r[i])
    super(ShapeByGeometry, self).__init__(edges)
