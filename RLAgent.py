import numpy as np
from Edge import Edge
from PointEnvironment.Pose import Pose
from PointEnvironment.Agent import Agent

class RLAgent(Agent):
  def __init__(self, id, pose=Pose(),  \
              defaultPose=False, collisionRadius=0.15, lenTrajectory=100):
    self.type     = "RLAgent"
    self.id       = id
    self.edges    = {}
    self.edgeControllers = {}
    super(RLAgent, self).__init__(id, pose=pose, defaultPose=defaultPose, \
      collisionRadius=collisionRadius, lenTrajectory=lenTrajectory)

  def initEdgeControllers(self, agent_id, controller):
    self.edgeControllers[agent_id] = controller

  def addEdges(self, edges):
    assert type(edges) == list
    for edge in edges:
      self.addEdge(edge)

  def addEdge(self, edge):
    assert isinstance(edge, Edge)
    assert edge.i.id == self.id
    self.edges[edge.j.id] = edge
    self.n_nhbrs  = len(self.edges)

  def train_edge_controller(self, nbr):
    self.edgeControllers[nbr].train_models()
