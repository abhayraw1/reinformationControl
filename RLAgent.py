import numpy as np
from Edge import Edge
from PointEnvironment.Pose import Pose
from PointEnvironment.Agent import Agent

class RLAgent(Agent):
  def __init__(self, id, train, pose=Pose(),  \
              defaultPose=False, collisionRadius=0.15, lenTrajectory=100):
    self.type     = "RLAgent"
    self.id       = id
    self.edges    = {}
    self.train    = train
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

  # training is done for only two agents at a time to ensure that averaging
  # velocities don't have any negative imact on training...


  # def act(self):
  #   action = np.zeros(2, 'f').reshape((1,2))
  #   for k in self.edgeControllers.keys():
  #     obs = self.edges[k].state().reshape((1, 3))
  #     action += controller.act(obs, train=self.train)
  #   action = action/len(self.edges)
  #   for k in self.edgeControllers.keys():
  #     next_state, reward, done = partial_env.step(action, k)
  #     if self.train:
  #       controller.replaybuffer.add(obs, action, reward, next_state, done)
  #       controller.train()
