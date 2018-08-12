import numpy as np
from PointEnvironment.Environment import PointEnvironment
from Edge import Edge
import HyperParams as HP
from Shape import ShapeByAgents, Shape

class FormationEnvironment(PointEnvironment):
  def __init__(self, targetshape, agents, num_iterations=100, dt=0.01):
    super(FormationEnvironment, self).__init__(num_iterations, dt, agents)
    self.agentpairs = [(i, j) for i in agents \
                      for j in agents if i.id != j.id]
    assert isinstance(targetshape, Shape)
    self.shape = ShapeByAgents(self.agentpairs)
    self.edges = self.shape.edges
    self.targetshape = targetshape
    self._bindEdgesToAgents()

  def stepAgent(self, agent_id, action):
    action = np.matrix(action)
    self.step({agent_id: action})
    self.updateEdges()

  def updateEdges(self):
    for i in self.edges.values():
      i.update()

  def cost(self, agent_id=None):
    cost = self.shape - self.targetshape
    # print "cost: ", cost
    if agent_id != None:
      print "||{}|| {}".format(agent_id, sum([cost[i, j] for (i,j) in cost.keys() if i == agent_id]))
      return sum([cost[i, j] for (i,j) in cost.keys() if i == agent_id])
    return cost

  def _bindEdgesToAgents(self):
    for i, j in self.edges.keys():
      self.agents[i].addEdge(self.edges[i, j])


class AgentObservedEnvironment:
  def __init__(self, world, agent):
    self.agent    = agent
    self.agent_id = agent.id
    self.world    = world

  def reset():
    self.agent.reset()
    return _getNextState()

  def step(self, action):
    prev_cost = self._getCost()
    self.world.stepAgent(self.agent_id, action)
    # prev_states = {i.j.id:i.state() for i in self.agent.edges.values()}
    reward = self._getAgentReward(prev_cost)
    return self._getNextState(), reward, self._isTerminal()

  def _getNextState(self):
    return {i.j.id:i.state() for i in self.agent.edges.values()}

  def _getCost(self):
    return self.world.cost(self.agent_id)


  def _getAgentReward(self, prev_cost=None):
    # print "total reward: {} [called from AOE for Agent:{}]".format(self.world.reward(), self.agent_id)
    cost = self._getCost()
    return (cost - prev_cost)*HP.REWARD_SCALE
    # return sum([reward[i, j] for (i, j) in reward.keys() if i == self.agent_id])

  def _isTerminal(self):
    return self.world.collisionOccured
