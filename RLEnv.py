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
    if agent_id != None:
      return sum([cost[i, j] for (i,j) in cost.keys() if i == agent_id])
    return cost

  def _bindEdgesToAgents(self):
    for i, j in self.edges.keys():
      self.agents[i].addEdge(self.edges[i, j])


class AgentObservedEnvironment:
  def __init__(self, world, agent):
    self.agent      = agent
    self.agent_id   = agent.id
    self.world      = world
    self.prev_cost  = 0
    self.current_st = None

  def reset(self):
    self.current_st = self._getNextState()
    return self.current_st

  def _step(self, action):
    self.prev_cost = self._getCost()
    self.world.stepAgent(self.agent_id, action)
    self.world.updateEdges()
    self.current_st = self._getNextState()
    return self.current_st

  def _getNextState(self):
    return {i.j.id:i.state(self.world.targetshape) for i in self.agent.edges.values()}

  def _getCost(self):
    return self.world.cost(self.agent_id)

  def getAgentReward(self):
    done = self._isTerminal()
    cost = self._getCost()
    if abs(cost) < 0.1:
      print "---------- FORMATION FORMED ----------", cost
      return 100, True
    if done:
      return -100, True
    reward = (cost - self.prev_cost)*HP.REWARD_SCALE
    self.prev_cost = cost
    return reward, done

  def _isTerminal(self):
    return self.world.collisionOccured

  def step(self, nbr, state):
    action = self.agent.edgeControllers[nbr].act(state)
    next_state = self._step(action)
    return action.reshape((HP.ACTION_DIM)), next_state

  def remember(self, nbr, memory):
    self.agent.edgeControllers[nbr].remember(*memory)
