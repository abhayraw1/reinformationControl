from PointEnvironment.Environment import PointEnvironment
from Edge import Edge
from Shape import ShapeByAgents, Shape

class FormationEnvironment(PointEnvironment):
  def __init__(self, targetshape, agents, num_iterations=100, dt=0.01):
    super(FormationEnvironment, self).__init__(num_iterations, dt, agents)
    agentpairs = [(i, j) for i in self.agents \
                      for j in self.agents if i.id != j.id]
    assert isinstance(targetshape, Shape)
    self.shape = ShapeByAgents(agentpairs)
    self.edges = self.shape.edges
    self.targetshape = targetshape
    self._bindEdgesToAgents()

  def stepAgent(self, action, agent_id):
    action = np.matrix(action)
    self.step({agent_id: action})
    self.updateEdges()

  def updateEdges(self):
    for i in self.edges.values():
      i.update()

  def reward(self):
    return self.shape - targetshape

  def _bindEdgesToAgents(self):
    for i, j in self.edges.keys():
      self.agents[i].addEdge(self.edges[i, j])


class AgentObservedEnvironment:
  def __init__(self, world, agent)
    self.agent    = agent
    self.agent_id = agent.id
    self.world    = world

  def reset():
    self.agent.reset()
    return _getNextState()

  def step(self, action):
    self.world.stepAgent({self.id: action})
    return _getNextState(), _getAgentReward(), isTerminal()

  def _getNextState(self):
    return {i:i.state() for i in self.agent.edges.values()}

  def _getAgentReward(self):
    reward = 0
    for k, v in world.reward().values():
      if k == self.agent_id:
        reward += v
    return reward

  def _isTerminal(self):
    return world.collisionOccured
