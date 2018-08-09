from PointEnvironment.Environment import PointEnvironment
from Edge import Edge
from Shape import ShapeByAgents, Shape

class FormationEnvironment(PointEnvironment):
  def __init__(self, targetshape, num_iterations=100, dt=0.01, agents={}):
    super(FormationEnvironment, self).__init__(num_iterations, dt, agents)
    agentpairs = [(i, j) for i in self.agents \
                      for j in self.agents if i.id != j.id]
    assert isinstance(targetshape, Shape)
    self.shape = ShapeByAgents(agentpairs)
    self.edges = self.shape.edges
    self.targetshape = targetshape

  def stepAgent(self, action, agent_id):
    action = np.matrix(action)
    self.step({agent_id: action})
    self.updateEdges()

  def updateEdges(self):
    for i in self.edges.values():
      i.update()

  def reward(self):
    return self.shape - targetshape


class AgentObservedEnvironment:
  def __init__(self, world, agent, shape)
    self.agent    = agent
    self.agent_id = agent.id
    self.world    = world

  def step(self, action, edge_pair):
    self.world.stepAgent({self.id: action})
    return _getNextState(edge_pair), _getAgentReward(), isTerminal()

  def _getNextState(self, pair):
    assert pair[0] = self.agent_id
    return self.edges[pair].state()

  def _getAgentReward(self):
    reward = 0
    for k, v in world.reward().values():
      if k == self.agent_id:
        reward += v
    return reward

  def _isTerminal(self):
    return world.collisionOccured
