from PointEnvironment.Environment import PointEnvironment

class FormationEnvironment(PointEnvironment):
  def __init__(self, num_iterations=100, dt=0.01, agents={}):
    super(FormationEnvironment, self).__init__(num_iterations, dt, agents)

  def stepAgent(self, action, agent_id):
    action = np.matrix(action)
    self.step({agent_id: action})

class AgentObservedEnvironment:
  def __init__(self, world, agent):
    self.agent  = agent
    self.id     = agent.id
    self.world  = world

  def step(self, action):
    self.world.stepAgent({self.id: action})
    return _getNextState(), _getAgentReward(), isTerminal()

  def _getNextState(self):

    pass

  def _getAgentReward(self):
    pass

  def _isTerminal(self):
    return world.collisionOccured
