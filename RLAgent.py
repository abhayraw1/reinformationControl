from Edge import AgentEdge

class RLAgent(AgentEdge):
  def __init__(self, id, agentpair, model):
    self.id    =
    self.model = model
    super(RLAgent, self).__init__(agentpair)

  def act_and_train(self, obs):
    # returns action performed by agent based on observation according to model
    pass

  def act(self, obs):
    pass

  def stop_episode(self, obs):
    pass
