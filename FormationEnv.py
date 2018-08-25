import numpy as np
from PointEnvironment.Pose import Pose
from PointEnvironment.Environment import PointEnvironment
from EdgeAgent import RLEdgeAgent
from MobileAgent import MobileAgent
from DDPG import DDPG
import HyperParams as HP

class FormationEnvironment(PointEnvironment):
  def __init__(self, targetShape, **kwargs):
    self.targetShape = targetShape
    self.LEADER      = 0
    self.genMobileAgents()
    self.populateEdgeAgentsInMobileAgents()
    super(FormationEnvironment, self).__init__(**kwargs)
    super(FormationEnvironment, self).addAgents(self.mobile_agents.values())

  def reset(self, poses=None):
    if poses == None: poses = {}
    super(FormationEnvironment, self).reset(poses)
    return

  def initEgdeModels(self, targetmodel):
    for agent in self.agents.values():
      agent.initEdgeAgentModel(targetmodel)

  def stepLeader(self, action):
    assert type(action) == list
    super(FormationEnvironment, self).step({self.LEADER:action})

  def step(self, actions=None):
    assert self.LEADER not in actions.keys(), "Call stepLeader to move leader"
    if actions == None: actions = {}
    super(FormationEnvironment, self).step(actions)
    experiences = {}
    for agent in self.agents.values():
      for e in agent.edge_agents:
        r = e.getReward()*HP.REWARD_SCALE
        r = sum(r)
        done = self.collisionOccured and agent.id in self.collisionbetween
        if done:
          r = -HP.REWARD_MAX*HP.REWARD_SCALE
        elif e.edge_maintained:
          r = HP.REWARD_MAX*HP.REWARD_SCALE
        if agent.id not in actions.keys(): actions[agent.id] = np.array([0.,0.])
        experiences[e.i.id, e.j.id] = [np.array(e.prevState), actions[agent.id], r, np.array(e.state), done]
        e.update()
    return experiences

  def genMobileAgents(self):
    ids = []
    for i, j in self.targetShape.keys():
      if i not in ids: ids.append(i)
      if j not in ids: ids.append(i)
    self.mobile_agents = {}
    for i in ids:
      self.mobile_agents[i] = MobileAgent(i, self.LEADER, pose=Pose(i))

  def populateEdgeAgentsInMobileAgents(self):
    for i in self.mobile_agents.values():
      edges = []
      for j, k in self.targetShape.items():
        if j[0] == i.id:
          edges.append(RLEdgeAgent(i, self.mobile_agents[j[1]], k))
      i.initEdgeAgents(edges)

if __name__ == '__main__':
  targetmodel = DDPG(train=False)
  targetshape = {(0,1): {'d': 2, 't': [0,-1]}, (1,0): {'d': 2, 't': [0, 1]}}
  visualOptions = { 'tailLength' : 4, # tail of agent's trajectory
                    'speedup' : 1, # realTime/simTime
                    'bounds': [-10,10,-10,10]# bounds of the environment [xmin, xmax, ymin, ymax]
                    }
  # options = { 'num_iterations' : 50, # no. of steps environment takes for one step call
  #             'dt' : 0.01, # dt in in kinametic update
  #             'visualize' : True, # show visualization?
  #             'visualOptions' : visualOptions # visual options
  #             }
  a = FormationEnvironment(targetshape, num_iterations=50, visualize=True, visualOptions=visualOptions)
  # for i in a.mobile_agents.values():
  #   i.initEdgeAgentModel(targetmodel)
  #   print [(k.model.target_model, k.model) for k in i.edge_agents]
  # print targetmodel
  a.startVisualiser()
  # a.reset()
  for i in a.agents.values():
    print i.pose
  from testmtg import move_to_pose
  print a.agents[0].edge_agents[0].state
  a.reset({0:Pose(-2, 0, 0), 1:Pose(2)})
  nn = ("\n"+"**"*40)*2
  f = open('testenv2.txt', "w+")
  s  = ''
  # done = False
  step = 0
  while step != 50:
    # print nn
    # print [j.pose for j in a.agents.values()]
    p = a.agents[1].pose.tolist() + [0, 5, 0]
    ax = move_to_pose(*p)
    print ax
    ex = a.step({1: [-1, 0]})
    for i, j in ex.items():
      # if i == (1,0): continue
      s = "\nFor EDGE {}:\n--STATE  :{}\n--ACTION :{}\n--REWARD :{}\n--NSTATE :{}\n--DONE   :{}\n".\
      format(i, *j)
      step += 1
      f.write(s+nn)
    step += 1

  # print HP.REWARD_MAX
    # s = raw_input()
