import numpy as np
from numpy import cos, sin
from DDPG import DDPG
import HyperParams as HP
from RLAgent import RLAgent
from Shape import ShapeByGeometry
from PointEnvironment.Pose import Pose
from RLEnv import FormationEnvironment, AgentObservedEnvironment

############### CONSTANTS ###############
NUM_AGENTS = 2
#########################################


######## TARGET SHAPE DEFINITION ########
geometry = {'coordinates': [[0,0], [2,0]],
            'orientations':[np.pi/2]*2 }
geometry['coordinates'] = map(np.array, geometry['coordinates'])

targetshape = ShapeByGeometry(geometry)
#########################################

def rnd(x = 5):
  return np.random.random()*x-x/2.

def randomtarget():
  geometry = {'coordinates': [[rnd(),rnd()], [rnd(),rnd()]],
              'orientations':[rnd(2*np.pi)]*2 }
  geometry['coordinates'] = map(np.array, geometry['coordinates'])

  return ShapeByGeometry(geometry)

############### AGENTS
agents_pose = [Pose(0,0), Pose(0, -2, -np.pi/2)]
agents = [RLAgent(i, pose=agents_pose[i]) for i in range(NUM_AGENTS)]
eval_agents = [RLAgent(i, pose=agents_pose[i]) for i in range(NUM_AGENTS)]

############### ENVIRONMENT
env = FormationEnvironment(targetshape, agents, num_iterations=HP.NUM_ITERATIONS, dt=HP.DT)
eval_env = FormationEnvironment(targetshape, eval_agents, num_iterations=HP.NUM_ITERATIONS, dt=HP.DT)

############### PARTIALLY OBSERVED ENVS
agent_observed_envs = {}
for agent in env.agents.values():
  agent_observed_envs[agent.id] = AgentObservedEnvironment(env, agent)

eval_agent_observed_envs = {}
for agent in eval_env.agents.values():
  eval_agent_observed_envs[agent.id] = AgentObservedEnvironment(env, agent)

############### MAIN TARGET MODEL
main_model = DDPG(train=False)
print "*"*80
print "MAINMODEL: ", main_model
############### INIT AGENT EDGES
for i, j in env.agentpairs:
  i.initEdgeControllers(j.id, DDPG(target_model=main_model))


for i, j in eval_env.agentpairs:
  i.initEdgeControllers(j.id, main_model)
print "*"*80
for i, j in env.agentpairs:
  print i.edgeControllers[j.id].target_model

for i, j in eval_env.agentpairs:
  print i.edgeControllers[j.id].target_model

############### GEN RANDOM POSE FOR AGENTS
def randPose():
  return Pose(*(np.random.random(3)*k - k/2.).tolist())
############### SAVE MODEL
def savemodel(eps):
  main_model.actor.model.save('results/model/actor/E_{}.h5'.format(eps))
  main_model.critic.model.save('results/model/critic/E_{}.h5'.format(eps))

############### SAVE STATS
def write_stats(x, y, file):
  f = open(file, 'a+')
  f.write('EPS: {}\nR {}\n\n'.format(x, y))
  f.close()

############### EVAL FUNCTION
def evaluate(num_times, eps_len, eps):
  trewards = []
  for i in xrange(num_times):
    print "EVAL RUN {}".format(i)
    r = {i.id:0 for i in eval_agents}
    terminal = []
    collision = False
    eval_env.targetshape = randomtarget()
    eval_env.reset({x.id:randPose() for x in eval_agents})
    for aoe in eval_agent_observed_envs.values():
      aoe.reset()
    for j in xrange(eps_len):
      for agent_id, aoe in eval_agent_observed_envs.items():
        for nbr, state in aoe.current_st.items():
          action, next_state = aoe.step(nbr, state)
          reward, done, reason = aoe.getAgentReward()
          r[agent_id] += reward
          if done and reason == 'c':
            collision = True
            break
          terminal.append(done)
        if collision:
          break
    trewards.append(sum(r.values()))
    print "--REWARD: ", sum(r.values())
  ttrr = np.mean(trewards)
  print "EVAL MEAN SCORE: ", ttrr
  write_stats(eps, ttrr, 'results/mean_scores')
  return ttrr


eps = 1
done = False
k = np.array([10, 10, np.pi])
best_mean_score = 0
while eps <= HP.NUM_EPS:
  print "EPISODE : {}".format(eps)
  t = 1
  r = {i.id:0 for i in agents}
  terminal = []
  collision = False
  env.targetshape = randomtarget()
  env.reset({x.id:randPose() for x in agents})
  for aoe in agent_observed_envs.values():
    aoe.reset()
  more_steps = int(np.random.random()*eps/20)
  while t <= (HP.MAX_EPS_LEN + more_steps):
    for agent_id, aoe in agent_observed_envs.items():
      for nbr, state in aoe.current_st.items():
        action, next_state = aoe.step(nbr, state)
        reward, done, reason = aoe.getAgentReward()
        r[agent_id] += reward
        aoe.remember(nbr, [state, action, reward, next_state[nbr], done])
        aoe.agent.train_edge_controller(nbr)
        if done and reason == 'c':
          collision = True
          break
        terminal.append(done)
      if collision:
        break
      if np.array(terminal).all():
        print "--------------------FULL FORMATION FORMED--------------------"
        print env.targetshape
        print env.shape
    t += 1
  write_stats(eps, r, 'results/scores')
  print "REWARD: ", sum(r.values())
  if (eps % HP.EVAL_INTERVAL) == 0:
    eval_score = evaluate(HP.NUM_EVALS, HP.EVAL_EPS_LEN, eps)
    if best_mean_score < eval_score:
      savemodel(eps)
      best_mean_score = eval_score

  for aoe in agent_observed_envs.values():
    for nbr in aoe.current_st.keys():
      for epoch in range(HP.NUM_EPOCS):
        aoe.agent.train_edge_controller(nbr)
  eps += 1
