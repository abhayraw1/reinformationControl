import numpy as np
from PointEnvironment.Utils import R
from FormationEnv import FormationEnvironment
from DDPG import DDPG

def genRandomPoints(seed=None):
  if seed is not None: np.random.seed(seed)
  pts = [(0,0)]
  r, th = np.random.random(2)
  r = r * 1.3 + 0.7
  th = th * np.pi + .5*np.pi
  x, y = r * np.array([np.cos(th), np.sin(th)])
  pts.append((x,y))
  r, th = np.random.random(2)
  r = r * 1.3 + 0.7
  th = th * np.pi - .5*np.pi
  x, y = r * np.array([np.cos(th), np.sin(th)])
  pts.append((x,y))
  return pts

def genTargetShape(pts=None, seed=None):
  if pts == None: pts = genRandomPoints(seed)
  pts = map(np.array, pts)
  ts  = {}
  r   = np.random.random()*np.pi
  print r
  for i, j in enumerate(pts):
    for k, l in enumerate(pts):
      if i == k: continue
      d = np.linalg.norm(l-j)
      t = (np.matrix(l-j)/d)*R(r)
      ts[i, k] = {'d': d, 't': t.tolist()[0]}
  return ts

def saveModel(model, eps, path):
  model.actor.model.save('{}/{}/E_{}.h5'.format(path, 'actor', eps))
  model.critic.model.save('{}/{}/E_{}.h5'.format(path, 'critic', eps))

def writeStats(filename, stat):
  with open(filename, 'a+') as f:
    f.write(stat)
    f.write('\n')

def runExperiment(env, num_eps, eps_len, model, num_epochs=1, eval=False, \
                  num_evals=10, eval_interval=50, eval_eps_len=None, result_path=''):
  mode = "eval" if eval else "training"
  print "hola! in {} mode".format(mode)
  episode     = 1
  best_score  = 0
  if eval: cumulative_reward = 0
  if eval and eval_eps_len is None: eval_eps_len = eps_len
  while episode <= num_eps:
    print "episode ", episode
    done = False
    t = 1
    r = 0
    env.targetshape = genTargetShape()
    env.reset() # random starting positions??
    while t <= eps_len and not done:
      # print "t", t
      if ((t-1) % (eps_len/2)) == 0:
        randomLeaderR = np.random.random()+1.5
        randomLeaderV = np.random.random()*0.2 + 0.2
        W = randomLeaderV/randomLeaderR*(2*int(np.random.random() > 0.5)-1)
        print "V: {}\tW: {}".format(randomLeaderV, W)
      t += 1
      env.stepLeader([randomLeaderV, W]) # v,w generated by some tarjectory
      f_actions = {_id:agent.getFinalAction(eval) \
        for _id, agent in env.agents.items() if _id != env.LEADER}
      for experience in env.step(f_actions).values():
        if not eval: model.replaybuffer.add(*experience)
        print experience[2], experience[-1]
        done = experience[-1]
        if done:
          print "episode {} t {}".format(episode, t)
          break
        r += experience[2]
    episode += 1
    if eval:
      cumulative_reward += r
      continue
    for agent in env.agents.values():
      for e in agent.edge_agents:
        [e.model.train_models() for x in xrange(num_epochs)]
    if ((episode-1) % eval_interval) == 0:
      eval_score = runExperiment(env, num_evals, eval_eps_len, model, eval=True)
      msg = "EVAL SCORE [EPISODE {}]: {}".format(episode-1, eval_score)
      writeStats(result_path+'/avg_scores.txt', msg)
      print msg
      if best_score < eval_score:
        saveModel(model, episode-1, result_path)
        best_score = eval_score
  if eval: return cumulative_reward

if __name__ == '__main__':
  import HyperParams as HP
  # print "REWARDMAX ", HP.REWARD_MAX
  tmodel = DDPG(train=False)
  visualOptions = { 'tailLength' : 4, # tail of agent's trajectory
                    'speedup' : 100, # realTime/simTime
                    'bounds': [-10,10,-10,10]# bounds of the environment [xmin, xmax, ymin, ymax]
                    }
  targetshape = genTargetShape(genRandomPoints())
  env = FormationEnvironment(targetshape, num_iterations=50, visualize=False, visualOptions=visualOptions)
  for i in env.agents.values():
    i.initEdgeAgentModel(tmodel)
  # env.startVisualiser()
  runExperiment(env, 40000, 30, tmodel, num_epochs=10, eval_interval=50, num_evals=10, eval_eps_len=50, result_path="results/run1")
