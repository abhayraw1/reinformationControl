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
  print pts
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

def writeStates(filename, stat):
  with open(filename, 'a+') as f:
    f.write(stat)
    f.write('\n')

def runExperiment(env, num_eps, eps_len, model, num_epocs=None, eval=False, \
                  num_evals=10, eval_interval=50, eval_eps_len=None, result_path=''):
  print "hola!"
  episode     = 1
  done        = False
  best_score  = 0
  if eval: cumulative_reward = 0
  while episode <= num_eps:
    print "episode ", episode
    t = 1
    r = 0
    env.targetshape = genTargetShape(seed=19)
    env.reset() # random starting positions??
    while t <= eps_len:
      print 't ', t
      env.stepLeader([0., 0.]) # v,w generated by some tarjectory
      f_actions = {agent.id:agent.getFinalAction(eval) for agent in env.agents.values()}
      for experience in env.step(f_actions):
        if not eval: model.replaybuffer.remember(*experience)
        r += experience[2]
    if eval:
      cumulative_reward += r
      continue
    for agent in env.agents.values():
      for e in agent.edge_agents:
        [e.model.train_models() for x in xrange(num_epocs)]
    if (episode % eval_interval) == 0:
      eval_score = runExperiment(env, num_evals, eval_eps_len, model, eval=True)
      if best_score < eval_score:
        saveModel(model, episode, path)
        best_score = eval_score
  if eval: return cumulative_reward/num_eps

if __name__ == '__main__':
  tmodel = DDPG(train=False)
  visualOptions = { 'tailLength' : 4, # tail of agent's trajectory
                    'speedup' : 1, # realTime/simTime
                    'bounds': [-10,10,-10,10]# bounds of the environment [xmin, xmax, ymin, ymax]
                    }
  targetshape = genTargetShape(genRandomPoints())
  env = FormationEnvironment(targetshape, num_iterations=50, visualize=True, visualOptions=visualOptions)
  for i in env.agents.values():
    i.initEdgeAgentModel(tmodel)
  env.startVisualiser()
  runExperiment(env, 1, 5, tmodel, 2)
