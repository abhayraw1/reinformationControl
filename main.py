from RLAgent import RLAgent
from Shape import ShapeByGeometry
from FormationEnvironment import FormationEnvironment

############### CONSTANTS ###############
NUM_AGENTS = 2
#########################################


######## TARGET SHAPE DEFINITION ########
geometry = {'coordinates': [[0,0], [2,0]],
            'orientations':[np.pi/2]*2 }

targetshape = ShapeByGeometry(geometry)
#########################################


############### AGENTS
agents = [RLAgent(i, train=True) for i in range(2)]

############### ENVIRONMENT
env = FormationEnvironment(targetshape, agents)

############### PARTIALLY OBSERVED ENVS
agent_observed_envs = {}
for agent in env.agents:
  agent_observed_envs[i.id] = AgentObservedEnvironment(env, agent)

main_model = DDPG()

for i, j in env.agents.keys():
  controller = DDPG()
  env.agents[i].initEdgeControllers(j, controller)
