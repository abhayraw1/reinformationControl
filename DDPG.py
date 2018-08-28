import numpy as np

import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam

from OU import OU
import HyperParams as HP
from Actor import ActorNetwork
from Critic import CriticNetwork
from ReplayBuffer import ReplayBuffer

class DDPG:
  @staticmethod
  def actorParams(sess, target):
    return {'sess': sess,
            'state_size':HP.STATE_DIM,
            'action_size':HP.ACTION_DIM,
            'tau':HP.ACTOR_TAU,
            'lr':HP.ACTOR_LR,
            'target': target}

  @staticmethod
  def criticParams(sess, target):
    return {'sess': sess,
            'state_size':HP.STATE_DIM,
            'action_size':HP.ACTION_DIM,
            'tau':HP.CRITIC_TAU,
            'lr':HP.CRITIC_LR,
            'target': target}

  def __init__(self, critic, target_model=None, train=True, replaybuffer=ReplayBuffer(HP.BUFFER_SIZE)):
    assert isinstance(target_model, DDPG) or target_model == None
    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    K.set_session(sess)
    target_actor = target_model.actor if target_model != None else None
    target_critic = target_model.critic if target_model != None else None
    self.actor = ActorNetwork(**DDPG.actorParams(sess, target_actor))
    self.critic = CriticNetwork(**DDPG.criticParams(sess, target_critic)) if target_model == None else critic
    self.target_model = target_model
    if target_model != None:
      self.replaybuffer = target_model.replaybuffer
    else:
      self.replaybuffer = replaybuffer
    self.train = train
    self.epsilon = 1

  def act(self, obs):
    obs = np.array(obs)
    action = self.actor.model.predict(obs.reshape(1,HP.STATE_DIM))
    action = action * (HP.MAX_ACTION - HP.MIN_ACTION) + HP.MIN_ACTION
    if self.train and self.epsilon > 0:
      action = self.addOU(action)
    return action

  def addOU(self, action):
    return action + OU.ou(action, HP.OU_MEAN, HP.OU_THETA, HP.OU_SIGMA)

  def train_models(self):
    batch = self.replaybuffer.getBatch(HP.BATCH_SIZE)
    experiences = [np.asarray([i[j] for i in batch]) for j in range(5)]
    states, actions, rewards, nstates, dones = experiences
    target_q = self.compute_target_q(nstates, rewards, dones)
    loss = self.critic.model.train_on_batch([states, actions], target_q)
    print "loss: ", loss, self.replaybuffer.count()
    a_for_grad = self.actor.model.predict(states)
    grads = self.critic.gradients(states, a_for_grad)
    self.actor.train(states, grads)
    self.actor.target_train()
    self.critic.target_train()

  def remember(self, obs, action, reward, next_obs, done):
    self.replaybuffer.add(obs, action, reward, next_obs, done)

  def compute_target_q(self, nstates, rewards, dones):
    target_q_values = self.critic.target_model.predict([nstates, \
                      self.actor.target_model.predict(nstates)])
    y_t = np.zeros(len(nstates))
    for idx, reward in enumerate(rewards):
      y_t[idx] = reward
      # print y_t[idx], reward, dones[idx]
      if not dones[idx]:
        y_t[idx] += HP.GAMMA*target_q_values[idx]
    # print y_t
    return y_t

  def copy_from_target(self):
    self.actor.copy_from_target()
    self.critic.copy_from_target()

  def save(self, location, epoch):
    self.actor.model.save(location+'/actor_model_{}.h5'.format(epoch))
    self.critic.model.save(location+'/critic_model_{}.h5'.format(epoch))

