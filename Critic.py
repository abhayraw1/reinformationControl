import numpy as np
import math
from keras.models import model_from_json, load_model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
from keras.utils import plot_model
import HyperParams as HP

class CriticNetwork(object):
  def __init__(self, sess, state_size, action_size, tau, lr, target):
    self.sess = sess
    self.tau = tau
    self.lr = lr
    self.action_size = action_size
    K.set_session(sess)
    self.model, self.action, self.state = self.create_critic_network(state_size, action_size)
    if target:
        self.target_model, self.target_action, self.target_state = target.model, target.action, target.state
    self.action_grads = tf.gradients(self.model.output, self.action) #GRADIENTS for policy update
    self.sess.run(tf.initialize_all_variables())

  def gradients(self, states, actions):
    return self.sess.run(self.action_grads, feed_dict={
      self.state: states,
      self.action: actions
    })[0]

  def target_train(self):
    critic_weights = self.model.get_weights()
    critic_target_weights = self.target_model.get_weights()
    for i in xrange(len(critic_weights)):
      critic_target_weights[i] = self.tau * critic_weights[i] + (1 - self.tau)* critic_target_weights[i]
    self.target_model.set_weights(critic_target_weights)

  def copy_from_target(self):
    # critic_weights = self.model.get_weights()
    critic_target_weights = self.target_model.get_weights()
    # for i in xrange(len(critic_weights)):
    #   critic_weights[i] = np.copy(critic_target_weights[i])
    self.model.set_weights(critic_target_weights)

  def create_critic_network(self, state_size,action_dim):
    print("Now we build the model")
    S = Input(shape=[state_size])
    A = Input(shape=[action_dim],name='action2')
    w1 = Dense(HP.CRITIC_N_NEURONS_L0, activation='relu')(S)
    a1 = Dense(HP.CRITIC_N_NEURONS_L1, activation='linear')(A)
    h1 = Dense(HP.CRITIC_N_NEURONS_L1, activation='linear')(w1)
    h2 = merge([h1,a1],mode='sum')
    h3 = Dense(HP.CRITIC_N_NEURONS_L1, activation='relu')(h2)
    V = Dense(1,activation='linear')(h3)
    model = Model(input=[S,A],output=V)
    adam = Adam(lr=self.lr)
    model.compile(loss='mse', optimizer=adam)
    return model, A, S
