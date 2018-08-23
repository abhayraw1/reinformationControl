import numpy as np
import math
from keras import initializers
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, Concatenate, Lambda
from keras.initializers import RandomNormal
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
import HyperParams as HP

class ActorNetwork(object):
  def __init__(self, sess, state_size, action_size, tau, lr, target):
    self.sess = sess
    self.tau = tau
    self.lr = lr
    K.set_session(sess)
    self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)
    if target:
      self.target_model, self.target_weights, self.target_state = target.model , target.weights, target.state
    self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
    self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
    grads = zip(self.params_grad, self.weights)
    self.optimize = tf.train.AdamOptimizer(lr).apply_gradients(grads)
    self.sess.run(tf.global_variables_initializer())

  def train(self, states, action_grads):
    self.sess.run(self.optimize, feed_dict={
      self.state: states,
      self.action_gradient: action_grads
    })

  def target_train(self):
    actor_weights = self.model.get_weights()
    actor_target_weights = self.target_model.get_weights()
    for i in xrange(len(actor_weights)):
      actor_target_weights[i] = self.tau * actor_weights[i] + (1 - self.tau)* actor_target_weights[i]
    self.target_model.set_weights(actor_target_weights)

  def copy_from_target(self):
    # actor_weights = self.model.get_weights()
    actor_target_weights = self.target_model.get_weights()
    # for i in xrange(len(actor_weights)):
    #   actor_weights[i] = np.copy(actor_target_weights[i])
    self.model.set_weights(actor_target_weights)

  def create_actor_network(self, state_size,action_dim):
    print("Building Actor Model")
    S = Input(shape=[state_size])
    h0 = Dense(HP.ACTOR_N_NEURONS_L0, activation='relu')(S)
    h1 = Dense(HP.ACTOR_N_NEURONS_L1, activation='relu')(h0)
    v = Dense(1,activation='sigmoid',kernel_initializer=RandomNormal())(h1)
    w = Dense(1,activation='tanh',kernel_initializer=RandomNormal())(h1)
    V = Concatenate()([v,w])
    model = Model(inputs=S,outputs=V)
    return model, model.trainable_weights, S
