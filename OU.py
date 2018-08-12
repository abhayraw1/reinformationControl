import random
import numpy as np

class OU(object):
  def __call__(self, x, mu, theta, sigma):
    return theta * (mu - x) + sigma * np.random.randn(1)
