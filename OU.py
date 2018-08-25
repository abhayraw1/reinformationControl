import random
import numpy as np

class OU:
  def __init__(self):
    pass

  @staticmethod
  def ou(x, mu, theta, sigma):
    x = np.array(x)
    return theta * (mu - x) + sigma * np.random.randn(x.shape[1])
