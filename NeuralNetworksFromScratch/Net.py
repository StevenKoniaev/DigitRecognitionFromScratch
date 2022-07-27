import math

from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import matplotlib
from matplotlib import pyplot

# Loading in training data

(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = np.array(train_X)
train_y = np.array(train_y)

