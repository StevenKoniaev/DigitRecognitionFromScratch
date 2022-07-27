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

def reLU(z):
    return np.maxiumum(z, 0)


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def d_sigmoid(z):
    return (np.exp(-z)) / ((np.exp(-z) + 1) ** 2)


def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def d_softmax(x):
    exp_element = np.exp(x - x.max())
    return exp_element / np.sum(exp_element, axis=0) * (1 - exp_element / np.sum(exp_element, axis=0))


def computer_cross_entropy_loss(Y, Y_hat):
    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1 / m) * L_sum
    return L


train_X = train_X / 255

batch_size = 1000
m = batch_size
batch_start = 0
batch_end = 1000
# batch_end += batch_size
# batch_start += batch_size
X = train_X[batch_start:batch_end]
X = X.reshape(batch_size, -1)
Y = train_y[batch_start:batch_end]

Y = np.array(Y)
X = np.array(X)
learning_rate = 1

n_neurons = 128
n_weights = 784

digits = 10

Y = Y.reshape(1, -1)

Y_new = np.zeros((Y.size, digits))
Y_new[np.arange(Y.size), Y] = 1

for i in range(1):
    # Horizontally hot encoded vectors
    # Already in .T
    W1 = np.random.rand(n_weights, n_neurons) * 0.1
    b1 = np.zeros((1, n_neurons))

    W2 = np.random.rand(n_neurons, digits) * 0.1
    b2 = np.zeros((1, digits))

    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)

    Z2 = np.dot(A1, W2) + b2

    A2 = softmax(Z2)


    cost = computer_cross_entropy_loss(Y_new, A2)


    dZ2 = (A2) - Y_new
    dW2 = (1. / m) * np.dot(dZ2.T, A1)
    db2 = (1. / m) * np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(W2, dZ2.T  )
    print(dA1.shape, " ", Z1.shape)
    dZ1 = dA1.T * sigmoid(Z1) * (1 - sigmoid(Z1))
    dW1 = (1. / m) * np.dot(X.T, dZ1)
    db1 = (1. / m) * np.sum(dZ1, axis=0, keepdims=True)

    W2 = W2 - learning_rate * dW2.T
    b2 = b2 - learning_rate * db2
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    batch_start += batch_size
    batch_end += batch_size
    print("Cost " ,cost)
  #  if (i % 100 == 0):
   #     print("Epoch", i, "cost: ", cost)

print("Final cost: ", cost)
