import math

from keras.datasets import mnist
from matplotlib import pyplot
import tensorflow as tf
import numpy as np



class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Already transposed Multiplied by 0.1 to further decrease values of the distribution
        self.output = None
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases




    def updateparameters(self, dW, db):
        learning_rate = 1
        self.weights = self.weights - learning_rate * dW
        self.biases = self.biases - learning_rate * db

class Activation_ReLU:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0



class Soft_Max():
    def forward(self, inputs):
        # Need to make sure exp doesn't make an overflow so will regulate it, will subtract max value.
        # Actual output will be exactly the same

        #THESE ARE COLUMN WISE
        inputs = inputs - np.max(inputs, axis=1, keepdims=True)
        exp_values = np.exp(inputs)
        self.probabilities = exp_values / (np.sum(exp_values, axis=1, keepdims=True))

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)

            jacobian_matrix = np.diaflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)

        data_loss = np.mean(sample_losses)

        return data_loss

class Loss_CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_probabilities = -np.log(correct_confidences)
        return negative_log_probabilities

def RELUbackward(dvalues):
    dvalues[dvalues <= 0] = 0
    return dvalues
def backward( X, Y, Y_new, Z1, A1, Z2, A2, W2):
    m = Y.shape[0]
    dz2 = A2 - Y_new
    dw2 = 1/m * np.dot( A1.T, dz2)
    db2 = 1/m * np.sum(dz2, axis = 0, keepdims=True)
    #print(dz2)
   # print(np.sum(dz2, axis = 0, keepdims=True))
    #print()

    dA1 = np.dot(dz2,  W2.T)
    dz1 = dA1 * RELUbackward(Z1)
    dw1 =  (1/m) * np.dot( X.T, dz1 )
    db1 = 1/m * np.sum(dz1, axis=0, keepdims=True)
    #print(dz1)

    return dw1, db1, dw2, db2

class NeuralNet():
    def __init__(self, digits=10):
        self.digits = digits
        self.training_rate = 1
        self.dense1 = Layer_Dense(784, 128)
        self.dense2 = Layer_Dense(128, 10)
        self.relu1 = Activation_ReLU()
        self.softmax = Soft_Max()
        self.loss = Loss_CategoricalCrossEntropy()

        pass
    def train(self):
        (train_X, train_y), (test_X, test_y) = mnist.load_data()
        train_X = np.array(train_X)
        train_y = np.array(train_y)

        train_X = train_X / 255
        train_X = train_X.reshape(-1, 784)
        X = train_X[0:60000]
        Y = train_y[0:60000]


        Y_new = np.zeros((Y.size, self.digits))
        Y_new[np.arange(Y.size), Y] = 1
        # print(np.sum(Y_new, axis=0))
        # print(Y)

        for i in range(2000):
            self.dense1.forward(X)
            self.relu1.forward(self.dense1.output)
            self.dense2.forward(self.relu1.output)
            self.softmax.forward(self.dense2.output)
            Cost = self.loss.calculate(self.softmax.probabilities, Y_new)

            dw1, db1, dw2, db2 = backward(X, Y, Y_new, self.dense1.output, self.relu1.output, self.dense2.output,
                                          self.softmax.probabilities, self.dense2.weights)
            self.dense1.weights = self.dense1.weights - self.training_rate * dw1
            self.dense1.biases = self.dense1.biases - self.training_rate * db1
            self.dense2.weights = self.dense2.weights - self.training_rate * dw2
            self.dense2.biases = self.dense2.biases - self.training_rate * db2

            if (i % 100 == 0):
                print("Epoch", i, "cost: ", Cost)

        self.test(test_X, test_y)
        pass
    def save(self):
        np.savez("NeuralNetworkInformation.npz", w1 = self.dense1.weights, b1 = self.dense1.biases,
                 w2 = self.dense2.weights,b2 = self.dense2.biases)
        pass
    def load(self):
        arr = np.load("NeuralNetworkInformation.npz")
        self.dense1.weights = arr['w1']
        self.dense1.biases = arr['b1']
        self.dense2.weights = arr['w2']
        self.dense2.biases = arr['b2']
        pass
    def test(self, test_X, test_y):
        # TEST
        test_X = test_X / 255
        test_X = test_X.reshape(-1, 784)

        Y_newTest = np.zeros((test_y.size, self.digits))
        Y_newTest[np.arange(test_y.size), test_y] = 1

        self.dense1.forward(test_X)
        self.relu1.forward(self.dense1.output)
        self.dense2.forward(self.relu1.output)
        self.softmax.forward(self.dense2.output)
        Cost = self.loss.calculate(self.softmax.probabilities, Y_newTest)

        print()
        print(self.softmax.probabilities[0] * 100)
        print("Highest probability is ", np.max(self.softmax.probabilities[0]) * 100, " neural network's guess is a ",
              np.argmax(self.softmax.probabilities[0]))
        print("Actual digit: ", test_y[0])
        print("Final Test COST: ", Cost)
        pass

    def drawTest(self, test_X):
        # TEST
        test_X = test_X / 255
        test_X = test_X.reshape(-1, 784)

        self.dense1.forward(test_X)
        self.relu1.forward(self.dense1.output)
        self.dense2.forward(self.relu1.output)
        self.softmax.forward(self.dense2.output)

        #print()
        #print(self.softmax.probabilities[0] * 100)
        #print("Highest probability is ", np.max(self.softmax.probabilities[0]) * 100, " neural network's guess is a ",
           #   np.argmax(self.softmax.probabilities[0]))
        return self.softmax.probabilities


#NN = NeuralNet()
#
#NN.save()


