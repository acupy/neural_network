import numpy as np
from scipy.optimize import minimize

from utils import sigmoid, sigmoid_gradient, encode_labels


class NeuralNetwork(object):
    def __init__(self, input_size, hidden_size, num_labels, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.learning_rate = learning_rate

    @classmethod
    def feed_forward(cls, X, theta1, theta2):
        m = X.shape[0]

        a1 = np.insert(X, 0, values=np.ones(m), axis=1)
        z2 = a1 * theta1.T
        a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
        z3 = a2 * theta2.T
        h = sigmoid(z3)

        return a1, z2, a2, z3, h

    def backpropagate(self, params, X, y):
        m = X.shape[0]
        X = np.matrix(X)
        y = np.matrix(y)

        theta1 = self.__unravel_theta1(params)
        theta2 = self.__unravel_theta2(params)

        # feed forward
        a1, z2, a2, z3, h = NeuralNetwork.feed_forward(X, theta1, theta2)

        J = 0

        # compute the cost
        for i in range(m):
            first_term = np.multiply(-y[i,:], np.log(h[i,:]))
            second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
            J += np.sum(first_term - second_term)

        J = J / m

        # cost regularization term
        J += (float(self.learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))

        # backpropagation

        delta1 = np.zeros(theta1.shape)
        delta2 = np.zeros(theta2.shape)

        for t in range(m):
            a1t = a1[t,:]
            z2t = z2[t,:]
            a2t = a2[t,:]
            ht = h[t,:]
            yt = y[t,:]

            d3t = ht - yt

            z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
            d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26)

            delta1 = delta1 + (d2t[:,1:]).T * a1t
            delta2 = delta2 + d3t.T * a2t

        delta1 = delta1 / m
        delta2 = delta2 / m

        # add the gradient regularization term
        delta1[:,1:] = delta1[:,1:] + (theta1[:,1:] * self.learning_rate) / m
        delta2[:,1:] = delta2[:,1:] + (theta2[:,1:] * self.learning_rate) / m

        # unravel the gradient matrices into a single array
        grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

        return J, grad

    def train(self, X, y):
        X = np.matrix(X)
        y_vectorized = encode_labels(y)
        params = self.__random_initialize_weights()

        # minimize the objective function
        fmin = minimize(fun=self.backpropagate, x0=params, args=(X, y_vectorized),
                        method='TNC', jac=True, options={'maxiter': 250})

        theta1 = self.__unravel_theta1(fmin.x)
        theta2 = self.__unravel_theta2(fmin.x)
        return theta1, theta2

    def get_prediction_accuracy(self, X, y, theta1, theta2):
            a1, z2, a2, z3, h = NeuralNetwork.feed_forward(X, theta1, theta2)
            y_pred = np.array(np.argmax(h, axis=1) + 1)

            correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
            accuracy = (sum(map(int, correct)) / float(len(correct)))
            print 'Accuracy: {0}%'.format(accuracy * 100)

    def __random_initialize_weights(self):
        epsilon_init = 0.25
        return (np.random.random(size=self.hidden_size * (self.input_size + 1) + self.num_labels * (self.hidden_size + 1)) - (2 * epsilon_init)) * epsilon_init

    def __unravel_theta1(self, params):
        return np.matrix(np.reshape(params[:self.hidden_size * (self.input_size + 1)], (self.hidden_size, (self.input_size + 1))))

    def __unravel_theta2(self, params):
        return np.matrix(np.reshape(params[self.hidden_size * (self.input_size + 1):], (self.num_labels, (self.hidden_size + 1))))


class NeuralNetworkException(Exception):
    pass
