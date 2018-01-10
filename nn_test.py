from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from nn import NeuralNetwork, NeuralNetworkException


def main(hidden_size, learning_rate):
    try:

        X, y = load_wine(return_X_y=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        y_train = y_train.reshape((y_train.shape[0],1))
        y_test = y_test.reshape((y_test.shape[0],1))

        # set input size to the number of features
        input_size = X_train.shape[1]
        # set number of labels
        num_labels = len(set(y_train.flatten()))

        the_neural_net = NeuralNetwork(input_size, hidden_size, num_labels, learning_rate)
        theta1, theta2 = the_neural_net.train(X_train, y_train)
        the_neural_net.get_prediction_accuracy(X_test, y_test, theta1, theta2)

    except NeuralNetworkException as ex:
        print 'ERROR :: {0}'.format(ex.message)


if __name__ == '__main__':

    hidden_size = 20
    learning_rate = 0.5

    main(hidden_size, learning_rate)
