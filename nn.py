import numpy as np
class NeuralNetwork(object):
    def __init__(self, structure=None, thetas=None):
        """
        :param structure: array of number of activation units per layer
        :param thetas: array of matrices
        """
        if structure == None and thetas == None:
            raise NeuralNetworkException('At least one of the two has to be' \
                                         'specified: structure or thetas')

        if structure and (not isinstance(structure, list) or len(structure) < 3):
            raise NeuralNetworkException('The neural net has to have ' \
                                         'at least 3 layers')
        self.structure = structure
        self.thetas = thetas if thetas else self.__random_initialize_weights()

        # z = np.array([[0,100,-100],[4,3,2],[-2,-3,-4]])
        # print 'sigmoid', NeuralNetwork.__sigmoid(z)

    def train():
        pass

    def predict():
        pass

    def __cost_function():
        pass

    def __random_initialize_weights(self):
        pass

    @classmethod
    def __sigmoid(cls, z):
        return 1.0 / (1.0 + np.exp(-z));

    def __sigmoid_gradient():
        pass

class NeuralNetworkException(Exception):
    pass
