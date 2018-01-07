from nn import NeuralNetwork, NeuralNetworkException
import numpy as np

def main():
    try:
        # 200*200 images, 25 units in the hidden layer, 2 classes
        nn = NeuralNetwork(structure=[40000,25,2])
    except NeuralNetworkException as ex:
        print 'ERROR :: {0}'.format(ex.message)

    print 'Neural Nets are awesome!!!'

if __name__ == '__main__':
    main()
