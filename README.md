## Neural Network

Find a simple example below. Make sure to write the code that loads the training data and set your `input_size`, `hidden_size` and `num_labels` accordingly.

```python
from nn import NeuralNetwork

input_size = 500
hidden_size = 30
num_labels = 4
learning_rate = 1

# Load training data
X, y = ... # TODO !!!

# Initialize Neural Network
the_neural_net = NeuralNetwork(input_size, hidden_size, num_labels, learning_rate)

# Find thetas
theta1, theta2 = the_neural_net.train(X, y)

# Test prediction accuracy
the_neural_net.get_prediction_accuracy(X, y, theta1, theta2)
```
