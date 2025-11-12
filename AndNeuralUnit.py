import numpy as np


class AndNeuralUnit:
    """A neural functional unit implementing AND computation."""

    def __init__(self):
        self.weights = np.array([[0.5, 0.5]])  # 1x2 weight matrix

    def relu(self, x):
        """ReLU activation function."""
        return max(0, x)

    def forward(self, input_array):
        """Forward propagation."""
        sum_result = np.sum(input_array * self.weights[0])
        return self.relu(sum_result)

    def predict(self, input_array):
        """Predict output based on threshold."""
        return 1.0 if self.forward(input_array) >= 1 else 0.0